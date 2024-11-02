pub mod algorithm;
pub mod framework;
pub mod state;

use crate::gpu::algorithm::PosResults;
use crate::gpu::state::WGPUState;
use crate::TILE_SIZE;
use algorithm::Algo;
use bytemuck::Zeroable;
use framework::*;
use image::{DynamicImage, RgbaImage};
use rustc_hash::FxHashSet;
use std::path::MAIN_SEPARATOR;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use walkdir::DirEntry;
use wgpu::{ImageCopyTexture, Maintain, TextureFormat};

#[derive(Default, Copy, Clone)]
struct GPUData {
    total1: f32,
    total2: f32,
}

unsafe impl Zeroable for GPUData {}
unsafe impl bytemuck::Pod for GPUData {}

#[derive(Clone)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub data: Arc<Vec<u8>>,
    pub smol_data: Arc<Vec<u8>>,
}

impl Tile {
    pub fn pos(&self) -> (u32, u32, u32) {
        (self.x, self.y, self.z)
    }
}

pub struct State {
    wgpu: WGPUState<GPUData>,
    algo: Algo,
    n_masks: usize,
    tiles: Vec<Tile>,
    tile_texs: Vec<GPUTexture>,
}

pub const TILE_CHUNK_SIZE: usize = 50;

impl State {
    pub async fn new(tile_size: (u32, u32), mask_size: (u32, u32), n_masks: usize) -> State {
        let wgpu = WGPUState::new().await;
        let device = &wgpu.device;

        let algo = Algo::new(device, tile_size, mask_size, n_masks);

        Self {
            wgpu,
            algo,
            n_masks,
            tiles: Default::default(),
            tile_texs: vec![],
        }
    }

    pub fn prepare(&mut self, tile_paths: &[DirEntry]) {
        use rayon::prelude::*;
        eprintln!("Reading {} tiles data from disk", tile_paths.len());
        self.tiles = tile_paths
            .par_iter()
            .map(|entry| {
                let path = entry.path();

                let tile_image_data =
                    std::fs::read(path).expect("could not read preprocessed data");

                let tile_smol_path = path
                    .display()
                    .to_string()
                    .replace("tiles_grad", "tiles_smol");
                let tile_smol_data =
                    std::fs::read(tile_smol_path).expect("could not read preprocessed data");

                let path_str = path.to_string_lossy();

                let parts = path_str.split(MAIN_SEPARATOR).collect::<Vec<_>>();
                let x = parts[parts.len() - 1]
                    .split_once(".")
                    .unwrap()
                    .0
                    .parse::<u32>()
                    .unwrap();
                let y = parts[parts.len() - 2].parse::<u32>().unwrap();
                let z = parts[parts.len() - 3].parse::<u32>().unwrap();

                Tile {
                    x,
                    y,
                    z,
                    data: Arc::new(tile_image_data),
                    smol_data: Arc::new(tile_smol_data),
                }
            })
            .collect::<Vec<_>>();

        eprintln!("preparing tile textures");
        self.tile_texs = (0..TILE_CHUNK_SIZE)
            .map(|_| {
                mk_tex_general(
                    &self.wgpu.device,
                    (TILE_SIZE, TILE_SIZE),
                    TextureFormat::Rgba8Unorm,
                    1,
                    3,
                )
            })
            .collect::<Vec<_>>();

        #[cfg(not(windows))]
        const GPU_RESULTS_FOLDER: &str = "data/results/gpu";

        #[cfg(windows)]
        const GPU_RESULTS_FOLDER: &str = "data\\results\\gpu";

        let _ = std::fs::remove_dir_all(GPU_RESULTS_FOLDER);
        std::fs::create_dir_all(GPU_RESULTS_FOLDER).unwrap();
    }

    pub fn run_on_image(
        &mut self,
        masks: &[(&RgbaImage, u32)],
        forbidden_tiles: &FxHashSet<(u32, u32, u32)>,
    ) -> (Vec<(u32, PosResults)>, std::time::Duration) {
        if masks.len() != self.n_masks {
            panic!("Expected {} masks, got {}", self.n_masks, masks.len());
        }
        let t_start = std::time::Instant::now();

        self.algo
            .best_pos
            .lock()
            .unwrap()
            .iter_mut()
            .for_each(PosResults::clear);

        let (mask_w, mask_h) = (masks[0].0.width(), masks[0].0.height());
        let mask_texs = (0..masks.len())
            .map(|_| {
                mk_tex_general(
                    &self.wgpu.device,
                    (mask_w, mask_h),
                    TextureFormat::Rgba8Unorm,
                    1,
                    2,
                )
            })
            .collect::<Vec<_>>();

        let mut mask_idx = Vec::with_capacity(masks.len());

        for ((mask, mask_i), mask_tex) in masks.iter().zip(mask_texs.iter()) {
            let copy = DynamicImage::ImageRgba8(RgbaImage::clone(mask));

            let mip = copy.resize(
                mask_tex.texture.size().width / 2,
                mask_tex.texture.size().height / 2,
                image::imageops::FilterType::Gaussian,
            );
            let pixels_mip = mip.to_rgba8();

            self.wgpu.queue.write_texture(
                mask_tex.texture.as_image_copy(),
                mask,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        mask_tex.texture.width() * mask_tex.format.block_copy_size(None).unwrap(),
                    ),
                    rows_per_image: Some(mask_tex.texture.height()),
                },
                mask_tex.texture.size(),
            );

            self.wgpu.queue.write_texture(
                ImageCopyTexture {
                    mip_level: 1,
                    ..mask_tex.texture.as_image_copy()
                },
                &pixels_mip,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        mask_tex.texture.width() / 2
                            * mask_tex.format.block_copy_size(None).unwrap(),
                    ),
                    rows_per_image: Some(mask_tex.texture.height() / 2),
                },
                wgpu::Extent3d {
                    width: mask_tex.texture.size().width / 2,
                    height: mask_tex.texture.size().height / 2,
                    depth_or_array_layers: 1,
                },
            );

            mask_idx.push(*mask_i);
        }

        WGPUState::modify_user_data(&self.wgpu.queue, &self.wgpu.user_data, &|u| {
            u.total1 = 0.0;
            u.total2 = 0.0;
        });

        let filtered_tiles = Arc::new(
            self.tiles
                .iter()
                .cloned()
                .filter(|tile| !forbidden_tiles.contains(&tile.pos()))
                .collect::<Vec<_>>(),
        );

        // image decoding thread
        let filtered_tiles_2 = filtered_tiles.clone();
        let (decoded_tiles_tx, decoded_tiles_rx) =
            crossbeam_channel::bounded::<Vec<(Vec<u8>, Vec<u8>)>>(10);
        rayon::spawn(move || {
            use rayon::prelude::*;

            for data_chunk in filtered_tiles_2.chunks(TILE_CHUNK_SIZE) {
                let decoded_chunk = data_chunk
                    .par_iter()
                    .map(|entry| {
                        let pixel_smol = image::load_from_memory(&entry.smol_data)
                            .expect("could not decode pixels");

                        let pixel_data =
                            image::load_from_memory(&entry.data).expect("could not decode pixels");

                        /*
                        let mut pixels_rg =
                            Vec::with_capacity(TILE_SIZE as usize * TILE_SIZE as usize * 2);

                        for pixel in pixel_data.to_rgb8().chunks_exact(3) {
                            pixels_rg.push(pixel[0]);
                            pixels_rg.push(pixel[1]);
                        }*/

                        (
                            pixel_data.to_rgba8().into_raw(),
                            pixel_smol.to_rgba8().into_raw(),
                        )
                    })
                    .collect();
                decoded_tiles_tx.send(decoded_chunk).unwrap();
            }
        });

        let mut result_bufs_waits = Vec::new();

        for tile_chunk in filtered_tiles.chunks(TILE_CHUNK_SIZE) {
            use rayon::prelude::*;
            let tile_texs = &self.tile_texs[..tile_chunk.len()];
            let decoded_tiles = decoded_tiles_rx.recv().unwrap();
            decoded_tiles.par_iter().zip(tile_texs.par_iter()).for_each(
                |((entry, smol), tile_tex)| {
                    self.wgpu.queue.write_texture(
                        tile_tex.texture.as_image_copy(),
                        &entry,
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(
                                tile_tex.texture.width()
                                    * tile_tex.format.block_copy_size(None).unwrap(),
                            ),
                            rows_per_image: Some(tile_tex.texture.height()),
                        },
                        tile_tex.texture.size(),
                    );

                    self.wgpu.queue.write_texture(
                        ImageCopyTexture {
                            mip_level: 2,
                            ..tile_tex.texture.as_image_copy()
                        },
                        &smol,
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(
                                tile_tex.texture.width() / 4
                                    * tile_tex.format.block_copy_size(None).unwrap(),
                            ),
                            rows_per_image: Some(tile_tex.texture.height() / 4),
                        },
                        wgpu::Extent3d {
                            width: tile_tex.texture.size().width / 4,
                            height: tile_tex.texture.size().height / 4,
                            depth_or_array_layers: 1,
                        },
                    );
                },
            );
            drop(decoded_tiles);

            let mut encoder = self
                .wgpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let pass_encoder = PassEncoder::new(&self.wgpu.device, &mut encoder, &self.wgpu.uni_bg);

            (self.algo.render_frame)(pass_encoder, &mask_texs, tile_texs);
            self.wgpu.queue.submit(Some(encoder.finish()));

            result_bufs_waits.push((self.algo.after_render)(
                &tile_chunk,
                &self.wgpu.device,
                &self.wgpu.queue,
            ));
            let to_wait = 10;
            if result_bufs_waits.len() >= to_wait {
                let to_wait: &Arc<AtomicU32> =
                    &result_bufs_waits[result_bufs_waits.len() - to_wait];
                while to_wait.load(Ordering::SeqCst) > 0 {
                    self.wgpu.device.poll(Maintain::Poll);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }

        for wait in result_bufs_waits {
            while wait.load(Ordering::SeqCst) > 0 {
                self.wgpu.device.poll(Maintain::Poll);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        let best_pos = self.algo.best_pos.lock().unwrap().clone();

        (
            mask_idx.into_iter().zip(best_pos.into_iter()).collect(),
            t_start.elapsed(),
        )
    }
}
