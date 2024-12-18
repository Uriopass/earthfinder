pub mod algorithm;
pub mod framework;
pub mod state;

use crate::data::{deform_width, extract_tile_pos, TilePos};
use crate::gpu::algorithm::{AlgoResult, TILE_CHUNK_SIZE};
use crate::gpu::state::WGPUState;
use crate::TILE_HEIGHT;
use algorithm::Algo;
use bytemuck::Zeroable;
use framework::*;
use image::RgbaImage;
use rustc_hash::FxHashSet;
use std::sync::Arc;
use walkdir::DirEntry;
use wgpu::{ImageCopyTexture, TextureFormat};

#[derive(Default, Copy, Clone)]
pub struct GPUData {
    pub total1: f32,
    pub total2: f32,
}

unsafe impl Zeroable for GPUData {}
unsafe impl bytemuck::Pod for GPUData {}

#[derive(Clone)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub width: u32,
    pub data: Arc<Vec<u8>>,
    pub smol_data: Arc<Vec<u8>>,
}

impl Tile {
    pub fn pos(&self) -> TilePos {
        (self.x, self.y, self.z)
    }
}

pub struct State {
    wgpu: WGPUState<GPUData>,
    algo: Algo,
    n_masks: usize,
    tiles: Vec<Tile>,
}

impl State {
    pub async fn new(mask_size: (u32, u32), n_masks: usize, n_extra_positions: usize) -> State {
        let wgpu = WGPUState::new().await;
        let device = &wgpu.device;

        let algo = Algo::new(device.clone(), mask_size, n_masks, n_extra_positions);

        Self {
            wgpu,
            algo,
            n_masks,
            tiles: Default::default(),
        }
    }

    pub fn prepare(&mut self, tile_paths: &[DirEntry]) {
        use rayon::prelude::*;
        eprint!("Reading {} tiles data from disk...", tile_paths.len());
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

                let (x, y, z) = extract_tile_pos(&path_str);

                Tile {
                    x,
                    y,
                    z,
                    width: deform_width(TILE_HEIGHT, y, z),
                    data: Arc::new(tile_image_data),
                    smol_data: Arc::new(tile_smol_data),
                }
            })
            .collect::<Vec<_>>();

        #[cfg(not(windows))]
        const GPU_RESULTS_FOLDER: &str = "data/results/gpu";

        #[cfg(windows)]
        const GPU_RESULTS_FOLDER: &str = "data\\results\\gpu";

        let _ = std::fs::remove_dir_all(GPU_RESULTS_FOLDER);
        std::fs::create_dir_all(GPU_RESULTS_FOLDER).unwrap();
        eprintln!("done");
    }

    pub fn run_on_image(
        &mut self,
        masks: &[(&RgbaImage, u32, &RgbaImage)],
        forbidden_tiles: &FxHashSet<TilePos>,
    ) -> (Vec<(u32, AlgoResult)>, std::time::Duration) {
        if masks.len() != self.n_masks {
            panic!("Expected {} masks, got {}", self.n_masks, masks.len());
        }
        let t_start = std::time::Instant::now();

        self.algo
            .result
            .lock()
            .unwrap()
            .iter_mut()
            .for_each(AlgoResult::clear);

        let (mask_w, mask_h) = (masks[0].0.width(), masks[0].0.height());
        let mask_texs = (0..masks.len())
            .map(|_| {
                mk_tex_general(
                    &self.wgpu.device,
                    (mask_w, mask_h),
                    TextureFormat::Rgba8Unorm,
                    1,
                    3,
                )
            })
            .collect::<Vec<_>>();

        let mut mask_idx = Vec::with_capacity(masks.len());

        for (&(mask, mask_i, last_tile_rgba), mask_tex) in masks.iter().zip(mask_texs.iter()) {
            let mip = image::imageops::resize(
                mask,
                mask_tex.texture.size().width / 2,
                mask_tex.texture.size().height / 2,
                image::imageops::FilterType::Gaussian,
            );

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
                &mip,
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

            self.wgpu.queue.write_texture(
                ImageCopyTexture {
                    mip_level: 2,
                    ..mask_tex.texture.as_image_copy()
                },
                &last_tile_rgba,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        mask_tex.texture.width() / 4
                            * mask_tex.format.block_copy_size(None).unwrap(),
                    ),
                    rows_per_image: Some(mask_tex.texture.height() / 4),
                },
                wgpu::Extent3d {
                    width: mask_tex.texture.size().width / 4,
                    height: mask_tex.texture.size().height / 4,
                    depth_or_array_layers: 1,
                },
            );

            mask_idx.push(mask_i);
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
            crossbeam_channel::bounded::<Vec<(Vec<u8>, Vec<u8>, u32)>>(10);
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
                            entry.width,
                        )
                    })
                    .collect();
                let Ok(()) = decoded_tiles_tx.send(decoded_chunk) else {
                    eprintln!("Error sending decoded tiles");
                    break;
                };
            }
        });

        for (tile_chunk, decoded_tiles) in filtered_tiles
            .chunks(TILE_CHUNK_SIZE)
            .zip(decoded_tiles_rx.iter())
        {
            (self.algo.render_frame)(&self.wgpu, &tile_chunk, &mask_texs, decoded_tiles);
        }

        let best_pos = (self.algo.finish)(&self.wgpu);

        (
            mask_idx.into_iter().zip(best_pos.into_iter()).collect(),
            t_start.elapsed(),
        )
    }
}
