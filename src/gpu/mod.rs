pub mod algorithm;
pub mod framework;
pub mod state;

use crate::gpu::algorithm::PosResult;
use crate::gpu::state::WGPUState;
use algorithm::Algo;
use bytemuck::Zeroable;
use framework::*;
use image::DynamicImage;
use std::path::MAIN_SEPARATOR;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use walkdir::DirEntry;
use wgpu::Maintain;

#[derive(Default, Copy, Clone)]
struct GPUData {
    total1: f32,
    total2: f32,
}

unsafe impl Zeroable for GPUData {}
unsafe impl bytemuck::Pod for GPUData {}

pub struct State {
    wgpu: WGPUState<GPUData>,
    algo: Algo,
    n_masks: usize,
    tile_data: Arc<Vec<Vec<u8>>>,
    tiles: Vec<(u32, u32, u32)>,
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
            tile_data: Default::default(),
            tiles: vec![],
        }
    }

    pub fn prepare(&mut self, tile_paths: &[DirEntry]) {
        eprintln!("Reading {} tiles data from disk", tile_paths.len());
        let mut tile_data = Vec::with_capacity(tile_paths.len());
        for entry in tile_paths {
            let path = entry.path();

            tile_data.push(std::fs::read(entry.path()).expect("could not read preprocessed data"));

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

            self.tiles.push((x, y, z));
        }
        self.tile_data = Arc::new(tile_data);
    }

    pub fn run_on_image(&mut self, masks: &[(DynamicImage, u32)]) -> Vec<(u32, PosResult)> {
        if masks.len() != self.n_masks {
            panic!("Expected {} masks, got {}", self.n_masks, masks.len());
        }
        let t_start = std::time::Instant::now();

        self.algo.best_score.lock().unwrap().fill(0.0);

        let (w, h) = (masks[0].0.width(), masks[0].0.height());
        eprintln!("preparing mask textures");
        let mask_texs = (0..masks.len())
            .map(|_| mk_tex(&self.wgpu.device, (w, h)))
            .collect::<Vec<_>>();

        let mut mask_idx = Vec::with_capacity(masks.len());

        for ((mask, mask_i), mask_tex) in masks.iter().zip(mask_texs.iter()) {
            let pixels = mask.to_rgba8().into_raw();

            self.wgpu.queue.write_texture(
                mask_tex.texture.as_image_copy(),
                &pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );

            mask_idx.push(*mask_i);
        }

        WGPUState::modify_user_data(&self.wgpu.queue, &self.wgpu.user_data, &|u| {
            u.total1 = 0.0;
            u.total2 = 0.0;
        });

        eprintln!("preparing tile textures");

        let tile_texs = (0..TILE_CHUNK_SIZE)
            .map(|_| mk_tex(&self.wgpu.device, (512, 512)))
            .collect::<Vec<_>>();

        #[cfg(not(windows))]
        const GPU_RESULTS_FOLDER: &str = "data/results/gpu";

        #[cfg(windows)]
        const GPU_RESULTS_FOLDER: &str = "data\\results\\gpu";

        let _ = std::fs::remove_dir_all(GPU_RESULTS_FOLDER);
        std::fs::create_dir_all(GPU_RESULTS_FOLDER).unwrap();

        let mut data_waits = Vec::new();

        let (decoded_tiles_tx, decoded_tiles_rx) = crossbeam_channel::bounded::<Vec<Vec<u8>>>(2);

        let tile_data_2 = self.tile_data.clone();

        rayon::spawn(move || {
            use rayon::prelude::*;

            for data_chunk in tile_data_2.chunks(TILE_CHUNK_SIZE) {
                let decoded_chunk = data_chunk
                    .par_iter()
                    .map(|entry| {
                        let pixel_data =
                            image::load_from_memory(entry).expect("could not decode pixels");

                        pixel_data.to_rgba8().into_raw()
                    })
                    .collect();
                decoded_tiles_tx.send(decoded_chunk).unwrap();
            }
        });

        eprintln!("processing tiles");
        for (data_chunk, tiles_chunk) in self
            .tile_data
            .chunks(TILE_CHUNK_SIZE)
            .zip(self.tiles.chunks(TILE_CHUNK_SIZE))
        {
            let tile_texs = &tile_texs[..data_chunk.len()];

            let decoded_tiles = decoded_tiles_rx.recv().unwrap();
            decoded_tiles
                .iter()
                .zip(tile_texs.iter())
                .for_each(|(entry, tile_tex)| {
                    self.wgpu.queue.write_texture(
                        tile_tex.texture.as_image_copy(),
                        &entry,
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(512 * 4),
                            rows_per_image: Some(512),
                        },
                        wgpu::Extent3d {
                            width: w,
                            height: h,
                            depth_or_array_layers: 1,
                        },
                    );
                });
            drop(decoded_tiles);

            let mut encoder = self
                .wgpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let pass_encoder = PassEncoder::new(&self.wgpu.device, &mut encoder, &self.wgpu.uni_bg);

            (self.algo.render_frame)(pass_encoder, &mask_texs, tile_texs);
            self.wgpu.queue.submit(Some(encoder.finish()));

            data_waits.push((self.algo.after_render)(
                &tiles_chunk,
                &self.wgpu.device,
                &self.wgpu.queue,
            ));
            let to_wait = 10;
            if data_waits.len() > to_wait {
                let to_wait: &Arc<AtomicU32> = &data_waits[data_waits.len() - to_wait - 1];
                while to_wait.load(Ordering::SeqCst) > 0 {
                    self.wgpu.device.poll(Maintain::Poll);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }

        for wait in data_waits {
            while wait.load(Ordering::SeqCst) > 0 {
                self.wgpu.device.poll(Maintain::Poll);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        println!(
            "Finished processing tile in {:.3}s                          ",
            t_start.elapsed().as_secs_f32()
        );

        let best_pos = self.algo.best_pos.lock().unwrap().clone();

        mask_idx.into_iter().zip(best_pos.into_iter()).collect()
    }
}
