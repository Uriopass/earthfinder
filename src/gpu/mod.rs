pub mod algorithm;
pub mod framework;
pub mod state;

use crate::gpu::algorithm::PosResult;
use crate::gpu::state::WGPUState;
use crate::raw::read_raw;
use algorithm::Algo;
use bytemuck::Zeroable;
use framework::*;
use image::DynamicImage;
use std::sync::atomic::Ordering;
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
        }
    }

    pub fn run_on_image(
        &mut self,
        masks: &[(DynamicImage, u32)],
        tile_paths: &[DirEntry],
    ) -> Vec<(u32, PosResult)> {
        if masks.len() != self.n_masks {
            panic!("Expected {} masks, got {}", self.n_masks, masks.len());
        }
        let t_start = std::time::Instant::now();

        *self.algo.best_score.lock().unwrap() = 0.0;

        let (w, h) = (masks[0].0.width(), masks[0].0.height());
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

        let tile_texs = (0..TILE_CHUNK_SIZE)
            .map(|_| mk_tex(&self.wgpu.device, (512, 512)))
            .collect::<Vec<_>>();

        let _ = std::fs::remove_dir_all("data/results/gpu");
        std::fs::create_dir_all("data/results/gpu").unwrap();

        let mut data_waits = Vec::new();

        for entry_chunk in tile_paths.chunks(TILE_CHUNK_SIZE) {
            let mut tiles = Vec::with_capacity(entry_chunk.len());

            let tile_texs = &tile_texs[..entry_chunk.len()];

            use rayon::prelude::*;

            entry_chunk.iter().for_each(|entry| {
                let path = entry.path();
                let pathstr = path.display().to_string();
                let parts = pathstr.split("/").collect::<Vec<_>>();
                let x = parts[parts.len() - 1]
                    .split_once(".")
                    .unwrap()
                    .0
                    .parse::<u32>()
                    .unwrap();
                let y = parts[parts.len() - 2].parse::<u32>().unwrap();
                let z = parts[parts.len() - 3].parse::<u32>().unwrap();

                tiles.push((x, y, z));
            });

            entry_chunk
                .par_iter()
                .zip(tile_texs.par_iter())
                .for_each(|(entry, tile_tex)| {
                    let path = entry.path();
                    let tile = read_raw(path);
                    let (w, h) = (tile.width(), tile.height());
                    let pixels = tile.into_raw();

                    self.wgpu.queue.write_texture(
                        tile_tex.texture.as_image_copy(),
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
                });

            let mut encoder = self
                .wgpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let pass_encoder = PassEncoder::new(&self.wgpu.device, &mut encoder, &self.wgpu.uni_bg);

            (self.algo.render_frame)(pass_encoder, &mask_texs, tile_texs);
            self.wgpu.queue.submit(Some(encoder.finish()));
            data_waits.push((self.algo.after_render)(
                &tiles,
                &self.wgpu.device,
                &self.wgpu.queue,
            ));
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
