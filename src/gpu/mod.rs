pub mod algorithm;
pub mod framework;
pub mod state;

use crate::gpu::algorithm::PosResult;
use crate::gpu::state::WGPUState;
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
}

impl State {
    pub async fn new(tile_size: (u32, u32), mask_size: (u32, u32)) -> State {
        let wgpu = WGPUState::new().await;
        let device = &wgpu.device;

        let algo = Algo::new(device, tile_size, mask_size);

        Self { wgpu, algo }
    }

    pub fn run_on_image(&mut self, mask: DynamicImage, tile_paths: &[DirEntry]) -> PosResult {
        *self.algo.best_score.lock().unwrap() = 0.0;

        let (w, h) = (mask.width(), mask.height());
        let mask_tex = mk_tex(&self.wgpu.device, (w, h));

        let pixels = mask.to_rgba8().into_raw();

        let mut total1 = 0.0;
        let mut total2 = 0.0;

        for pixel in pixels.chunks_exact(4) {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;

            total1 += r;
            total2 += g;
        }

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

        const CHUNK_SIZE: usize = 100;

        WGPUState::modify_user_data(&self.wgpu.queue, &self.wgpu.user_data, &|u| {
            u.total1 = total1;
            u.total2 = total2;
        });

        let tile_texs = (0..CHUNK_SIZE)
            .map(|_| mk_tex(&self.wgpu.device, (512, 512)))
            .collect::<Vec<_>>();

        let mut i = 0;
        let total = tile_paths.len();

        let _ = std::fs::remove_dir_all("data/results/gpu");
        std::fs::create_dir_all("data/results/gpu").unwrap();

        let mut data_waits = Vec::new();

        let t_start = std::time::Instant::now();

        for entry_chunk in tile_paths.chunks(100) {
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
                    let tile = image::open(path).unwrap();
                    let (w, h) = (tile.width(), tile.height());
                    let pixels = tile.to_rgba8().into_raw();

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

            (self.algo.render_frame)(pass_encoder, &mask_tex, tile_texs);
            self.wgpu.queue.submit(Some(encoder.finish()));
            data_waits.push((self.algo.after_render)(
                &tiles,
                &self.wgpu.device,
                &self.wgpu.queue,
            ));

            i += entry_chunk.len();
            let elapsed = t_start.elapsed();
            let eta = elapsed / i as u32 * (total - i) as u32;
            print!(
                "Processing tile {}/{} ETA:{:.0}s\r",
                i,
                total,
                eta.as_secs_f32()
            );
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

        *self.algo.best_pos.lock().unwrap()
    }
}
