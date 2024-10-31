pub mod algorithm;
pub mod framework;
pub mod state;

use crate::gpu::state::WGPUState;
use algorithm::Algo;
use framework::*;
use image::DynamicImage;
use walkdir::DirEntry;

pub struct State {
    wgpu: WGPUState<f32>,
    algo: Algo,
}

impl State {
    pub async fn new(tile_size: (u32, u32), mask_size: (u32, u32)) -> State {
        let wgpu = WGPUState::new().await;
        let device = &wgpu.device;

        let algo = Algo::new(device, tile_size, mask_size);

        Self { wgpu, algo }
    }

    pub fn run_on_image(&mut self, mask: DynamicImage, tile_paths: &[DirEntry]) {
        let (w, h) = (mask.width(), mask.height());
        let mask_tex = mk_tex(&self.wgpu.device, (w, h));

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
        let tile_tex = mk_tex(&self.wgpu.device, (512, 512));

        let mut i = 0;
        let total = tile_paths.len();

        let t_start = std::time::Instant::now();

        std::fs::create_dir_all("data/results/gpu").unwrap();

        for entry in tile_paths {
            i += 1;

            if i % 100 == 0 {
                let elapsed = t_start.elapsed();
                let eta = elapsed / i as u32 * (total - i) as u32;
                println!(
                    "Processing tile {}/{} ETA:{:.0}s",
                    i,
                    total,
                    eta.as_secs_f32()
                );
            }

            let path = entry.path();
            let pathstr = path.display().to_string();
            let parts = pathstr.split("/").collect::<Vec<_>>();
            let x = parts[parts.len() - 1]
                .split_once(".")
                .unwrap()
                .0
                .parse()
                .unwrap();
            let y = parts[parts.len() - 2].parse().unwrap();

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

            let mut encoder = self
                .wgpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let pass_encoder = PassEncoder::new(&self.wgpu.device, &mut encoder, &self.wgpu.uni_bg);

            (self.algo.render_frame)(pass_encoder, &mask_tex, &tile_tex);
            self.wgpu.queue.submit(Some(encoder.finish()));
            (self.algo.after_render)(path, (x, y), &self.wgpu.device, &self.wgpu.queue);
        }
    }
}
