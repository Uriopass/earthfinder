#![allow(clippy::type_complexity)]

use crate::gpu::framework::*;
use image::RgbImage;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::{CommandEncoderDescriptor, Device, MapMode, Queue};

#[derive(Debug, Default, Copy, Clone)]
pub struct PosResult {
    pub tile_x: u32,
    pub tile_y: u32,
    pub tile_z: u32,
    pub x: u32,
    pub y: u32,
}

pub struct Algo {
    pub render_frame: Box<dyn FnMut(PassEncoder, &GPUTextureInner, &[GPUTexture])>,
    pub after_render: Box<dyn FnMut(&[(u32, u32, u32)], &Device, &Queue) -> Arc<AtomicU32>>,
    pub best_pos: Arc<Mutex<PosResult>>,
    pub best_score: Arc<Mutex<f32>>,
}

const STEP_SIZE: usize = 2;

impl Algo {
    pub fn new(device: &Device, tile_size: (u32, u32), mask_size: (u32, u32)) -> Algo {
        let result_size = (
            (tile_size.0 - mask_size.0) / STEP_SIZE as u32,
            (tile_size.1 - mask_size.1) / STEP_SIZE as u32,
        );
        let tex_result_size = (wgpu::util::align_to(result_size.0, 64), result_size.1);

        let result_frames = Arc::new(
            (0..100)
                .map(|_| mk_tex_f32(device, tex_result_size))
                .collect::<Vec<_>>(),
        );
        let result_frames_2 = result_frames.clone();

        let best_pos = Arc::new(Mutex::new(PosResult::default()));
        let best_score = Arc::new(Mutex::new(0.0));

        Algo {
            best_pos: best_pos.clone(),
            best_score: best_score.clone(),
            render_frame: Box::new(move |mut encoder, mask_tex, tile_texs| {
                for (tex, result_tex) in tile_texs.iter().zip(result_frames.iter()) {
                    encoder.pass("main_pass", result_tex, &[mask_tex, tex]);
                }
            }),

            after_render: Box::new(move |tile_paths, device, queue| {
                let result_bufs = (0..tile_paths.len())
                    .map(|_| mk_buffer_dst(device, tex_result_size.0 * tex_result_size.1 * 4))
                    .collect::<Vec<_>>();

                let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("after_render enc"),
                });

                for (result_buf, result_tex) in result_bufs.iter().zip(result_frames_2.iter()) {
                    enc.copy_texture_to_buffer(
                        result_tex.texture.as_image_copy(),
                        wgpu::ImageCopyBuffer {
                            buffer: &result_buf,
                            layout: wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(4 * tex_result_size.0),
                                rows_per_image: Some(tex_result_size.1),
                            },
                        },
                        result_tex.texture.size(),
                    );
                }

                queue.submit(Some(enc.finish()));

                let wait_for_data = Arc::new(AtomicU32::new(tile_paths.len() as u32));

                for (result_buf, (tile_x, tile_y, tile_z)) in
                    result_bufs.iter().zip(tile_paths.iter().copied())
                {
                    let result_buf_cpy = result_buf.clone();

                    let best_score = best_score.clone();
                    let best_pos = best_pos.clone();
                    let wfd = wait_for_data.clone();

                    result_buf.slice(..).map_async(MapMode::Read, move |done| {
                        if done.is_err() {
                            eprintln!("Failed to map buffer");
                            return;
                        }

                        wfd.fetch_sub(1, Ordering::SeqCst);

                        let data: &[u8] = &result_buf_cpy.slice(..).get_mapped_range();
                        let data: &[f32] = bytemuck::cast_slice(data);

                        for (y, row) in data.chunks(tex_result_size.0 as usize).enumerate() {
                            for (x, pixel) in row.iter().enumerate() {
                                if x >= result_size.0 as usize {
                                    break;
                                }
                                let score = *pixel;
                                if score > *best_score.lock().unwrap() {
                                    *best_score.lock().unwrap() = score;
                                    *best_pos.lock().unwrap() = PosResult {
                                        tile_x,
                                        tile_y,
                                        tile_z,
                                        x: x as u32,
                                        y: y as u32,
                                    };
                                    /*eprintln!(
                                        "New best score: {} at {:?}",
                                        *best_score.lock().unwrap(),
                                        *best_pos.lock().unwrap()
                                    );*/
                                }
                            }
                        }

                        result_buf_cpy.destroy();
                    });
                }

                wait_for_data
            }),
        }
    }
}

impl PosResult {
    pub fn to_image(self, mask_size: (u32, u32)) -> RgbImage {
        let mut img = RgbImage::new(mask_size.0, mask_size.1);

        let path = format!(
            "data/tiles/{}/{}/{}.png",
            self.tile_z, self.tile_y, self.tile_x
        );
        let tile_data = image::open(&path).unwrap().to_rgb8();

        for yy in 0..mask_size.1 {
            for xx in 0..mask_size.0 {
                let pixel = *tile_data.get_pixel(
                    (self.x * STEP_SIZE as u32) + xx,
                    (self.y * STEP_SIZE as u32) + yy,
                );

                img.put_pixel(xx, yy, pixel);
            }
        }

        img
    }
}
