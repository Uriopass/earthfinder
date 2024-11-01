#![allow(clippy::type_complexity)]

use crate::gpu::framework::*;
use crate::gpu::TILE_CHUNK_SIZE;
use image::RgbImage;
use std::path::PathBuf;
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
    pub render_frame: Box<dyn FnMut(PassEncoder, &[GPUTexture], &[GPUTexture])>,
    pub after_render: Box<dyn FnMut(&[(u32, u32, u32)], &Device, &Queue) -> Arc<AtomicU32>>,
    pub best_pos: Arc<Mutex<Vec<PosResult>>>,
    pub best_score: Arc<Mutex<Vec<f32>>>,
}

const STEP_SIZE: usize = 2;

impl Algo {
    pub fn new(
        device: &Device,
        tile_size: (u32, u32),
        mask_size: (u32, u32),
        n_masks: usize,
    ) -> Algo {
        let result_size = (
            (tile_size.0 - mask_size.0) / STEP_SIZE as u32,
            (tile_size.1 - mask_size.1) / STEP_SIZE as u32,
        );
        let tex_result_size = (wgpu::util::align_to(result_size.0, 64), result_size.1);

        let result_frames = Arc::new(
            (0..TILE_CHUNK_SIZE * n_masks)
                .map(|_| mk_tex_f32(device, tex_result_size))
                .collect::<Vec<_>>(),
        );
        let result_frames_2 = result_frames.clone();

        let best_pos = Arc::new(Mutex::new(
            (0..n_masks).map(|_| PosResult::default()).collect(),
        ));
        let best_score = Arc::new(Mutex::new((0..n_masks).map(|_| 0.0).collect()));

        Algo {
            best_pos: best_pos.clone(),
            best_score: best_score.clone(),
            render_frame: Box::new(move |mut encoder, mask_texs, tile_texs| {
                let mut i = 0;
                for mask_tex in mask_texs {
                    for tile_tex in tile_texs {
                        let result_tex = &result_frames[i];
                        encoder.pass("main_pass", result_tex, &[mask_tex, tile_tex]);
                        i += 1;
                    }
                }
            }),

            after_render: Box::new(move |tile_paths, device, queue| {
                let result_bufs = (0..tile_paths.len() * n_masks)
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

                let wait_for_data =
                    Arc::new(AtomicU32::new(tile_paths.len() as u32 * n_masks as u32));

                let mut i_buf = 0;
                for mask_i in 0..n_masks {
                    for (tile_x, tile_y, tile_z) in tile_paths.iter().copied() {
                        let result_buf = &result_bufs[i_buf].clone();
                        i_buf += 1;

                        let result_buf_cpy = result_buf.clone();

                        let best_score = best_score.clone();
                        let best_pos = best_pos.clone();
                        let wfd = wait_for_data.clone();

                        result_buf.slice(..).map_async(MapMode::Read, move |done| {
                            if done.is_err() {
                                eprintln!("Failed to map buffer");
                                return;
                            }

                            let data: &[u8] = &result_buf_cpy.slice(..).get_mapped_range();
                            let data: &[f32] = bytemuck::cast_slice(data);

                            let mut tile_best_score = 0.0;
                            let mut local_best_pos = PosResult::default();

                            for (y, row) in data.chunks(tex_result_size.0 as usize).enumerate() {
                                for (x, pixel) in row.iter().enumerate() {
                                    if x >= result_size.0 as usize {
                                        break;
                                    }
                                    let score = *pixel;
                                    if score > tile_best_score {
                                        tile_best_score = score;
                                        local_best_pos = PosResult {
                                            tile_x,
                                            tile_y,
                                            tile_z,
                                            x: x as u32,
                                            y: y as u32,
                                        };
                                    }
                                }
                            }

                            if tile_best_score > best_score.lock().unwrap()[mask_i] {
                                best_score.lock().unwrap()[mask_i] = tile_best_score;
                                best_pos.lock().unwrap()[mask_i] = local_best_pos;
                            }

                            wfd.fetch_sub(1, Ordering::SeqCst);

                            result_buf_cpy.destroy();
                        });
                    }
                }

                wait_for_data
            }),
        }
    }
}

impl PosResult {
    pub fn to_image(self, mask_size: (u32, u32)) -> RgbImage {
        let mut img = RgbImage::new(mask_size.0, mask_size.1);

        let mut path = PathBuf::new();

        path.push("data");
        path.push("tiles");
        path.push(self.tile_z.to_string());
        path.push(self.tile_y.to_string());
        path.push(format!("{}.png", self.tile_x));
        let tile_data = image::open(&path)
            .unwrap_or_else(|e| {
                panic!("Could not open image {}: {}", path.display(), e);
            })
            .to_rgb8();

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
