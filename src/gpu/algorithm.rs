#![allow(clippy::type_complexity)]

use crate::gpu::framework::*;
use crate::gpu::{Tile, TILE_CHUNK_SIZE};
use crate::TILE_SIZE;
use image::{Rgb32FImage, RgbImage, RgbaImage};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, CommandEncoderDescriptor, Device, MapMode, Queue};

#[derive(Debug, Copy, Clone)]
pub struct PosResult {
    pub tile_x: u32,
    pub tile_y: u32,
    pub tile_z: u32,
    pub x: u32,
    pub y: u32,
    pub score: f32,
}

impl PosResult {
    pub fn tile_pos(&self) -> (u32, u32, u32) {
        (self.tile_x, self.tile_y, self.tile_z)
    }
}

impl Default for PosResult {
    fn default() -> Self {
        Self {
            tile_x: u32::MAX,
            tile_y: u32::MAX,
            tile_z: u32::MAX,
            x: 0,
            y: 0,
            score: f32::NEG_INFINITY,
        }
    }
}

#[derive(Clone)]
pub struct PosResults {
    top_k: usize,
    top_results: Vec<PosResult>,
}

impl PosResults {
    pub fn new(top_k: usize) -> PosResults {
        PosResults {
            top_k,
            top_results: vec![PosResult::default(); top_k],
        }
    }

    pub fn clear(&mut self) {
        self.top_results.clear();
        for _ in 0..self.top_k {
            self.top_results.push(PosResult::default());
        }
    }

    pub fn results(&self) -> &[PosResult] {
        &self.top_results
    }

    pub fn insert(&mut self, pos: PosResult) {
        let mut i = self.top_k - 1;
        if pos.score < self.top_results[i].score {
            return;
        }
        self.top_results[i] = pos;
        while i > 0 && self.top_results[i].score > self.top_results[i - 1].score {
            self.top_results.swap(i, i - 1);
            i -= 1;
        }
    }
}

pub struct Algo {
    pub render_frame: Box<dyn FnMut(PassEncoder, &[GPUTexture], &[GPUTexture])>,
    pub after_render: Box<dyn FnMut(&[Tile], &Device, &Queue) -> Arc<AtomicU32>>,
    pub best_pos: Arc<Mutex<Vec<PosResults>>>,
}

const STEP_SIZE: usize = 1;

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
            (0..n_masks)
                .map(|_| PosResults::new(n_masks + 10))
                .collect(),
        ));

        let free_buffers: Arc<Mutex<Vec<Arc<Buffer>>>> = Arc::new(Mutex::new(
            (0..20 * TILE_CHUNK_SIZE * n_masks)
                .map(|_| mk_buffer_dst(device, tex_result_size.0 * tex_result_size.1 * 4))
                .collect(),
        ));

        Algo {
            best_pos: best_pos.clone(),
            render_frame: Box::new(move |mut encoder, mask_texs, tile_texs| {
                let mut i = 0;
                for tile_tex in tile_texs {
                    for mask_tex in mask_texs {
                        let result_tex = &result_frames[i];
                        encoder.pass("main_pass", result_tex, &[mask_tex, tile_tex]);
                        i += 1;
                    }
                }
            }),

            after_render: Box::new(move |tile_paths, device, queue| {
                let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("after_render enc"),
                });

                let mut buffers_lock = free_buffers.lock().unwrap();
                let mut buffers = Vec::with_capacity(tile_paths.len() * n_masks);
                while buffers.len() < buffers.capacity() {
                    buffers.push(buffers_lock.pop().expect("not enough free buffers"));
                }
                drop(buffers_lock);

                for (result_buf, result_tex) in buffers.iter().zip(result_frames_2.iter()) {
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

                let wait_for_data = Arc::new(AtomicU32::new(0));

                let mut i_buf = 0;
                for tile in tile_paths.iter() {
                    for mask_i in 0..n_masks {
                        wait_for_data.fetch_add(1, Ordering::SeqCst);
                        let result_buf = &buffers[i_buf].clone();
                        i_buf += 1;

                        let result_buf_cpy = result_buf.clone();

                        let best_pos = best_pos.clone();
                        let free_buffers = free_buffers.clone();
                        let wfd = Arc::clone(&wait_for_data);

                        let (tile_x, tile_y, tile_z) = (tile.x, tile.y, tile.z);

                        result_buf.slice(..).map_async(MapMode::Read, move |done| {
                            if done.is_err() {
                                eprintln!("Failed to map buffer");
                                return;
                            }

                            let slice = result_buf_cpy.slice(..).get_mapped_range();
                            let data: &[u8] = &slice;
                            let data: &[f32] = bytemuck::cast_slice(data);

                            assert_eq!(
                                data.len(),
                                tex_result_size.0 as usize * tex_result_size.1 as usize
                            );

                            let mut tile_best_pos = PosResult::default();
                            tile_best_pos.tile_x = tile_x;
                            tile_best_pos.tile_y = tile_y;
                            tile_best_pos.tile_z = tile_z;

                            for (y, row) in data.chunks(tex_result_size.0 as usize).enumerate() {
                                for (x, pixel) in row.iter().enumerate() {
                                    if x >= result_size.0 as usize {
                                        break;
                                    }

                                    let score = *pixel;
                                    if score > tile_best_pos.score {
                                        tile_best_pos.x = x as u32;
                                        tile_best_pos.y = y as u32;
                                        tile_best_pos.score = score;
                                    }
                                }
                            }

                            best_pos.lock().unwrap()[mask_i].insert(tile_best_pos);
                            drop(slice);
                            result_buf_cpy.unmap();
                            free_buffers.lock().unwrap().push(result_buf_cpy);

                            wfd.fetch_sub(1, Ordering::SeqCst);
                        });
                    }
                }

                wait_for_data
            }),
        }
    }
}

impl PosResult {
    pub fn calc_error(&self, mask_data: &RgbaImage, mut adderror: impl FnMut(u32, u32, f32)) {
        let path_grad = format!(
            "data/tiles_grad/{}/{}/{}.png",
            self.tile_z, self.tile_y, self.tile_x
        );
        let tile_grad = image::open(&path_grad).unwrap().to_rgb8();

        for yy in 0..mask_data.height() {
            for xx in 0..mask_data.width() {
                let tile_pixel = *tile_grad.get_pixel(self.x + xx, self.y + yy);
                let mask_pixel = mask_data.get_pixel(xx, yy);

                let pr = tile_pixel[0] as f32 / 255.0;
                let pg = tile_pixel[1] as f32 / 255.0;

                let mr = mask_pixel[0] as f32 / 255.0;
                let mg = mask_pixel[1] as f32 / 255.0;
                let mb = mask_pixel[2] as f32 / 255.0;

                let diff = [
                    f32::max(0.0, mr - pr),
                    f32::max(0.0, mg - pg),
                    0.5 * mb * (pr * pr + pg * pg),
                ];

                let error = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2]).sqrt();

                adderror(xx, yy, error);
            }
        }
    }

    pub fn to_rgba(self, mask_dims: (u32, u32)) -> RgbaImage {
        let path_tile = format!(
            "data/tiles/{}/{}/{}.png",
            self.tile_z, self.tile_y, self.tile_x
        );
        let tile = image::open(&path_tile).unwrap().to_rgba8();

        let mut mask_rgba = RgbaImage::new(mask_dims.0, mask_dims.1);

        for yy in 0..mask_dims.1 {
            for xx in 0..mask_dims.0 {
                let pixel = *tile.get_pixel(self.x + xx, self.y + yy);
                mask_rgba.put_pixel(xx, yy, pixel);
            }
        }

        mask_rgba
    }

    pub fn to_image(self, mask_data: &RgbaImage, avg_error: &Rgb32FImage) -> RgbImage {
        const Z_UP: u32 = 1;
        const UPSCALE: u32 = 1 << Z_UP;

        let mask_size = mask_data.dimensions();

        let mut tiles = Vec::with_capacity(UPSCALE as usize * UPSCALE as usize);

        let mut path = PathBuf::new();
        path.push("data");
        path.push("tiles");
        path.push((self.tile_z + Z_UP).to_string());

        let path_grad = format!(
            "data/tiles_grad/{}/{}/{}.png",
            self.tile_z, self.tile_y, self.tile_x
        );
        let tile_grad = image::open(&path_grad).unwrap().to_rgb8();

        for up_tile_y in 0..UPSCALE {
            path.push((self.tile_y * UPSCALE + up_tile_y).to_string());
            for up_tile_x in 0..UPSCALE {
                path.push(format!("{}.png", self.tile_x * UPSCALE + up_tile_x));
                let tile_data = image::open(&path)
                    .unwrap_or_else(|e| {
                        panic!("Could not open image {}: {}", path.display(), e);
                    })
                    .to_rgb8();
                tiles.push(tile_data);
                path.pop();
            }
            path.pop();
        }

        let mut img = RgbImage::new(mask_size.0 * UPSCALE * 4, mask_size.1 * UPSCALE);

        for yy in 0..mask_size.1 * UPSCALE {
            for xx in 0..mask_size.0 * UPSCALE {
                let up_x = (self.x * STEP_SIZE as u32) * UPSCALE + xx;
                let up_y = (self.y * STEP_SIZE as u32) * UPSCALE + yy;

                let tile_x = up_x / TILE_SIZE;
                let tile_y = up_y / TILE_SIZE;

                let tile = &tiles[(tile_y * UPSCALE + tile_x) as usize];
                let x = up_x % TILE_SIZE;
                let y = up_y % TILE_SIZE;

                let pixel = *tile.get_pixel(x, y);
                img.put_pixel(xx, yy, pixel);
            }
        }

        for yy in 0..mask_size.1 * UPSCALE {
            for xx in 0..mask_size.0 * UPSCALE {
                let pixel = *mask_data.get_pixel(xx / UPSCALE, yy / UPSCALE);
                img.put_pixel(
                    xx + mask_size.0 * UPSCALE,
                    yy,
                    From::from([pixel.0[0], pixel.0[1], pixel.0[2]]),
                );

                let pixel_grad = *tile_grad.get_pixel(
                    (self.x * STEP_SIZE as u32) + xx / UPSCALE,
                    (self.y * STEP_SIZE as u32) + yy / UPSCALE,
                );
                img.put_pixel(xx + mask_size.0 * UPSCALE * 2, yy, pixel_grad);

                let pixel_err =
                    (avg_error.get_pixel(xx / UPSCALE, yy / UPSCALE).0[0] * 255.0) as u8;
                img.put_pixel(
                    xx + mask_size.0 * UPSCALE * 3,
                    yy,
                    From::from([pixel_err, pixel_err, pixel_err]),
                );
            }
        }

        img
    }
}
