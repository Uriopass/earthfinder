#![allow(clippy::type_complexity)]

use crate::gpu::framework::*;
use image::RgbImage;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use wgpu::{CommandEncoderDescriptor, Device, MapMode, Queue};

pub struct Algo {
    pub render_frame: Box<dyn FnMut(PassEncoder, &GPUTextureInner, &GPUTextureInner)>,
    pub after_render: Box<dyn FnMut(&Path, (u32, u32), &Device, &Queue)>,
}

impl Algo {
    pub fn new(device: &Device, tile_size: (u32, u32), mask_size: (u32, u32)) -> Algo {
        let result_size = (tile_size.0 - mask_size.0, tile_size.1 - mask_size.1);
        let mut tex_result_size = result_size;
        if tex_result_size.0 % 64 != 0 {
            tex_result_size.0 += 64 - tex_result_size.0 % 64;
        }

        let result_frame = mk_tex(device, tex_result_size);

        let best_pos = Arc::new(Mutex::new((0, 0, 0, 0)));
        let best_score = Arc::new(Mutex::new(0));

        Algo {
            render_frame: Box::new(move |mut encoder, mask_tex, tile_tex| {
                encoder.pass("main_pass", result_frame, &[mask_tex, tile_tex]);
            }),

            after_render: Box::new(move |tile_path, tile_pos, device, queue| {
                let result_buf = mk_buffer_dst(device, tex_result_size.0 * tex_result_size.1 * 4);

                let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("after_render enc"),
                });

                enc.copy_texture_to_buffer(
                    result_frame.texture.as_image_copy(),
                    wgpu::ImageCopyBuffer {
                        buffer: &result_buf,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * tex_result_size.0),
                            rows_per_image: Some(tex_result_size.1),
                        },
                    },
                    result_frame.texture.size(),
                );

                queue.submit(Some(enc.finish()));

                let wait_for_data = Arc::new(AtomicBool::new(false));
                let wfd = wait_for_data.clone();

                let result_buf_cpy = result_buf.clone();
                let best_score = best_score.clone();
                let best_pos = best_pos.clone();

                let tile_path = tile_path.to_path_buf();

                result_buf.slice(..).map_async(MapMode::Read, move |done| {
                    wfd.store(true, std::sync::atomic::Ordering::SeqCst);
                    if done.is_err() {
                        eprintln!("Failed to map buffer");
                        return;
                    }

                    let data: &[u8] = &result_buf_cpy.slice(..).get_mapped_range();
                    let data: &[[u8; 4]] = bytemuck::cast_slice(data);

                    for (y, row) in data.chunks(tex_result_size.0 as usize).enumerate() {
                        for (x, pixel) in row.iter().enumerate() {
                            if x >= result_size.0 as usize {
                                break;
                            }
                            let score = pixel[0];
                            if score > *best_score.lock().unwrap() {
                                *best_score.lock().unwrap() = score;
                                *best_pos.lock().unwrap() =
                                    (tile_pos.0, tile_pos.1, x as u32, y as u32);
                                eprintln!(
                                    "New best score: {} at {:?}",
                                    *best_score.lock().unwrap(),
                                    *best_pos.lock().unwrap()
                                );

                                let mut img = RgbImage::new(mask_size.0, mask_size.1);

                                let tile_path =
                                    tile_path.to_string_lossy().replace("tiles_oklab", "tiles");
                                let tile_data = image::open(&tile_path).unwrap().to_rgb8();

                                for yy in 0..mask_size.1 {
                                    for xx in 0..mask_size.0 {
                                        let pixel =
                                            *tile_data.get_pixel(x as u32 + xx, y as u32 + yy);

                                        img.put_pixel(xx, yy, pixel);
                                    }
                                }

                                img.save(format!("data/results/gpu/best_{}.png", score))
                                    .unwrap();
                            }
                        }
                    }
                });

                /*while !wait_for_data.load(std::sync::atomic::Ordering::SeqCst) {
                    device.poll(wgpu::Maintain::Wait);
                    std::thread::yield_now();
                }*/
            }),
        }
    }
}
