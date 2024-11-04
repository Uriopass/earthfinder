#![allow(clippy::type_complexity)]

use crate::gpu::framework::*;
use crate::gpu::state::WGPUState;
use crate::gpu::{GPUData, Tile};
use crate::TILE_SIZE;
use image::{Rgb32FImage, RgbImage, RgbaImage};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::{Device, Extent3d, ImageCopyTexture, Maintain, MapMode, Origin3d, TextureFormat};

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
    pub render_frame:
        Box<dyn FnMut(&WGPUState<GPUData>, &[Tile], &[GPUTexture], Vec<(Vec<u8>, Vec<u8>)>)>,
    pub finish: Box<dyn FnMut(&WGPUState<GPUData>) -> Vec<PosResults>>,
    pub best_pos: Arc<Mutex<Vec<PosResults>>>,
}

pub const STEP_SIZE: usize = 1;
const TILE_BATCHES_IN_PARALLEL: usize = 20;
pub const CHUNK_MULT: u32 = 4;
pub const TILE_CHUNK_SIZE: usize = (CHUNK_MULT * CHUNK_MULT) as usize;

impl Algo {
    pub fn new(
        device: Arc<Device>,
        tile_size: (u32, u32),
        mask_size: (u32, u32),
        n_masks: usize,
    ) -> Algo {
        let result_size = (
            (CHUNK_MULT * tile_size.0 - mask_size.0 * CHUNK_MULT) / STEP_SIZE as u32,
            (CHUNK_MULT * tile_size.1 - mask_size.1 * CHUNK_MULT) / STEP_SIZE as u32,
        );
        let tex_result_size = (
            wgpu::util::align_to(result_size.0, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / 4),
            result_size.1,
        );

        let result_frames = (0..n_masks)
            .map(|_| mk_tex_f32(&device, tex_result_size))
            .collect::<Vec<_>>();

        let batched_tile_tex = mk_tex_general(
            &device,
            (CHUNK_MULT * tile_size.0, CHUNK_MULT * tile_size.1),
            TextureFormat::Rgba8Unorm,
            1,
            3,
        );

        let best_pos = Arc::new(Mutex::new(
            (0..n_masks).map(|_| PosResults::new(n_masks + 1)).collect(),
        ));
        let best_pos_2 = best_pos.clone();

        let free_buffers = Arc::new(Mutex::new(
            (0..(TILE_BATCHES_IN_PARALLEL + 1) * n_masks)
                .map(|_| {
                    mk_buffer_dst(
                        &device,
                        tex_result_size.0
                            * tex_result_size.1
                            * batched_tile_tex.format.block_copy_size(None).unwrap(),
                    )
                })
                .collect::<Vec<_>>(),
        ));

        let result_bufs_waits = Arc::new(Mutex::new(Vec::new()));
        let buf_waits_2 = Arc::clone(&result_bufs_waits);

        Algo {
            best_pos: best_pos.clone(),
            render_frame: Box::new(
                move |wgpu: &WGPUState<GPUData>, tile_paths: &[Tile], mask_texs, decoded_tiles| {
                    decoded_tiles
                        .iter()
                        .enumerate()
                        .for_each(|(batch_i, (entry, smol))| {
                            wgpu.queue.write_texture(
                                ImageCopyTexture {
                                    texture: &batched_tile_tex.texture,
                                    mip_level: 0,
                                    origin: Origin3d {
                                        x: (batch_i as u32) % CHUNK_MULT * tile_size.0,
                                        y: (batch_i as u32) / CHUNK_MULT * tile_size.1,
                                        z: 0,
                                    },
                                    aspect: Default::default(),
                                },
                                &entry,
                                wgpu::ImageDataLayout {
                                    offset: 0,
                                    bytes_per_row: Some(
                                        tile_size.0
                                            * batched_tile_tex
                                                .format
                                                .block_copy_size(None)
                                                .unwrap(),
                                    ),
                                    rows_per_image: Some(tile_size.0),
                                },
                                Extent3d {
                                    width: tile_size.0,
                                    height: tile_size.1,
                                    depth_or_array_layers: 1,
                                },
                            );

                            wgpu.queue.write_texture(
                                ImageCopyTexture {
                                    texture: &batched_tile_tex.texture,
                                    mip_level: 2,
                                    origin: Origin3d {
                                        x: ((batch_i as u32) % CHUNK_MULT * tile_size.0) / 4,
                                        y: ((batch_i as u32) / CHUNK_MULT * tile_size.1) / 4,
                                        z: 0,
                                    },
                                    aspect: Default::default(),
                                },
                                &smol,
                                wgpu::ImageDataLayout {
                                    offset: 0,
                                    bytes_per_row: Some(
                                        (tile_size.0 / 4)
                                            * batched_tile_tex
                                                .format
                                                .block_copy_size(None)
                                                .unwrap(),
                                    ),
                                    rows_per_image: Some(tile_size.1 / 4),
                                },
                                Extent3d {
                                    width: tile_size.0 / 4,
                                    height: tile_size.1 / 4,
                                    depth_or_array_layers: 1,
                                },
                            );
                        });

                    wgpu.queue.submit([]);

                    drop(decoded_tiles);
                    let mut enc = wgpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut pass_encoder =
                            PassEncoder::new(&wgpu.device, &mut enc, &wgpu.uni_bg);

                        let mut i = 0;
                        for mask_tex in mask_texs {
                            let result_tex = &result_frames[i];
                            pass_encoder.pass(
                                "main_pass",
                                result_tex,
                                &[mask_tex, batched_tile_tex],
                            );
                            i += 1;
                        }
                    }
                    wgpu.queue.submit(Some(enc.finish()));

                    let mut enc = wgpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    let mut buffers_lock = free_buffers.lock().unwrap();
                    let mut buffers = Vec::with_capacity(n_masks);
                    while buffers.len() < buffers.capacity() {
                        buffers.push(buffers_lock.pop().expect("not enough free buffers"));
                    }
                    drop(buffers_lock);

                    for (result_buf, result_tex) in buffers.iter().zip(result_frames.iter()) {
                        enc.copy_texture_to_buffer(
                            result_tex.texture.as_image_copy(),
                            wgpu::ImageCopyBuffer {
                                buffer: &result_buf,
                                layout: wgpu::ImageDataLayout {
                                    offset: 0,
                                    bytes_per_row: Some(
                                        result_tex.format.block_copy_size(None).unwrap()
                                            * tex_result_size.0,
                                    ),
                                    rows_per_image: Some(tex_result_size.1),
                                },
                            },
                            result_tex.texture.size(),
                        );
                    }

                    wgpu.queue.submit(Some(enc.finish()));

                    let wait_for_data = Arc::new(AtomicU32::new(0));

                    let mut i_buf = 0;
                    for mask_i in 0..n_masks {
                        wait_for_data.fetch_add(1, Ordering::SeqCst);

                        let result_buf = buffers[i_buf].clone();
                        i_buf += 1;

                        let result_buf_cpy = result_buf.clone();

                        let best_pos = best_pos.clone();
                        let free_buffers = free_buffers.clone();
                        let wfd = Arc::clone(&wait_for_data);

                        let tile_poses = tile_paths.iter().map(|t| t.pos()).collect::<Vec<_>>();

                        result_buf.slice(..).map_async(MapMode::Read, move |done| {
                            if done.is_err() {
                                eprintln!("Failed to map buffer");
                                return;
                            }
                            rayon::spawn(move || {
                                let slice = result_buf_cpy.slice(..).get_mapped_range();
                                let data: &[u8] = &slice;
                                let data: &[f32] = bytemuck::cast_slice(data);

                                assert_eq!(
                                    data.len(),
                                    tex_result_size.0 as usize * tex_result_size.1 as usize
                                );

                                let mut tile_best_pos = PosResult::default();
                                for (y, row) in data.chunks(tex_result_size.0 as usize).enumerate()
                                {
                                    for (x, pixel) in row.iter().enumerate() {
                                        if x >= result_size.0 as usize {
                                            break;
                                        }

                                        let score = *pixel;
                                        if score > tile_best_pos.score {
                                            let batch_x =
                                                x / (tile_size.0 as usize - mask_size.0 as usize);
                                            let batch_y =
                                                y / (tile_size.1 as usize - mask_size.1 as usize);
                                            let i_tile = batch_x + batch_y * CHUNK_MULT as usize;

                                            if i_tile >= tile_poses.len() {
                                                break;
                                            }

                                            let (tile_x, tile_y, tile_z) = tile_poses[i_tile];

                                            tile_best_pos.tile_x = tile_x;
                                            tile_best_pos.tile_y = tile_y;
                                            tile_best_pos.tile_z = tile_z;
                                            tile_best_pos.x =
                                                (x as u32) % (tile_size.0 - mask_size.0);
                                            tile_best_pos.y =
                                                (y as u32) % (tile_size.1 - mask_size.1);
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
                        });
                    }
                    result_bufs_waits.lock().unwrap().push(wait_for_data);

                    let mut rwait = result_bufs_waits.lock().unwrap();
                    if rwait.len() >= TILE_BATCHES_IN_PARALLEL {
                        let to_wait = rwait.remove(0);
                        while to_wait.load(Ordering::SeqCst) > 0 {
                            wgpu.device.poll(Maintain::Wait);
                        }
                    }
                },
            ),
            finish: Box::new(move |wgpu| {
                let rwait = buf_waits_2.lock().unwrap();

                for wait in rwait.iter() {
                    while wait.load(Ordering::SeqCst) > 0 {
                        wgpu.device.poll(Maintain::Wait);
                    }
                }
                best_pos_2.lock().unwrap().clone()
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

    pub fn to_image(self, mask_data: &RgbaImage, avg_error: &Rgb32FImage, debug: bool) -> RgbImage {
        const Z_UP: u32 = 2;
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

        let img_width = if debug {
            mask_size.0 * UPSCALE * 4
        } else {
            mask_size.0 * UPSCALE
        };

        let mut img = RgbImage::new(img_width, mask_size.1 * UPSCALE);

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

        if debug {
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
        }

        img
    }
}
