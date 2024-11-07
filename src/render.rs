use crate::data::{deform_width, parse_csv, TilePos};
use crate::gpu::algorithm::{PosResult, STEP_SIZE};
use crate::tiles_grad::zero_fill;
use crate::TILE_HEIGHT;
use image::imageops::FilterType;
use image::{GenericImageView, ImageError, RgbImage};
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::{stdout, ErrorKind, Write};
use std::sync::atomic::AtomicUsize;

pub fn tiles_needed(mask_size: (u32, u32), result: &PosResult, z_up: u32) -> FxHashSet<TilePos> {
    let upscale = 1 << z_up;
    let mut tiles = FxHashSet::default();

    let deform_w = deform_width(TILE_HEIGHT, result.tile_y, result.tile_z);

    for yy in 0..mask_size.1 * upscale {
        for xx in 0..mask_size.0 * upscale {
            let up_x = (result.x * STEP_SIZE as u32) * upscale + (xx as f32 * result.zoom) as u32;
            let up_y = (result.y * STEP_SIZE as u32) * upscale + (yy as f32 * result.zoom) as u32;

            let tile_x = result.tile_x * upscale + up_x / deform_w;
            let tile_y = result.tile_y * upscale + up_y / TILE_HEIGHT;

            tiles.insert((tile_x, tile_y, result.tile_z + z_up));
        }
    }

    tiles
}

pub fn fetch_tiles_to_cache<'a>(mask_idx: u32, tiles: &FxHashSet<TilePos>) {
    use rayon::prelude::*;
    tiles.par_iter().for_each(|(x, y, z)| {
        let path = format!("./data/tiles/{z}/{y}/{x}.tif");
        let path_png = format!("./data/tiles/{z}/{y}/{x}.png");

        if !std::fs::exists(&path_png).unwrap() {
            std::fs::create_dir_all(format!("./data/tiles/{z}/{y}")).unwrap();
            let aws_path = format!("s3://eox-s2maps/tiles/{z}/{y}/{x}.tif");

            let output = std::process::Command::new("aws")
                .arg("s3")
                .arg("cp")
                .arg(&aws_path)
                .arg(&path)
                .arg("--request-payer")
                .arg("requester")
                .output()
                .map_err(|e| {
                    panic!(
                        "Failed to fetch tile {z}/{y}/{x} for mask {mask_idx}: {:?}",
                        e
                    )
                })
                .unwrap();

            stdout().write_all(&output.stderr).unwrap();
            stdout().write_all(&output.stdout).unwrap();
            if !output.status.success() {
                eprintln!("Failed to fetch tile {z}/{y}/{x} for mask {mask_idx}");
                return;
            }

            let _ = std::process::Command::new("magick")
                .arg("mogrify")
                .arg("-format")
                .arg("png")
                .arg(&path)
                .output();

            let _ = std::fs::remove_file(&path);

            println!("Fetched tile: {z}/{y}/{x} for mask {mask_idx}");
        }
    });
}

fn render_final<'a>(
    mask_idx: u32,
    mask_size: (u32, u32),
    result: &PosResult,
    tiles_needed: impl Iterator<Item = &'a TilePos>,
) {
    let mut z_up = 5.min(13 - result.tile_z);
    let upscale = 1 << z_up;

    let deform_w = deform_width(TILE_HEIGHT, result.tile_y, result.tile_z);
    let tiles = tiles_needed
        .map(|pos @ &(x, y, z)| {
            let path = format!("./data/tiles/{z}/{y}/{x}.png");
            let mut image = image::open(&path)
                .unwrap_or_else(|e| {
                    if let ImageError::IoError(io_err) = &e {
                        if io_err.kind() == ErrorKind::UnexpectedEof {
                            eprintln!("Removing corrupted tile: {z}/{y}/{x}");
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                    panic!(
                        "Failed to open tile {z}/{y}/{x} for mask {mask_idx}: {:?}",
                        e
                    )
                })
                .to_rgb8();
            image = zero_fill(image).unwrap_or_else(|img| img);
            image = image::imageops::resize(&image, deform_w, TILE_HEIGHT, FilterType::Lanczos3);
            (pos, image)
        })
        .collect::<FxHashMap<_, _>>();

    let thumbnail_size = 40;
    let real_frame = image::open(format!("data/bad_apple_frames/bad_apple_{mask_idx:03}.png"))
        .unwrap()
        .resize(4 * thumbnail_size, 3 * thumbnail_size, FilterType::Lanczos3)
        .to_rgb8();

    let img_w = mask_size.0 * upscale;
    let img_h = mask_size.1 * upscale;
    let mut img = RgbImage::new(img_w, img_h);

    for yy in 0..mask_size.1 * upscale {
        for xx in 0..mask_size.0 * upscale {
            let up_x = (result.x * STEP_SIZE as u32) * upscale + (xx as f32 * result.zoom) as u32;
            let up_y = (result.y * STEP_SIZE as u32) * upscale + (yy as f32 * result.zoom) as u32;

            let up_tile_x = up_x / deform_w;
            let up_tile_y = up_y / TILE_HEIGHT;

            let tile = &tiles[&(
                result.tile_x * upscale + up_tile_x,
                result.tile_y * upscale + up_tile_y,
                result.tile_z + z_up,
            )];

            let up_x = ((result.x * STEP_SIZE as u32) * upscale) as f32 + (xx as f32 * result.zoom);
            let up_y = ((result.y * STEP_SIZE as u32) * upscale) as f32 + (yy as f32 * result.zoom);

            let up_x_fract = up_x.fract();
            let up_y_fract = up_y.fract();

            let x = up_x.floor() as u32 % deform_w;
            let y = up_y.floor() as u32 % TILE_HEIGHT;

            let x1 = (x + 1).min(deform_w - 1);
            let y1 = (y + 1).min(TILE_HEIGHT - 1);

            let mut pixel = [0, 0, 0];

            let pix1 = tile.get_pixel(x, y);
            let pix2 = tile.get_pixel(x1, y);
            let pix3 = tile.get_pixel(x, y1);
            let pix4 = tile.get_pixel(x1, y1);

            for i in 0..3 {
                pixel[i] = (pix1[i] as f32 * (1.0 - up_x_fract) * (1.0 - up_y_fract)
                    + pix2[i] as f32 * up_x_fract * (1.0 - up_y_fract)
                    + pix3[i] as f32 * (1.0 - up_x_fract) * up_y_fract
                    + pix4[i] as f32 * up_x_fract * up_y_fract) as u8;
            }

            img.put_pixel(xx, yy, image::Rgb(pixel));
        }
    }

    while z_up < 6 {
        img = image::imageops::resize(
            &img,
            img.width() * 2,
            img.height() * 2,
            FilterType::Lanczos3,
        );
        z_up += 1;
    }

    let offset_w = img.width() - real_frame.width();
    let offset_h = img.height() - real_frame.height();

    for yy in 0..real_frame.height() {
        for xx in 0..real_frame.width() {
            let pixel = real_frame.get_pixel(xx, yy);
            img.put_pixel(offset_w + xx, offset_h + yy, *pixel);
        }
    }

    img.save(format!("data/render_final/{}.png", mask_idx))
        .unwrap();
}

pub fn render(path: &str) {
    use rayon::prelude::*;

    let _ = std::fs::create_dir_all("data/render_final");

    let mask_example = image::open("data/bad_apple_masks/bad_apple_1.png").unwrap();
    let mask_size = mask_example.dimensions();

    let csv = std::fs::read_to_string(path).unwrap();
    let frames = parse_csv(&csv);

    let i = AtomicUsize::new(0);
    frames.par_iter().for_each(|frame| {
        let needed = tiles_needed(mask_size, &frame.result, 5.min(13 - frame.result.tile_z));
        fetch_tiles_to_cache(frame.frame, &needed);
        render_final(frame.frame, mask_size, &frame.result, needed.iter());
        let val = i.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if val % 100 == 0 {
            eprintln!("{} / {}", val, frames.len());
        }
    });
}
