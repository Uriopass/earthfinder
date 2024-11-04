use crate::gpu::algorithm::STEP_SIZE;
use crate::TILE_SIZE;
use image::imageops::FilterType;
use image::{GenericImageView, RgbImage};
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::{stdout, Write};

const Z_UP: u32 = 4;
const UPSCALE: u32 = 1 << Z_UP;

fn tiles_needed(
    mask_size: (u32, u32),
    tile_x: u32,
    tile_y: u32,
    tile_z: u32,
    x: u32,
    y: u32,
) -> FxHashSet<(u32, u32, u32)> {
    let mut tiles = FxHashSet::default();

    for yy in 0..mask_size.1 * UPSCALE {
        for xx in 0..mask_size.0 * UPSCALE {
            let up_x = (x * STEP_SIZE as u32) * UPSCALE + xx;
            let up_y = (y * STEP_SIZE as u32) * UPSCALE + yy;

            let tile_x = tile_x * UPSCALE + up_x / TILE_SIZE;
            let tile_y = tile_y * UPSCALE + up_y / TILE_SIZE;

            tiles.insert((tile_x, tile_y, tile_z + Z_UP));
        }
    }

    tiles
}

pub fn fetch_tiles_to_cache<'a>(tiles: &FxHashSet<(u32, u32, u32)>) {
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
                .expect("Failed to fetch tile");

            stdout().write_all(&output.stderr).unwrap();
            stdout().write_all(&output.stdout).unwrap();
            if !output.status.success() {
                eprintln!("Failed to fetch tile: {z}/{y}/{x}");
                return;
            }

            eprintln!("Converting tile: {z}/{y}/{x} to png");

            std::process::Command::new("magick")
                .arg("mogrify")
                .arg("-format")
                .arg("png")
                .arg(&path)
                .output()
                .expect("Failed to convert tile");

            std::fs::remove_file(&path).expect("Failed to remove tif file");

            println!("Fetched tile: {z}/{y}/{x}");
        }
    });
}

fn render_final<'a>(
    mask_idx: u32,
    mask_size: (u32, u32),
    tile_x: u32,
    tile_y: u32,
    tile_z: u32,
    x: u32,
    y: u32,
    tiles_needed: impl Iterator<Item = &'a (u32, u32, u32)>,
) {
    let tiles = tiles_needed
        .map(|pos @ &(x, y, z)| {
            let path = format!("./data/tiles/{z}/{y}/{x}.png");
            (pos, image::open(path).unwrap().to_rgb8())
        })
        .collect::<FxHashMap<_, _>>();

    let thumbnail_size = 40;
    let real_frame = image::open(format!("data/bad_apple_frames/bad_apple_{mask_idx:03}.png"))
        .unwrap()
        .resize(4 * thumbnail_size, 3 * thumbnail_size, FilterType::Lanczos3)
        .to_rgb8();

    let img_w = mask_size.0 * UPSCALE;
    let img_h = mask_size.1 * UPSCALE;
    let mut img = RgbImage::new(img_w, img_h);

    for yy in 0..mask_size.1 * UPSCALE {
        for xx in 0..mask_size.0 * UPSCALE {
            let up_x = (x * STEP_SIZE as u32) * UPSCALE + xx;
            let up_y = (y * STEP_SIZE as u32) * UPSCALE + yy;

            let up_tile_x = up_x / TILE_SIZE;
            let up_tile_y = up_y / TILE_SIZE;

            let tile = &tiles[&(
                tile_x * UPSCALE + up_tile_x,
                tile_y * UPSCALE + up_tile_y,
                tile_z + Z_UP,
            )];
            let x = up_x % TILE_SIZE;
            let y = up_y % TILE_SIZE;

            img.put_pixel(xx, yy, *tile.get_pixel(x, y));
        }
    }

    img = image::imageops::resize(&img, img_w * 2, img_h * 2, FilterType::Lanczos3);

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
    let _ = std::fs::create_dir_all("data/render_final");

    let mask_example = image::open("data/bad_apple_masks/bad_apple_1.png").unwrap();
    let mask_size = mask_example.dimensions();

    let csv = std::fs::read_to_string(path).unwrap();
    let mut lines = csv.lines();
    lines.next().unwrap(); // skip header

    for line in lines {
        let mut parts = line.split(',');
        let frame = parts.next().unwrap().trim().trim().parse::<u32>().unwrap();
        let tile_x = parts.next().unwrap().trim().parse::<u32>().unwrap();
        let tile_y = parts.next().unwrap().trim().parse::<u32>().unwrap();
        let tile_z = parts.next().unwrap().trim().parse::<u32>().unwrap();
        let x = parts.next().unwrap().trim().parse::<u32>().unwrap();
        let y = parts.next().unwrap().trim().parse::<u32>().unwrap();
        let score = parts.next().unwrap().trim().parse::<f32>().unwrap();
        let time = parts.next().unwrap().trim().parse::<f32>().unwrap();
        println!(
            "Frame: {frame}, Tile: ({tile_x}, {tile_y}, {tile_z}), Position: ({x}, {y}), Score: {score}, Time: {time}s",
        );

        let needed = tiles_needed(mask_size, tile_x, tile_y, tile_z, x, y);
        fetch_tiles_to_cache(&needed);
        render_final(
            frame,
            mask_size,
            tile_x,
            tile_y,
            tile_z,
            x,
            y,
            needed.iter(),
        );
    }
}
