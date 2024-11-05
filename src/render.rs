use crate::data::deform_width;
use crate::gpu::algorithm::STEP_SIZE;
use crate::TILE_HEIGHT;
use image::imageops::FilterType;
use image::{GenericImageView, RgbImage};
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::{stdout, Write};
use std::sync::atomic::AtomicUsize;

const Z_UP: u32 = 4;
const UPSCALE: u32 = 1 << Z_UP;

pub fn tiles_needed(
    mask_size: (u32, u32),
    tile_x: u32,
    tile_y: u32,
    tile_z: u32,
    zoom: f32,
    x: u32,
    y: u32,
    z_up: u32,
) -> FxHashSet<(u32, u32, u32)> {
    let upscale = 1 << z_up;
    let mut tiles = FxHashSet::default();

    let deform_w = deform_width(TILE_HEIGHT, tile_y, tile_z);

    for yy in 0..mask_size.1 * upscale {
        for xx in 0..mask_size.0 * upscale {
            let up_x = (x * STEP_SIZE as u32) * upscale + (xx as f32 * zoom) as u32;
            let up_y = (y * STEP_SIZE as u32) * upscale + (yy as f32 * zoom) as u32;

            let tile_x = tile_x * upscale + up_x / deform_w;
            let tile_y = tile_y * upscale + up_y / TILE_HEIGHT;

            tiles.insert((tile_x, tile_y, tile_z + z_up));
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
                .map_err(|e| panic!("Failed to fetch tile: {z}/{y}/{x} {:?}", e))
                .unwrap();

            stdout().write_all(&output.stderr).unwrap();
            stdout().write_all(&output.stdout).unwrap();
            if !output.status.success() {
                eprintln!("Failed to fetch tile: {z}/{y}/{x}");
                return;
            }

            let _ = std::process::Command::new("magick")
                .arg("mogrify")
                .arg("-format")
                .arg("png")
                .arg(&path)
                .output();

            let _ = std::fs::remove_file(&path);

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
    zoom: f32,
    x: u32,
    y: u32,
    tiles_needed: impl Iterator<Item = &'a (u32, u32, u32)>,
) {
    let deform_w = deform_width(TILE_HEIGHT, tile_y, tile_z);
    let tiles = tiles_needed
        .map(|pos @ &(x, y, z)| {
            let path = format!("./data/tiles/{z}/{y}/{x}.png");
            let image = image::open(path).unwrap().to_rgb8();
            let image =
                image::imageops::resize(&image, deform_w, TILE_HEIGHT, FilterType::Lanczos3);
            (pos, image)
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
            let up_x = (x * STEP_SIZE as u32) * UPSCALE + (xx as f32 * zoom) as u32;
            let up_y = (y * STEP_SIZE as u32) * UPSCALE + (yy as f32 * zoom) as u32;

            let up_tile_x = up_x / deform_w;
            let up_tile_y = up_y / TILE_HEIGHT;

            let tile = &tiles[&(
                tile_x * UPSCALE + up_tile_x,
                tile_y * UPSCALE + up_tile_y,
                tile_z + Z_UP,
            )];
            let x = up_x % deform_w;
            let y = up_y % TILE_HEIGHT;

            img.put_pixel(xx, yy, *tile.get_pixel(x, y));
        }
    }

    // todo: for zoom 8 and 7, we don't need to use lvl 12, we can use lvl 13 and avoid upscale !
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

struct FrameData {
    frame: u32,
    tile_x: u32,
    tile_y: u32,
    tile_z: u32,
    zoom: f32,
    x: u32,
    y: u32,
    _score: f32,
    _time: f32,
}

pub fn render(path: &str) {
    use rayon::prelude::*;

    let _ = std::fs::create_dir_all("data/render_final");

    let mask_example = image::open("data/bad_apple_masks/bad_apple_1.png").unwrap();
    let mask_size = mask_example.dimensions();

    let csv = std::fs::read_to_string(path).unwrap();
    let mut lines = csv.lines();
    lines.next().unwrap(); // skip header
    let frames = lines
        .map(|line| {
            let mut parts = line.split(',');
            let frame = parts.next().unwrap().trim().trim().parse::<u32>().unwrap();
            let tile_x = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let tile_y = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let tile_z = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let zoom = parts.next().unwrap().trim().parse::<f32>().unwrap();
            let x = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let y = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let score = parts.next().unwrap().trim().parse::<f32>().unwrap();
            let time = parts.next().unwrap().trim().parse::<f32>().unwrap();

            FrameData {
                frame,
                tile_x,
                tile_y,
                tile_z,
                zoom,
                x,
                y,
                _score: score,
                _time: time,
            }
        })
        .collect::<Vec<_>>();

    let i = AtomicUsize::new(0);
    frames.par_iter().for_each(|frame| {
        let needed = tiles_needed(
            mask_size,
            frame.tile_x,
            frame.tile_y,
            frame.tile_z,
            frame.zoom,
            frame.x,
            frame.y,
            Z_UP,
        );
        fetch_tiles_to_cache(&needed);
        render_final(
            frame.frame,
            mask_size,
            frame.tile_x,
            frame.tile_y,
            frame.tile_z,
            frame.zoom,
            frame.x,
            frame.y,
            needed.iter(),
        );
        let val = i.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if val % 100 == 0 {
            eprintln!("{} / {}", val, frames.len());
        }
    });
}
