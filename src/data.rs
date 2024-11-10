use crate::gpu::algorithm::PosResult;
use image::RgbaImage;
use std::f32::consts::{FRAC_PI_2, PI};
use walkdir::DirEntry;

/// (x, y, z)
pub type TilePos = (u32, u32, u32);

/// How much to warp width to get orthonormal distances from a tile
/// Always between [0-1]
pub fn deformation(y: u32, z: u32) -> f32 {
    let n_y_tiles = (1 << (z - 1)) as f32;
    let latitude = FRAC_PI_2 - (y as f32 / n_y_tiles) * PI;

    latitude.cos()
}

pub struct FrameData {
    pub frame: u32,
    pub result: PosResult,
}

pub fn parse_csv(content: &str) -> Vec<FrameData> {
    content
        .lines()
        .skip(1) // skip header
        .filter(|line| !line.trim().is_empty())
        .filter(|line| !line.starts_with('#'))
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

            FrameData {
                frame,
                result: PosResult {
                    tile_x,
                    tile_y,
                    tile_z,
                    zoom,
                    x,
                    y,
                    score,
                },
            }
        })
        .collect::<Vec<_>>()
}

pub fn deform_width(width: u32, y: u32, z: u32) -> u32 {
    (width as f32 * deformation(y, z)).ceil() as u32
}

pub fn extract_tile_pos(path_str: &str) -> TilePos {
    let parts = path_str
        .split([std::path::MAIN_SEPARATOR, '/'])
        .collect::<Vec<_>>();
    let x = parts[parts.len() - 1]
        .split_once(".")
        .unwrap()
        .0
        .parse::<u32>()
        .unwrap();
    let y = parts[parts.len() - 2].parse::<u32>().unwrap();
    let z = parts[parts.len() - 3].parse::<u32>().unwrap();
    (x, y, z)
}

#[allow(dead_code)]
pub fn debug_entry(tile_x: u32, tile_y: u32, tile_z: u32) -> Vec<DirEntry> {
    print!(
        "reading debug tile entry {} {} {}...",
        tile_x, tile_y, tile_z
    );
    let entries: Vec<_> = walkdir::WalkDir::new(format!(
        "data/tiles_grad{}{}{}{}",
        std::path::MAIN_SEPARATOR,
        tile_z,
        std::path::MAIN_SEPARATOR,
        tile_y
    ))
    .into_iter()
    .filter_map(|v| match v {
        Ok(entry) => Some(entry),
        Err(e) => {
            eprintln!("Error reading entry: {}", e);
            None
        }
    })
    .filter(|entry| {
        let path = entry.path().display().to_string();
        if path.ends_with(format!("{}.png", tile_x).as_str()) {
            println!("found");
            true
        } else {
            false
        }
    })
    .collect();

    println!("done ({} entries)", entries.len());

    entries
}

pub fn tile_grad_entries(zoom_levels: &[u32]) -> Vec<DirEntry> {
    print!("reading tile grad entries {:?} ...", zoom_levels);
    let mut entries = Vec::with_capacity(50000);

    for z in zoom_levels {
        entries.extend(
            walkdir::WalkDir::new(format!("data/tiles_grad{}{}", std::path::MAIN_SEPARATOR, z))
                .into_iter()
                .filter_map(|v| match v {
                    Ok(entry) => Some(entry),
                    Err(e) => {
                        eprintln!("Error reading entry: {}", e);
                        None
                    }
                })
                .filter(|entry| {
                    let path = entry.path().display().to_string();
                    entry.file_type().is_file()
                        && (path.ends_with(".png")
                            || path.ends_with(".jpeg")
                            || path.ends_with(".jpg"))
                }),
        );
    }

    println!("done ({} entries)", entries.len());

    entries
}

pub fn mask_i(i: u32) -> RgbaImage {
    image::open(format!("data/bad_apple_masks/bad_apple_{}.png", i))
        .unwrap()
        .to_rgba8()
}

pub fn sanity_check() {
    let check_7 = format!("data/tiles_grad/7/27/35.png");
    let check_8 = format!("data/tiles_grad/8/54/70.png");
    let check_9 = format!("data/tiles_grad/9/108/140.png");

    let tiles_grad_rs = format!("src/tiles_grad.rs");

    fn systime(path: &str) -> Option<std::time::SystemTime> {
        std::fs::metadata(path).ok().and_then(|m| m.modified().ok())
    }

    let check_7_time = systime(&check_7);
    let check_8_time = systime(&check_8);
    let check_9_time = systime(&check_9);

    let tiles_grad_rs_time = systime(&tiles_grad_rs);

    if check_7_time < tiles_grad_rs_time {
        eprintln!("/!\\ Warning: tiles_grad 7 is older than tiles_grad.rs");
    }

    if check_8_time < tiles_grad_rs_time {
        eprintln!("/!\\ Warning: tiles_grad 8 is older than tiles_grad.rs");
    }

    if check_9_time < tiles_grad_rs_time {
        eprintln!("/!\\ Warning: tiles_grad 9 is older than tiles_grad.rs");
    }

    let check_mask = format!("data/bad_apple_masks/bad_apple_5.png");

    let gen_mask_rs = format!("src/gen_mask.rs");

    let check_mask_time = systime(&check_mask);
    let gen_mask_rs_time = systime(&gen_mask_rs);

    if check_mask_time < gen_mask_rs_time {
        eprintln!("/!\\ Warning: mask 5 is older than gen_mask.rs");
    }
}
