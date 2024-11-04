use image::RgbaImage;
use std::f32::consts::{FRAC_PI_2, PI};
use walkdir::DirEntry;

/// How much to warp width to get orthonormal distances from a tile
/// Always between [0-1]
pub fn deformation(z: u32, y: u32) -> f32 {
    let n_y_tiles = (1 << (z - 1)) as f32;
    let latitude = FRAC_PI_2 - (y as f32 / n_y_tiles) * PI;

    latitude.cos()
}

pub fn tile_grad_entries(zoom_levels: &[u32]) -> Vec<DirEntry> {
    print!("reading tile grad entries... {:?}", zoom_levels);
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
