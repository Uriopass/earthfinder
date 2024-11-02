use image::RgbaImage;
use walkdir::DirEntry;

pub fn tile_grad_entries() -> Vec<DirEntry> {
    print!("reading tile grad entries... ");
    let zoom_levels = [7];

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
