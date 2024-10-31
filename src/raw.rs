use image::RgbaImage;
use std::path::Path;

pub fn read_raw(path: &Path) -> RgbaImage {
    let content = std::fs::read(path).unwrap_or_else(|e| {
        panic!("Could not read file {}: {}", path.display(), e);
    });

    RgbaImage::from_raw(512, 512, content).unwrap()
}
