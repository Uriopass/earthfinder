use crate::ROOT;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

pub fn preprocess() {
    let i = AtomicU32::new(0);
    let entries: Vec<_> = walkdir::WalkDir::new(ROOT).into_iter().collect();
    let to_process = entries.len();

    for entry in entries.iter() {
        let Ok(entry) = entry else {
            continue;
        };
        let ftype = entry.file_type();

        if ftype.is_dir() {
            let mut path = entry.path().display().to_string();
            path = path.replace("tiles", "tiles_oklab");
            let _ = std::fs::create_dir_all(path);
            continue;
        }
    }

    entries.par_iter().for_each(|entry| {
        let Ok(entry) = entry else {
            return;
        };
        if !entry.file_type().is_file() {
            return;
        }
        let path = entry.path();

        if !path.display().to_string().ends_with(".png") {
            return;
        }

        let v = i.fetch_add(1, Ordering::Relaxed);
        if v % 1000 == 0 {
            eprintln!(
                "{} images processed ({:.0}%)",
                v,
                v as f32 / to_process as f32 * 100.0
            );
        }

        let mut image = match image::open(path) {
            Ok(image) => image.to_rgb8(),
            Err(e) => panic!("Could not open image {}: {}", path.display(), e),
        };

        for pixel in image.pixels_mut() {
            let [r, g, b] = pixel.0;
            if r == 0 && g == 0 && b == 0 {
                return;
            }
            let lab = oklab::srgb_to_oklab(oklab::Rgb::new(r, g, b));
            *pixel = image::Rgb([
                (lab.l * 255.0) as u8,
                ((lab.b + 0.5) * 255.0) as u8,
                ((lab.a + 0.5) * 255.0) as u8,
            ]);
        }

        let mut path_string = path.display().to_string();

        path_string = path_string.replace("tiles", "tiles_oklab");

        image.save(&path_string).unwrap_or_else(|e| {
            panic!("Could not save image {}: {}", path_string, e);
        });
    });
}
