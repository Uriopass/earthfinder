use crate::ROOT;
use image::{GenericImage, GenericImageView, Rgb, Rgb32FImage};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
const BA_SIZE: u32 = 32;

pub fn process() {
    let i = AtomicU32::new(0);
    let t_start = Mutex::new(std::time::Instant::now());

    let max_score = Mutex::new(0.0);

    let to_walk = ROOT.replace("tiles", "tiles_oklab");

    let entries: Vec<_> = walkdir::WalkDir::new(to_walk).into_iter().collect();
    let to_process = entries.len();

    let bad_apple_pos = match image::open("data/bad_apple_pos.png") {
        Ok(image) => image.to_rgb8(),
        Err(e) => panic!("Could not open bad apple: {}", e),
    };
    let bad_apple_neg = match image::open("data/bad_apple_neg.png") {
        Ok(image) => image.to_rgb8(),
        Err(e) => panic!("Could not open bad apple: {}", e),
    };

    let mut bad_apple_f32: Rgb32FImage = Rgb32FImage::new(64, 64);

    for y in 0..BA_SIZE {
        for x in 0..BA_SIZE {
            let mut pos = bad_apple_pos.get_pixel(x, y).0[0] as f32 / 255.0;
            let mut neg = bad_apple_neg.get_pixel(x, y).0[0] as f32 / 255.0;

            pos = pos.sqrt();
            neg = neg.sqrt();

            bad_apple_f32.put_pixel(x, y, [pos, neg, 0.0].into());
        }
    }

    let mut tiles_scores: Vec<_> = entries
        .par_iter()
        .flat_map(|entry| {
            let Ok(entry) = entry else {
                return Default::default();
            };
            if !entry.file_type().is_file() {
                return Default::default();
            }
            let path = entry.path();

            if !path.display().to_string().ends_with(".png") {
                return Default::default();
            }

            let v = i.fetch_add(1, Ordering::Relaxed);
            if v % 100 == 0 {
                let t_now = std::time::Instant::now();
                let t_elapsed = t_now.duration_since(*t_start.lock().unwrap());
                let t_elapsed = t_elapsed.as_secs_f32();
                let t_remaining = t_elapsed / v as f32 * to_process as f32 - t_elapsed;
                eprintln!(
                    "{} images processed ({:.0}%) ETA: {}s",
                    v,
                    v as f32 / to_process as f32 * 100.0,
                    t_remaining
                );
            }

            let image = match image::open(path) {
                Ok(image) => image.to_rgb8(),
                Err(e) => panic!("Could not open image {}: {}", path.display(), e),
            };

            let mut image_f32 = Rgb32FImage::new(512, 512);

            for y in 0..512 {
                for x in 0..512 {
                    let [r, g, b] = image.get_pixel(x, y).0;
                    image_f32.put_pixel(
                        x,
                        y,
                        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0].into(),
                    );
                }
            }

            let mut total_pos = 0.0;
            let mut total_neg = 0.0;

            for y in 0..BA_SIZE {
                for x in 0..BA_SIZE {
                    let Rgb([pos, neg, ..]) = bad_apple_f32.get_pixel(x, y);

                    total_pos += pos;
                    total_neg += neg;
                }
            }

            let mut best_convol_score = 0.0;
            let mut best_convol_x = 0;
            let mut best_convol_y = 0;

            for convol_y in (0..512 - BA_SIZE).step_by(8) {
                for convol_x in (0..512 - BA_SIZE).step_by(8) {
                    let mut sum_pos = [0.0, 0.0, 0.0];
                    let mut sum_neg = [0.0, 0.0, 0.0];

                    for y in 0..BA_SIZE {
                        for x in 0..BA_SIZE {
                            let Rgb([pos, neg, ..]) =
                                unsafe { bad_apple_f32.unsafe_get_pixel(x, y) };
                            let pixel =
                                unsafe { image_f32.unsafe_get_pixel(x + convol_x, y + convol_y) };
                            let [r, g, b] = pixel.0;

                            sum_pos[0] += r * pos;
                            sum_pos[1] += g * pos;
                            sum_pos[2] += b * pos;

                            sum_neg[0] += r * neg;
                            sum_neg[1] += g * neg;
                            sum_neg[2] += b * neg;
                        }
                    }

                    let avg_white = [
                        sum_pos[0] / total_pos,
                        sum_pos[1] / total_pos,
                        sum_pos[2] / total_pos,
                    ];
                    let avg_black = [
                        sum_neg[0] / total_neg,
                        sum_neg[1] / total_neg,
                        sum_neg[2] / total_neg,
                    ];

                    let score = (avg_black[0] - avg_white[0]).abs()
                        + (avg_black[1] - avg_white[1]).abs()
                        + (avg_black[2] - avg_white[2]).abs();

                    if score > *max_score.lock().unwrap() {
                        *max_score.lock().unwrap() = score;
                        eprintln!("New max score: {} at {}", score, path.display());

                        export_convol(1337, &path, convol_x, convol_y);
                    }

                    if score > best_convol_score {
                        best_convol_score = score;
                        best_convol_x = convol_x;
                        best_convol_y = convol_y;
                    }
                }
            }

            drop(image);

            Some((
                OrderedFloat(best_convol_score),
                path.to_path_buf(),
                best_convol_x,
                best_convol_y,
            ))
        })
        .collect();

    tiles_scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    std::fs::create_dir_all("data/results").unwrap();

    for (i, (score, path, convol_x, convol_y)) in tiles_scores[..10].iter().enumerate() {
        println!("{}: {} {}-{}", score, path.display(), convol_x, convol_y);

        export_convol(i, path, *convol_x, *convol_y);
    }
}

fn export_convol(i: usize, path: &Path, convol_x: u32, convol_y: u32) {
    let image = image::open(path.display().to_string().replace("tiles_oklab", "tiles")).unwrap();

    let mut subtile = image::DynamicImage::new_rgba8(BA_SIZE, BA_SIZE);

    for y in 0..BA_SIZE {
        for x in 0..BA_SIZE {
            let pixel = image.get_pixel(x + convol_x, y + convol_y);
            subtile.put_pixel(x, y, pixel);
        }
    }

    subtile.save(format!("data/results/{}.png", i)).unwrap();
}
