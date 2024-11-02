use image::imageops::FilterType;
use image::RgbImage;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

fn gaussian(x: i32, y: i32, sigma: f32) -> f32 {
    let sigma2 = sigma * sigma;
    let x2 = x as f32 * x as f32;
    let y2 = y as f32 * y as f32;
    ((-x2 - y2) / (2.0 * sigma2)).exp()
}

pub fn gen_masks() {
    let i = AtomicU32::new(0);
    let entries: Vec<_> = walkdir::WalkDir::new("data/bad_apple_frames")
        .into_iter()
        .collect();
    let to_process = entries.len();

    std::fs::create_dir_all("data/bad_apple_masks").unwrap_or_else(|e| {
        panic!("Could not create directory: {}", e);
    });

    let conv_size: i32 = 9;

    let sigma_first = 1.0;
    let sigma_second = 0.5;
    let sigma_third = 4.0;
    let sigma_fourth = 2.0;

    let mut gfirst = Vec::new();
    let mut gsecond = Vec::new();
    let mut gthird = Vec::new();
    let mut gfourth = Vec::new();

    let mut total_first = 0.0;
    let mut total_second = 0.0;
    let mut total_third = 0.0;
    let mut total_fourth = 0.0;

    for dy in -conv_size..=conv_size {
        for dx in -conv_size..=conv_size {
            let g1 = gaussian(dx, dy, sigma_first);
            gfirst.push(g1);
            total_first += g1;

            let g2 = gaussian(dx, dy, sigma_second);
            gsecond.push(g2);
            total_second += g2;

            let g3 = gaussian(dx, dy, sigma_third);
            gthird.push(g3);
            total_third += g3;

            let g4 = gaussian(dx, dy, sigma_fourth);
            gfourth.push(g4);
            total_fourth += g4;
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

        let mask_id = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .split('_')
            .last()
            .unwrap()
            .parse::<u32>()
            .unwrap();

        let v = i.fetch_add(1, Ordering::Relaxed);
        if v % 500 == 0 {
            eprintln!(
                "{} bad apple images processed ({:.0}%)",
                v,
                v as f32 / to_process as f32 * 100.0
            );
        }

        let ba = match image::open(path) {
            Ok(mut image) => {
                image = image.resize(
                    image.width() / 30,
                    image.height() / 30,
                    FilterType::Lanczos3,
                );

                image.to_rgb8()
            }
            Err(e) => panic!("Could not open image {}: {}", path.display(), e),
        };

        let mut mask_image = RgbImage::new(ba.width(), ba.height());

        for y in 0..ba.height() {
            for x in 0..ba.width() {
                let mut sum_first = 0.0;
                let mut sum_second = 0.0;
                let mut sum_third = 0.0;
                let mut sum_fourth = 0.0;

                for dy in -conv_size..=conv_size {
                    let yy = (y as i32 + dy).clamp(0, ba.height() as i32 - 1);
                    for dx in -conv_size..=conv_size {
                        let xx = (x as i32 + dx).clamp(0, ba.width() as i32 - 1);

                        let pixel = ba.get_pixel(xx as u32, yy as u32);

                        let v = (pixel.0[0] > 210) as u8 as f32;
                        let g1 = gfirst
                            [((dy + conv_size) * (2 * conv_size + 1) + dx + conv_size) as usize];
                        let g2 = gsecond
                            [((dy + conv_size) * (2 * conv_size + 1) + dx + conv_size) as usize];
                        let g3 = gthird
                            [((dy + conv_size) * (2 * conv_size + 1) + dx + conv_size) as usize];
                        let g4 = gfourth
                            [((dy + conv_size) * (2 * conv_size + 1) + dx + conv_size) as usize];

                        sum_first += v * g1;
                        sum_second += v * g2;
                        sum_third += v * g3;
                        sum_fourth += v * g4;
                    }
                }

                let first = sum_first / total_first;
                let second = sum_second / total_second;
                let third = sum_third / total_third;
                let fourth = sum_fourth / total_fourth;

                let mask1 = (first - second).abs().clamp(0.0, 1.0) * 255.0;
                let mask2 =
                    ((third - fourth).abs() - (first - second).abs()).clamp(0.0, 1.0) * 255.0;

                mask_image.put_pixel(x, y, From::from([mask1 as u8, mask2 as u8, 0]));
            }
        }

        let path_string = format!("data/bad_apple_masks/bad_apple_{}.png", mask_id);

        mask_image.save(&path_string).unwrap_or_else(|e| {
            panic!("Could not save image {}: {}", path_string, e);
        });
    });
}
