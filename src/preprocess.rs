use crate::ROOT;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{GenericImage, GenericImageView, GrayImage, ImageBuffer, Rgb, Rgb32FImage};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

pub fn preprocess() {
    let i = AtomicU32::new(0);
    let entries: Vec<_> = walkdir::WalkDir::new(ROOT).into_iter().collect();
    let to_process = entries.len();

    let _ = std::fs::remove_dir_all("data/tiles_grad").unwrap();

    for entry in entries.iter() {
        let Ok(entry) = entry else {
            continue;
        };
        let ftype = entry.file_type();

        if ftype.is_dir() {
            let mut path = entry.path().display().to_string();
            path = path.replace("tiles", "tiles_grad");
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

        let image = match image::open(path) {
            Ok(image) => image.to_rgb8(),
            Err(e) => panic!("Could not open image {}: {}", path.display(), e),
        };

        let mut image_f32 = Rgb32FImage::new(image.width(), image.height());

        for (x, y, pixel) in image.enumerate_pixels() {
            let [r, g, b] = pixel.0;
            if r == 0 && g == 0 && b == 0 {
                return;
            }
            let lab = oklab::srgb_to_oklab(oklab::Rgb::new(r, g, b));

            if lab.a.abs() > 0.25 || lab.b.abs() > 0.25 {
                return;
            }

            unsafe {
                image_f32.unsafe_put_pixel(x, y, Rgb::<f32>::from([lab.l, lab.a, lab.b]));
            }
        }

        let mut gradient_image: GrayImage = ImageBuffer::new(image.width(), image.height());

        const CONV_SIZE: i32 = 3;

        for y in 0..image.height() {
            for x in 0..image.width() {
                if x < CONV_SIZE as u32
                    || y < CONV_SIZE as u32
                    || x >= image.width() - CONV_SIZE as u32
                    || y >= image.height() - CONV_SIZE as u32
                {
                    gradient_image.put_pixel(x, y, From::from([0]));
                    continue;
                }

                let mut gx = [0.0, 0.0, 0.0];
                let mut gy = [0.0, 0.0, 0.0];

                for dy in -CONV_SIZE..=CONV_SIZE {
                    let ny = y as i32 + dy;
                    if ny < 0 || ny >= image.height() as i32 {
                        continue;
                    }
                    for dx in -CONV_SIZE..=CONV_SIZE {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i32 + dx;
                        if nx < 0 || nx >= image.width() as i32 {
                            continue;
                        }
                        let pixel = unsafe { image_f32.unsafe_get_pixel(nx as u32, ny as u32) };

                        if dx != 0 {
                            gx[0] += pixel[0] / dx as f32;
                            gx[1] += pixel[1] / dx as f32;
                            gx[2] += pixel[2] / dx as f32;
                        }

                        if dy != 0 {
                            gy[0] += pixel[0] / dy as f32;
                            gy[1] += pixel[1] / dy as f32;
                            gy[2] += pixel[2] / dy as f32;
                        }
                    }
                }

                let gx_norm = gx[0] * gx[0] + gx[1] * gx[1] + gx[2] * gx[2];
                let gy_norm = gy[0] * gy[0] + gy[1] * gy[1] + gy[2] * gy[2];
                let v = (gx_norm + gy_norm) * 50.0;

                unsafe {
                    gradient_image.unsafe_put_pixel(x, y, From::from([v as u8]));
                }
            }
        }

        let mut path_string = path.display().to_string();

        path_string = path_string.replace("tiles", "tiles_grad");

        let mut path = PathBuf::from(path_string);
        path.set_extension("png");

        let file_writer = std::fs::File::create(&path).unwrap_or_else(|e| {
            panic!("Could not create file {}: {}", path.display(), e);
        });
        let mut bufwriter = std::io::BufWriter::new(file_writer);

        /*gradient_image
        .write_with_encoder(JpegEncoder::new_with_quality(&mut bufwriter, 95))
        .expect("Could not write image");*/

        gradient_image
            .write_with_encoder(PngEncoder::new_with_quality(
                &mut bufwriter,
                CompressionType::Best,
                FilterType::Adaptive,
            ))
            .expect("Could not write image");
    });
}
