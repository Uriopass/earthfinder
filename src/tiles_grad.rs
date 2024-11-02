use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{
    DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgb, Rgb32FImage, RgbImage,
};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

static SHOW_ORIGINAL: bool = false;

pub fn gen_tiles_grad(z: u32) {
    let i = AtomicU32::new(0);
    let entries: Vec<_> = walkdir::WalkDir::new(format!("data/tiles/{}", z))
        .into_iter()
        .collect();
    let to_process = entries.len();

    let _ = std::fs::remove_dir_all(format!("data/tiles_grad/{}", z));
    let _ = std::fs::remove_dir_all(format!("data/tiles_smol/{}", z));

    for entry in entries.iter() {
        let Ok(entry) = entry else {
            continue;
        };
        let ftype = entry.file_type();

        if ftype.is_dir() {
            let mut path = entry.path().display().to_string();
            path = path.replace("tiles", "tiles_grad");
            let _ = std::fs::create_dir_all(path);

            let mut path = entry.path().display().to_string();
            path = path.replace("tiles", "tiles_smol");
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

        let path_str = path.display().to_string();
        let parts = path_str
            .split(std::path::MAIN_SEPARATOR)
            .collect::<Vec<_>>();
        let y = parts[parts.len() - 2].parse::<u32>().unwrap();

        let latitude = 90.0 - (y as f32 / (1 << (z - 1)) as f32) * 180.0;

        if latitude.abs() > 70.0 {
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

        let tmp = DynamicImage::ImageRgb8(image);
        let image_smol = tmp.resize(
            tmp.width() / 4,
            tmp.height() / 4,
            image::imageops::FilterType::Gaussian,
        );
        let DynamicImage::ImageRgb8(image) = tmp else {
            unreachable!("by construction");
        };

        let mut image_f32 = Rgb32FImage::new(image.width(), image.height());

        let mut n_zeros = 0;

        for (x, y, pixel) in image.enumerate_pixels() {
            let [r, g, b] = pixel.0;

            if r == 0 && g == 0 && b == 0 {
                unsafe {
                    image_f32.unsafe_put_pixel(x, y, Rgb::<f32>::from([0.0, 0.0, 0.0]));
                }
                n_zeros += 1;
                continue;
            }

            let lab = oklab::srgb_to_oklab(oklab::Rgb::new(r, g, b));

            if lab.a.abs() > 0.25 || lab.b.abs() > 0.25 {
                return;
            }

            unsafe {
                image_f32.unsafe_put_pixel(x, y, Rgb::<f32>::from([lab.l, lab.a, lab.b]));
            }
        }

        if n_zeros as f32 > 0.05 * image.width() as f32 * image.height() as f32 {
            return;
        }

        let gradient_w = if SHOW_ORIGINAL {
            image.width() * 2
        } else {
            image.width()
        };
        let mut gradient_image: RgbImage = ImageBuffer::new(gradient_w, image.height());

        const CONV_SIZE: i32 = 1;

        if SHOW_ORIGINAL {
            for y in 0..image.height() {
                for x in 0..image.width() {
                    gradient_image.put_pixel(x + image.width(), y, *image.get_pixel(x, y));
                }
            }
        }

        for y in 0..image.height() {
            for x in 0..image.width() {
                if x < CONV_SIZE as u32
                    || y < CONV_SIZE as u32
                    || x >= image.width() - CONV_SIZE as u32
                    || y >= image.height() - CONV_SIZE as u32
                {
                    gradient_image.put_pixel(x, y, From::from([0, 0, 0]));
                    continue;
                }

                let mut gx = [0.0, 0.0, 0.0];
                let mut gy = [0.0, 0.0, 0.0];

                let mut has_zero = false;

                'outer: for dy in -CONV_SIZE..=CONV_SIZE {
                    let ny = y as i32 + dy;
                    for dx in -CONV_SIZE..=CONV_SIZE {
                        let nx = x as i32 + dx;
                        let pixel = unsafe { image_f32.unsafe_get_pixel(nx as u32, ny as u32) };

                        if pixel[0] == 0.0 && pixel[1] == 0.0 && pixel[2] == 0.0 {
                            has_zero = true;
                            break 'outer;
                        }

                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        if dx != 0 {
                            let dx_mult = ((1 << dx.abs()) * dx.signum()) as f32;
                            gx[0] += pixel[0] / dx_mult;
                            gx[1] += pixel[1] / dx_mult;
                            gx[2] += pixel[2] / dx_mult;
                        }

                        if dy != 0 {
                            let dy_mult = ((1 << dy.abs()) * dy.signum()) as f32;
                            gy[0] += pixel[0] / dy_mult;
                            gy[1] += pixel[1] / dy_mult;
                            gy[2] += pixel[2] / dy_mult;
                        }
                    }
                }

                // reserve this value for zero input
                if has_zero {
                    gradient_image.put_pixel(x, y, From::from([255, 0, 0]));
                    continue;
                }

                let gx_norm = (gx[0] * gx[0] + gx[1] * gx[1] + gx[2] * gx[2]).sqrt();
                let gy_norm = (gy[0] * gy[0] + gy[1] * gy[1] + gy[2] * gy[2]).sqrt();

                let pix: Rgb<u8> =
                    From::from([(gx_norm * 170.0) as u8, (gy_norm * 170.0) as u8, 0]);

                if pix.0[0] == 255 {
                    gradient_image.put_pixel(x, y, From::from([254, 0, 0]));
                    continue;
                }

                unsafe {
                    gradient_image.unsafe_put_pixel(x, y, pix);
                }
            }
        }

        {
            let path_smol_str = path
                .display()
                .to_string()
                .replace("data/tiles", "data/tiles_smol");

            let mut path_smol = PathBuf::from(path_smol_str);
            path_smol.set_extension("png");

            image_smol.save(path_smol).unwrap();
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