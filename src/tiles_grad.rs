use crate::data::{deform_width, deformation, extract_tile_pos};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{GenericImage, GenericImageView, ImageBuffer, Rgb, Rgb32FImage, RgbImage};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

static SHOW_ORIGINAL: bool = false;

pub fn gen_tiles_grad(tile_z: u32) {
    let i = AtomicU32::new(0);
    let entries: Vec<_> = walkdir::WalkDir::new(format!("data/tiles/{}", tile_z))
        .into_iter()
        .collect();
    let mut to_process = 0;

    let _ = std::fs::remove_dir_all(format!("data/tiles_grad/{}", tile_z));
    let _ = std::fs::remove_dir_all(format!("data/tiles_smol/{}", tile_z));

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
        } else {
            to_process += 1;
        }
    }

    let skipped = AtomicUsize::new(0);

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
        let (_, tile_y, _) = extract_tile_pos(&path_str);
        let latitude = 90.0 - (tile_y as f32 / (1 << (tile_z - 1)) as f32) * 180.0;

        if latitude.abs() > 70.0 {
            skipped.fetch_add(1, Ordering::Relaxed);
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

        let orig_image = match image::open(path) {
            Ok(image) => image.to_rgb8(),
            Err(e) => panic!("Could not open image {}: {}", path.display(), e),
        };

        let new_w = deform_width(orig_image.width(), tile_y, tile_z);
        let deformation_coeff = deformation(tile_y, tile_z);

        let mut image = image::imageops::resize(
            &orig_image,
            new_w,
            orig_image.height(),
            image::imageops::FilterType::Lanczos3,
        );

        let mut image_smol = image::imageops::resize(
            &image,
            image.width() / 4,
            image.height() / 4,
            image::imageops::FilterType::Gaussian,
        );

        for (x, y, pixel) in orig_image.enumerate_pixels() {
            if pixel.0 == [0, 0, 0] {
                let newx = ((x as f32) * deformation_coeff) as u32;
                image.put_pixel(newx, y, *pixel);
                image_smol.put_pixel((newx / 4).min(image_smol.width() - 1), y / 4, *pixel);
            }
        }

        let mut image_f32 = Rgb32FImage::new(image.width(), image.height());

        let mut n_zeros = 0;

        let mut n_not_blue = 0;

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
                skipped.fetch_add(1, Ordering::Relaxed);
                return;
            }

            if (lab.a > 0.2 || lab.b > 0.0) && lab.l < 0.92 {
                n_not_blue += 1;
            }

            unsafe {
                image_f32.unsafe_put_pixel(x, y, Rgb::<f32>::from([lab.l, lab.a, lab.b]));
            }
        }

        if n_zeros as f32 > 0.02 * image.width() as f32 * image.height() as f32 {
            skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }

        if n_not_blue < (32.0 * deformation(tile_y, tile_z)) as u32 {
            skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let gradient_w = if SHOW_ORIGINAL {
            image.width() * 2
        } else {
            image.width()
        };
        let mut gradient_image: RgbImage = ImageBuffer::new(gradient_w, image.height());

        const CONV_SIZE: i32 = 2;

        if SHOW_ORIGINAL {
            for y in 0..image.height() {
                for x in 0..image.width() {
                    gradient_image.put_pixel(x + image.width(), y, *image.get_pixel(x, y));
                }
            }
        }

        let mut max_grad: f32 = 0.0;

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

                        let mult = 1.0 / (2 * dx.abs() + 2 * dy.abs()) as f32;

                        if dx != 0 {
                            let dx_mult = mult * dx.signum() as f32;
                            gx[0] += pixel[0] * dx_mult;
                            gx[1] += pixel[1] * dx_mult;
                            gx[2] += pixel[2] * dx_mult;
                        }

                        if dy != 0 {
                            let dy_mult = mult * dy.signum() as f32;
                            gy[0] += pixel[0] * dy_mult;
                            gy[1] += pixel[1] * dy_mult;
                            gy[2] += pixel[2] * dy_mult;
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

                max_grad = max_grad.max(gx_norm).max(gy_norm);

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

    eprintln!(
        "Skipped {}/{} images",
        skipped.load(Ordering::SeqCst),
        to_process
    );
}
