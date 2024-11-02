use crate::gpu::State;
use crate::{data, TILE_SIZE};
use image::{DynamicImage, GrayImage, Rgb32FImage, RgbaImage};

static SAVE_ERROR: bool = false;

pub fn gpu_all() {
    std::fs::create_dir_all("data/results/frames").unwrap();

    let mask_example = data::mask_i(3350);
    let mask_dims = (mask_example.width(), mask_example.height());
    let mask_chunk_size = 1;
    let mut state = pollster::block_on(State::new(
        (TILE_SIZE, TILE_SIZE),
        mask_dims,
        mask_chunk_size,
    ));

    let mask_idxs = (3350..3350 + 30 * 15).collect::<Vec<_>>();

    let entries = data::tile_grad_entries();

    state.prepare(&entries);
    drop(entries);

    let mut last_tile_rgb = RgbaImage::new(mask_dims.0 / 4, mask_dims.1 / 4);

    let mut avg_error = Rgb32FImage::new(mask_dims.0, mask_dims.1);
    avg_error.fill(0.5);

    for mask_idx in mask_idxs {
        let mut mask = data::mask_i(mask_idx);

        mask.enumerate_pixels_mut().for_each(|(x, y, p)| {
            let apply_error = |v| {
                let err = avg_error.get_pixel(x, y).0[0];
                (v as f32 * (0.2 + err * 0.8)) as u8
            };

            p.0[0] = apply_error(p.0[0]);
            p.0[1] = apply_error(p.0[1]);
        });

        let (results, elapsed) =
            state.run_on_image(&[(&mask, mask_idx, &last_tile_rgb)], &Default::default());
        let result = results[0].1.results()[0];

        println!(
            "Frame {} -> {:?} score: {:.4} (in {:.2}s)",
            mask_idx,
            result.tile_pos(),
            result.score,
            elapsed.as_secs_f32()
        );

        avg_error.pixels_mut().for_each(|p| {
            p.0[0] *= 0.5;
        });
        result.calc_error(&mask, |x, y, err| {
            avg_error.get_pixel_mut(x, y).0[0] += err;
        });

        if SAVE_ERROR {
            let mut error_show = GrayImage::new(mask_dims.0, mask_dims.1);
            avg_error.enumerate_pixels().for_each(|(x, y, p)| {
                error_show.put_pixel(x, y, image::Luma([((p.0[0] * 255.0) as u8).min(255)]));
            });

            error_show
                .save(format!("data/results/frames/{}_avg_error.png", mask_idx))
                .unwrap();
        }

        let img = result.to_image(&mask, &avg_error);
        img.save(format!("data/results/frames/{}.png", mask_idx))
            .unwrap();

        let img_rgb = result.to_rgba(mask.dimensions());
        last_tile_rgb = DynamicImage::ImageRgba8(img_rgb)
            .resize(
                mask_dims.0 / 4,
                mask_dims.1 / 4,
                image::imageops::FilterType::Gaussian,
            )
            .to_rgba8();
    }

    /*
    for mask_idxs in mask_idxs.chunks_exact(mask_chunk_size) {
        let mut masks = Vec::with_capacity(mask_idxs.len());

        for mask_i in mask_idxs {
            masks.push((data::mask_i(*mask_i), *mask_i));
        }

        let (results, elapsed) = state.run_on_image(&masks, &Default::default());

        for (mask_i, result) in results {
            let result = result.results()[0];
            //forbidden_tiles.insert(result.tile_pos());

            println!(
                "Frame {} -> {:?} score: {:.4} (in {:.2}s)",
                mask_i,
                result.tile_pos(),
                result.score,
                elapsed.as_secs_f32()
            );

            let img = result.to_image(mask_i, mask_dims);
            img.save(format!("data/results/frames/{}.png", mask_i))
                .unwrap();
        }
    } */
}
