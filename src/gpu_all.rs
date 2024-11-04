use crate::gpu::State;
use crate::{data, TILE_SIZE};
use image::{DynamicImage, GrayImage, Rgb32FImage, RgbaImage};
use rustc_hash::FxHashSet;
use std::fs::File;
use std::io::Write;

static SAVE_ERROR: bool = false;

pub fn gpu_all(zs: &[u32]) {
    let _ = std::fs::create_dir_all("data/results/frames");
    let _ = std::fs::create_dir_all("data/results/frames_debug");

    let mask_example = data::mask_i(3350);
    let mask_dims = (mask_example.width(), mask_example.height());
    let mask_chunk_size = 1;
    let mut state = pollster::block_on(State::new(
        (TILE_SIZE, TILE_SIZE),
        mask_dims,
        mask_chunk_size,
    ));

    //let mask_idxs = (3350..3350 + 30 * 15).collect::<Vec<_>>();
    let mask_idxs = (50..2000).collect::<Vec<_>>();

    let entries = data::tile_grad_entries(zs);

    state.prepare(&entries);
    drop(entries);

    let mut last_tile_rgb = RgbaImage::new(mask_dims.0 / 4, mask_dims.1 / 4);

    let mut avg_error = Rgb32FImage::new(mask_dims.0, mask_dims.1);
    avg_error.fill(0.1);

    let mut forbidden_tile_ring = Vec::with_capacity(30);

    let mut forbidden_tiles = FxHashSet::default();

    let result_csv = File::create("data/results/out.csv").unwrap();
    let mut bufwriter = std::io::BufWriter::new(result_csv);

    writeln!(&mut bufwriter, "Frame,tile_x,tile_y,tile_z,x,y,score,time").unwrap();

    for mask_idx in mask_idxs {
        let mut mask = data::mask_i(mask_idx);

        mask.enumerate_pixels_mut().for_each(|(x, y, p)| {
            let apply_error = |v| {
                let err = avg_error.get_pixel(x, y).0[0];
                (v as f32 * (0.5 + err * 0.5)) as u8
            };

            p.0[0] = apply_error(p.0[0]);
            p.0[1] = apply_error(p.0[1]);
        });

        let (results, elapsed) =
            state.run_on_image(&[(&mask, mask_idx, &last_tile_rgb)], &forbidden_tiles);
        let result = results[0].1.results()[0];

        forbidden_tile_ring.push(result.tile_pos());
        forbidden_tiles.insert(result.tile_pos());

        if forbidden_tile_ring.len() >= 15 {
            forbidden_tiles.remove(&forbidden_tile_ring.remove(0));
        }

        println!(
            "Frame {}: ({:>3},{:>3},{}) ({:>3},{:>3}) score:{:>7.4} t:{:.2}s",
            mask_idx,
            result.tile_x,
            result.tile_y,
            result.tile_z,
            result.x,
            result.y,
            result.score,
            elapsed.as_secs_f32()
        );

        writeln!(
            &mut bufwriter,
            "{},{:>3},{:>3},{},{:>3},{:>3},{:>7.4},{:.2}",
            mask_idx,
            result.tile_x,
            result.tile_y,
            result.tile_z,
            result.x,
            result.y,
            result.score,
            elapsed.as_secs_f32()
        )
        .unwrap();

        let _ = bufwriter.flush();

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

        let img_debug = result.to_image(&mask, &avg_error, true);
        img_debug
            .save(format!("data/results/frames_debug/{}.png", mask_idx))
            .unwrap();

        let img = result.to_image(&mask, &avg_error, false);
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
