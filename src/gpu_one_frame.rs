use crate::data;
use crate::gpu::State;
use image::Rgb32FImage;

pub fn gpu_one_frame(zs: &[u32]) {
    let mask_ids = vec![4318];
    let mut masks = mask_ids
        .iter()
        .map(|&i| (data::mask_i(i), i))
        .collect::<Vec<_>>();
    let mask_size = masks[0].0.dimensions();
    let mut state = pollster::block_on(State::new(mask_size, masks.len()));

    let first_result = crate::gpu::algorithm::PosResult {
        tile_x: 221,
        tile_y: 35,
        tile_z: 8,
        x: 43,
        y: 56,
        score: -1.97,
        zoom: 1.5,
    };
    let last_tile_rgb = first_result.to_rgba_quarter(mask_size);
    //let last_tile_rgb = RgbaImage::new(mask_size.0 / 4, mask_size.1 / 4);

    let entries = data::tile_grad_entries(zs);
    //let entries = data::debug_entry(70, 68, 8);
    state.prepare(&entries);

    let mut avg_error = Rgb32FImage::new(mask_size.0, mask_size.1);
    avg_error.fill(0.5);
    avg_error.pixels_mut().for_each(|p| {
        p.0[0] *= 0.5;
    });

    for (mask, _) in &mut masks {
        mask.enumerate_pixels_mut().for_each(|(x, y, p)| {
            let apply_error = |v| {
                let err = avg_error.get_pixel(x, y).0[0];
                (v as f32 * (0.7 + err * 0.3)) as u8
            };

            p.0[0] = apply_error(p.0[0]);
            p.0[1] = apply_error(p.0[1]);
        });
    }

    let (results, elapsed_gpu) = state.run_on_image(
        &masks
            .iter()
            .map(|(data, i)| (data, *i, &last_tile_rgb))
            .collect::<Vec<_>>(),
        &Default::default(),
    );
    eprintln!("processing took: {:.2}s", elapsed_gpu.as_secs_f32());

    let _ = std::fs::create_dir_all("data/results").unwrap();

    //let mut forbidden_tiles = FxHashSet::default();
    for (mask_i, (mask_idx, results)) in results.iter().enumerate() {
        for (i, result) in results.results().iter().enumerate() {
            eprintln!(
                "{}: {} (z{}) {:?} {} {}",
                i,
                result.score,
                result.zoom,
                result.tile_pos(),
                result.x,
                result.y
            );
            let img = result.to_image(&masks[mask_i].0, &avg_error, true);
            img.save(format!("data/results/gpu_of_{}_{}.png", mask_idx, i))
                .unwrap();
        }
    }
}
