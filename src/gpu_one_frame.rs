use crate::data;
use crate::gpu::State;
use image::{Rgb32FImage, RgbaImage};

pub fn gpu_one_frame(zs: &[u32]) {
    let mask_ids = vec![239, 4318];
    // let mask_ids = vec![239, 4318];
    let masks = mask_ids
        .iter()
        .map(|&i| (data::mask_i(i), i))
        .collect::<Vec<_>>();
    let mask_size = masks[0].0.dimensions();
    let mut state = pollster::block_on(State::new(mask_size, masks.len()));

    let entries = data::tile_grad_entries(zs);
    state.prepare(&entries);

    let last_tile = RgbaImage::new(mask_size.0 / 4, mask_size.1 / 4);

    let (results, elapsed) = state.run_on_image(
        &masks
            .iter()
            .map(|(data, i)| (data, *i, &last_tile))
            .collect::<Vec<_>>(),
        &Default::default(),
    );
    eprintln!("processing took: {:.2}s", elapsed.as_secs_f32());

    let _ = std::fs::create_dir_all("data/results").unwrap();

    let avg_error = Rgb32FImage::new(mask_size.0, mask_size.1);

    //let mut forbidden_tiles = FxHashSet::default();
    for (mask_i, (mask_idx, results)) in results.iter().enumerate() {
        for (i, result) in results.results().iter().enumerate().take(10) {
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
