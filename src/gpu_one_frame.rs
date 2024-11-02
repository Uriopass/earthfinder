use crate::gpu::State;
use crate::{data, TILE_SIZE};

pub fn gpu_one_frame() {
    let mask1 = data::mask_i(239);
    let mask2 = data::mask_i(4318);
    let mask_size = (mask1.width(), mask1.height());

    //let masks = &vec![(mask1, 239), (mask2, 4318)];
    let masks = &vec![(mask2, 4318)];
    let mut state = pollster::block_on(State::new((TILE_SIZE, TILE_SIZE), mask_size, masks.len()));

    let entries = data::tile_grad_entries();
    state.prepare(&entries);
    let (results, elapsed) = state.run_on_image(masks, &Default::default());
    eprintln!("processing took: {:.2}s", elapsed.as_secs_f32());

    let _ = std::fs::create_dir_all("data/results").unwrap();

    //let mut forbidden_tiles = FxHashSet::default();
    for (mask_idx, results) in results {
        for (i, result) in results.results().iter().enumerate().take(10) {
            eprintln!(
                "{}: {} {:?} {} {}",
                i,
                result.score,
                result.tile_pos(),
                result.x,
                result.y
            );
            let img = result.to_image(mask_idx, mask_size);
            img.save(format!("data/results/gpu_of_{}_{}.png", mask_idx, i))
                .unwrap();
        }
    }
}
