use crate::gpu::State;
use crate::{ROOT, TILE_SIZE};

pub fn gpu_one_frame() {
    let mask1 = image::open("data/bad_apple_masks/bad_apple_239.png").unwrap();
    let mask2 = image::open("data/bad_apple_masks/bad_apple_4318.png").unwrap();
    let mask_size = (mask1.width(), mask1.height());

    //let masks = &vec![(mask1, 239), (mask2, 4318)];
    let masks = &vec![(mask2, 4318)];
    let mut state = pollster::block_on(State::new((TILE_SIZE, TILE_SIZE), mask_size, masks.len()));

    let mut entries: Vec<_> = walkdir::WalkDir::new(ROOT.replace("tiles", "tiles_grad"))
        .into_iter()
        .filter_map(|v| v.ok())
        .collect();

    entries.retain(|entry| {
        entry.file_type().is_file() && entry.path().display().to_string().ends_with(".png")
    });

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
