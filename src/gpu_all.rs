use crate::gpu::State;
use crate::ROOT;

pub fn gpu_all() {
    std::fs::create_dir_all("data/results/frames").unwrap();

    let tile_size = (512, 512);
    let mask_example = image::open("data/bad_apple_masks/bad_apple_3350.png").unwrap();
    let mask_dims = (mask_example.width(), mask_example.height());
    let mask_chunk_size = 2;
    let mut state = pollster::block_on(State::new(tile_size, mask_dims, mask_chunk_size));

    let mask_idxs = (3350..3350 + 30 * 15).collect::<Vec<_>>();

    eprintln!("Reading entries");
    let mut entries: Vec<_> = walkdir::WalkDir::new(ROOT.replace("tiles", "tiles_oklab"))
        .into_iter()
        .filter_map(|v| v.ok())
        .collect();
    entries.retain(|entry| {
        entry.file_type().is_file() && entry.path().display().to_string().ends_with(".jpg")
    });

    state.prepare(&entries);
    drop(entries);

    for mask_idxs in mask_idxs.chunks_exact(mask_chunk_size) {
        let mut masks = Vec::with_capacity(mask_idxs.len());

        for mask_i in mask_idxs {
            let mask =
                image::open(format!("data/bad_apple_masks/bad_apple_{}.png", mask_i)).unwrap();
            masks.push((mask, *mask_i));
        }

        let res = state.run_on_image(&masks);

        for (mask_i, res) in res {
            res.to_image(mask_dims)
                .save(format!("data/results/frames/{}.png", mask_i))
                .unwrap();
        }
    }
}
