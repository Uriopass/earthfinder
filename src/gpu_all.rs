use crate::gpu::State;
use crate::ROOT;

pub fn gpu_all() {
    std::fs::create_dir_all("data/results/frames").unwrap();

    let tile_size = (512, 512);
    let mask_example = image::open("data/bad_apple_masks/bad_apple_3350.png").unwrap();
    let mask_dims = (mask_example.width(), mask_example.height());
    let mut state = pollster::block_on(State::new(tile_size, mask_dims));

    for mask_i in 3350..3350 + 30 * 15 {
        let mask = image::open(format!("data/bad_apple_masks/bad_apple_{}.png", mask_i)).unwrap();

        let mut entries: Vec<_> = walkdir::WalkDir::new(ROOT.replace("tiles", "tiles_oklab"))
            .into_iter()
            .filter_map(|v| v.ok())
            .collect();

        entries.retain(|entry| {
            entry.file_type().is_file() && entry.path().display().to_string().ends_with(".png")
        });

        let res = state.run_on_image(mask, &entries);

        res.to_image(mask_dims)
            .save(format!("data/results/frames/{}.png", mask_i))
            .unwrap();
    }
}
