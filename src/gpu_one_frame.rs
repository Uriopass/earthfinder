use crate::gpu::State;
use crate::ROOT;

pub fn gpu_one_frame() {
    let mask = image::open("data/bad_apple_masks/bad_apple_239.png").unwrap();
    let tile_size = (512, 512);

    let mut state = pollster::block_on(State::new(tile_size, (mask.width(), mask.height())));

    let mut entries: Vec<_> = walkdir::WalkDir::new(ROOT.replace("tiles", "tiles_oklab"))
        .into_iter()
        .filter_map(|v| v.ok())
        .collect();

    entries.retain(|entry| {
        entry.file_type().is_file() && entry.path().display().to_string().ends_with(".png")
    });

    state.run_on_image(mask, &entries);
}
