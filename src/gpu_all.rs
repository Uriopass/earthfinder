use crate::gpu::State;
use crate::{data, TILE_SIZE};
use rustc_hash::FxHashSet;

pub fn gpu_all() {
    std::fs::create_dir_all("data/results/frames").unwrap();

    let mask_example = data::mask_i(3350);
    let mask_dims = (mask_example.width(), mask_example.height());
    let mask_chunk_size = 2;
    let mut state = pollster::block_on(State::new(
        (TILE_SIZE, TILE_SIZE),
        mask_dims,
        mask_chunk_size,
    ));

    let mask_idxs = (3350..3350 + 30 * 15).collect::<Vec<_>>();

    let entries = data::tile_grad_entries();

    state.prepare(&entries);
    drop(entries);

    let forbidden_tiles = FxHashSet::default();

    for mask_idxs in mask_idxs.chunks_exact(mask_chunk_size) {
        let mut masks = Vec::with_capacity(mask_idxs.len());

        for mask_i in mask_idxs {
            masks.push((data::mask_i(*mask_i), *mask_i));
        }

        let (results, elapsed) = state.run_on_image(&masks, &forbidden_tiles);

        for (mask_i, result) in results {
            let result = result
                .results()
                .iter()
                .find(|r| !forbidden_tiles.contains(&r.tile_pos()))
                .unwrap();
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
    }
}
