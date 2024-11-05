use crate::data;
use crate::gpu::algorithm::PosResult;
use crate::gpu::State;
use image::{GrayImage, Rgb32FImage, RgbaImage};
use rustc_hash::FxHashSet;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::time::Instant;

static SAVE_ERROR: bool = false;

pub fn gpu_all(zs: &[u32]) {
    let _ = std::fs::create_dir_all("data/results/frames");
    let _ = std::fs::create_dir_all("data/results/frames_debug");

    let mask_example = data::mask_i(1);
    let mask_dims = (mask_example.width(), mask_example.height());
    let mask_chunk_size = 1;
    let mut state = pollster::block_on(State::new(mask_dims, mask_chunk_size));

    let mask_idxs = (1..6000).collect::<Vec<_>>();

    let entries = data::tile_grad_entries(zs);

    state.prepare(&entries);
    drop(entries);

    let mut last_tile_rgb = RgbaImage::new(mask_dims.0 / 4, mask_dims.1 / 4);

    let mut avg_error = Rgb32FImage::new(mask_dims.0, mask_dims.1);
    avg_error.fill(0.5);

    let mut forbidden_tile_ring = Vec::with_capacity(30);

    let mut forbidden_tiles = FxHashSet::default();

    let mut result_csv = File::options()
        .read(true)
        .write(true)
        .create(true)
        .open("data/results/out.csv")
        .unwrap();

    let csv_content = {
        let mut bufreader = std::io::BufReader::new(&result_csv);
        let mut content = String::new();
        bufreader.read_to_string(&mut content).unwrap();
        content
    };

    let file_existed = !csv_content.is_empty();

    let mut frames_already_done = csv_content
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut parts = line.split(',');
            let frame = parts.next().unwrap().trim().trim().parse::<u32>().unwrap();
            let tile_x = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let tile_y = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let tile_z = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let zoom = parts.next().unwrap().trim().parse::<f32>().unwrap();
            let x = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let y = parts.next().unwrap().trim().parse::<u32>().unwrap();
            let score = parts.next().unwrap().trim().parse::<f32>().unwrap();

            (
                frame,
                PosResult {
                    tile_x,
                    tile_y,
                    tile_z,
                    zoom,
                    x,
                    y,
                    score,
                },
            )
        })
        .collect::<Vec<_>>();
    frames_already_done.sort_unstable_by_key(|(frame, _)| *frame);

    for (last, next) in frames_already_done
        .iter()
        .zip(frames_already_done.iter().skip(1))
    {
        if last.0 == next.0 {
            panic!("Frame {} was calculated twice. Please fix CSV", last.0);
        }
    }

    if !frames_already_done.is_empty() {
        eprintln!("Some frames were already calculated and will be skipped:");
        let mut ranges_already_done = vec![];
        let mut end = frames_already_done[0].0;
        let mut begin = end;
        for &(frame, _) in frames_already_done.iter().skip(1) {
            if frame != end + 1 {
                ranges_already_done.push((begin, end));
                begin = frame;
            }
            end = frame;
        }
        ranges_already_done.push((begin, end));

        for (begin, end) in ranges_already_done {
            if begin == end {
                eprintln!("  Frame  {}", begin);
            } else {
                eprintln!("  Frames {}-{}", begin, end);
            }
        }
    }

    result_csv.seek(std::io::SeekFrom::End(0)).unwrap();

    let mut bufwriter = std::io::BufWriter::new(result_csv);

    if !file_existed {
        writeln!(
            &mut bufwriter,
            "Frame,tile_x,tile_y,tile_z,zoom,x,y,score,time"
        )
        .unwrap();
    } else {
        writeln!(&mut bufwriter, "").unwrap();
    }

    let mut prev_result: Option<PosResult> = None;

    let mut t_start = Instant::now();

    for mask_idx in mask_idxs {
        if let Ok(idx) = frames_already_done.binary_search_by_key(&mask_idx, |(frame, _)| *frame) {
            let (_, res) = frames_already_done[idx];
            forbidden_tile_ring.push(res.tile_pos());
            forbidden_tiles.insert(res.tile_pos());

            if forbidden_tile_ring.len() >= 10 {
                forbidden_tiles.remove(&forbidden_tile_ring.remove(0));
            }
            last_tile_rgb = res.to_rgba_quarter(mask_dims);
            continue;
        }

        let mut mask = data::mask_i(mask_idx);

        avg_error.pixels_mut().for_each(|p| {
            p.0[0] *= 0.5;
        });
        prev_result.map(|result| {
            result.calc_error(&mask, |x, y, err| {
                avg_error.get_pixel_mut(x, y).0[0] += err;
            })
        });

        mask.enumerate_pixels_mut().for_each(|(x, y, p)| {
            let apply_error = |v| {
                let err = avg_error.get_pixel(x, y).0[0];
                (v as f32 * (0.7 + err * 0.3)) as u8
            };

            p.0[0] = apply_error(p.0[0]);
            p.0[1] = apply_error(p.0[1]);
        });

        let (results, elapsed_gpu) =
            state.run_on_image(&[(&mask, mask_idx, &last_tile_rgb)], &forbidden_tiles);
        let result = results[0].1.results()[0];

        forbidden_tile_ring.push(result.tile_pos());
        forbidden_tiles.insert(result.tile_pos());

        if forbidden_tile_ring.len() >= 10 {
            forbidden_tiles.remove(&forbidden_tile_ring.remove(0));
        }

        let t_total = t_start.elapsed();
        t_start = Instant::now();
        println!(
            "Frame {}: ({:>3},{:>3},{},z{}) ({:>3},{:>3}) score:{:>7.4} t:{:.2}s ({:.2}s gpu)",
            mask_idx,
            result.tile_x,
            result.tile_y,
            result.tile_z,
            result.zoom,
            result.x,
            result.y,
            result.score,
            t_total.as_secs_f32(),
            elapsed_gpu.as_secs_f32()
        );

        writeln!(
            &mut bufwriter,
            "{},{},{},{},{},{},{},{:.6},{:.2}",
            mask_idx,
            result.tile_x,
            result.tile_y,
            result.tile_z,
            result.zoom,
            result.x,
            result.y,
            result.score,
            elapsed_gpu.as_secs_f32()
        )
        .unwrap();

        let _ = bufwriter.flush();

        last_tile_rgb = result.to_rgba_quarter(mask.dimensions());

        let avg_error_cpy = avg_error.clone();

        rayon::spawn(move || {
            if SAVE_ERROR {
                let mut error_show = GrayImage::new(mask_dims.0, mask_dims.1);
                avg_error_cpy.enumerate_pixels().for_each(|(x, y, p)| {
                    error_show.put_pixel(x, y, image::Luma([((p.0[0] * 255.0) as u8).min(255)]));
                });

                error_show
                    .save(format!("data/results/frames/{}_avg_error.png", mask_idx))
                    .unwrap();
            }
            let img_debug = result.to_image(&mask, &avg_error_cpy, true);
            img_debug
                .save(format!("data/results/frames_debug/{}.png", mask_idx))
                .unwrap();

            let img = result.to_image(&mask, &avg_error_cpy, false);
            img.save(format!("data/results/frames/{}.png", mask_idx))
                .unwrap();
        });

        prev_result = Some(result);
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
