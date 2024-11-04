use renderdoc::{RenderDoc, V141};
use std::process::ExitCode;

mod data;
mod gen_mask;
mod gpu;
mod gpu_all;
mod gpu_one_frame;
mod render;
mod tiles_grad;

pub const TILE_HEIGHT: u32 = 512;

fn parse_zoom_levels() -> Vec<u32> {
    let mut zs = vec![7, 8, 9];

    if std::env::args().len() > 2 {
        zs = std::env::args()
            .skip(2)
            .map(|arg| arg.parse::<u32>().expect("Invalid zoom level"))
            .collect::<Vec<_>>();
    }

    zs
}

fn main() -> ExitCode {
    /*
    let n_threads = std::thread::available_parallelism().unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads.get() / 2)
        .build_global()
        .unwrap();*/

    if !std::fs::exists("data").unwrap_or(false) {
        eprintln!("data directory not found, are you at the root of the project?");
        return ExitCode::FAILURE;
    }

    let command = std::env::args().nth(1).expect("No command provided");

    let mut rd: Option<RenderDoc<V141>> = RenderDoc::new().ok();

    if let Some(rd) = &mut rd {
        rd.start_frame_capture(std::ptr::null(), std::ptr::null());
    } else {
        eprintln!("RenderDoc not found, skipping frame capture");
    }

    match command.as_str() {
        "tiles_grad" => {
            let zs = parse_zoom_levels();
            if zs.len() == 0 {
                eprintln!("No zoom levels provided");
                return ExitCode::FAILURE;
            }

            for z in zs {
                tiles_grad::gen_tiles_grad(z);
            }
        }
        "gen_mask" => gen_mask::gen_masks(),
        "gpu_one_frame" => gpu_one_frame::gpu_one_frame(&parse_zoom_levels()),
        "gpu" => gpu_all::gpu_all(&parse_zoom_levels()),
        "render" => {
            let path = std::env::args().nth(2).unwrap_or_else(|| {
                static DEFAULT_PATH: &str = "data/results/out.csv";
                eprintln!("No path provided, using default: {}", DEFAULT_PATH);
                DEFAULT_PATH.to_string()
            });
            render::render(&path);
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            eprintln!("Available commands are: tiles_grad, gen_mask, gpu_one_frame, gpu");
            std::process::exit(1);
        }
    }

    if let Some(rd) = &mut rd {
        rd.end_frame_capture(std::ptr::null(), std::ptr::null());
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    ExitCode::SUCCESS
}
