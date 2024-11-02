use renderdoc::{RenderDoc, V141};
use std::process::ExitCode;

mod cpu_one_frame;
mod gen_mask;
mod gpu;
mod gpu_all;
mod gpu_one_frame;
mod preprocess;

#[cfg(not(windows))]
pub const ROOT: &str = "data/tiles/9";

#[cfg(windows)]
pub const ROOT: &str = "data\\tiles\\9";

pub const TILE_SIZE: u32 = 512;

fn main() -> ExitCode {
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
        "preprocess" => preprocess::preprocess(),
        "cpu_one_frame" => cpu_one_frame::process(),
        "gen_mask" => gen_mask::gen_masks(),
        "gpu_one_frame" => gpu_one_frame::gpu_one_frame(),
        "gpu" => gpu_all::gpu_all(),
        _ => {
            eprintln!("Unknown command: {}", command);
            eprintln!("Available commands are: preprocess");
            std::process::exit(1);
        }
    }

    if let Some(rd) = &mut rd {
        rd.end_frame_capture(std::ptr::null(), std::ptr::null());
    }

    std::thread::sleep(std::time::Duration::from_secs(1));

    ExitCode::SUCCESS
}
