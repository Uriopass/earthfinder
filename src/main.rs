mod cpu_one_frame;
mod gen_mask;
mod gpu;
mod gpu_all;
mod gpu_one_frame;
mod preprocess;
mod raw;

pub const ROOT: &str = "data/tiles/9";

fn main() {
    let command = std::env::args().nth(1).expect("No command provided");

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
}
