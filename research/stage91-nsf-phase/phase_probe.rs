#[path = "../../models/tts-with-rvc-rs/src/chunk_plan.rs"]
mod chunk_plan;
#[path = "../../models/tts-with-rvc-rs/src/chunked.rs"]
mod chunked;
#[path = "../../models/tts-with-rvc-rs/src/nsf_phase.rs"]
mod nsf_phase;

fn main() {
    let f0: Vec<f32> = (0..9_000)
        .map(|i| if i % 17 < 3 { 0.0 } else { 110.0 + (i * 73 % 481) as f32 * 0.2 })
        .collect();
    let global = nsf_phase::GlobalNsfPhase::new(&f0, 0.0, 40_000, 400).unwrap();
    for first in [0, 300, 600, 900, 1_200, 2_700, 4_500, 6_300, 8_100] {
        println!("{first}\t{:.10}", global.phase_at(first - 40).unwrap());
    }
}
