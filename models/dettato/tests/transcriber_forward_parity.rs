use burn::tensor::Device;
use dettato_rs::{
    checkpoint::{Checkpoint, ModelPaths},
    hcqt::Hcqt,
    transcriber::net::{LaneNet, Settings},
};
use serde_json::Value;
use std::{f32::consts::PI, path::PathBuf, time::Instant};

#[test]
#[ignore = "released checkpoint required: DETTATO_WEIGHTS"]
fn short_transcriber_matches_original_pytorch() {
    let root = PathBuf::from(std::env::var("DETTATO_WEIGHTS").expect("DETTATO_WEIGHTS"));
    let paths = ModelPaths::in_directory(root, "transcriber").unwrap();
    let config = paths.load_config().unwrap();
    let network = LaneNet::load(
        &Checkpoint::open(paths.weights).unwrap(),
        &Device::default(),
        Settings::from_config(&config).unwrap(),
    )
    .unwrap();
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/transcriber_forward_reference.json")).unwrap();
    let count = reference["samples"].as_u64().unwrap() as usize;
    let samples = (0..count)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            (0.07 * (2.0 * PI * 220.0 * t).sin() + 0.025 * (2.0 * PI * 440.0 * t).sin())
                * (0.8 + 0.2 * i as f32 / count as f32)
        })
        .collect::<Vec<_>>();
    let start = Instant::now();
    let result = network
        .infer_features(&Hcqt::default_16k().extract(&samples))
        .unwrap();
    eprintln!("short Transcriber Burn forward: {:?}", start.elapsed());
    assert_eq!(result.maps.frames(), 10);
    for (index, head) in ["onset", "frame", "offset", "onset2", "frame2", "offset2"]
        .into_iter()
        .enumerate()
    {
        let expected = reference["heads"][head].as_array().unwrap();
        let mut max_abs = 0.0f32;
        for (t, frame) in expected.iter().enumerate() {
            for (p, value) in frame.as_array().unwrap().iter().enumerate() {
                let v = value.as_f64().unwrap() as f32;
                max_abs = max_abs.max((result.maps.heads[index][t * 88 + p] - v).abs());
            }
        }
        eprintln!("{head}: maximum absolute difference {max_abs:.8}");
        assert!(max_abs < 0.01, "{head}: max abs = {max_abs}");
    }
}
