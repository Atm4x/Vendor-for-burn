use burn::tensor::Device;
use dettato_rs::grouper::net::{NoteNet, TRANSCRIBER_TAP_DIM, WindowInput};
use serde_json::Value;
use std::f32::consts::PI;

#[test]
#[ignore = "released checkpoint required: DETTATO_WEIGHTS"]
fn grouper_notenet_matches_original_pytorch() {
    let root = std::env::var("DETTATO_WEIGHTS").expect("DETTATO_WEIGHTS");
    let network = NoteNet::from_directory(root, &Device::default()).unwrap();
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/grouper_forward_reference.json")).unwrap();
    let samples = (0..32000)
        .map(|i| 0.08 * (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let mut note = [0.0f32; 12];
    note[0] = 0.25;
    note[1] = 0.75;
    note[2] = 69.0;
    note[3] = 1.0;
    note[4] = 0.95;
    note[5] = 0.9;
    let output = network
        .infer(&WindowInput {
            audio16k: samples,
            notes: vec![note; 4],
            taps: vec![[[0.0f32; TRANSCRIBER_TAP_DIM]; 3]; 4],
        })
        .unwrap();
    for (name, rows) in [
        ("probabilities", &output.probabilities),
        ("embeddings", &output.embeddings),
        ("hidden", &output.hidden),
    ] {
        let mut max_error = 0.0f32;
        for (i, row) in rows.iter().enumerate() {
            let expected = reference[name][i].as_array().unwrap();
            assert_eq!(row.len(), expected.len());
            for (v, target) in row.iter().zip(expected) {
                max_error = max_error.max((v - target.as_f64().unwrap() as f32).abs());
            }
        }
        eprintln!("Grouper {name}: maximum absolute difference {max_error:.8}");
        assert!(max_error < 0.005, "{name}: max abs = {max_error}");
    }
    let mut error = 0.0f32;
    for (v, expected) in output
        .keep_logits
        .iter()
        .zip(reference["keep_logits"].as_array().unwrap())
    {
        error = error.max((v - expected.as_f64().unwrap() as f32).abs());
    }
    eprintln!("Grouper keep logits: maximum absolute difference {error:.8}");
    assert!(error < 0.005, "keep max abs = {error}");
}
