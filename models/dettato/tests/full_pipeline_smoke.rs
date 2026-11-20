use burn::tensor::Device;
use dettato_rs::pipeline::Dettato;

#[test]
#[ignore = "end-to-end released-weight test; requires DETTATO_WEIGHTS and a fast backend"]
fn short_mixed_audio_produces_readable_multitrack_midi() {
    let root = std::env::var("DETTATO_WEIGHTS").expect("DETTATO_WEIGHTS required");
    let mut pipeline = Dettato::from_directory(root, &Device::default()).unwrap();
    pipeline.set_merge_same_instrument(false);
    pipeline.transcriber.set_shared_cache_budget_bytes(256 * 1024 * 1024);
    let sample_rate = 48_000u32;
    let audio: Vec<f32> = (0..sample_rate / 4).map(|i| {
        let t = i as f32 / sample_rate as f32;
        0.12 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
            + 0.07 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
    }).collect();
    let result = pipeline.transcribe(&audio, sample_rate).unwrap();
    assert!((result.transcription.seconds - 0.25).abs() < 1e-6);
    assert!(!result.info.drums_skipped_low_sample_rate);
    assert_eq!(result.info.grouped_notes,
        result.transcription.tracks.iter().filter(|t| !t.is_drum).map(|t| t.notes.len()).sum::<usize>());
    let midi = result.transcription.to_midi_bytes().unwrap();
    assert_eq!(&midi[..4], b"MThd");
    assert_eq!(u16::from_be_bytes([midi[8], midi[9]]), 1);
    assert_eq!(u16::from_be_bytes([midi[10], midi[11]]) as usize,
        result.transcription.tracks.len() + 1);
}
