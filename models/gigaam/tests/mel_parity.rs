use gigaam_rs::audio::log_mel;

fn floats(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn log_mel_matches_torchaudio_reference() {
    let audio = floats(include_bytes!("fixtures/log_mel_input_16k.f32"));
    let expected = floats(include_bytes!("fixtures/log_mel_torchaudio64.f32"));
    let calculated = log_mel(&audio, 64).unwrap();
    assert_eq!(calculated.n_mels, 64);
    assert_eq!(calculated.frames, 28);
    assert_eq!(calculated.values.len(), expected.len());
    let errors = calculated
        .values
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs());
    let max_error = errors.clone().fold(0f32, f32::max);
    let mean_error = errors.sum::<f32>() / expected.len() as f32;
    assert!(max_error < 0.002, "max log-mel error: {max_error}");
    assert!(mean_error < 5e-5, "mean log-mel error: {mean_error}");
}
