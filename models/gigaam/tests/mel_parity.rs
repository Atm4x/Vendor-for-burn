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

#[test]
fn log_mel_matches_official_v3_uncentered_torchaudio() {
    use gigaam_rs::audio::{MelConfig, log_mel_with_config};

    let audio = floats(include_bytes!("fixtures/log_mel_input_16k.f32"));
    let expected = floats(include_bytes!("fixtures/log_mel_torchaudio_v3_320.f32"));
    let config = MelConfig {
        n_fft: 320,
        win_length: 320,
        hop_length: 160,
        center: false,
        ..MelConfig::default()
    };
    let calculated = log_mel_with_config(&audio, 64, config).unwrap();
    assert_eq!(calculated.frames, 26);
    assert_eq!(calculated.values.len(), expected.len());
    let max_error = calculated
        .values
        .iter()
        .zip(expected.iter())
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error < 0.003,
        "official v3 max log-mel error: {max_error}"
    );
}

#[test]
fn log_mel_matches_torchaudio_when_window_is_shorter_than_fft() {
    use gigaam_rs::audio::{MelConfig, log_mel_with_config};

    let audio = floats(include_bytes!("fixtures/log_mel_input_16k.f32"));
    let expected = floats(include_bytes!(
        "fixtures/log_mel_torchaudio_unaligned_512.f32"
    ));
    let config = MelConfig {
        n_fft: 512,
        win_length: 320,
        hop_length: 160,
        center: false,
        ..MelConfig::default()
    };
    let calculated = log_mel_with_config(&audio, 64, config).unwrap();
    assert_eq!(calculated.frames, 25);
    assert_eq!(calculated.values.len(), expected.len());
    let max_error = calculated
        .values
        .iter()
        .zip(expected.iter())
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error < 0.003,
        "padded-window max log-mel error: {max_error}"
    );
}

#[test]
fn log_mel_matches_torchaudio_slaney_odd_fft() {
    use gigaam_rs::audio::{MelConfig, MelNorm, MelScale, log_mel_with_config};
    let audio = floats(include_bytes!("fixtures/log_mel_input_16k.f32"));
    let expected = floats(include_bytes!("fixtures/log_mel_slaney_odd.f32"));
    let config = MelConfig {
        n_fft: 321,
        win_length: 319,
        hop_length: 160,
        center: true,
        lowfreq: 80.0,
        highfreq: Some(7200.0),
        mel_scale: MelScale::Slaney,
        mel_norm: Some(MelNorm::Slaney),
    };
    let calculated = log_mel_with_config(&audio, 16, config).unwrap();
    assert_eq!(calculated.values.len(), expected.len());
    let max = calculated
        .values
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Slaney/odd FFT: max={max:.8e}");
    assert!(max < 0.003);
}
