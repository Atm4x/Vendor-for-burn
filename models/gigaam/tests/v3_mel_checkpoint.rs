#[cfg(any(feature = "cpu", feature = "flex"))]
#[test]
fn embedded_v3_stft_checkpoint_matches_explicit_manifest() {
    use burn::tensor::Device;
    use gigaam_rs::{GigaAm, audio, inspect_native_manifest};
    use std::path::Path;

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let ckpt = fixture_dir.join("v3_mel_checkpoint/v3_ctc.ckpt");
    let explicit = fixture_dir.join("backend/ctc-v3-mel");
    let wav = fixture_dir.join("backend/tone.wav");

    let manifest = inspect_native_manifest(&ckpt).unwrap();
    assert_eq!(manifest["mel"]["n_fft"], 320);
    assert_eq!(manifest["mel"]["win_length"], 320);
    assert_eq!(manifest["mel"]["hop_length"], 160);
    assert_eq!(manifest["mel"]["center"], false);

    #[cfg(feature = "cpu")]
    let device = Device::cpu();
    #[cfg(all(not(feature = "cpu"), feature = "flex"))]
    let device = Device::flex();

    let embedded = GigaAm::load(&ckpt, device.clone()).unwrap();
    let configured = GigaAm::load(&explicit, device).unwrap();
    let pcm = audio::load_audio_16k(&wav).unwrap();
    let embedded_mel = embedded.extract_features_16k(&pcm).unwrap();
    let explicit_mel = configured.extract_features_16k(&pcm).unwrap();
    assert_eq!(embedded_mel.frames, explicit_mel.frames);
    assert_eq!(embedded_mel.values, explicit_mel.values);

    let (embedded_logits, embedded_frames) = embedded.infer_logits_16k(&pcm).unwrap();
    let (configured_logits, configured_frames) = configured.infer_logits_16k(&pcm).unwrap();
    assert_eq!(embedded_frames, configured_frames);
    assert_eq!(embedded_logits.len(), configured_logits.len());
    let max_error = embedded_logits
        .iter()
        .zip(configured_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error < 1e-6,
        "checkpoint versus sidecar max error: {max_error}"
    );
}
