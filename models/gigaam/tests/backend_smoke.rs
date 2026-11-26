#![cfg(any(feature = "cpu", feature = "flex"))]

use anyhow::Result;
use burn::tensor::Device;
use gigaam_rs::{GigaAm, ModelKind, audio};
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend")
}

fn run_model_suite(device: Device) -> Result<()> {
    let fixtures = fixtures();
    let pcm = audio::load_audio_16k(&fixtures.join("tone.wav"))?;
    assert_eq!(pcm.len(), 6400);

    let ctc = GigaAm::load(&fixtures.join("ctc"), device.clone())?;
    assert_eq!(ctc.model_kind(), ModelKind::Ctc);
    let (logits, frames) = ctc.infer_logits_16k(&pcm)?;
    assert_eq!(frames, 11);
    assert_eq!(logits.len(), frames * 4);
    assert!(logits.iter().all(|v| v.is_finite()));
    let transcript = ctc.transcribe_16k(&pcm)?;
    assert!(transcript.token_ids.iter().all(|&token| token < 4));

    let ssl = GigaAm::load(&fixtures.join("ssl"), device.clone())?;
    assert_eq!(ssl.model_kind(), ModelKind::Ssl);
    let (embeddings, ssl_frames, channels) = ssl.infer_embeddings_16k(&pcm)?;
    assert_eq!([ssl_frames, channels], [11, 8]);
    assert_eq!(embeddings.len(), ssl_frames * channels);
    assert!(embeddings.iter().all(|v| v.is_finite()));

    let emo = GigaAm::load(&fixtures.join("emo"), device.clone())?;
    assert_eq!(emo.model_kind(), ModelKind::Emo);
    let scores = emo.emotion_probs_16k(&pcm)?;
    assert_eq!(scores.len(), 3);
    assert!((scores.iter().map(|(_, value)| value).sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(scores.iter().all(|(_, value)| value.is_finite()));

    let rnnt = GigaAm::load(&fixtures.join("rnnt"), device)?;
    assert_eq!(rnnt.model_kind(), ModelKind::Rnnt);
    let decoded = rnnt.transcribe_16k(&pcm)?;
    assert!(decoded.token_ids.iter().all(|&token| token < 4));
    Ok(())
}

#[cfg(feature = "cpu")]
#[test]
fn cubecl_cpu_executes_all_model_families() -> Result<()> {
    run_model_suite(Device::cpu())
}

#[cfg(feature = "flex")]
#[test]
fn opt_in_flex_executes_all_model_families() -> Result<()> {
    run_model_suite(Device::flex())
}
