#![cfg(feature = "flex")]
use anyhow::Result;
use burn::tensor::Device;
use gigaam_rs::{
    GigaAm, audio,
    model_registry::CacheOptions,
    service::GigaAmService,
};
use std::path::Path;

#[test]
fn finetuned_checkpoint_uses_base_architecture_and_own_weights() -> Result<()> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let base_path = fixtures.join("v3_mel_checkpoint/v3_ctc.ckpt");
    let cache = std::env::temp_dir().join(format!(
        "gigaam-finetuned-cache-{}", std::process::id()
    ));
    std::fs::create_dir_all(&cache)?;
    std::fs::copy(&base_path, cache.join("v3_ctc.ckpt"))?;
    let options = CacheOptions {
        directory: cache.clone(),
        download_if_missing: false,
        validate_md5: false,
    };
    let audio_path = fixtures.join("backend/tone.wav");
    let pcm = audio::load_audio_16k(&audio_path)?;
    let expected = GigaAm::load(&base_path, Device::flex())?.transcribe_16k(&pcm)?;
    for filename in ["missing_cfg.ckpt", "external_decoder.ckpt"] {
        let fine = fixtures.join("finetune").join(filename);
        let model = GigaAmService::load_named(fine.to_str().unwrap(), Device::flex(), &options)?;
        assert_eq!(model.transcribe_short_16k(&pcm, false)?.text, expected.text);
    }
    std::fs::remove_dir_all(cache)?;
    Ok(())
}
