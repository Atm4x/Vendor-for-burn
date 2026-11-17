#![cfg(feature = "flex")]
use anyhow::Result;
use burn::tensor::Device;
use gigaam_rs::{GigaAm, audio};
use std::path::Path;

#[test]
fn padded_batch_encoder_matches_individual_with_unequal_lengths() -> Result<()> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend");
    let pcm = audio::load_audio_16k(&base.join("tone.wav"))?;
    let recordings: [&[f32]; 3] = [&pcm, &pcm[..4800], &pcm[..2400]];
    for family in ["ctc", "rnnt", "ssl", "emo"] {
        let model = GigaAm::load(&base.join(family), Device::flex())?;
        let output = model.infer_encoder_batch_16k(&recordings)?;
        assert_eq!(output.len(), recordings.len());
        for (index, sample) in recordings.iter().enumerate() {
            let expected = model.infer_encoder_16k(sample)?;
            assert_eq!(
                (output[index].1, output[index].2),
                (expected.1, expected.2),
                "{family}, sample {index}: shape mismatch"
            );
            let max_abs = output[index]
                .0
                .iter()
                .zip(expected.0.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 0.03,
                "{family}, sample {index}: max abs diff {max_abs}"
            );
        }
    }
    Ok(())
}

#[test]
fn padded_batch_sizes_match_python_reference_1_2_4_16() -> Result<()> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend");
    let pcm = audio::load_audio_16k(&base.join("tone.wav"))?;
    let model = GigaAm::load(&base.join("ctc-v3-mel"), Device::flex())?;
    for batch_size in [1usize, 2, 4, 16] {
        let examples: Vec<&[f32]> = (0..batch_size)
            .map(|index| &pcm[..(1600 + (index * 307) % 4600)])
            .collect();
        let encoded = model.infer_encoder_batch_16k(&examples)?;
        assert_eq!(encoded.len(), batch_size);
        for (index, sample) in examples.iter().enumerate() {
            let expected = model.infer_encoder_16k(sample)?;
            let max_abs = encoded[index]
                .0
                .iter()
                .zip(expected.0)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 0.03,
                "batch={batch_size} item={index} diff={max_abs}"
            );
        }
    }
    Ok(())
}

#[test]
fn padded_batch_asr_keeps_original_decoding_and_order() -> Result<()> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend");
    let pcm = audio::load_audio_16k(&base.join("tone.wav"))?;
    let recordings: [&[f32]; 3] = [&pcm[..2400], &pcm, &pcm[..4800]];
    for family in ["ctc", "rnnt"] {
        let model = GigaAm::load(&base.join(family), Device::flex())?;
        let batch = model.transcribe_batch_16k(&recordings)?;
        let timed = model.transcribe_batch_with_timestamps_16k(&recordings)?;
        assert_eq!(batch.len(), 3);
        assert_eq!(timed.len(), 3);
        for (index, pcm) in recordings.iter().enumerate() {
            let expected = model.transcribe_16k(pcm)?;
            assert_eq!(batch[index], expected, "{family}, sample {index}");
            assert_eq!(
                timed[index],
                model.transcribe_with_timestamps_16k(pcm)?,
                "{family}, timed sample {index}"
            );
        }
    }
    Ok(())
}
