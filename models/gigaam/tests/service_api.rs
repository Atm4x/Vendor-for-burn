#![cfg(feature = "flex")]

use anyhow::Result;
use burn::tensor::Device;
use gigaam_rs::{
    ModelKind,
    audio::{self, SpeechRegion, VadGrouping},
    service::{GigaAmService, Segmentation, TranscribeOptions},
};
use serde_json::Value;
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend")
}

#[test]
fn persistent_api_preserves_segmentation_and_model_lifetime() -> Result<()> {
    let base = fixtures();
    let pcm = audio::load_audio_16k(&base.join("tone.wav"))?;
    let service = GigaAmService::load(&base.join("ctc"), Device::flex())?;
    assert_eq!(service.status().model_kind, ModelKind::Ctc);
    assert_eq!(service.status().sample_rate, 16_000);

    let initial = service.transcribe_16k(&pcm, TranscribeOptions::default())?;
    let repeat = service.transcribe_16k(&pcm, TranscribeOptions::default())?;
    assert_eq!(initial, repeat);
    assert_eq!(initial.segments.len(), 1);
    assert!(initial.segments[0].words.is_none());

    let short_result = service.transcribe_short_16k(&pcm, false)?;
    assert_eq!(short_result.text, initial.text);
    assert_eq!(short_result.words, None);
    assert_eq!(initial.words().len(), 0);
    assert!(!initial.has_word_timestamps());
    let timed_result = service.transcribe_short_16k(&pcm, true)?;
    assert!(timed_result.words.is_some());

    let limited = GigaAmService::load(&base.join("ctc"), Device::flex())?.with_batch_size(2)?;
    assert!(
        GigaAmService::load(&base.join("ctc"), Device::flex())?
            .with_batch_size(0)
            .is_err()
    );
    let items: Vec<&[f32]> = (0..5).map(|i| &pcm[..pcm.len() - i * 500]).collect();
    let expected: Vec<_> = items
        .iter()
        .map(|item| limited.transcribe_short_16k(item, true))
        .collect::<Result<_>>()?;
    let actual = limited.transcribe_batch_16k(&items, true)?;
    assert_eq!(actual.len(), 5);
    for (expected, actual) in expected.into_iter().zip(actual) {
        assert_eq!(expected.text, actual.text);
        assert_eq!(expected.words, actual.segments[0].words);
    }

    let short_batch = service.transcribe_batch_16k(&[pcm.as_slice(), &pcm[1200..]], true)?;
    assert_eq!(short_batch.len(), 2);
    assert_eq!(short_batch[0].text, initial.text);
    assert_eq!(short_batch[0].segments[0].words.as_ref().is_some(), true);
    assert_eq!(
        short_batch[1].duration_seconds,
        (pcm.len() - 1200) as f64 / 16_000.0
    );

    let too_long = vec![0.0_f32; 25 * 16_000 + 1];
    let error = service
        .transcribe_16k(&too_long, TranscribeOptions::default())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("external VAD segments for longform")
    );
    assert!(
        service
            .transcribe_16k(&[], TranscribeOptions::default())
            .is_err()
    );

    let speech = [SpeechRegion {
        start_sample: 800,
        end_sample: 5600,
    }];
    let selected = service.transcribe_16k(
        &pcm,
        TranscribeOptions {
            segmentation: Segmentation::SpeechRegions(&speech),
            word_timestamps: true,
        },
    )?;
    assert_eq!(selected.segments.len(), 1);
    assert_eq!(selected.segments[0].start, 0.05);
    assert_eq!(selected.segments[0].end, 0.35);
    assert!(selected.segments[0].words.is_some());
    assert_eq!(selected.sample_rate, 16_000);

    let canonical_long = service.transcribe_longform_regions_16k(&pcm, &speech, true, None)?;
    assert_eq!(canonical_long, selected);
    assert!(canonical_long.has_word_timestamps());
    let tiny_speech = [SpeechRegion {
        start_sample: 800,
        end_sample: 2400,
    }];
    let tiny_result = service.transcribe_longform_regions_16k(&pcm, &tiny_speech, false, None)?;
    assert!(tiny_result.segments.is_empty());
    assert_eq!(canonical_long.words().len(), selected.words().len());

    let grouped = service.transcribe_16k(
        &pcm,
        TranscribeOptions {
            segmentation: Segmentation::GroupedSpeechRegions {
                regions: &speech,
                grouping: VadGrouping::default(),
            },
            word_timestamps: false,
        },
    )?;
    assert_eq!(grouped.segments.len(), 1);
    assert_eq!(grouped.segments[0].start, 0.05);
    assert_eq!(grouped.segments[0].end, 0.35);

    let empty = service.transcribe_16k(
        &pcm,
        TranscribeOptions {
            segmentation: Segmentation::SpeechRegions(&[]),
            word_timestamps: false,
        },
    )?;
    assert!(empty.segments.is_empty());
    assert_eq!(empty.text, "");
    Ok(())
}

#[test]
fn wrong_model_operation_fails_before_inference() -> Result<()> {
    let base = fixtures();
    let pcm = audio::load_audio_16k(&base.join("tone.wav"))?;
    let ssl = GigaAmService::load(&base.join("ssl"), Device::flex())?;
    assert!(
        ssl.transcribe_16k(&pcm, TranscribeOptions::default())
            .is_err()
    );
    let embedding = ssl.embed_16k(&pcm)?;
    assert_eq!((embedding.frames, embedding.channels), (11, 8));
    assert_eq!(embedding.data.len(), 88);

    let emo = GigaAmService::load(&base.join("emo"), Device::flex())?;
    let values = emo.emotions_16k(&pcm)?;
    assert_eq!(values.len(), 3);
    let mapped = emo.emotions_dict_16k(&pcm)?;
    assert_eq!(mapped.len(), 3);
    for (name, prob) in values {
        assert_eq!(mapped.get(&name), Some(&prob));
    }
    assert!(emo.embed_16k(&pcm).is_err());
    Ok(())
}

#[test]
fn jsonl_serve_reuses_model_and_survives_bad_request() -> Result<()> {
    let base = fixtures();
    let mut process = Command::new(env!("CARGO_BIN_EXE_gigaam-rs"))
        .args([
            "serve",
            base.join("ctc").to_str().unwrap(),
            "--device",
            "flex",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let audio_path = base.join("tone.wav");
    {
        let mut input = process.stdin.take().unwrap();
        writeln!(input, "{}", r#"{"id":1,"method":"status"}"#)?;
        writeln!(
            input,
            "{}",
            serde_json::json!({"id":2,"method":"transcribe","audio_path":audio_path})
        )?;
        writeln!(
            input,
            "{}",
            r#"{"id":3,"method":"transcribe","audio_path":"nonexistent.wav"}"#
        )?;
        writeln!(
            input,
            "{}",
            serde_json::json!({"id":4,"method":"transcribe","audio_path":audio_path,"word_timestamps":true,"segments":[{"start":0.05,"end":0.35}]})
        )?;
        writeln!(
            input,
            "{}",
            serde_json::json!({"id":5,"method":"transcribe_longform","audio_path":audio_path,"segments":[],"word_timestamps":true})
        )?;
        writeln!(
            input,
            "{}",
            serde_json::json!({"id":6,"method":"transcribe_batch","audio_paths":[audio_path,audio_path]})
        )?;
        writeln!(input, "{}", r#"{"id":7,"method":"shutdown"}"#)?;
    }
    let output = process.wait_with_output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses: Vec<Value> = String::from_utf8(output.stdout)?
        .lines()
        .map(serde_json::from_str)
        .collect::<serde_json::Result<_>>()?;
    assert_eq!(responses.len(), 7);
    assert_eq!(responses[0]["result"]["model_kind"], "ctc");
    assert_eq!(responses[1]["ok"], true);
    assert_eq!(responses[2]["ok"], false);
    assert_eq!(responses[3]["result"]["segments"][0]["start"], 0.05);
    assert_eq!(responses[3]["result"]["segments"][0]["end"], 0.35);
    assert_eq!(
        responses[4]["result"]["segments"].as_array().unwrap().len(),
        0
    );
    assert_eq!(responses[5]["result"].as_array().unwrap().len(), 2);
    assert_eq!(responses[6]["result"]["stopped"], true);
    Ok(())
}

#[test]
fn encoder_export_accepts_every_upstream_model_family() -> Result<()> {
    let base = fixtures();
    let pcm = audio::load_audio_16k(&base.join("tone.wav"))?;
    for family in ["ctc", "rnnt", "ssl", "emo"] {
        let model = gigaam_rs::GigaAm::load(&base.join(family), Device::flex())?;
        let (encoded, frames, channels) = model.infer_encoder_16k(&pcm)?;
        assert!(frames > 0 && channels == 8, "{family}");
        assert_eq!(encoded.len(), frames * channels, "{family}");
        assert!(encoded.iter().all(|value| value.is_finite()), "{family}");
    }
    Ok(())
}
