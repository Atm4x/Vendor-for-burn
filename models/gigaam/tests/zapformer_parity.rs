#![cfg(any(feature = "cpu", feature = "flex"))]

use anyhow::Result;
use burn::tensor::{Device, Tensor};
use gigaam_rs::{
    weights::Weights,
    zapformer::{
        ZapformerEncoder,
        config::{CheckpointConfig, HeadConfig, HeadsConfig, ZapformerConfig},
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    file: String,
    config: Value,
    lengths: Vec<usize>,
    output_lengths: Vec<usize>,
}

fn compare(label: &str, actual: Tensor<3>, expected: Tensor<3>) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{label}");
    let actual = actual.into_data().try_to_vec::<f32>()?;
    let expected = expected.into_data().try_to_vec::<f32>()?;
    let mut max = 0.0f32;
    let mut squared = 0.0f64;
    let mut signal = 0.0f64;
    for (a, b) in actual.iter().zip(&expected) {
        assert!(a.is_finite() && b.is_finite(), "{label}: non-finite output");
        let diff = (a - b).abs();
        max = max.max(diff);
        squared += f64::from(diff).powi(2);
        signal += f64::from(*b).powi(2);
    }
    let rmse = (squared / actual.len() as f64).sqrt();
    let snr = 10.0 * (signal / squared.max(f64::MIN_POSITIVE)).log10();
    eprintln!("{label}: max={max:.8e} rmse={rmse:.8e} snr={snr:.2}dB");
    assert!(max < 2e-5 && rmse < 3e-6, "{label}: max {max}, RMSE {rmse}");
    Ok(())
}

fn run(device: Device) -> Result<()> {
    let cases: Vec<Case> = serde_json::from_str(include_str!("fixtures/zapformer/cases.json"))?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zapformer");
    for case in cases {
        let config = ZapformerConfig::from_upstream(&case.config)?;
        let mut weights = Weights::open(&root.join(&case.file))?;
        let model = ZapformerEncoder::load(&mut weights, "encoder", &device, config)?;
        let input = weights.tensor::<3>("reference.input", &device)?;
        let (front, lengths) = model.frontend.forward(input.clone(), &case.lengths)?;
        assert_eq!(lengths, case.output_lengths);
        compare(
            &format!("{}/frontend", case.file),
            front,
            weights.tensor("reference.frontend", &device)?,
        )?;
        let (output, lengths) = model.forward(input, &case.lengths)?;
        assert_eq!(lengths, case.output_lengths);
        compare(
            &format!("{}/encoder", case.file),
            output,
            weights.tensor("reference.output", &device)?,
        )?;
    }
    Ok(())
}

#[cfg(feature = "flex")]
#[test]
fn flex_matches_upstream_zapformer() -> Result<()> {
    run(Device::flex())
}
#[cfg(feature = "cpu")]
#[test]
fn cubecl_cpu_matches_upstream_zapformer() -> Result<()> {
    run(Device::cpu())
}

#[test]
fn typed_checkpoint_retains_independent_head_geometries_and_contexts() -> Result<()> {
    let cases: Vec<Case> = serde_json::from_str(include_str!("fixtures/zapformer/cases.json"))?;
    let rnnt = |classes: usize| json!({"_target_":"gigaam.decoder.RNNTHead", "decoder":{"pred_hidden":16,"pred_rnn_layers":1,"num_classes":classes},"joint":{"enc_hidden":8,"pred_hidden":16,"joint_hidden":12,"num_classes":classes}});
    let mut cfg = json!({"encoder":cases[0].config,"preprocessor":{"features":8,"center":false,"n_fft":320,"win_length":320}, "heads":{"raw":rnnt(3),"e2e":rnnt(5)},"decodings":{"raw":{"_target_":"gigaam.decoding.RNNTGreedyDecoding","vocabulary":["a","b"]},"e2e":{"_target_":"gigaam.decoding.RNNTGreedyDecoding","model_path":"train/e2e.model"}},"language_conditioning":{"languages":["ru","en"],"num_slots":3,"embedding_dim":4,"film":{"enabled":true,"placement":"all"}},"streaming":{"contexts":[{"future_context_ms":160,"left_context":32}]}});
    let parsed = CheckpointConfig::from_upstream(&cfg)?;
    assert!(matches!(
        parsed.heads,
        HeadsConfig::Dual {
            raw: HeadConfig::Rnnt { num_classes: 3, .. },
            e2e: HeadConfig::Rnnt { num_classes: 5, .. }
        }
    ));
    assert_eq!(parsed.encoder.stacks[1].dim, 12);
    assert_eq!(parsed.encoder.contexts.len(), 4);
    assert_eq!(parsed.encoder.stacks[0].ff_multiple, 2);
    assert!(parsed.language_conditioning.is_some() && parsed.streaming.is_some());
    cfg["encoder"]["query_head_dim"] = json!([4, 3]);
    assert!(CheckpointConfig::from_upstream(&cfg).is_err());
    cfg["encoder"]["query_head_dim"] = json!(4);
    cfg["heads"]["e2e"]["joint"]["num_classes"] = json!(6);
    assert!(CheckpointConfig::from_upstream(&cfg).is_err());
    Ok(())
}

#[derive(Deserialize)]
struct DualInput {
    input: Vec<f32>,
    encoded: Vec<f32>,
    lengths: Vec<usize>,
    output_lengths: Vec<usize>,
    shape: [usize; 3],
    tags: Vec<gigaam_rs::zapformer::language::LanguageTag>,
}
#[derive(Deserialize)]
struct DualResult {
    raw: Vec<(String, Vec<usize>, Vec<usize>)>,
    e2e: Vec<(String, Vec<usize>, Vec<usize>)>,
}

fn run_dual(device: Device) -> Result<()> {
    use burn::tensor::TensorData;
    use gigaam_rs::{
        decode::Transcription,
        zapformer::{HeadTranscriptions, ZapformerModel},
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zapformer");
    let input: DualInput =
        serde_json::from_str(include_str!("fixtures/zapformer/dual_input.json"))?;
    for kind in ["ctc", "rnnt"] {
        let path = root.join(format!("dual_{kind}.ckpt"));
        let inspected = gigaam_rs::inspect_native_manifest(&path)?;
        assert_eq!(inspected["heads"]["kind"], "dual");
        let model = ZapformerModel::load_checkpoint(&path, device.clone())?;
        let x = Tensor::<3>::from_data(TensorData::new(input.input.clone(), [3, 35, 8]), &device);
        let (encoded, lengths) = model.encode_features(x, &input.lengths, Some(&input.tags))?;
        assert_eq!(lengths, input.output_lengths);
        compare(
            &format!("dual_{kind}/conditioned"),
            encoded.clone(),
            Tensor::from_data(TensorData::new(input.encoded.clone(), input.shape), &device),
        )?;
        let expected: DualResult = serde_json::from_str(&std::fs::read_to_string(
            root.join(format!("dual_{kind}.json")),
        )?)?;
        let convert = |values: Vec<(String, Vec<usize>, Vec<usize>)>| {
            values
                .into_iter()
                .map(|(text, token_ids, frames)| Transcription {
                    text,
                    token_ids,
                    frames,
                })
                .collect()
        };
        assert_eq!(
            model.decode_encoded(encoded, &lengths)?,
            HeadTranscriptions::Dual {
                raw: convert(expected.raw),
                e2e: convert(expected.e2e)
            }
        );
    }
    Ok(())
}
#[cfg(feature = "flex")]
#[test]
fn flex_loads_native_dual_heads_and_language_conditioning() -> Result<()> {
    run_dual(Device::flex())
}
#[cfg(feature = "cpu")]
#[test]
fn cubecl_cpu_loads_native_dual_heads_and_language_conditioning() -> Result<()> {
    run_dual(Device::cpu())
}
