use burn::tensor::Device;
use std::{env, fs, process::ExitCode};
use tts_with_rvc_rs::{TextFrontFixture, attention::AttentionFixture, encoder::EncoderLayerFixture, text_encoder::TextEncoderFixture, max_abs_error};

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or("missing command")?;
    if command == "inspect-pth" {
        let path = args.next().ok_or("missing checkpoint path")?;
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        println!("checkpoint={} tensor_count={}", path, checkpoint.tensor_count());
        for name in args {
            let (shape, values) = checkpoint.tensor_f32(&name)?;
            println!("tensor={name} shape={shape:?} elements={} first={:?}", values.len(), &values[..values.len().min(4)]);
        }
        return Ok(());
    }
    if !matches!(command.as_str(), "verify-text-front" | "verify-attention" | "verify-encoder-layer" | "verify-text-encoder" | "verify-flow" | "verify-text-encoder-pth" | "verify-decoder-stage" | "verify-nsf-decoder" | "verify-hubert-conv" | "verify-hubert-frontend" | "verify-hubert-transformer" | "verify-hubert-stack" | "verify-hubert-position" | "verify-hubert-encoder" | "verify-hubert-model" | "verify-rmvpe-head" | "verify-rmvpe-encoder" | "verify-rmvpe-intermediate" | "verify-rmvpe-decoder" | "verify-rmvpe-unet" | "verify-rmvpe-gru" | "verify-rmvpe-network") {
        return Err("usage: tts-with-rvc-rs verify-text-front|verify-attention|verify-encoder-layer|verify-text-encoder <fixture.json> [flex|cpu|cuda|wgpu]".into());
    }
    let path = args.next().ok_or("missing fixture or checkpoint path")?;
    let flow_fixture_path = if command == "verify-flow" || command == "verify-text-encoder-pth" || command == "verify-decoder-stage" || command == "verify-nsf-decoder" || command == "verify-hubert-conv" || command == "verify-hubert-frontend" || command == "verify-hubert-transformer" || command == "verify-hubert-stack" || command == "verify-hubert-position" || command == "verify-hubert-encoder" || command == "verify-hubert-model" || command == "verify-rmvpe-head" || command == "verify-rmvpe-encoder" || command == "verify-rmvpe-intermediate" || command == "verify-rmvpe-decoder" || command == "verify-rmvpe-unet" || command == "verify-rmvpe-gru" || command == "verify-rmvpe-network" {
        Some(args.next().ok_or("missing flow fixture path")?)
    } else { None };
    let backend = args.next().unwrap_or_else(|| "flex".to_owned());
    let device = match backend.as_str() {
        #[cfg(feature = "flex")]
        "flex" => Device::flex(),
        #[cfg(feature = "cpu")]
        "cpu" => Device::cpu(),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(Default::default()),
        other => return Err(format!("backend {other} is not compiled into this binary")),
    };
    if command == "verify-hubert-conv" {
        use burn::tensor::{Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_model(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing HuBERT reference")?;
        let fixture: serde_json::Value = serde_json::from_str(&fs::read_to_string(fixture_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["waveform"].clone()).map_err(|e| e.to_string())?;
        let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).map_err(|e| e.to_string())?;
        let model = tts_with_rvc_rs::hubert::HubertFirstConv::load(&checkpoint, &device)?;
        let actual = model.forward(Tensor::from_data(TensorData::new(input, shape), &device))
            .into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        println!("backend={backend} max_abs_hubert_conv={error:.9} tensor_count={}", checkpoint.tensor_count());
        if !error.is_finite() || error > 1e-4 { return Err("HuBERT Conv1d/GroupNorm/GELU parity exceeded tolerance".into()); }
        return Ok(());
    }
    if command == "verify-hubert-frontend" {
        use burn::tensor::{Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_model(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing frontend reference")?;
        let fixture: serde_json::Value = serde_json::from_str(&fs::read_to_string(reference_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let waveform: Vec<f32> = serde_json::from_value(fixture["waveform"].clone()).map_err(|e| e.to_string())?;
        let input_shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|e| e.to_string())?;
        let expected_shape: [usize; 3] = serde_json::from_value(fixture["expected_shape"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).map_err(|e| e.to_string())?;
        let model = tts_with_rvc_rs::hubert::HubertFrontend::load(&checkpoint, &device)?;
        let result = model.forward(Tensor::from_data(TensorData::new(waveform, input_shape), &device));
        if result.dims() != expected_shape {
            return Err(format!("HuBERT frontend expected shape {expected_shape:?}, got {:?}", result.dims()));
        }
        let actual = result.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        println!("backend={backend} max_abs_hubert_frontend={error:.9} tensor_count={}", checkpoint.tensor_count());
        if !error.is_finite() || error > 1e-4 {
            return Err("HuBERT frontend Torch/Burn parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-hubert-transformer" || command == "verify-hubert-stack" {
        use burn::tensor::{Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_model(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing transformer reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|e| e.to_string())?,
        ).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let valid: Vec<f32> = serde_json::from_value(fixture["valid"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).map_err(|e| e.to_string())?;
        let unmasked: Vec<f32> = serde_json::from_value(fixture["expected_unmasked"].clone()).map_err(|e| e.to_string())?;
        let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|e| e.to_string())?;
        let heads: usize = serde_json::from_value(fixture["heads"].clone()).map_err(|e| e.to_string())?;
        let input = Tensor::from_data(TensorData::new(input, shape), &device);
        let valid = Tensor::from_data(TensorData::new(valid, [shape[0], shape[1]]), &device);
        let (actual, actual_unmasked) = if command == "verify-hubert-stack" {
            let layers: usize = serde_json::from_value(fixture["layers"].clone()).map_err(|e| e.to_string())?;
            let stack = tts_with_rvc_rs::hubert_transformer::HubertTransformerStack::load(&checkpoint, layers, heads, &device)?;
            (stack.forward(input.clone(), Some(valid)), stack.forward(input, None))
        } else {
            let layer = tts_with_rvc_rs::hubert_transformer::HubertTransformerLayer::load(&checkpoint, 0, heads, &device)?;
            (layer.forward(input.clone(), Some(valid)), layer.forward(input, None))
        };
        let actual = actual.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let actual_unmasked = actual_unmasked.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        let unmasked_error = max_abs_error(&actual_unmasked, &unmasked);
        println!("backend={backend} max_abs_{command}={error:.9} max_abs_unmasked={unmasked_error:.9}");
        if !error.is_finite() || error > 1e-4 || !unmasked_error.is_finite() || unmasked_error > 1e-4 {
            return Err("HuBERT Transformer Torch/Burn parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-hubert-model" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_model::HubertFeatureModel};
        let checkpoint = RvcCheckpoint::open_model(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing HuBERT waveform reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let read = |field: &str| -> Result<Vec<f32>, String> {
            serde_json::from_value(fixture[field].clone()).map_err(|error| error.to_string())
        };
        let input_shape: [usize; 3] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|error| error.to_string())?;
        let output_shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|error| error.to_string())?;
        let heads: usize = serde_json::from_value(fixture["heads"].clone()).map_err(|error| error.to_string())?;
        let layers: usize = serde_json::from_value(fixture["layers"].clone()).map_err(|error| error.to_string())?;
        let waveform = Tensor::from_data(TensorData::new(read("waveform")?, input_shape), &device);
        let valid = Tensor::from_data(TensorData::new(read("valid")?, [output_shape[0], output_shape[1]]), &device);
        let model = HubertFeatureModel::load(&checkpoint, layers, heads, &device)?;
        let result = model.forward(waveform.clone(), Some(valid));
        let unmasked_result = model.forward(waveform, None);
        if result.dims() != output_shape || unmasked_result.dims() != output_shape {
            return Err("HuBERT complete waveform forward produced incorrect output shape".into());
        }
        let actual = result.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let unmasked = unmasked_result.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&actual, &read("expected")?);
        let unmasked_error = max_abs_error(&unmasked, &read("expected_unmasked")?);
        println!("backend={backend} max_abs_hubert_model={error:.9} max_abs_unmasked={unmasked_error:.9} layers={layers} frames={}", output_shape[1]);
        if !error.is_finite() || !unmasked_error.is_finite() || error > 1e-4 || unmasked_error > 1e-4 {
            return Err(format!("HuBERT waveform → features parity exceeded tolerance: {error:.8}, {unmasked_error:.8}"));
        }
        return Ok(());
    }
    if command == "verify-hubert-position" || command == "verify-hubert-encoder" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_position::HubertEncoder};
        let checkpoint = RvcCheckpoint::open_model(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing positional reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let read = |field: &str| -> Result<Vec<f32>, String> {
            serde_json::from_value(fixture[field].clone()).map_err(|error| error.to_string())
        };
        let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|error| error.to_string())?;
        let heads: usize = serde_json::from_value(fixture["heads"].clone()).map_err(|error| error.to_string())?;
        let layers: usize = serde_json::from_value(fixture["layers"].clone()).map_err(|error| error.to_string())?;
        let x = Tensor::from_data(TensorData::new(read("input")?, shape), &device);
        let valid = Tensor::from_data(TensorData::new(read("valid")?, [shape[0], shape[1]]), &device);
        let model = HubertEncoder::load(&checkpoint, layers, heads, &device)?;
        let (actual, unmasked, expected, expected_unmasked) = if command == "verify-hubert-position" {
            (model.position_only(x.clone(), Some(valid)), model.position_only(x, None),
             read("expected_position")?, read("expected_position_unmasked")?)
        } else {
            (model.forward(x.clone(), Some(valid)), model.forward(x, None),
             read("expected")?, read("expected_unmasked")?)
        };
        let actual = actual.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let unmasked = unmasked.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&actual, &expected);
        let unmasked_error = max_abs_error(&unmasked, &expected_unmasked);
        println!("backend={backend} max_abs_{command}={error:.9} max_abs_unmasked={unmasked_error:.9} layers={layers}");
        if !error.is_finite() || !unmasked_error.is_finite() || error > 1e-4 || unmasked_error > 1e-4 {
            return Err(format!("HuBERT positional/encoder parity exceeded tolerance: {error:.8}, {unmasked_error:.8}"));
        }
        return Ok(());
    }
    if command == "verify-rmvpe-gru" || command == "verify-rmvpe-network" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe::salience_to_f0, rmvpe::RmvpeHead, rmvpe_gru::{RmvpeBiGru, RmvpeNetwork}};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing RMVPE GRU reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let read = |name: &str| -> Result<Vec<f32>, String> {
            serde_json::from_value(fixture[name].clone()).map_err(|error| error.to_string())
        };
        let (actual, expected, expected_f0) = if command == "verify-rmvpe-gru" {
            let shape: [usize; 3] = serde_json::from_value(fixture["gru_input_shape"].clone()).map_err(|error| error.to_string())?;
            let expected_shape: [usize; 3] = serde_json::from_value(fixture["gru_output_shape"].clone()).map_err(|error| error.to_string())?;
            let gru = RmvpeBiGru::load(&checkpoint, &device)?;
            let input = Tensor::from_data(TensorData::new(read("gru_input")?, shape), &device);
            let encoded = gru.forward(input);
            if encoded.dims() != expected_shape { return Err("RMVPE GRU output shape mismatch".into()); }
            let encoded = encoded.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let gru_error = max_abs_error(&encoded, &read("expected_gru")?);
            if !gru_error.is_finite() || gru_error > 1e-4 { return Err(format!("RMVPE bidirectional GRU error {gru_error}")); }
            println!("backend={backend} rmvpe_gru_input={} hidden={} max_abs_gru={gru_error:.9}", gru.input_size(), gru.hidden_size());
            let head = RmvpeHead::load(&checkpoint, &device)?;
            let output = head.forward(Tensor::from_data(TensorData::new(encoded, expected_shape), &device));
            (output.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?, read("expected_salience")?, read("expected_f0")?)
        } else {
            let shape: [usize; 3] = serde_json::from_value(fixture["mel_shape"].clone()).map_err(|error| error.to_string())?;
            let levels: usize = serde_json::from_value(fixture["unet_levels"].clone()).map_err(|error| error.to_string())?;
            let middle: usize = serde_json::from_value(fixture["middle_levels"].clone()).map_err(|error| error.to_string())?;
            let model = RmvpeNetwork::load(&checkpoint, levels, middle, shape[1], &device)?;
            let input = Tensor::from_data(TensorData::new(read("mel")?, shape), &device);
            let output = model.forward(input);
            let expected_shape: [usize; 3] = serde_json::from_value(fixture["expected_full_shape"].clone()).map_err(|error| error.to_string())?;
            if output.dims() != expected_shape { return Err(format!("RMVPE network output shape mismatch {:?}", output.dims())); }
            (output.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?, read("expected_full")?, read("expected_full_f0")?)
        };
        let error = max_abs_error(&actual, &expected);
        let f0 = salience_to_f0(&actual, 0.03)?;
        let f0_error = max_abs_error(&f0, &expected_f0);
        println!("backend={backend} max_abs_{command}={error:.9} max_abs_f0={f0_error:.9}");
        if !error.is_finite() || error > 1e-4 || !f0_error.is_finite() || f0_error > 1e-3 {
            return Err(format!("RMVPE GRU/network parity exceeded tolerance: {error}, {f0_error}"));
        }
        return Ok(());
    }
    if command == "verify-rmvpe-unet" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeUnet};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing RMVPE full U-Net reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|e| e.to_string())?,
        ).map_err(|e| e.to_string())?;
        let shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let output_shape: [usize; 4] = serde_json::from_value(fixture["output_shape"].clone()).map_err(|e| e.to_string())?;
        let feature_shape: [usize; 3] = serde_json::from_value(fixture["feature_shape"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).map_err(|e| e.to_string())?;
        let expected_features: Vec<f32> = serde_json::from_value(fixture["expected_features"].clone()).map_err(|e| e.to_string())?;
        let levels: usize = serde_json::from_value(fixture["encoder_levels"].clone()).map_err(|e| e.to_string())?;
        let middle: usize = serde_json::from_value(fixture["intermediate_levels"].clone()).map_err(|e| e.to_string())?;
        let model = RmvpeUnet::load(&checkpoint, levels, middle, &device)?;
        let x = Tensor::from_data(TensorData::new(input, shape), &device);
        let output = model.forward(x.clone());
        if output.dims() != output_shape { return Err(format!("RMVPE U-Net output shape mismatch: {:?}", output.dims())); }
        let actual = output.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        let features = model.forward_gru_features(x);
        if features.dims() != feature_shape { return Err(format!("RMVPE CNN-to-BiGRU feature shape mismatch: {:?}", features.dims())); }
        let actual_features = features.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let feature_error = max_abs_error(&actual_features, &expected_features);
        println!("backend={backend} rmvpe_unet_levels={} max_abs_rmvpe_unet={error:.9} max_abs_rmvpe_cnn_features={feature_error:.9}", model.encoder_levels());
        if !error.is_finite() || !feature_error.is_finite() || error > 1e-4 || feature_error > 1e-4 {
            return Err("RMVPE full U-Net/CNN Torch/Burn parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-rmvpe-decoder" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeDecoder};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing RMVPE decoder reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|e| e.to_string())?,
        ).map_err(|e| e.to_string())?;
        let input_shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let mut skips = Vec::new();
        for item in fixture["skips"].as_array().ok_or("missing RMVPE decoder skips")? {
            let shape: [usize; 4] = serde_json::from_value(item["shape"].clone()).map_err(|e| e.to_string())?;
            let values: Vec<f32> = serde_json::from_value(item["values"].clone()).map_err(|e| e.to_string())?;
            skips.push(Tensor::from_data(TensorData::new(values, shape), &device));
        }
        let levels = fixture["levels"].as_array().ok_or("missing RMVPE decoder levels")?;
        let model = RmvpeDecoder::load(&checkpoint, levels.len(), [2, 2], &device)?;
        let mut maximum = 0.0_f32;
        let outputs = model.forward_with_levels(Tensor::from_data(TensorData::new(input, input_shape), &device), &skips);
        for (index, (tensor, expected)) in outputs.into_iter().zip(levels).enumerate() {
            let shape: [usize; 4] = serde_json::from_value(expected["shape"].clone()).map_err(|e| e.to_string())?;
            if tensor.dims() != shape { return Err(format!("RMVPE Decoder stage {index}: shape mismatch")); }
            let values: Vec<f32> = serde_json::from_value(expected["output"].clone()).map_err(|e| e.to_string())?;
            let actual = tensor.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
            let error = max_abs_error(&actual, &values);
            println!("backend={backend} rmvpe_decoder_stage={index} max_abs={error:.9}");
            maximum = maximum.max(error);
            if !error.is_finite() || error > 1e-4 { return Err(format!("RMVPE Decoder stage {index}: Torch/Burn parity exceeded tolerance")); }
        }
        println!("backend={backend} rmvpe_decoder_stages={} max_abs_rmvpe_decoder={maximum:.9}", model.stages());
        return Ok(());
    }
    if command == "verify-rmvpe-intermediate" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeIntermediate};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing RMVPE Intermediate reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let expected = fixture["levels"].as_array().ok_or("missing RMVPE Intermediate levels")?;
        let model = RmvpeIntermediate::load(&checkpoint, expected.len(), &device)?;
        let values = model.forward_with_levels(Tensor::from_data(TensorData::new(input, shape), &device));
        let mut maximum = 0.0_f32;
        for (level, (tensor, reference)) in values.into_iter().zip(expected).enumerate() {
            let output_shape: [usize; 4] = serde_json::from_value(reference["shape"].clone()).map_err(|e| e.to_string())?;
            if tensor.dims() != output_shape { return Err(format!("RMVPE Intermediate level {level}: shape mismatch")); }
            let reference_output: Vec<f32> = serde_json::from_value(reference["output"].clone()).map_err(|e| e.to_string())?;
            let actual = tensor.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
            let error = max_abs_error(&actual, &reference_output);
            maximum = maximum.max(error);
            println!("backend={backend} rmvpe_intermediate_level={level} max_abs={error:.9}");
            if !error.is_finite() || error > 1e-4 { return Err(format!("RMVPE Intermediate level {level}: Torch/Burn parity exceeded tolerance")); }
        }
        println!("backend={backend} rmvpe_intermediate_levels={} max_abs_rmvpe_intermediate={maximum:.9}", model.levels());
        return Ok(());
    }
    if command == "verify-rmvpe-encoder" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeEncoderStage};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing RMVPE U-Net reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let input_shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let stages = fixture["stages"].as_array().ok_or("missing U-Net stages")?;
        if stages.is_empty() || stages.len() > 5 { return Err("invalid RMVPE encoder stages".into()); }
        let mut current = Tensor::from_data(TensorData::new(input, input_shape), &device);
        let mut maximum = 0.0_f32;
        for (index, expected) in stages.iter().enumerate() {
            let model = RmvpeEncoderStage::load(&checkpoint, index, &device)?;
            let result = model.forward(current);
            let skip_shape: [usize; 4] = serde_json::from_value(expected["skip_shape"].clone()).map_err(|e| e.to_string())?;
            let pooled_shape: [usize; 4] = serde_json::from_value(expected["pooled_shape"].clone()).map_err(|e| e.to_string())?;
            if result.skip.dims() != skip_shape || result.pooled.dims() != pooled_shape {
                return Err(format!("RMVPE encoder stage {index} shape mismatch"));
            }
            let read = |field: &str| -> Result<Vec<f32>, String> { serde_json::from_value(expected[field].clone()).map_err(|e| e.to_string()) };
            let error_skip = max_abs_error(&result.skip.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?, &read("skip")?);
            let error_pooled = max_abs_error(&result.pooled.clone().into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?, &read("pooled")?);
            maximum = maximum.max(error_skip).max(error_pooled);
            println!("backend={backend} rmvpe_encoder_stage={index} blocks={} max_abs_skip={error_skip:.9} max_abs_pooled={error_pooled:.9}", model.blocks());
            if !error_skip.is_finite() || !error_pooled.is_finite() || error_skip > 1e-4 || error_pooled > 1e-4 {
                return Err(format!("RMVPE U-Net stage {index} Torch/Burn parity exceeded tolerance"));
            }
            current = result.pooled;
        }
        println!("backend={backend} rmvpe_encoder_stages={} max_abs_rmvpe_encoder={maximum:.9}", stages.len());
        return Ok(());
    }
    if command == "verify-rmvpe-head" {
        use burn::tensor::{Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_root(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing RMVPE reference")?;
        let fixture: serde_json::Value = serde_json::from_str(&fs::read_to_string(fixture_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["features"].clone()).map_err(|e| e.to_string())?;
        let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected_salience"].clone()).map_err(|e| e.to_string())?;
        let expected_f0: Vec<f32> = serde_json::from_value(fixture["expected_f0"].clone()).map_err(|e| e.to_string())?;
        let model = tts_with_rvc_rs::rmvpe::RmvpeHead::load(&checkpoint, &device)?;
        let actual = model.forward(Tensor::from_data(TensorData::new(input, shape), &device))
            .into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        let f0 = tts_with_rvc_rs::rmvpe::salience_to_f0(&actual, 0.03)?;
        let f0_error = max_abs_error(&f0, &expected_f0);
        println!("backend={backend} max_abs_rmvpe_salience={error:.9} max_abs_f0={f0_error:.9} tensor_count={}", checkpoint.tensor_count());
        if !error.is_finite() || error > 1e-4 || !f0_error.is_finite() || f0_error > 1e-3 {
            return Err("RMVPE output head / F0 parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-nsf-decoder" {
        let reference_path = flow_fixture_path.as_ref().ok_or("missing NSF decoder reference")?;
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        let data = fs::read_to_string(reference_path).map_err(|error| error.to_string())?;
        let reference: tts_with_rvc_rs::decoder::NsfDecoderFixture = serde_json::from_str(&data).map_err(|error| error.to_string())?;
        let decoder = reference.load(&checkpoint, &device)?;
        let (input, harmonic, g) = reference.tensors(&device);
        let actual = decoder.forward(input, harmonic, g).into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&actual, &reference.expected_waveform);
        println!("backend={backend} stages={} max_abs_waveform={error:.9}", decoder.stage_count());
        if !error.is_finite() || error > 1e-4 {
            return Err("Torch/Burn NSF waveform decoder parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-decoder-stage" {
        let reference_path = flow_fixture_path.as_ref().ok_or("missing decoder reference")?;
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        let data = fs::read_to_string(reference_path).map_err(|error| error.to_string())?;
        let reference: tts_with_rvc_rs::decoder::DecoderStageFixture = serde_json::from_str(&data).map_err(|error| error.to_string())?;
        let stage = reference.load(&checkpoint, &device)?;
        let (input, source) = reference.tensors(&device);
        let upsampled = stage.upsample_only(input.clone()).into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let up_error = max_abs_error(&upsampled, &reference.expected_upsampled);
        let result = stage.forward(input, source).into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&result, &reference.expected_output);
        println!("backend={backend} stage={} resblocks={} max_abs_upsample={up_error:.9} max_abs_decoder_stage={error:.9}", reference.stage, stage.residual_count());
        if !up_error.is_finite() || up_error > 1e-4 || !error.is_finite() || error > 1e-4 {
            return Err("Torch/Burn NSF decoder stage parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-flow" {
        let flow_fixture_path = flow_fixture_path.as_ref().ok_or("missing flow reference")?;
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        let data = fs::read_to_string(flow_fixture_path).map_err(|error| error.to_string())?;
        let fixture: tts_with_rvc_rs::flow::FlowFixture = serde_json::from_str(&data).map_err(|error| error.to_string())?;
        let flow = tts_with_rvc_rs::flow::ResidualCouplingBlock::load(&checkpoint, &device)?;
        let (input, mask, condition) = fixture.tensors(&device);
        let forward = flow.forward(input, mask.clone(), Some(condition.clone()), false);
        let actual = forward.clone().into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&actual, &fixture.expected_forward);
        let reverse = flow.forward(forward, mask, Some(condition), true);
        let actual = reverse.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let reverse_error = max_abs_error(&actual, &fixture.expected_reverse);
        println!("backend={backend} flow_layers={} max_abs_flow_forward={error:.9} max_abs_flow_reverse={reverse_error:.9}", flow.len());
        if !error.is_finite() || error > 1e-4 || !reverse_error.is_finite() || reverse_error > 1e-4 {
            return Err("Torch/Burn ResidualCouplingBlock parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-text-encoder-pth" {
        use burn::tensor::{Int, Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing TextEncoder reference")?;
        let data = fs::read_to_string(reference_path).map_err(|error| error.to_string())?;
        let fixture: TextEncoderFixture = serde_json::from_str(&data).map_err(|error| error.to_string())?;
        let model = tts_with_rvc_rs::text_encoder::TextEncoder::load(&checkpoint, &device)?;
        let (batch, time, channels) = (fixture.front.batch, fixture.front.time, fixture.front.in_channels);
        let phone = Tensor::from_data(TensorData::new(fixture.front.phone.clone(), [batch, time, channels]), &device);
        let pitch = Tensor::<2, Int>::from_data(TensorData::new(fixture.front.pitch.clone(), [batch, time]), &device);
        let verify = |out: tts_with_rvc_rs::text_encoder::TextEncoderOutput, mean: &[f32], logs: &[f32], mask: &[f32]| -> Result<f32, String> {
            let actual_mean = out.mean.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let actual_logs = out.log_scale.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let actual_mask = out.mask.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let deviation = max_abs_error(&actual_mean, mean).max(max_abs_error(&actual_logs, logs));
            let mask_error = max_abs_error(&actual_mask, mask);
            if !deviation.is_finite() || deviation > 1e-4 || mask_error != 0.0 {
                return Err(format!("native TextEncoder parity failed: deviation={deviation}, mask={mask_error}"));
            }
            Ok(deviation)
        };
        let forward = model.forward(phone.clone(), Some(pitch.clone()), &fixture.front.lengths, None);
        let error = verify(forward, &fixture.expected_mean, &fixture.expected_log_scale, &fixture.expected_mask)?;
        let skipped = model.forward(phone, Some(pitch), &fixture.front.lengths, Some(fixture.skip_head));
        let skip_error = verify(skipped, &fixture.expected_skip_mean, &fixture.expected_skip_log_scale, &fixture.expected_skip_mask)?;
        println!("backend={backend} max_abs_native_text_encoder={error:.9} max_abs_native_skip_head={skip_error:.9}");
        return Ok(());
    }
    let data = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    if command == "verify-text-encoder" {
        let fixture: TextEncoderFixture = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        let output = fixture.forward(&device, None);
        let mean = output.mean.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let logs = output.log_scale.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let mask = output.mask.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&mean, &fixture.expected_mean)
            .max(max_abs_error(&logs, &fixture.expected_log_scale));
        let mask_error = max_abs_error(&mask, &fixture.expected_mask);
        println!("backend={backend} max_abs_text_encoder={error:.9} max_abs_mask={mask_error:.9}");
        if !error.is_finite() || error > 1e-4 || mask_error > 0.0 {
            return Err("Torch/Burn full TextEncoder parity exceeded tolerance".into());
        }
        let skipped = fixture.forward(&device, Some(fixture.skip_head));
        let mean = skipped.mean.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let logs = skipped.log_scale.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let mask = skipped.mask.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&mean, &fixture.expected_skip_mean)
            .max(max_abs_error(&logs, &fixture.expected_skip_log_scale));
        let mask_error = max_abs_error(&mask, &fixture.expected_skip_mask);
        println!("backend={backend} max_abs_skip_head={error:.9} max_abs_skip_mask={mask_error:.9}");
        if !error.is_finite() || error > 1e-4 || mask_error > 0.0 {
            return Err("Torch/Burn TextEncoder skip_head parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-encoder-layer" {
        let fixture: EncoderLayerFixture = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        let actual = fixture.forward(&device).into_data().try_into_vec::<f32>()
            .map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &fixture.expected_encoder);
        println!("backend={backend} max_abs_encoder_layer={error:.9}");
        if !error.is_finite() || error > 1e-4 {
            return Err("Torch/Burn encoder layer parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-attention" {
        let fixture: AttentionFixture = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        let actual = fixture.forward(&device).into_data().try_into_vec::<f32>()
            .map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &fixture.expected);
        println!("backend={backend} max_abs_attention={error:.9}");
        if !error.is_finite() || error > 1e-4 {
            return Err("Torch/Burn attention parity exceeded tolerance".into());
        }
        return Ok(());
    }
    let fixture: TextFrontFixture = serde_json::from_str(&data).map_err(|e| e.to_string())?;
    let result = fixture.forward(&device);
    let actual_features = result.features.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
    let actual_mask = result.mask.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
    let feature_error = max_abs_error(&actual_features, &fixture.expected_features);
    let mask_error = max_abs_error(&actual_mask, &fixture.expected_mask);
    println!("backend={backend} max_abs_features={feature_error:.9} max_abs_mask={mask_error:.9}");
    if !feature_error.is_finite() || feature_error > 1e-4 || mask_error > 0.0 {
        return Err("Torch/Burn parity exceeded tolerance".into());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
