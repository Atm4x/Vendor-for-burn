use burn::tensor::{Device, Tensor, TensorData};
use tts_with_rvc_rs::{TextFrontFixture, fused_add_tanh_sigmoid_multiply, max_abs_error};

#[test]
fn text_encoder_front_matches_pytorch_reference() {
    let fixture: TextFrontFixture = serde_json::from_str(include_str!("fixtures/text_front.json")).unwrap();
    let result = fixture.forward(&Device::flex());
    assert_eq!(result.features.dims(), [fixture.batch, fixture.hidden_channels, fixture.time]);
    assert_eq!(result.mask.dims(), [fixture.batch, 1, fixture.time]);
    let features = result.features.into_data().try_into_vec::<f32>().unwrap();
    let mask = result.mask.into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&features, &fixture.expected_features) <= 1e-5);
    assert_eq!(max_abs_error(&mask, &fixture.expected_mask), 0.0);
}

#[test]
fn wavenet_gate_matches_pytorch_reference() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/wavenet_gate.json")).unwrap();
    let read = |key: &str| -> Vec<f32> {
        fixture[key].as_array().unwrap().iter().map(|value| value.as_f64().unwrap() as f32).collect()
    };
    let device = Device::flex();
    let a = Tensor::from_data(TensorData::new(read("a"), [1, 4, 3]), &device);
    let b = Tensor::from_data(TensorData::new(read("b"), [1, 4, 3]), &device);
    let result = fused_add_tanh_sigmoid_multiply(a, b, 2)
        .into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&result, &read("expected")) <= 1e-6);
}

#[test]
fn relative_attention_matches_pytorch_reference() {
    let fixture: tts_with_rvc_rs::attention::AttentionFixture =
        serde_json::from_str(include_str!("fixtures/relative_attention.json")).unwrap();
    let result = fixture.forward(&Device::flex());
    assert_eq!(result.dims(), [fixture.batch, fixture.channels, fixture.time]);
    let actual = result.into_data().try_into_vec::<f32>().unwrap();
    let error = max_abs_error(&actual, &fixture.expected);
    assert!(error <= 1e-4, "relative attention max absolute deviation {error}");
}

#[test]
fn encoder_layer_matches_pytorch_reference() {
    let fixture: tts_with_rvc_rs::encoder::EncoderLayerFixture =
        serde_json::from_str(include_str!("fixtures/encoder_layer.json")).unwrap();
    let output = fixture.forward(&Device::flex());
    assert_eq!(output.dims(), [fixture.attention.batch, fixture.attention.channels, fixture.attention.time]);
    let actual = output.into_data().try_into_vec::<f32>().unwrap();
    let error = max_abs_error(&actual, &fixture.expected_encoder);
    assert!(error <= 1e-4, "encoder layer max absolute deviation {error}");
}

#[test]
fn full_text_encoder_matches_pytorch_reference() {
    use tts_with_rvc_rs::text_encoder::TextEncoderFixture;
    let fixture: TextEncoderFixture =
        serde_json::from_str(include_str!("fixtures/text_encoder.json")).unwrap();
    assert_eq!(fixture.layers.len(), 6);
    let device = Device::flex();
    let check = |output: tts_with_rvc_rs::text_encoder::TextEncoderOutput,
                 expected_mean: &[f32], expected_log_scale: &[f32], expected_mask: &[f32]| {
        let mean = output.mean.into_data().try_into_vec::<f32>().unwrap();
        let log_scale = output.log_scale.into_data().try_into_vec::<f32>().unwrap();
        let mask = output.mask.into_data().try_into_vec::<f32>().unwrap();
        let error = max_abs_error(&mean, expected_mean)
            .max(max_abs_error(&log_scale, expected_log_scale));
        assert!(error <= 1e-4, "full TextEncoder max abs deviation {error}");
        assert_eq!(max_abs_error(&mask, expected_mask), 0.0);
    };
    check(fixture.forward(&device, None),
          &fixture.expected_mean, &fixture.expected_log_scale, &fixture.expected_mask);
    check(fixture.forward(&device, Some(fixture.skip_head)),
          &fixture.expected_skip_mean, &fixture.expected_skip_log_scale,
          &fixture.expected_skip_mask);
}

#[test]
fn direct_pytorch_checkpoint_reads_half_and_reconstructs_weight_norm() {
    use tts_with_rvc_rs::checkpoint::RvcCheckpoint;
    let checkpoint = RvcCheckpoint::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_checkpoint.pth")).unwrap();
    assert_eq!(checkpoint.tensor_count(), 5);
    assert!(!checkpoint.contains("other"));
    let (shape, values) = checkpoint.tensor_f32("enc_p.emb_pitch.weight").unwrap();
    assert_eq!(shape, vec![1, 3]);
    assert_eq!(values, vec![1.5, -2.25, 3.0]);
    assert_eq!(checkpoint.tensor_f32("roundtrip.bfloat16").unwrap().1, vec![1.5, -2.25]);
    let values = checkpoint.weight_norm_conv1d("flow.flows.0.enc.in_layers.0", &Device::flex())
        .unwrap().into_data().try_into_vec::<f32>().unwrap();
    let expected = [1.2, 1.6, 0.0, 0.0, 0.0, 0.0, 3.0_f32 / 5.0_f32.sqrt(), 6.0_f32 / 5.0_f32.sqrt()];
    assert!(max_abs_error(&values, &expected) < 1e-6);
    assert!(checkpoint.tensor_f32("nonexistent").is_err());
}

#[test]
fn residual_coupling_flow_matches_original_pytorch_model() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, flow::{FlowFixture, ResidualCouplingBlock}};
    let checkpoint = RvcCheckpoint::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_flow.pth")).unwrap();
    let reference: FlowFixture = serde_json::from_str(include_str!("fixtures/flow.json")).unwrap();
    let device = Device::flex();
    let flow = ResidualCouplingBlock::load(&checkpoint, &device).unwrap();
    assert_eq!(flow.len(), 2);
    let (x, mask, condition) = reference.tensors(&device);
    let forward = flow.forward(x, mask.clone(), Some(condition.clone()), false);
    let actual = forward.clone().into_data().try_into_vec::<f32>().unwrap();
    let error = max_abs_error(&actual, &reference.expected_forward);
    assert!(error <= 1e-4, "flow forward max absolute deviation {error}");
    let reverse = flow.forward(forward, mask, Some(condition), true);
    let actual = reverse.into_data().try_into_vec::<f32>().unwrap();
    let error = max_abs_error(&actual, &reference.expected_reverse);
    assert!(error <= 1e-4, "flow reverse max absolute deviation {error}");
}

#[test]
fn native_checkpoint_text_encoder_matches_original_pytorch() {
    use burn::tensor::Int;
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, text_encoder::{TextEncoder, TextEncoderFixture}};
    let checkpoint = RvcCheckpoint::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_text_encoder.pth")).unwrap();
    let fixture: TextEncoderFixture = serde_json::from_str(include_str!("fixtures/text_encoder.json")).unwrap();
    let device = Device::flex();
    let model = TextEncoder::load(&checkpoint, &device).unwrap();
    let (batch, time, channels) = (fixture.front.batch, fixture.front.time, fixture.front.in_channels);
    let phone = Tensor::from_data(TensorData::new(fixture.front.phone.clone(), [batch, time, channels]), &device);
    let pitch = burn::tensor::Tensor::<2, Int>::from_data(
        TensorData::new(fixture.front.pitch.clone(), [batch, time]), &device,
    );
    let output = model.forward(phone.clone(), Some(pitch.clone()), &fixture.front.lengths, None);
    let mean = output.mean.into_data().try_into_vec::<f32>().unwrap();
    let log_scale = output.log_scale.into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&mean, &fixture.expected_mean) < 1e-4);
    assert!(max_abs_error(&log_scale, &fixture.expected_log_scale) < 1e-4);
    let skipped = model.forward(phone, Some(pitch), &fixture.front.lengths, Some(fixture.skip_head));
    let mean = skipped.mean.into_data().try_into_vec::<f32>().unwrap();
    let log_scale = skipped.log_scale.into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&mean, &fixture.expected_skip_mean) < 1e-4);
    assert!(max_abs_error(&log_scale, &fixture.expected_skip_log_scale) < 1e-4);
}

#[test]
fn decoder_first_stage_matches_original_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, decoder::DecoderStageFixture};
    let checkpoint = RvcCheckpoint::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_decoder.pth")).unwrap();
    let reference: DecoderStageFixture = serde_json::from_str(include_str!("fixtures/decoder_stage.json")).unwrap();
    let device = Device::flex();
    let stage = reference.load(&checkpoint, &device).unwrap();
    assert_eq!(stage.residual_count(), 3);
    let (input, source) = reference.tensors(&device);
    let upsampled = stage.upsample_only(input.clone()).into_data().try_into_vec::<f32>().unwrap();
    let up_error = max_abs_error(&upsampled, &reference.expected_upsampled);
    assert!(up_error.is_finite() && up_error < 1e-4, "ConvTranspose1d error {up_error}");
    let output = stage.forward(input, source).into_data().try_into_vec::<f32>().unwrap();
    let error = max_abs_error(&output, &reference.expected_output);
    assert!(error.is_finite() && error < 1e-4, "DecoderStage error {error}");
}

#[test]
fn complete_nsf_decoder_with_external_harmonic_matches_original_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, decoder::NsfDecoderFixture};
    let checkpoint = RvcCheckpoint::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_nsf_decoder.pth")).unwrap();
    let reference: NsfDecoderFixture = serde_json::from_str(include_str!("fixtures/nsf_decoder.json")).unwrap();
    let device = Device::flex();
    let decoder = reference.load(&checkpoint, &device).unwrap();
    assert_eq!(decoder.stage_count(), reference.upsample_rates.len());
    let (input, source, condition) = reference.tensors(&device);
    let actual = decoder.forward(input, source, condition).into_data().try_into_vec::<f32>().unwrap();
    let deviation = max_abs_error(&actual, &reference.expected_waveform);
    assert!(deviation.is_finite() && deviation < 1e-4, "NSF decoder error {deviation}");
}

#[test]
fn hubert_first_conv_matches_torch_group_norm_gelu() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert::HubertFirstConv};
    let checkpoint = RvcCheckpoint::open_model(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_hubert_head.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/hubert_conv.json")).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["waveform"].clone()).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let device = Device::flex();
    let model = HubertFirstConv::load(&checkpoint, &device).unwrap();
    let actual = model.forward(Tensor::from_data(TensorData::new(input, shape), &device))
        .into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&actual, &expected) < 1e-4);
}

#[test]
fn rmvpe_output_head_and_local_centroid_match_torch_numpy() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe::{RmvpeHead, salience_to_f0}};
    let checkpoint = RvcCheckpoint::open_root(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_rmvpe_head.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_head.json")).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["features"].clone()).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected_salience"].clone()).unwrap();
    let f0_expected: Vec<f32> = serde_json::from_value(fixture["expected_f0"].clone()).unwrap();
    let device = Device::flex();
    let model = RmvpeHead::load(&checkpoint, &device).unwrap();
    let actual = model.forward(Tensor::from_data(TensorData::new(input, shape), &device))
        .into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&actual, &expected) < 1e-4);
    let f0 = salience_to_f0(&actual, 0.03).unwrap();
    assert!(max_abs_error(&f0, &f0_expected) < 1e-3);
    assert_eq!(salience_to_f0(&vec![0.0; 360], 0.03).unwrap(), vec![0.0]);
    assert!(salience_to_f0(&vec![f32::NAN; 360], 0.03).is_err());
}

#[test]
fn seven_layer_hubert_frontend_matches_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert::HubertFrontend};
    let checkpoint = RvcCheckpoint::open_model(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_hubert_frontend.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/hubert_frontend.json")).unwrap();
    let waveform: Vec<f32> = serde_json::from_value(fixture["waveform"].clone()).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).unwrap();
    let expected_shape: [usize; 3] = serde_json::from_value(fixture["expected_shape"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let device = Device::flex();
    let model = HubertFrontend::load(&checkpoint, &device).unwrap();
    let actual = model.forward(Tensor::from_data(TensorData::new(waveform, shape), &device));
    assert_eq!(actual.dims(), expected_shape);
    assert!(max_abs_error(&actual.into_data().try_into_vec::<f32>().unwrap(), &expected) < 1e-4);
}

#[test]
fn hubert_transformer_layer_matches_pytorch_with_and_without_padding() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_transformer::HubertTransformerLayer};
    let checkpoint = RvcCheckpoint::open_model(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_hubert_transformer.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/hubert_transformer.json")).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).unwrap();
    let valid: Vec<f32> = serde_json::from_value(fixture["valid"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let expected_unmasked: Vec<f32> = serde_json::from_value(fixture["expected_unmasked"].clone()).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).unwrap();
    let heads: usize = serde_json::from_value(fixture["heads"].clone()).unwrap();
    let device = Device::flex();
    let model = HubertTransformerLayer::load(&checkpoint, 0, heads, &device).unwrap();
    let x = Tensor::from_data(TensorData::new(input, shape), &device);
    let mask = Tensor::from_data(TensorData::new(valid, [shape[0], shape[1]]), &device);
    let actual_masked = model.forward(x.clone(), Some(mask)).into_data().try_into_vec::<f32>().unwrap();
    let actual_unmasked = model.forward(x, None).into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&actual_masked, &expected) < 1e-4);
    assert!(max_abs_error(&actual_unmasked, &expected_unmasked) < 1e-4);
}

#[test]
fn hubert_transformer_stack_matches_torch_with_padding() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_transformer::HubertTransformerStack};
    let checkpoint = RvcCheckpoint::open_model(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_hubert_stack.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/hubert_stack.json")).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).unwrap();
    let valid: Vec<f32> = serde_json::from_value(fixture["valid"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let expected_unmasked: Vec<f32> = serde_json::from_value(fixture["expected_unmasked"].clone()).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).unwrap();
    let heads: usize = serde_json::from_value(fixture["heads"].clone()).unwrap();
    let layers: usize = serde_json::from_value(fixture["layers"].clone()).unwrap();
    let device = Device::flex();
    let model = HubertTransformerStack::load(&checkpoint, layers, heads, &device).unwrap();
    assert_eq!(model.len(), 3);
    let x = Tensor::from_data(TensorData::new(input, shape), &device);
    let mask = Tensor::from_data(TensorData::new(valid, [shape[0], shape[1]]), &device);
    let actual_masked = model.forward(x.clone(), Some(mask)).into_data().try_into_vec::<f32>().unwrap();
    let actual_unmasked = model.forward(x, None).into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&actual_masked, &expected) < 1e-4);
    assert!(max_abs_error(&actual_unmasked, &expected_unmasked) < 1e-4);
}

#[test]
fn hubert_positional_conv_and_three_layer_encoder_match_torch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_position::HubertEncoder};
    let checkpoint = RvcCheckpoint::open_model(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_hubert_position.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/hubert_position.json")).unwrap();
    let get = |name: &str| -> Vec<f32> { serde_json::from_value(fixture[name].clone()).unwrap() };
    let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).unwrap();
    let heads: usize = serde_json::from_value(fixture["heads"].clone()).unwrap();
    let layers: usize = serde_json::from_value(fixture["layers"].clone()).unwrap();
    let device = Device::flex();
    let model = HubertEncoder::load(&checkpoint, layers, heads, &device).unwrap();
    assert_eq!(model.layers(), 3);
    let input = Tensor::from_data(TensorData::new(get("input"), shape), &device);
    let valid = Tensor::from_data(TensorData::new(get("valid"), [shape[0], shape[1]]), &device);
    let position = model.position_only(input.clone(), Some(valid.clone()))
        .into_data().try_into_vec::<f32>().unwrap();
    let position_unmasked = model.position_only(input.clone(), None)
        .into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&position, &get("expected_position")) < 1e-4);
    assert!(max_abs_error(&position_unmasked, &get("expected_position_unmasked")) < 1e-4);
    let encoded = model.forward(input.clone(), Some(valid))
        .into_data().try_into_vec::<f32>().unwrap();
    let encoded_unmasked = model.forward(input, None)
        .into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&encoded, &get("expected")) < 1e-4);
    assert!(max_abs_error(&encoded_unmasked, &get("expected_unmasked")) < 1e-4);
}

#[test]
fn hubert_waveform_to_content_matches_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_model::HubertFeatureModel};
    let checkpoint = RvcCheckpoint::open_model(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_hubert_model.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/hubert_model.json")).unwrap();
    let get = |field: &str| -> Vec<f32> { serde_json::from_value(fixture[field].clone()).unwrap() };
    let input_shape: [usize; 3] = serde_json::from_value(fixture["input_shape"].clone()).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).unwrap();
    let layers: usize = serde_json::from_value(fixture["layers"].clone()).unwrap();
    let heads: usize = serde_json::from_value(fixture["heads"].clone()).unwrap();
    let device = Device::flex();
    let model = HubertFeatureModel::load(&checkpoint, layers, heads, &device).unwrap();
    assert_eq!(model.output_layers(), 3);
    let waveform = Tensor::from_data(TensorData::new(get("waveform"), input_shape), &device);
    let valid = Tensor::from_data(TensorData::new(get("valid"), [shape[0], shape[1]]), &device);
    let masked = model.forward(waveform.clone(), Some(valid));
    assert_eq!(masked.dims(), shape);
    let unmasked = model.forward(waveform, None);
    let masked = masked.into_data().try_into_vec::<f32>().unwrap();
    let unmasked = unmasked.into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&masked, &get("expected")) < 1e-4);
    assert!(max_abs_error(&unmasked, &get("expected_unmasked")) < 1e-4);
}

#[test]
fn rmvpe_first_two_unet_encoder_stages_match_original_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeEncoderStage};
    let checkpoint = RvcCheckpoint::open_root(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_rmvpe_encoder.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_encoder.json")).unwrap();
    let shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).unwrap();
    let device = Device::flex();
    let mut current = Tensor::from_data(TensorData::new(input, shape), &device);
    let stages = fixture["stages"].as_array().unwrap();
    assert_eq!(stages.len(), 2);
    for (index, expected) in stages.iter().enumerate() {
        let model = RmvpeEncoderStage::load(&checkpoint, index, &device).unwrap();
        assert_eq!(model.blocks(), 2);
        let result = model.forward(current);
        let expected_skip: Vec<f32> = serde_json::from_value(expected["skip"].clone()).unwrap();
        let expected_pool: Vec<f32> = serde_json::from_value(expected["pooled"].clone()).unwrap();
        let shape_skip: [usize; 4] = serde_json::from_value(expected["skip_shape"].clone()).unwrap();
        let shape_pool: [usize; 4] = serde_json::from_value(expected["pooled_shape"].clone()).unwrap();
        assert_eq!(result.skip.dims(), shape_skip);
        assert_eq!(result.pooled.dims(), shape_pool);
        let error_skip = max_abs_error(&result.skip.into_data().try_into_vec::<f32>().unwrap(), &expected_skip);
        let error_pool = max_abs_error(&result.pooled.clone().into_data().try_into_vec::<f32>().unwrap(), &expected_pool);
        assert!(error_skip.is_finite() && error_skip < 1e-4, "RMVPE stage {index} skip parity {error_skip}");
        assert!(error_pool.is_finite() && error_pool < 1e-4, "RMVPE stage {index} pooled parity {error_pool}");
        current = result.pooled;
    }
}

#[test]
fn rmvpe_intermediate_residual_groups_match_original_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeIntermediate};
    let checkpoint = RvcCheckpoint::open_root(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_rmvpe_intermediate.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_intermediate.json")).unwrap();
    let shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).unwrap();
    let levels = fixture["levels"].as_array().unwrap();
    assert_eq!(levels.len(), 2);
    let device = Device::flex();
    let model = RmvpeIntermediate::load(&checkpoint, levels.len(), &device).unwrap();
    assert_eq!(model.levels(), 2);
    let tensors = model.forward_with_levels(Tensor::from_data(TensorData::new(input, shape), &device));
    for (index, (tensor, expected)) in tensors.into_iter().zip(levels).enumerate() {
        let expected_shape: [usize; 4] = serde_json::from_value(expected["shape"].clone()).unwrap();
        let expected_output: Vec<f32> = serde_json::from_value(expected["output"].clone()).unwrap();
        assert_eq!(tensor.dims(), expected_shape);
        let actual = tensor.into_data().try_into_vec::<f32>().unwrap();
        let error = max_abs_error(&actual, &expected_output);
        assert!(error.is_finite() && error < 1e-4, "RMVPE Intermediate level {index} deviation {error}");
    }
}

#[test]
fn rmvpe_unet_decoder_matches_original_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeDecoder};
    let checkpoint = RvcCheckpoint::open_root(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_rmvpe_decoder.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_decoder.json")).unwrap();
    let device = Device::flex();
    let input_shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).unwrap();
    let skips: Vec<Tensor<4>> = fixture["skips"].as_array().unwrap().iter().map(|s| {
        let shape: [usize; 4] = serde_json::from_value(s["shape"].clone()).unwrap();
        let values: Vec<f32> = serde_json::from_value(s["values"].clone()).unwrap();
        Tensor::from_data(TensorData::new(values, shape), &device)
    }).collect();
    let levels = fixture["levels"].as_array().unwrap();
    assert_eq!(levels.len(), 2);
    let decoder = RmvpeDecoder::load(&checkpoint, levels.len(), [2, 2], &device).unwrap();
    assert_eq!(decoder.stages(), 2);
    let outputs = decoder.forward_with_levels(Tensor::from_data(TensorData::new(input, input_shape), &device), &skips);
    for (index, (actual, expected)) in outputs.into_iter().zip(levels).enumerate() {
        let expected_shape: [usize; 4] = serde_json::from_value(expected["shape"].clone()).unwrap();
        let expected_values: Vec<f32> = serde_json::from_value(expected["output"].clone()).unwrap();
        assert_eq!(actual.dims(), expected_shape);
        let actual_values = actual.into_data().try_into_vec::<f32>().unwrap();
        let error = max_abs_error(&actual_values, &expected_values);
        assert!(error.is_finite() && error < 1e-4, "RMVPE decoder stage {index} deviation {error}");
    }
}

#[test]
fn complete_rmvpe_unet_and_cnn_match_original_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeUnet};
    let checkpoint = RvcCheckpoint::open_root(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_rmvpe_unet.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_unet.json")).unwrap();
    let input_shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).unwrap();
    let output_shape: [usize; 4] = serde_json::from_value(fixture["output_shape"].clone()).unwrap();
    let features_shape: [usize; 3] = serde_json::from_value(fixture["feature_shape"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let expected_features: Vec<f32> = serde_json::from_value(fixture["expected_features"].clone()).unwrap();
    let levels = fixture["encoder_levels"].as_u64().unwrap() as usize;
    let middle = fixture["intermediate_levels"].as_u64().unwrap() as usize;
    let device = Device::flex();
    let model = RmvpeUnet::load(&checkpoint, levels, middle, &device).unwrap();
    assert_eq!(model.encoder_levels(), 2);
    let x = Tensor::from_data(TensorData::new(input, input_shape), &device);
    let logits = model.forward(x.clone());
    assert_eq!(logits.dims(), output_shape);
    let logits = logits.into_data().try_into_vec::<f32>().unwrap();
    let features = model.forward_gru_features(x);
    assert_eq!(features.dims(), features_shape);
    let features = features.into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&logits, &expected) < 1e-4);
    assert!(max_abs_error(&features, &expected_features) < 1e-4);
}

#[test]
fn rmvpe_bidirectional_gru_matches_original_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_gru::RmvpeBiGru};
    let checkpoint = RvcCheckpoint::open_root(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_rmvpe_gru.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_gru.json")).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["gru_input_shape"].clone()).unwrap();
    let output_shape: [usize; 3] = serde_json::from_value(fixture["gru_output_shape"].clone()).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["gru_input"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected_gru"].clone()).unwrap();
    let device = Device::flex();
    let gru = RmvpeBiGru::load(&checkpoint, &device).unwrap();
    assert_eq!(gru.input_size(), shape[2]);
    assert_eq!(gru.hidden_size(), 256);
    let actual = gru.forward(Tensor::from_data(TensorData::new(input, shape), &device));
    assert_eq!(actual.dims(), output_shape);
    let values = actual.into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&values, &expected) < 1e-4);
}

#[test]
fn rmvpe_unet_bigru_and_salience_head_match_original_pytorch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe::salience_to_f0, rmvpe_gru::RmvpeNetwork};
    let checkpoint = RvcCheckpoint::open_root(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_rmvpe_gru.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_gru.json")).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["mel_shape"].clone()).unwrap();
    let expected_shape: [usize; 3] = serde_json::from_value(fixture["expected_full_shape"].clone()).unwrap();
    let input: Vec<f32> = serde_json::from_value(fixture["mel"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected_full"].clone()).unwrap();
    let expected_f0: Vec<f32> = serde_json::from_value(fixture["expected_full_f0"].clone()).unwrap();
    let device = Device::flex();
    let model = RmvpeNetwork::load(&checkpoint, 2, 2, shape[1], &device).unwrap();
    let actual = model.forward(Tensor::from_data(TensorData::new(input, shape), &device));
    assert_eq!(actual.dims(), expected_shape);
    let values = actual.into_data().try_into_vec::<f32>().unwrap();
    assert!(max_abs_error(&values, &expected) < 1e-4);
    let f0 = salience_to_f0(&values, 0.03).unwrap();
    assert!(max_abs_error(&f0, &expected_f0) < 1e-3);
}

#[test]
fn rmvpe_stft_mel_matches_original_pytorch_librosa() {
    use tts_with_rvc_rs::rmvpe_mel::RmvpeMelSpectrogram;
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_mel.json")).unwrap();
    let audio: Vec<f32> = serde_json::from_value(fixture["audio"].clone()).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["mel_shape"].clone()).unwrap();
    let expected: Vec<f32> = serde_json::from_value(fixture["expected_mel"].clone()).unwrap();
    let mel = RmvpeMelSpectrogram::new();
    let (actual, frames) = mel.forward(&audio).unwrap();
    assert_eq!([1, 128, frames], shape);
    assert!(max_abs_error(&actual, &expected) < 5e-4);
    assert!(mel.forward(&audio[..512]).is_err());
    assert!(mel.forward(&[f32::NAN; 1024]).is_err());
}

#[test]
fn nsf_sine_generation_and_linear_source_match_original_torch() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, nsf_source::NsfSourceFixture};
    let checkpoint = RvcCheckpoint::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_nsf_decoder.pth")).unwrap();
    let fixture: NsfSourceFixture = serde_json::from_str(include_str!("fixtures/nsf_source.json")).unwrap();
    let (sine_error, voiced_error, noise_error, merged_error) = fixture.run(&checkpoint, &Device::flex()).unwrap();
    assert!(sine_error.is_finite() && sine_error < 1e-4, "SineGen deviation {sine_error}");
    assert_eq!(voiced_error, 0.0);
    assert!(noise_error.is_finite() && noise_error < 1e-6, "SineGen noise deviation {noise_error}");
    assert!(merged_error.is_finite() && merged_error < 1e-4, "NSF SourceModule deviation {merged_error}");
}

#[test]
fn nsf_f0_to_waveform_matches_original_generator() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, nsf_source::NsfFromF0Fixture};
    let checkpoint = RvcCheckpoint::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_nsf_decoder.pth")).unwrap();
    let fixture: NsfFromF0Fixture = serde_json::from_str(include_str!("fixtures/nsf_f0_decoder.json")).unwrap();
    let error = fixture.run(&checkpoint, &Device::flex()).unwrap();
    assert!(error.is_finite() && error < 1e-4, "NSF full F0→waveform deviation {error}");
}

#[test]
fn unsupported_multiharmonic_nsf_fails_explicitly() {
    use tts_with_rvc_rs::nsf_source::SineGen;
    assert!(SineGen::new(40000, 2).is_err());
    assert!(SineGen::new(0, 0).is_err());
}

#[test]
fn tensor_batch_hubert_matches_individual_waveforms() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_model::HubertFeatureModel};
    let checkpoint = RvcCheckpoint::open_model(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_hubert_model.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/hubert_model.json")).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["input_shape"].clone()).unwrap();
    let waveform: Vec<f32> = serde_json::from_value(fixture["waveform"].clone()).unwrap();
    let device = Device::flex();
    let model = HubertFeatureModel::load(&checkpoint, 3, 2, &device).unwrap();
    let input = Tensor::from_data(TensorData::new(waveform, shape), &device);
    let batched = model.forward(input.clone(), None).into_data().try_into_vec::<f32>().unwrap();
    let mut individual = Vec::new();
    for index in 0..shape[0] {
        let sample = input.clone().slice([index..index + 1, 0..1, 0..shape[2]]);
        individual.extend(model.forward(sample, None).into_data().try_into_vec::<f32>().unwrap());
    }
    assert!(max_abs_error(&batched, &individual) <= 1e-4);
}

#[test]
fn tensor_batch_rmvpe_network_matches_individual_mels() {
    use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_gru::RmvpeNetwork};
    let checkpoint = RvcCheckpoint::open_root(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_rmvpe_gru.pth")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/rmvpe_gru.json")).unwrap();
    let shape: [usize; 3] = serde_json::from_value(fixture["mel_shape"].clone()).unwrap();
    let waveform: Vec<f32> = serde_json::from_value(fixture["mel"].clone()).unwrap();
    let device = Device::flex();
    let model = RmvpeNetwork::load(&checkpoint, 2, 2, shape[1], &device).unwrap();
    let input = Tensor::from_data(TensorData::new(waveform, shape), &device);
    let second = input.clone() * 0.87;
    let batch = Tensor::cat(vec![input.clone(), second.clone()], 0);
    let batched = model.forward(batch).into_data().try_into_vec::<f32>().unwrap();
    let mut individual = Vec::new();
    individual.extend(model.forward(input).into_data().try_into_vec::<f32>().unwrap());
    individual.extend(model.forward(second).into_data().try_into_vec::<f32>().unwrap());
    assert!(max_abs_error(&batched, &individual) <= 2e-4);
}

#[test]
fn tensor_batch_nsf_from_f0_matches_individual_waveforms() {
    use tts_with_rvc_rs::{
        checkpoint::RvcCheckpoint, decoder::NsfWaveformDecoder,
        nsf_source::{NsfFromF0Fixture, SourceModuleHnNSF},
    };
    let checkpoint = RvcCheckpoint::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synthetic_nsf_decoder.pth")).unwrap();
    let fixture: NsfFromF0Fixture = serde_json::from_str(include_str!("fixtures/nsf_f0_decoder.json")).unwrap();
    let device = Device::flex();
    let decoder = NsfWaveformDecoder::load(
        &checkpoint, &fixture.upsample_rates, fixture.kernels_per_stage, fixture.dilations, &device,
    ).unwrap();
    let source = SourceModuleHnNSF::load(&checkpoint, fixture.sample_rate, 0, &device).unwrap();
    let [_, channels, frames] = fixture.input_shape;
    let upsample = fixture.upsample_rates.iter().product::<usize>();
    let input = Tensor::from_data(TensorData::new(fixture.input.clone(), [1, channels, frames]), &device);
    let f0 = Tensor::from_data(TensorData::new(fixture.f0.clone(), [1, frames]), &device);
    let noise = Tensor::from_data(TensorData::new(fixture.gaussian_noise.clone(), [1, frames * upsample, 1]), &device);
    let phases = Tensor::from_data(TensorData::new(fixture.initial_phases, [1, 1, 1]), &device);
    let condition = fixture.condition.as_ref().map(|values| Tensor::from_data(
        TensorData::new(values.clone(), fixture.condition_shape.unwrap()), &device));
    let single = decoder.forward_from_f0(input.clone(), f0.clone(), condition.clone(), &source,
        phases.clone(), noise.clone()).unwrap();
    let altered = decoder.forward_from_f0(input.clone() * 0.9, f0.clone() * 0.91,
        condition.clone(), &source, phases.clone(), noise.clone()).unwrap();
    let batched_condition = condition.map(|x| Tensor::cat(vec![x.clone(), x], 0));
    let batched = decoder.forward_from_f0(
        Tensor::cat(vec![input.clone(), input * 0.9], 0),
        Tensor::cat(vec![f0.clone(), f0 * 0.91], 0), batched_condition, &source,
        phases, Tensor::cat(vec![noise.clone(), noise], 0),
    ).unwrap();
    let expected = Tensor::cat(vec![single, altered], 0).into_data().try_into_vec::<f32>().unwrap();
    let actual = batched.into_data().try_into_vec::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(max_abs_error(&actual, &expected) < 2e-4);
}
