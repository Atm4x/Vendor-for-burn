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
