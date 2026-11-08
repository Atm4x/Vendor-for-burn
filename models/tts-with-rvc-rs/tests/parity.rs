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
