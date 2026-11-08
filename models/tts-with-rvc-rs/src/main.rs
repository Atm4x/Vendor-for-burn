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
    if !matches!(command.as_str(), "verify-text-front" | "verify-attention" | "verify-encoder-layer" | "verify-text-encoder" | "verify-flow") {
        return Err("usage: tts-with-rvc-rs verify-text-front|verify-attention|verify-encoder-layer|verify-text-encoder <fixture.json> [flex|cpu|cuda|wgpu]".into());
    }
    let path = args.next().ok_or("missing fixture or checkpoint path")?;
    let flow_fixture_path = if command == "verify-flow" {
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
    let data = fs::read_to_string(path).map_err(|e| e.to_string())?;
    if let Some(flow_fixture_path) = flow_fixture_path {
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
