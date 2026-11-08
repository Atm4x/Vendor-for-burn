use burn::tensor::Device;
use std::{env, fs, process::ExitCode};
use tts_with_rvc_rs::{TextFrontFixture, max_abs_error};

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    if args.next().as_deref() != Some("verify-text-front") {
        return Err("usage: tts-with-rvc-rs verify-text-front <fixture.json> [flex|cpu|cuda|wgpu]".into());
    }
    let path = args.next().ok_or("missing fixture path")?;
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
    let fixture: TextFrontFixture = serde_json::from_str(&fs::read_to_string(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
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
