use std::{
    io::Write,
    process::{Command, Stdio},
};

use serde_json::{Value, json};

fn run_process(requests: &[Value]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_silero-tts"))
        .arg("--stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Starting Silero worker");

    {
        let stdin = child.stdin.as_mut().unwrap();
        for request in requests {
            writeln!(stdin, "{}", serde_json::to_string(request).unwrap()).unwrap();
        }
    }
    let result = child.wait_with_output().unwrap();
    assert!(result.status.success(), "Worker exit: {}", String::from_utf8_lossy(&result.stderr));
    let output = String::from_utf8(result.stdout).unwrap();
    let responses: Vec<Value> = output.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(responses.len(), requests.len());
    responses
}

#[test]
fn binary_preserves_line_framing_and_request_ids() {
    let result = run_process(&[
        json!({"command": "capabilities", "request_id": "cap"}),
        json!({"command": "status", "request_id": "stat"}),
        json!({"command": "shutdown", "request_id": "bye"}),
    ]);
    assert_eq!(result[0]["request_id"], "cap");
    assert_eq!(result[0]["data"]["persistent_weights"], true);
    assert_eq!(result[1]["request_id"], "stat");
    assert_eq!(result[1]["data"]["state"], "unloaded");
    assert_eq!(result[2]["request_id"], "bye");
}

#[cfg(feature = "reference-flex")]
#[test]
fn same_process_reuses_loaded_weights_and_releases_them() {
    let Ok(model_dir) = std::env::var("SILERO_PARITY_DIR") else { return };
    let result = run_process(&[
        json!({"command": "load", "request_id": "load", "model_dir": model_dir, "backend": "flex"}),
        json!({"command": "infer", "request_id": "first", "speaker": "aidar", "sample_rate": 8000, "input": {"kind": "accented", "text": "а"}}),
        json!({"command": "infer", "request_id": "second", "speaker": "aidar", "sample_rate": 8000, "input": {"kind": "accented", "text": "а"}}),
        json!({"command": "status", "request_id": "stat"}),
        json!({"command": "unload", "request_id": "unload"}),
        json!({"command": "shutdown", "request_id": "bye"}),
    ]);
    for response in &result {
        assert_eq!(response["ok"], true, "{response}");
    }
    assert_eq!(result[0]["data"]["state"], "loaded");
    let samples = result[1]["data"]["sample_count"].as_u64().unwrap();
    assert!(samples > 0);
    assert_eq!(result[2]["data"]["sample_count"], samples);
    assert_eq!(result[1]["data"]["audio"]["data_base64"], result[2]["data"]["audio"]["data_base64"]);
    assert_eq!(result[3]["data"]["state"], "ready");
    assert_eq!(result[3]["data"]["successful_inferences"], 2);
    assert_eq!(result[4]["data"]["state"], "unloaded");
}

#[cfg(feature = "reference-flex")]
#[test]
#[ignore = "requires official v5_ru.pt; set SILERO_V5_RU_CHECKPOINT"]
fn official_russian_checkpoint_worker_pcm() {
    let checkpoint = std::env::var("SILERO_V5_RU_CHECKPOINT").expect("checkpoint path");
    let result = run_process(&[
        json!({"command": "load", "request_id": "load", "checkpoint": checkpoint, "backend": "flex"}),
        json!({"command": "infer", "request_id": "first", "speaker": "aidar", "sample_rate": 8000, "input": {"kind": "text", "text": "Привет."}}),
        json!({"command": "infer", "request_id": "second", "speaker": "aidar", "sample_rate": 8000, "input": {"kind": "text", "text": "Привет."}}),
        json!({"command": "unload", "request_id": "unload"}),
        json!({"command": "shutdown", "request_id": "bye"}),
    ]);
    for response in &result { assert_eq!(response["ok"], true, "{response}"); }
    assert!(result[1]["data"]["sample_count"].as_u64().unwrap() > 0);
    let audio = &result[1]["data"]["audio"]["data_base64"];
    assert!(audio.as_str().is_some_and(|s| !s.is_empty()));
    assert_eq!(audio, &result[2]["data"]["audio"]["data_base64"]);
    assert_eq!(result[3]["data"]["state"], "unloaded");
}

#[cfg(feature = "reference-flex")]
#[test]
#[ignore = "requires original v3_en.pt; set SILERO_V3_EN_CHECKPOINT"]
fn official_english_checkpoint_worker_lifecycle() {
    use base64::{Engine, engine::general_purpose::STANDARD};
    let checkpoint = std::env::var("SILERO_V3_EN_CHECKPOINT").expect("checkpoint path");
    let result = run_process(&[
        json!({"command": "capabilities", "request_id": "cap"}),
        json!({"command": "load", "request_id": "load", "checkpoint": checkpoint, "backend": "flex"}),
        json!({"command": "warmup", "request_id": "warm"}),
        json!({"command": "status", "request_id": "ready"}),
        json!({"command": "infer", "request_id": "first", "speaker": "en_0", "sample_rate": 8000, "input": {"kind": "text", "text": "Hello, world!"}}),
        json!({"command": "infer", "request_id": "second", "speaker": "en_0", "sample_rate": 8000, "input": {"kind": "text", "text": "Hello, world!"}}),
        json!({"command": "infer", "request_id": "wrong", "speaker": "aidar", "sample_rate": 8000, "input": {"kind": "text", "text": "Hello, world!"}}),
        json!({"command": "unload", "request_id": "unload"}),
        json!({"command": "shutdown", "request_id": "bye"}),
    ]);
    for (i, response) in result.iter().enumerate() { assert_eq!(response["ok"], i != 6, "{response}"); }
    assert_eq!(result[1]["data"]["model_variant"], "v3_hifigan");
    assert_eq!(result[3]["data"]["state"], "ready");
    assert_eq!(result[4]["data"]["sample_count"], 7800);
    let audio = result[4]["data"]["audio"]["data_base64"].as_str().unwrap();
    assert_eq!(audio, result[5]["data"]["audio"]["data_base64"].as_str().unwrap());
    let bytes = STANDARD.decode(audio).unwrap();
    let values = bytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>();
    assert!(values.iter().all(|x| x.is_finite()) && values.iter().any(|x| x.abs() > 0.01));
    assert_eq!(result[7]["data"]["state"], "unloaded");
}
