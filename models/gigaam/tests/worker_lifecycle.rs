#![cfg(feature = "flex")]

use anyhow::Result;
use serde_json::{Value, json};
use std::{io::Write, path::{Path, PathBuf}, process::{Command, Stdio}};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend")
}

#[test]
fn worker_load_unload_reload_and_protocol() -> Result<()> {
    let root = fixtures();
    let mut child = Command::new(env!("CARGO_BIN_EXE_gigaam-rs"))
        .args(["worker", "--device", "flex", "--batch-size", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let audio_path = root.join("tone.wav");
    let out = std::env::temp_dir().join(format!("gigaam-worker-{}.f32", std::process::id()));
    let requests = vec![
        json!({"id":1, "method":"status"}),
        json!({"id":2, "method":"transcribe", "audio_path":audio_path}),
        json!({"id":3, "method":"load", "model":root.join("ctc")}),
        json!({"id":4, "method":"transcribe", "audio_path":audio_path}),
        json!({"id":5, "method":"load", "model":root.join("ssl")}),
        json!({"id":6, "method":"unload"}),
        json!({"id":7, "method":"load", "model":root.join("ssl")}),
        json!({"id":8, "method":"status"}),
        json!({"id":9, "method":"transcribe", "audio_path":audio_path}),
        json!({"id":10, "method":"embed", "audio_path":audio_path, "output_path":out}),
        json!({"id":11, "method":"unload"}),
        json!({"id":12, "method":"status"}),
        json!({"id":13, "method":"shutdown"}),
    ];
    {
        let mut input = child.stdin.take().unwrap();
        for request in &requests {
            writeln!(input, "{request}")?;
        }
    }
    let output = child.wait_with_output()?;
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let responses: Vec<Value> = String::from_utf8(output.stdout)?
        .lines().map(serde_json::from_str).collect::<serde_json::Result<_>>()?;
    assert_eq!(responses.len(), requests.len());
    for (i, response) in responses.iter().enumerate() {
        assert_eq!(response["id"], requests[i]["id"]);
    }
    assert_eq!(responses[0]["result"]["state"], "unloaded");
    assert_eq!(responses[1]["ok"], false);
    assert_eq!(responses[2]["result"]["state"], "ready");
    assert_eq!(responses[3]["ok"], true);
    assert_eq!(responses[4]["ok"], false);
    assert_eq!(responses[5]["result"]["unloaded"], true);
    assert_eq!(responses[6]["result"]["model_kind"], "ssl");
    assert_eq!(responses[7]["result"]["batch_size"], 2);
    assert_eq!(responses[8]["ok"], false);
    assert_eq!(responses[9]["ok"], true);
    assert_eq!(responses[10]["result"]["unloaded"], true);
    assert_eq!(responses[11]["result"]["state"], "unloaded");
    assert_eq!(responses[12]["result"]["stopped"], true);
    std::fs::remove_file(&out)?;
    Ok(())
}
