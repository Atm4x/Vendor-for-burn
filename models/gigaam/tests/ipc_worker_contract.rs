#![cfg(feature = "flex")]

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend")
}

fn snapshot(root: &Path, file: &str) -> Result<Value> {
    let bytes = fs::read(root.join(file))?;
    Ok(json!({
        "file_name":file,
        "size_bytes":bytes.len(),
        "sha256":format!("{:x}", Sha256::digest(bytes))
    }))
}

fn run_case(model: &str, model_id: &str, kind: &str) -> Result<()> {
    let fixture = fixture_root();
    let temp = std::env::temp_dir().join(format!(
        "gigaam-stdio-contract-{}-{}-{}", model, std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&temp)?;
    let result = (|| -> Result<()> {
        fs::copy(fixture.join("tone.wav"), temp.join("tone.wav"))?;
        let good = snapshot(&temp, "tone.wav")?;
        let mut bad = good.clone();
        bad["sha256"] = json!("0".repeat(64));
        let mut escaped = good.clone();
        escaped["file_name"] = json!("../tone.wav");
        let ids: Vec<String> = (0..15).map(|i| format!("00000000-0000-4000-8000-{i:012}")).collect();
        let tasks = vec![
            ("capabilities",json!({})),
            ("infer",json!({"gigaam_version":1,"kind":kind,"input":good})),
            ("load",json!({"model_id":"gigaam_wrong"})),
            ("load",json!({"model_id":format!("gigaam_{model_id}")})),
            ("status",json!({})),
            ("warmup",json!({})),
            ("status",json!({})),
            ("infer",json!({"gigaam_version":1,"kind":kind,"input":bad})),
            ("infer",json!({"gigaam_version":1,"kind":kind,"input":escaped})),
            ("infer",json!({"gigaam_version":1,"kind":kind,"input":good})),
            ("load",json!({"model_id":format!("gigaam_{model_id}")})),
            ("unload",json!({})),
            ("status",json!({})),
            ("shutdown",json!({})),
        ];
        let mut child = Command::new(env!("CARGO_BIN_EXE_gigaam-rs"))
            .args([
                "--ipc-stdio", "--model-id", &format!("gigaam_{model_id}"),
                "--backend", "cpu", "--model-dir", fixture.join(model).to_str().context("fixture path is UTF-8")?,
                "--input-dir", temp.to_str().context("temporary path is UTF-8")?,
                "--output-dir", temp.to_str().context("temporary path is UTF-8")?,
            ])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
        {
            let mut stdin = child.stdin.take().context("worker has no stdin")?;
            for (i,(command,payload)) in tasks.iter().enumerate() {
                let line = json!({"version":1,"id":ids[i],"command":command,"payload":payload});
                writeln!(stdin,"{line}")?;
            }
        }
        let output = child.wait_with_output()?;
        ensure!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let replies: Vec<Value> = String::from_utf8(output.stdout)?.lines()
            .map(serde_json::from_str).collect::<serde_json::Result<_>>()?;
        ensure!(replies.len() == tasks.len(), "unexpected response count: {}", replies.len());
        for (i,reply) in replies.iter().enumerate() {
            ensure!(reply["version"] == 1 && reply["id"] == ids[i], "wrong envelope: {reply}");
        }
        let expected = [true,false,false,true,true,true,true,false,false,true,false,true,true,true];
        for (i,ok) in expected.into_iter().enumerate() {
            ensure!(replies[i]["ok"] == ok, "unexpected result #{i}: {}", replies[i]);
        }
        ensure!(replies[4]["result"]["phase"] == "loaded", "status after load is wrong");
        ensure!(replies[6]["result"]["phase"] == "ready", "status after warmup is wrong");
        ensure!(replies[12]["result"]["phase"] == "started", "status after unload is wrong");
        ensure!(replies[9]["result"]["kind"] == kind, "wrong inference result");
        if kind == "embeddings" {
            let result = &replies[9]["result"];
            let descriptor = &result["output"];
            let filename = descriptor["file_name"].as_str().context("missing output")?;
            let stored = fs::read(temp.join(filename))?;
            ensure!(descriptor["size_bytes"] == stored.len(), "incorrect output size");
            ensure!(descriptor["sha256"] == format!("{:x}",Sha256::digest(&stored)), "incorrect output digest");
        }
        Ok(())
    })();
    fs::remove_dir_all(&temp)?;
    result
}

#[test]
fn ctc_matches_beat_worker_lifecycle() -> Result<()> { run_case("ctc", "v3_ctc", "asr") }
#[test]
fn ssl_matches_beat_worker_lifecycle() -> Result<()> { run_case("ssl", "v3_ssl", "embeddings") }
#[test]
fn emotion_matches_beat_worker_lifecycle() -> Result<()> { run_case("emo", "emo", "emotion") }

#[test]
fn worker_rejects_unsupported_backend_before_writing_stdout() -> Result<()> {
    let fixture = fixture_root();
    let output = Command::new(env!("CARGO_BIN_EXE_gigaam-rs"))
        .args(["--ipc-stdio", "--model-id", "gigaam_v3_ctc", "--backend", "cuda", "--model-dir"])
        .arg(fixture.join("ctc"))
        .args(["--input-dir"]).arg(&fixture)
        .output()?;
    ensure!(!output.status.success() && output.stdout.is_empty(), "unsupported backend must fail cleanly");
    Ok(())
}
