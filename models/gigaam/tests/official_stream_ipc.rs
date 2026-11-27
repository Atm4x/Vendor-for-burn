#![cfg(feature = "flex")]

use anyhow::{Context, Result, ensure};
use burn::tensor::Device;
use gigaam_rs::{
    GigaAm,
    service::WordResult,
    zapformer::{StreamPacket, StreamTranscription, language::LanguageTag},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "gigaam-official-ipc-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

fn uid(i: usize) -> String {
    format!("00000000-0000-4000-8000-{i:012}")
}
fn head(value: StreamTranscription) -> Value {
    json!({
        "text":value.text,"token_ids":value.token_ids,"frames":value.frames,
        "words":value.words.map(|w| w.into_iter().map(WordResult::from).collect::<Vec<_>>())
    })
}
fn heads(value: StreamPacket) -> Value {
    match value {
        StreamPacket::Single(v) => json!({"single":head(v)}),
        StreamPacket::Dual { raw, e2e } => json!({"raw":head(raw),"e2e":head(e2e)}),
    }
}
fn snapshot(root: &Path, name: &str, pcm: &[f32]) -> Result<Value> {
    let bytes: Vec<u8> = pcm.iter().flat_map(|v|v.to_le_bytes()).collect();
    fs::write(root.join(name), &bytes)?;
    Ok(json!({"file_name":name,"size_bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}))
}

fn run_case(name: &str, language: Option<&str>) -> Result<()> {
    let root = PathBuf::from(std::env::var("GIGAAM_OFFICIAL_DIR")?).join(name);
    let pcm: Vec<f32> = (0..16000).map(|i|((i as f32*0.067).sin()+(i as f32*0.101).cos())*0.1).collect();
    let chunks: Vec<_> = pcm.chunks(1280).collect();
    let tag = language.map(|l| LanguageTag::Tag(l.into()));
    let mut expected = Vec::new();
    {
        let model=GigaAm::load(&root.join(format!("{name}.ckpt")),Device::flex())?;
        let mut stream=model.start_stream(Some(160),tag.as_ref(),true)?;
        for chunk in &chunks { expected.push(heads(stream.push(chunk,false)?)); }
        expected.push(heads(stream.flush()?));
    }
    let temp=Scratch::new()?;
    let model_id=format!("gigaam_{name}");
    let start=json!({"gigaam_version":1,"future_context_ms":160,"language":language,"word_timestamps":true});
    let mut tasks=vec![
        ("capabilities",json!({}),true),
        ("stream_start",start.clone(),false),
        ("load",json!({"model_id":model_id}),true),
        ("stream_start",start.clone(),false),
        ("warmup",json!({}),true),
        ("stream_start",start.clone(),true),
        ("stream_start",start.clone(),false),
    ];
    let stream_id=uid(5);
    let bad=snapshot(&temp.0,"nan.f32",&[f32::NAN])?;
    tasks.push(("stream_push",json!({"stream_id":stream_id,"packet_index":0,"input":bad}),false));
    let mut positions=Vec::new();
    for (i,chunk) in chunks.iter().enumerate() {
        let input=snapshot(&temp.0,&format!("p{i}.f32"),chunk)?;
        if i==0 {
            tasks.push(("stream_push",json!({"stream_id":stream_id,"packet_index":1,"input":input}),false));
            let mut corrupt=input.clone();
            corrupt["sha256"]=json!("0".repeat(64));
            tasks.push(("stream_push",json!({"stream_id":stream_id,"packet_index":0,"input":corrupt}),false));
        }
        positions.push(tasks.len());
        tasks.push(("stream_push",json!({"stream_id":stream_id,"packet_index":i,"input":input}),true));
    }
    let finish=json!({"stream_id":stream_id,"packet_index":chunks.len()});
    positions.push(tasks.len());
    tasks.push(("stream_finish",finish.clone(),true));
    tasks.push(("stream_finish",finish,false));
    let next_id=uid(tasks.len());
    tasks.push(("stream_start",start.clone(),true));
    tasks.push(("stream_cancel",json!({"stream_id":stream_id}),false));
    tasks.push(("stream_cancel",json!({"stream_id":next_id}),true));
    let unload_id=uid(tasks.len());
    tasks.push(("stream_start",start,true));
    tasks.push(("unload",json!({}),true));
    tasks.push(("stream_cancel",json!({"stream_id":unload_id}),false));
    tasks.push(("status",json!({}),true));
    tasks.push(("shutdown",json!({}),true));
    let mut child=Command::new(env!("CARGO_BIN_EXE_gigaam-rs"))
        .args(["--ipc-stdio","--backend","cpu","--model-id",&model_id,"--model-dir"])
        .arg(&root).arg("--input-dir").arg(&temp.0)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    {
        let mut input=child.stdin.take().context("worker stdin missing")?;
        for (i,(command,payload,_)) in tasks.iter().enumerate() {
            writeln!(input,"{}",json!({"version":1,"id":uid(i),"command":command,"payload":payload}))?;
        }
    }
    let output=child.wait_with_output()?;
    ensure!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    let replies: Vec<Value>=String::from_utf8(output.stdout)?.lines().map(serde_json::from_str).collect::<serde_json::Result<_>>()?;
    assert_eq!(replies.len(),tasks.len());
    for (i,reply) in replies.iter().enumerate() {
        assert_eq!(reply["id"],uid(i));
        assert_eq!(reply["ok"],tasks[i].2,"{name}: request {i}: {reply}");
    }
    assert_eq!(replies[0]["result"]["gigaam_capability"]["streaming"]["pcm_format"],"f32le");
    let mut consumed=0;
    for (i,position) in positions.into_iter().enumerate() {
        let result=&replies[position]["result"];
        if i<chunks.len() { consumed+=chunks[i].len(); }
        assert_eq!(result["stream_id"],stream_id);
        assert_eq!(result["packet_index"],i);
        assert_eq!(result["accepted_samples"],consumed);
        assert_eq!(result["finished"],i==chunks.len());
        assert_eq!(result["heads"],expected[i],"{name}: packet {i}");
    }
    assert_eq!(replies[replies.len()-2]["result"]["phase"],"started");
    assert!(replies[replies.len()-2]["result"]["active_stream"].is_null());
    eprintln!("OFFICIAL_IPC_PASS {name}");
    Ok(())
}

#[test]
#[ignore = "requires verified official checkpoint directory in GIGAAM_OFFICIAL_DIR"]
fn official_v4_ctc_stream_ipc() -> Result<()> { run_case("v4_ctc",None) }

#[test]
#[ignore = "requires verified official checkpoint directory in GIGAAM_OFFICIAL_DIR"]
fn official_multilingual_stream_ipc() -> Result<()> { run_case("multilingual_v2_rnnt",Some("ru")) }
