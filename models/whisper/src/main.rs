#![forbid(unsafe_code)]

mod mel;
mod model;
mod transcribe;
mod weights;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::Device;
use native_worker_audio::decode_pcm;
use native_worker_ipc::{Snapshot, absolute_path, serve, validate_model_payload};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use transcribe::{Options, Transcriber};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Infer {
    asr_version: u32,
    input: Snapshot,
    language: Option<String>,
}

struct Worker {
    model_id: String,
    weights: PathBuf,
    tokenizer: PathBuf,
    generation_config: PathBuf,
    input_dir: PathBuf,
    device: Device,
    backend: String,
    model: Option<Transcriber>,
    ready: bool,
}

impl Worker {
    fn command(&mut self, command: &str, payload: Value) -> Result<Value> {
        validate_model_payload(command, &payload, &self.model_id).map_err(anyhow::Error::msg)?;
        match command {
            "capabilities" => Ok(json!({"service":"asr","asr_contract_version":1,
                "formats":["wav"],"backend":self.backend,"backends":[self.backend],"model_id":self.model_id,"max_audio_seconds":120})),
            "status" => {
                let phase = match (self.model.is_some(), self.ready) {
                    (_, true) => "ready",
                    (true, false) => "loaded",
                    _ => "started",
                };
                Ok(json!({"phase":phase,"backend":self.backend,"model_id":self.model_id}))
            }
            "load" => {
                ensure!(self.model.is_none(), "model is already loaded");
                let model = Transcriber::load(
                    &self.weights,
                    &self.tokenizer,
                    &self.generation_config,
                    &self.device,
                )?;
                self.device.sync()?;
                self.model = Some(model);
                Ok(json!({"loaded":true}))
            }
            "warmup" => {
                let model = self.model.as_ref().context("model is not loaded")?;
                let pcm: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.1).sin() * 0.01).collect();
                model.transcribe(
                    &pcm,
                    &Options {
                        max_new_tokens: 1,
                        ..Default::default()
                    },
                )?;
                self.device.sync()?;
                self.ready = true;
                Ok(json!({"warm":true}))
            }
            "infer" => {
                let model = self
                    .model
                    .as_ref()
                    .filter(|_| self.ready)
                    .context("model is not ready")?;
                let request: Infer = serde_json::from_value(payload)?;
                ensure!(
                    request.asr_version == 1 && request.input.file_name.ends_with(".wav"),
                    "unsupported ASR version or audio format"
                );
                let data = request
                    .input
                    .read(&self.input_dir)
                    .map_err(anyhow::Error::msg)?;
                let samples = decode_pcm(&data, 16_000, 120)?;
                let options = Options {
                    language: request.language.filter(|code| code != "auto"),
                    ..Default::default()
                };
                let text = model.transcribe(&samples, &options)?;
                Ok(
                    json!({"asr_version":1,"kind":"asr","text":text,"sample_rate_hz":16000,
                    "duration_seconds":samples.len() as f64 / 16000.0}),
                )
            }
            "unload" | "shutdown" => {
                self.ready = false;
                self.model = None;
                Ok(json!({}))
            }
            _ => bail!("unknown worker command"),
        }
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    ensure!(
        args.next().as_deref() == Some("--ipc-stdio"),
        "expected --ipc-stdio"
    );
    let mut options = BTreeMap::new();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .filter(|v| !v.starts_with("--"))
            .with_context(|| format!("missing value for {flag}"))?;
        ensure!(
            options.insert(flag, value).is_none(),
            "duplicate launch option"
        );
    }
    let mut take = |flag: &str| {
        options
            .remove(flag)
            .with_context(|| format!("missing {flag}"))
    };
    let model_id = take("--model-id")?;
    ensure!(!model_id.trim().is_empty(), "invalid --model-id");
    let backend = take("--backend")?;
    let device = match backend.as_str() {
        "cpu" => Device::flex(),
        "cuda" => Device::cuda(0),
        "wgpu" => Device::wgpu(Default::default()),
        _ => bail!("unsupported backend {backend}"),
    };
    let mut worker = Worker {
        model_id,
        weights: absolute_path(Path::new(&take("--checkpoint")?), false)
            .map_err(anyhow::Error::msg)?,
        tokenizer: absolute_path(Path::new(&take("--tokenizer")?), false)
            .map_err(anyhow::Error::msg)?,
        generation_config: absolute_path(Path::new(&take("--generation-config")?), false)
            .map_err(anyhow::Error::msg)?,
        input_dir: absolute_path(Path::new(&take("--input-dir")?), true)
            .map_err(anyhow::Error::msg)?,
        backend,
        device,
        model: None,
        ready: false,
    };
    ensure!(
        options.is_empty(),
        "unknown launch options: {:?}",
        options.keys()
    );
    serve(|cmd, body| worker.command(cmd, body).map_err(|e| format!("{e:#}")))
        .map_err(anyhow::Error::msg)
}
