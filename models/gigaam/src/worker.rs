//! Persistent inference-worker protocol; model ownership is independent of a request.
use crate::{jsonl, serve};
use anyhow::{Context, Result, ensure};
use burn::tensor::Device;
use gigaam_rs::{
    model_registry::{CacheOptions, known_models},
    service::GigaAmService,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct LoadRequest {
    model: String,
    #[serde(default)]
    batch_size: Option<usize>,
}

struct Worker {
    model: Option<GigaAmService>,
    model_spec: Option<String>,
    device: Device,
    cache: CacheOptions,
    python_audio: bool,
    batch_size: usize,
}

impl Worker {
    fn status(&self) -> Value {
        json!({
            "protocol_version": 1,
            "state": if self.model.is_some() { "ready" } else { "unloaded" },
            "model": self.model_spec,
            "model_kind": self.model.as_ref().map(|model| model.status().model_kind),
            "sample_rate": 16_000,
            "batch_size": self.batch_size,
            "external_vad": true,
        })
    }

    fn execute(&mut self, request: Value) -> Result<(Value, bool)> {
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .context("request requires a string method")?;
        match method {
            "status" => Ok((self.status(), false)),
            "list_models" => Ok((json!({
                "names": known_models().iter().map(|(name, _)| *name).collect::<Vec<_>>()
            }), false)),
            "load" => {
                ensure!(
                    self.model.is_none(),
                    "model already loaded; call unload first"
                );
                let load: LoadRequest = serde_json::from_value(request)?;
                ensure!(!load.model.trim().is_empty(), "model must not be empty");
                let batch_size = load.batch_size.unwrap_or(self.batch_size);
                ensure!(
                    (1..=256).contains(&batch_size),
                    "batch_size must be 1..=256"
                );
                let service =
                    GigaAmService::load_named(&load.model, self.device.clone(), &self.cache)?
                        .with_batch_size(batch_size)?;
                self.model_spec = Some(load.model);
                self.batch_size = batch_size;
                self.model = Some(service);
                Ok((self.status(), false))
            }
            "unload" => {
                let was_loaded = self.model.take().is_some();
                self.model_spec = None;
                Ok((json!({"unloaded": was_loaded}), false))
            }
            "shutdown" => Ok((json!({"stopped": true}), true)),
            _ => {
                let parsed: serve::Request = serde_json::from_value(request)?;
                let service = self
                    .model
                    .as_ref()
                    .context("model not loaded; call load first")?;
                Ok((
                    serve::execute(service, parsed.command, self.python_audio)?,
                    false,
                ))
            }
        }
    }
}

pub fn run(device: Device, cache: CacheOptions, python_audio: bool, batch_size: usize) -> Result<()> {
    ensure!(
        (1..=256).contains(&batch_size),
        "batch_size must be 1..=256"
    );
    let mut worker = Worker {
        model: None,
        model_spec: None,
        device,
        cache,
        python_audio,
        batch_size,
    };
    jsonl::run(|request| worker.execute(request))
}
