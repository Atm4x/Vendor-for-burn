#![forbid(unsafe_code)]

mod encoder;
mod qwen;
mod semantic;

use anyhow::{Context, Result, bail, ensure};
use native_worker_ipc::{absolute_path, serve, validate_model_payload};
use semantic::{Artifacts, Model, Pooling, normalize};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const MAX_ITEMS: usize = 32;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Embed {
    texts: Vec<String>,
    purpose: Purpose,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Purpose {
    Query,
    Document,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rerank {
    query: String,
    candidates: Vec<String>,
}

struct Worker {
    model_id: String,
    backend: String,
    artifacts: Artifacts,
    query_prefix: String,
    document_prefix: String,
    reranker: bool,
    model: Option<Model>,
    ready: bool,
}

impl Worker {
    fn command(&mut self, command: &str, payload: Value) -> Result<Value> {
        validate_model_payload(command, &payload, &self.model_id).map_err(anyhow::Error::msg)?;
        match command {
            "capabilities" => Ok(
                json!({"service": if self.reranker {"reranker"} else {"embedding"},
                "semantic_contract_version":1,"backend":self.backend,"backends":[self.backend],"model_id":self.model_id,
                "max_tokens":self.artifacts.max_tokens,"max_items":MAX_ITEMS}),
            ),
            "status" => {
                let phase = match (self.model.is_some(), self.ready) {
                    (_, true) => "ready",
                    (true, false) => "loaded",
                    _ => "started",
                };
                Ok(json!({"phase":phase,"model_id":self.model_id,"backend":self.backend}))
            }
            "load" => {
                ensure!(self.model.is_none(), "model is already loaded");
                self.model = Some(self.artifacts.load(self.reranker, &self.backend)?);
                Ok(json!({"loaded":true}))
            }
            "warmup" => {
                let model = self.model.as_ref().context("model is not loaded")?;
                if self.reranker {
                    model.rank("sky", "the sky is blue")?;
                } else {
                    model.embed("warmup")?;
                }
                self.ready = true;
                Ok(json!({"warm":true}))
            }
            "infer" => {
                let model = self
                    .model
                    .as_ref()
                    .filter(|_| self.ready)
                    .context("model is not ready")?;
                if self.reranker {
                    let request: Rerank = serde_json::from_value(payload)?;
                    ensure!(
                        valid(&request.query)
                            && (1..=MAX_ITEMS).contains(&request.candidates.len())
                            && request.candidates.iter().all(|s| valid(s)),
                        "invalid query or candidates"
                    );
                    let scores = request
                        .candidates
                        .iter()
                        .map(|doc| model.rank(&request.query, doc))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(
                        json!({"model":self.model_id,"protocol":model.rerank_protocol(),"scores":scores}),
                    )
                } else {
                    let request: Embed = serde_json::from_value(payload)?;
                    ensure!(
                        (1..=MAX_ITEMS).contains(&request.texts.len())
                            && request.texts.iter().all(|s| valid(s)),
                        "invalid texts"
                    );
                    let prefix = match request.purpose {
                        Purpose::Query => &self.query_prefix,
                        Purpose::Document => &self.document_prefix,
                    };
                    let vectors = request
                        .texts
                        .iter()
                        .map(|text| model.embed(&format!("{prefix}{text}")))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(
                        json!({"model":self.model_id,"protocol":"local-transformers-v1","dimensions":vectors[0].len(),"vectors":vectors}),
                    )
                }
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

fn valid(text: &str) -> bool {
    !text.trim().is_empty() && text.chars().count() <= 8192
}

fn path(value: &str) -> Result<PathBuf> {
    absolute_path(Path::new(value), false).map_err(anyhow::Error::msg)
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
            .with_context(|| format!("missing value for {flag}"))?;
        ensure!(
            options.insert(flag, value).is_none(),
            "duplicate launch option"
        );
    }
    let query_prefix = options.remove("--query-prefix").unwrap_or_default();
    let document_prefix = options.remove("--document-prefix").unwrap_or_default();
    let instruction = options.remove("--instruction");
    ensure!(
        query_prefix.len() <= 8192
            && document_prefix.len() <= 8192
            && instruction.as_ref().is_none_or(|x| valid(x)),
        "invalid prefix or instruction length"
    );
    let positive = options
        .remove("--positive-label")
        .map(|label| label.parse())
        .transpose()?;
    let max_tokens: usize = options
        .remove("--max-tokens")
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(512);
    ensure!(
        (32..=8192).contains(&max_tokens),
        "max tokens must be in 32..=8192"
    );
    let pooling = match (
        options.remove("--pooling"),
        options.remove("--pooling-config"),
    ) {
        (Some(name), None) => Some(Pooling::parse(&name)?),
        (None, Some(value)) => Some(Pooling::from_file(&path(&value)?)?),
        (None, None) => None,
        (Some(_), Some(_)) => bail!("use one pooling option"),
    };
    let mut take = |flag: &str| {
        options
            .remove(flag)
            .with_context(|| format!("missing {flag}"))
    };
    let model_id = take("--model-id")?;
    ensure!(valid(&model_id), "invalid model ID");
    let reranker = match take("--task")?.as_str() {
        "embedding" => false,
        "reranker" => true,
        _ => bail!("expected --task embedding|reranker"),
    };
    ensure!(
        !reranker || (pooling.is_none() && query_prefix.is_empty() && document_prefix.is_empty()),
        "pooling and text prefixes apply to embeddings"
    );
    ensure!(
        reranker || (positive.is_none() && instruction.is_none()),
        "positive label and instruction apply to reranking"
    );
    let backend = take("--backend")?;
    ensure!(
        matches!(backend.as_str(), "cpu" | "cuda" | "wgpu"),
        "unsupported backend {backend}"
    );
    let artifacts = Artifacts {
        checkpoint: path(&take("--checkpoint")?)?,
        config: path(&take("--config")?)?,
        tokenizer: path(&take("--tokenizer")?)?,
        pooling,
        positive,
        instruction,
        max_tokens,
    };
    ensure!(
        options.is_empty(),
        "unknown launch options: {:?}",
        options.keys()
    );
    let mut worker = Worker {
        model_id,
        backend,
        artifacts,
        query_prefix,
        document_prefix,
        reranker,
        model: None,
        ready: false,
    };
    serve(|cmd, body| worker.command(cmd, body).map_err(|e| format!("{e:#}")))
        .map_err(anyhow::Error::msg)
}
