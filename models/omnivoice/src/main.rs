#![forbid(unsafe_code)]
#![recursion_limit = "256"]

mod config;
mod duration;
mod omnivoice;
mod tokenizer;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use config::OmniVoiceConfig;
use native_worker_audio::{decode_pcm, encode_pcm16};
use native_worker_ipc::{Snapshot, absolute_path, serve, validate_model_payload};
use omnivoice::{OmniVoiceModel, OmniVoiceModelConfig};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use tokenizer::config::HiggsAudioV2TokenizerConfig;

const RATE: usize = 24_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Infer {
    tts_version: u32,
    text: String,
    reference: Snapshot,
    reference_text: String,
    language: Option<String>,
    steps: usize,
    seed: u64,
}

struct Artifacts {
    weights: PathBuf,
    config: PathBuf,
    text_tokenizer: PathBuf,
    codec_weights: PathBuf,
    codec_config: PathBuf,
}

impl Artifacts {
    fn load(&self, device: &Device) -> Result<OmniVoiceModel> {
        let config: OmniVoiceConfig = serde_json::from_slice(&std::fs::read(&self.config)?)?;
        config.llm_config.validate()?;
        let codec: HiggsAudioV2TokenizerConfig =
            serde_json::from_slice(&std::fs::read(&self.codec_config)?)?;
        ensure!(
            config.num_audio_codebook == 8
                && config.audio_mask_id == 1024
                && config.audio_vocab_size == 1025
                && config.audio_codebook_weights.len() == 8
                && codec.sample_rate == RATE
                && codec.codebook_size == 1024,
            "unsupported OmniVoice codec topology"
        );
        let tokens =
            tokenizers::Tokenizer::from_file(&self.text_tokenizer).map_err(anyhow::Error::msg)?;
        ensure!(
            tokens.get_vocab_size(true) <= config.llm_config.vocab_size,
            "text tokenizer exceeds vocabulary"
        );
        let mut model = OmniVoiceModelConfig::new().init(&config, &codec, tokens, device);
        let mut voice_store = SafetensorsStore::from_file(&self.weights)
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(true);
        let results = model
            .load_from(&mut voice_store)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            !results.applied.is_empty()
                && results.errors.is_empty()
                && results
                    .missing
                    .iter()
                    .all(|(path, _)| path.starts_with("audio_tokenizer.")),
            "OmniVoice weights: {results}"
        );
        let mut codec_store = SafetensorsStore::from_file(&self.codec_weights)
            .with_from_adapter(PyTorchToBurnAdapter);
        let results = model
            .audio_tokenizer
            .load_from(&mut codec_store)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            !results.applied.is_empty() && results.errors.is_empty(),
            "audio tokenizer weights: {results}"
        );
        device.sync()?;
        Ok(model)
    }
}

struct Worker {
    model_id: String,
    artifacts: Artifacts,
    input: PathBuf,
    output: PathBuf,
    device: Device,
    backend: String,
    model: Option<OmniVoiceModel>,
    ready: bool,
}

impl Worker {
    fn command(&mut self, command: &str, payload: Value) -> Result<Value> {
        validate_model_payload(command, &payload, &self.model_id).map_err(anyhow::Error::msg)?;
        match command {
            "capabilities" => Ok(
                json!({"service":"tts","tts_contract_version":1,"model_id":self.model_id,
                "backend":self.backend,"backends":[self.backend],"sample_rate_hz":RATE,"reference_formats":["wav"],"requires_reference_text":true}),
            ),
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
                self.model = Some(self.artifacts.load(&self.device)?);
                Ok(json!({"loaded":true}))
            }
            "warmup" => {
                let model = self.model.as_ref().context("model is not loaded")?;
                let pcm: Vec<f32> = (0..RATE).map(|i| (i as f32 * 0.12).sin() * 0.05).collect();
                self.synthesize(model, pcm, "Hello.", "Hello.", Some("en"), 2, 0)?;
                self.ready = true;
                Ok(json!({"warm":true}))
            }
            "infer" => {
                let model = self
                    .model
                    .as_ref()
                    .filter(|_| self.ready)
                    .context("model is not ready")?;
                let req: Infer = serde_json::from_value(payload)?;
                ensure!(
                    req.tts_version == 1
                        && (1..=64).contains(&req.steps)
                        && req.reference.file_name.ends_with(".wav"),
                    "invalid TTS version, steps or reference format"
                );
                ensure!(
                    !req.text.trim().is_empty()
                        && req.text.len() <= 1800
                        && !req.reference_text.trim().is_empty()
                        && req.reference_text.len() <= 1800,
                    "invalid text length"
                );
                let bytes = req
                    .reference
                    .read(&self.input)
                    .map_err(anyhow::Error::msg)?;
                let pcm = decode_pcm(&bytes, RATE, 30)?;
                let audio = self.synthesize(
                    model,
                    pcm,
                    &req.text,
                    &req.reference_text,
                    req.language.as_deref(),
                    req.steps,
                    req.seed,
                )?;
                let wav = encode_pcm16(&audio, RATE as u32)?;
                let output =
                    Snapshot::write(&self.output, "wav", &wav).map_err(anyhow::Error::msg)?;
                Ok(
                    json!({"tts_version":1,"output":output,"sample_rate_hz":RATE,"samples":audio.len()}),
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

    fn synthesize(
        &self,
        model: &OmniVoiceModel,
        mut pcm: Vec<f32>,
        text: &str,
        reference_text: &str,
        language: Option<&str>,
        steps: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let samples = pcm.len() / 960 * 960;
        ensure!(samples >= 4800, "reference is shorter than 200 ms");
        pcm.truncate(samples);
        self.device.seed(seed);
        let input = Tensor::<2>::from_data(TensorData::new(pcm, [1, samples]), &self.device);
        let codes = model.get_ref_audio_codes(input);
        let result = model.generate_speech(
            text,
            reference_text,
            codes,
            samples as f32 / RATE as f32,
            language,
            steps,
            3.0,
        )?;
        result
            .into_data()
            .try_to_vec::<f32>()
            .map_err(anyhow::Error::msg)
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
    let model_id = options
        .remove("--model-id")
        .filter(|id| !id.trim().is_empty())
        .context("missing --model-id")?;
    let backend = options.remove("--backend").context("missing --backend")?;
    let device = match backend.as_str() {
        "cpu" => Device::flex(),
        "cuda" => Device::cuda(0),
        "wgpu" => Device::wgpu(Default::default()),
        _ => bail!("unsupported backend {backend}"),
    };
    let mut path = |flag: &str, directory| -> Result<PathBuf> {
        let value = options
            .remove(flag)
            .with_context(|| format!("missing {flag}"))?;
        absolute_path(Path::new(&value), directory).map_err(anyhow::Error::msg)
    };
    let artifacts = Artifacts {
        weights: path("--checkpoint", false)?,
        config: path("--config", false)?,
        text_tokenizer: path("--tokenizer", false)?,
        codec_weights: path("--codec-checkpoint", false)?,
        codec_config: path("--codec-config", false)?,
    };
    let mut worker = Worker {
        model_id,
        artifacts,
        input: path("--input-dir", true)?,
        output: path("--output-dir", true)?,
        device,
        backend,
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
