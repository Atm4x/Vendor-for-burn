use crate::{audio::wav_pcm16, runtime::SileroTts, worker::{BackendName, Input}};
use burn::tensor::Device;
use native_worker_ipc::{absolute_path, serve, Snapshot};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::{Path, PathBuf}, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Infer {
    silero_version: u32,
    input: Input,
    speaker: String,
    sample_rate: u32,
}

struct Session {
    model: SileroTts,
    ready: bool,
    successful_inferences: u64,
}

struct Worker {
    checkpoint: PathBuf,
    output_dir: PathBuf,
    device: Device,
    backend: String,
    gpu_id: Option<String>,
    session: Option<Session>,
}

impl Worker {
    fn status(&self) -> Value {
        let phase = match &self.session {
            None => "started",
            Some(session) if session.ready => "ready",
            Some(_) => "loaded",
        };
        json!({"phase": phase, "backend": self.backend, "gpu_id": self.gpu_id,
            "model_variant": self.session.as_ref().map(|s| s.model.variant()),
            "speaker_names": self.session.as_ref().map(|s| s.model.speakers().collect::<Vec<_>>()),
            "successful_inferences": self.session.as_ref().map_or(0, |s| s.successful_inferences)})
    }

    fn command(&mut self, name: &str, payload: Value) -> Result<Value, String> {
        if name != "infer" && payload != json!({}) {
            return Err("unexpected command payload".into());
        }
        match name {
            "capabilities" => Ok(json!({
                "service": "tts", "silero_contract_version": 1,
                "backend": self.backend, "gpu_id": self.gpu_id,
                "model_variants": [
                    {"id":"v5_ru", "inputs":["text","ssml","accented","symbol_durations"]},
                    {"id":"v3_hifigan", "inputs":["text"]},
                    {"id":"v4_hifigan", "inputs":["text"]},
                    {"id":"v5_convnext_plain", "inputs":["text"]}
                ],
                "sample_rates": [8000,24000,48000], "outputs": ["wav_pcm16"],
                "maximum_input_characters": crate::longform::MAX_INPUT_CHARS,
                "maximum_output_seconds": crate::longform::MAX_AUDIO_SECONDS,
                "cancel_supported": false,
            })),
            "load" => {
                if self.session.is_some() { return Err("model is already loaded".into()); }
                let model = SileroTts::load(&self.checkpoint, &self.device).map_err(|e| format!("{e:#}"))?;
                self.device.sync().map_err(|e| e.to_string())?;
                self.session = Some(Session { model, ready: false, successful_inferences: 0 });
                Ok(self.status())
            }
            "warmup" => {
                let session = self.session.as_mut().ok_or("model is not loaded")?;
                if session.ready { return Err("model is already ready".into()); }
                let (text, speaker) = session.model.warmup_input().map_err(|e| format!("{e:#}"))?;
                session.model.infer_text(&text, &speaker, 8000).map_err(|e| format!("{e:#}"))?;
                self.device.sync().map_err(|e| e.to_string())?;
                session.ready = true;
                Ok(self.status())
            }
            "status" => Ok(self.status()),
            "infer" => self.infer(serde_json::from_value(payload).map_err(|e| e.to_string())?),
            "unload" => {
                if self.session.is_none() { return Err("model is not loaded".into()); }
                self.session = None;
                Ok(self.status())
            }
            "shutdown" => { self.session = None; Ok(json!({})) }
            _ => Err("unknown worker command".into()),
        }
    }

    fn infer(&mut self, request: Infer) -> Result<Value, String> {
        if request.silero_version != 1 { return Err("unsupported Silero request version".into()); }
        let session = self.session.as_mut().filter(|s| s.ready).ok_or("model is not ready")?;
        let start = Instant::now();
        let model = &session.model;
        let speaker = &request.speaker;
        let rate = request.sample_rate;
        let samples = match request.input {
            Input::Text { text, options } => model.infer_text_with_options(&text, speaker, rate, &options),
            Input::Ssml { ssml, options } => model.infer_ssml_with_options(&ssml, speaker, rate, &options),
            Input::Accented { text } => model.infer_accented(&text, speaker, rate),
            Input::SymbolDurations { text, durations, options } => model.infer_text_with_symbol_durations(&text, speaker, rate, &durations, &options),
            #[cfg(feature = "ruaccent")]
            Input::RuaccentText { .. } => return Err("external RUAccent is not supported by IPC v1".into()),
        }.map_err(|e| format!("{e:#}"))?;
        self.device.sync().map_err(|e| e.to_string())?;
        let compute_ms = start.elapsed().as_secs_f64() * 1000.0;
        if samples.is_empty() { return Err("empty PCM output".into()); }
        let wav = wav_pcm16(&samples, rate).map_err(|e| e.to_string())?;
        let audio = Snapshot::write(&self.output_dir, "wav", &wav)?;
        session.successful_inferences += 1;
        Ok(json!({"silero_version": 1, "sample_rate_hz": rate, "sample_count": samples.len(),
            "compute_ms": compute_ms, "audio": audio}))
    }
}

fn select_device(backend: &str, gpu_id: Option<&str>) -> Result<(Device, Option<String>), String> {
    let name: BackendName = serde_json::from_value(json!(backend)).map_err(|_| format!("unsupported backend: {backend}"))?;
    #[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
    if let Some(id) = gpu_id {
        let target: gpu_device_resolver::Backend = backend.parse().map_err(|e: gpu_device_resolver::ResolveError| e.to_string())?;
        let id: gpu_device_resolver::GpuId = id.parse().map_err(|e: gpu_device_resolver::ResolveError| e.to_string())?;
        let resolved = gpu_device_resolver::resolve(target, id).map_err(|e| e.to_string())?;
        return Ok((resolved.device, Some(id.to_string())));
    }
    #[cfg(not(any(feature = "cubecl-cuda", feature = "cubecl-wgpu")))]
    if gpu_id.is_some() { return Err("physical GPU selection requires a compiled GPU backend".into()); }
    Ok((name.device(0).map_err(|e| e.to_string())?, None))
}

pub fn run(args: &[String]) -> Result<(), String> {
    let mut options = BTreeMap::new();
    for pair in args.chunks(2) {
        if pair.len() != 2 || !pair[0].starts_with("--") || pair[1].starts_with("--") {
            return Err("expected --flag value pairs".into());
        }
        if options.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err(format!("duplicate option {}", pair[0]));
        }
    }
    let mut take = |name: &str| options.remove(name).ok_or_else(|| format!("missing {name}"));
    let backend = take("--backend")?.to_owned();
    let checkpoint = absolute_path(Path::new(take("--checkpoint")?), false)?;
    let _input_dir = absolute_path(Path::new(take("--input-dir")?), true)?;
    let output_dir = absolute_path(Path::new(take("--output-dir")?), true)?;
    let gpu_id = options.remove("--gpu-id");
    if let Some(flag) = options.keys().next() { return Err(format!("unknown option {flag}")); }
    let (device, gpu_id) = select_device(&backend, gpu_id)?;
    let mut worker = Worker { checkpoint, output_dir, device, backend, gpu_id, session: None };
    serve(|name, payload| worker.command(name, payload))
}
