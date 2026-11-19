use std::{
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{audio::write_wav_pcm16, runtime::{SileroTts, TextOptions}};

const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendName {
    Cpu,
    Cuda,
    Wgpu,
    Flex,
}

impl BackendName {
    fn device(&self, index: usize) -> Result<Device> {
        match self {
            Self::Cpu => {
                #[cfg(feature = "cubecl-cpu")]
                { Ok(Device::cpu()) }
                #[cfg(not(feature = "cubecl-cpu"))]
                { bail!("CubeCL CPU backend was not compiled into this executable") }
            }
            Self::Cuda => {
                #[cfg(feature = "cubecl-cuda")]
                { Ok(Device::cuda(index)) }
                #[cfg(not(feature = "cubecl-cuda"))]
                { let _ = index; bail!("CubeCL CUDA backend was not compiled into this executable") }
            }
            Self::Wgpu => {
                #[cfg(feature = "cubecl-wgpu")]
                { let _ = index; Ok(Device::wgpu(Default::default())) }
                #[cfg(not(feature = "cubecl-wgpu"))]
                { let _ = index; bail!("CubeCL WGPU backend was not compiled into this executable") }
            }
            Self::Flex => {
                #[cfg(feature = "reference-flex")]
                { let _ = index; Ok(Device::flex()) }
                #[cfg(not(feature = "reference-flex"))]
                { let _ = index; bail!("Diagnostic Flex backend was not compiled into this executable") }
            }
        }
    }
}

fn compiled_backends() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut available = Vec::new();
    #[cfg(feature = "cubecl-cpu")]
    available.push("cpu");
    #[cfg(feature = "cubecl-cuda")]
    available.push("cuda");
    #[cfg(feature = "cubecl-wgpu")]
    available.push("wgpu");
    #[cfg(feature = "reference-flex")]
    available.push("flex");
    available
}

pub fn capabilities() -> Value {
    json!({
        "protocol_version": PROTOCOL_VERSION,
        "model": "silero-v5-ru",
        "available_backends": compiled_backends(),
        "preferred_backend": if cfg!(feature = "cubecl-cpu") { Some("cpu") } else if cfg!(feature = "cubecl-wgpu") { Some("wgpu") } else if cfg!(feature = "cubecl-cuda") { Some("cuda") } else { None },
        "diagnostic_backend": if cfg!(feature = "reference-flex") { Some("flex") } else { None },
        "speaker_names": ["aidar", "baya", "kseniya", "eugene", "xenia"],
        "sample_rates": [8000, 24000, 48000],
        "inference_inputs": ["text", "ssml", "accented", "symbol_durations"],
        "optional_text_providers": if cfg!(feature = "ruaccent") { vec!["ruaccent"] } else { vec![] },
        "outputs": ["inline_f32le", "wav_file"],
        "cancel_supported": false,
        "persistent_weights": true,
        "long_form": {
            "automatic_chunking": true,
            "target_characters": crate::longform::TARGET_CHARS,
            "maximum_input_characters": crate::longform::MAX_INPUT_CHARS,
            "maximum_output_seconds": crate::longform::MAX_AUDIO_SECONDS,
            "adaptive_acoustic_limit_retries": true,
            "streaming": false,
        },
    })
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Input {
    Text { text: String, #[serde(default)] options: TextOptions },
    #[cfg(feature = "ruaccent")]
    RuaccentText { text: String },
    Ssml { ssml: String, #[serde(default)] options: TextOptions },
    Accented { text: String },
    SymbolDurations {
        text: String,
        durations: Vec<(String, u32)>,
        #[serde(default)] options: TextOptions,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioOutput {
    InlineF32le,
    WavFile { path: PathBuf },
}

impl Default for AudioOutput {
    fn default() -> Self {
        Self::InlineF32le
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Request {
    Capabilities { request_id: String },
    Load {
        request_id: String,
        model_dir: PathBuf,
        backend: BackendName,
        #[serde(default)] device_index: usize,
        #[cfg(feature = "ruaccent")]
        ruaccent_dir: Option<PathBuf>,
        #[cfg(feature = "ruaccent")]
        ruaccent_omograph_model: Option<String>,
    },
    Warmup {
        request_id: String,
        #[serde(default = "default_warmup_text")] text: String,
        #[serde(default = "default_warmup_speaker")] speaker: String,
        #[serde(default = "default_sample_rate")] sample_rate: u32,
    },
    Infer {
        request_id: String,
        input: Input,
        speaker: String,
        sample_rate: u32,
        #[serde(default)] output: AudioOutput,
    },
    Status { request_id: String },
    Unload { request_id: String },
    Cancel { request_id: String, target_request_id: String },
    Shutdown { request_id: String },
}

fn default_sample_rate() -> u32 { 8000 }
fn default_warmup_text() -> String { "На двери замок.".into() }
fn default_warmup_speaker() -> String { "aidar".into() }

impl Request {
    fn id(&self) -> &str {
        match self {
            Self::Capabilities { request_id }
            | Self::Load { request_id, .. }
            | Self::Warmup { request_id, .. }
            | Self::Infer { request_id, .. }
            | Self::Status { request_id }
            | Self::Unload { request_id }
            | Self::Cancel { request_id, .. }
            | Self::Shutdown { request_id } => request_id,
        }
    }
}

#[derive(Serialize)]
struct Response {
    request_id: Option<String>,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl Response {
    fn success(id: impl Into<String>, data: Value) -> Self {
        Self { request_id: Some(id.into()), ok: true, data: Some(data), error: None }
    }

    fn failure(id: Option<String>, error: impl std::fmt::Display) -> Self {
        Self { request_id: id, ok: false, data: None, error: Some(error.to_string()) }
    }
}

struct Loaded {
    model: SileroTts,
    model_dir: PathBuf,
    backend: BackendName,
    device_index: usize,
    warmed: bool,
    successful_inferences: u64,
    #[cfg(feature = "ruaccent")]
    ruaccent: Option<ruaccent_rs::RuAccent>,
}

#[derive(Default)]
pub struct Worker {
    loaded: Option<Loaded>,
}

impl Worker {
    fn status(&self) -> Value {
        match &self.loaded {
            None => json!({"state": "unloaded", "protocol_version": PROTOCOL_VERSION}),
            Some(l) => json!({
                "state": if l.warmed { "ready" } else { "loaded" },
                "backend": l.backend,
                "device_index": l.device_index,
                "model_dir": l.model_dir,
                "successful_inferences": l.successful_inferences,
                "protocol_version": PROTOCOL_VERSION,
            }),
        }
    }

    fn execute(&mut self, request: Request) -> Result<(Value, bool)> {
        match request {
            Request::Capabilities { .. } => Ok((capabilities(), false)),
            Request::Load {
                model_dir, backend, device_index,
                #[cfg(feature = "ruaccent")]
                ruaccent_dir,
                #[cfg(feature = "ruaccent")]
                ruaccent_omograph_model,
                ..
            } => {
                if self.loaded.is_some() {
                    bail!("A model is already loaded; unload before loading a different model");
                }
                let device = backend.device(device_index)?;
                let canonical = model_dir.canonicalize()
                    .with_context(|| format!("Model directory is inaccessible: {}", model_dir.display()))?;
                let model = SileroTts::load(&canonical, &device)
                    .with_context(|| format!("Loading model from {}", canonical.display()))?;
                #[cfg(feature = "ruaccent")]
                let ruaccent = match ruaccent_dir {
                    Some(path) => {
                        let path = path.canonicalize().with_context(|| {
                            format!("RUAccent model directory is inaccessible: {}", path.display())
                        })?;
                        Some(ruaccent_rs::RuAccent::load_with_omograph(
                            &path, ruaccent_omograph_model.as_deref().unwrap_or("tiny"),
                        ).map_err(|e| anyhow::anyhow!("RUAccent initialization failed: {e}"))?)
                    },
                    None => {
                        anyhow::ensure!(ruaccent_omograph_model.is_none(),
                            "ruaccent_omograph_model requires ruaccent_dir");
                        None
                    }
                };
                self.loaded = Some(Loaded {
                    model, model_dir: canonical, backend, device_index,
                    warmed: false, successful_inferences: 0,
                    #[cfg(feature = "ruaccent")]
                    ruaccent,
                });
                Ok((self.status(), false))
            }
            Request::Warmup { text, speaker, sample_rate, .. } => {
                let loaded = self.loaded.as_mut().context("Model is not loaded")?;
                let start = Instant::now();
                let samples = loaded.model.infer_text(&text, &speaker, sample_rate)?;
                loaded.warmed = true;
                Ok((json!({
                    "state": "ready", "sample_count": samples.len(),
                    "sample_rate": sample_rate, "elapsed_ms": start.elapsed().as_millis(),
                }), false))
            }
            Request::Infer { input, speaker, sample_rate, output, .. } => {
                let loaded = self.loaded.as_mut().context("Model is not loaded")?;
                let start = Instant::now();
                let samples = match input {
                    Input::Text { text, options } => loaded.model.infer_text_with_options(&text, &speaker, sample_rate, &options)?,
                    #[cfg(feature = "ruaccent")]
                    Input::RuaccentText { text } => {
                        let ruaccent = loaded.ruaccent.as_mut().context(
                            "RUAccent not loaded: supply ruaccent_dir in the load request",
                        )?;
                        loaded.model.infer_text_with_ruaccent(&text, &speaker, sample_rate, ruaccent)?
                    },
                    Input::Ssml { ssml, options } => loaded.model.infer_ssml_with_options(&ssml, &speaker, sample_rate, &options)?,
                    Input::Accented { text } => loaded.model.infer_accented(&text, &speaker, sample_rate)?,
                    Input::SymbolDurations { text, durations, options } => loaded.model.infer_text_with_symbol_durations(&text, &speaker, sample_rate, &durations, &options)?,
                };
                let audio = match output {
                    AudioOutput::InlineF32le => {
                        let mut bytes = Vec::with_capacity(samples.len() * 4);
                        for sample in &samples {
                            bytes.extend_from_slice(&sample.to_le_bytes());
                        }
                        json!({"format": "f32le", "data_base64": STANDARD.encode(bytes)})
                    }
                    AudioOutput::WavFile { path } => {
                        write_wav_pcm16(Path::new(&path), &samples, sample_rate)?;
                        json!({"format": "wav_pcm16", "path": path})
                    }
                };
                loaded.warmed = true;
                loaded.successful_inferences += 1;
                Ok((json!({
                    "state": "ready", "sample_count": samples.len(), "sample_rate": sample_rate,
                    "elapsed_ms": start.elapsed().as_millis(), "audio": audio,
                }), false))
            }
            Request::Status { .. } => Ok((self.status(), false)),
            Request::Unload { .. } => {
                self.loaded = None;
                Ok((self.status(), false))
            }
            Request::Cancel { target_request_id, .. } => {
                Ok((json!({
                    "supported": false,
                    "target_request_id": target_request_id,
                    "reason": "Inference is synchronous and cannot be interrupted between CubeCL kernels in this worker version",
                }), false))
            }
            Request::Shutdown { .. } => {
                self.loaded = None;
                Ok((json!({"state": "shutdown"}), true))
            }
        }
    }

    fn process(&mut self, request: Request) -> (Response, bool) {
        let id = request.id().to_string();
        match self.execute(request) {
            Ok((data, shutdown)) => (Response::success(id, data), shutdown),
            Err(error) => (Response::failure(Some(id), format!("{error:#}")), false),
        }
    }
}

fn read_frame<R: BufRead>(reader: &mut R, frame: &mut Vec<u8>) -> io::Result<Option<bool>> {
    frame.clear();
    let mut oversized = false;
    let mut consumed = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if consumed { Ok(Some(!oversized)) } else { Ok(None) };
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let count = newline.map(|idx| idx + 1).unwrap_or(available.len());
        if !oversized {
            if frame.len() + count > MAX_REQUEST_BYTES {
                oversized = true;
                frame.clear();
            } else {
                frame.extend_from_slice(&available[..count]);
            }
        }
        reader.consume(count);
        consumed = true;
        if newline.is_some() {
            return Ok(Some(!oversized));
        }
    }
}

pub fn serve<R: BufRead, W: Write>(mut reader: R, mut writer: W) -> io::Result<()> {
    let mut worker = Worker::default();
    let mut frame = Vec::new();
    loop {
        let Some(valid_length) = read_frame(&mut reader, &mut frame)? else { break };
        let (reply, shutdown) = if !valid_length {
            (Response::failure(None, "Request exceeds 1 MiB"), false)
        } else {
            match serde_json::from_slice::<Request>(&frame) {
                Ok(request) => worker.process(request),
                Err(error) => (Response::failure(None, format!("Invalid request: {error}")), false),
            }
        };
        serde_json::to_writer(&mut writer, &reply)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        if shutdown { break; }
    }
    Ok(())
}

#[cfg(all(test, feature = "ruaccent"))]
mod ruaccent_protocol_tests {
    use super::*;

    #[test]
    fn explicit_provider_is_opt_in() {
        let req: Request = serde_json::from_str(r#"{"command":"infer","request_id":"ra","speaker":"aidar","sample_rate":8000,"input":{"kind":"ruaccent_text","text":"На двери замок."}}"#).unwrap();
        assert!(matches!(req, Request::Infer { input: Input::RuaccentText { .. }, .. }));
        let original: Request = serde_json::from_str(r#"{"command":"infer","request_id":"old","speaker":"aidar","sample_rate":8000,"input":{"kind":"text","text":"На двери замок."}}"#).unwrap();
        assert!(matches!(original, Request::Infer { input: Input::Text { .. }, .. }));
        assert_eq!(capabilities()["optional_text_providers"][0], "ruaccent");
    }

    #[test]
    fn optional_model_pack_is_not_loaded_by_default() {
        let legacy: Request = serde_json::from_str(r#"{"command":"load","request_id":"old","backend":"flex","model_dir":"/tmp/silero"}"#).unwrap();
        assert!(matches!(legacy, Request::Load { ruaccent_dir: None, ruaccent_omograph_model: None, .. }));
        let selected: Request = serde_json::from_str(r#"{"command":"load","request_id":"new","backend":"flex","model_dir":"/tmp/silero","ruaccent_dir":"/tmp/ruaccent","ruaccent_omograph_model":"tiny"}"#).unwrap();
        assert!(matches!(selected, Request::Load { ruaccent_dir: Some(_), .. }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exchange(commands: &str) -> Vec<Value> {
        let mut output = Vec::new();
        serve(commands.as_bytes(), &mut output).unwrap();
        String::from_utf8(output).unwrap().lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap()).collect()
    }

    #[cfg(feature = "cubecl-cpu")]
    #[test]
    fn cubecl_cpu_executes_tensor_matmul() {
        use burn::tensor::Tensor;
        let device = Device::cpu();
        let input = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let result = input.clone().matmul(input).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[test]
    fn capabilities_without_loading_tensor_runtime() {
        let responses = exchange("{\"command\":\"capabilities\",\"request_id\":\"caps\"}\n{\"command\":\"shutdown\",\"request_id\":\"end\"}\n");
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["ok"], true);
        assert_eq!(responses[0]["data"]["protocol_version"], 1);
        assert_eq!(responses[0]["data"]["cancel_supported"], false);
        assert_eq!(responses[1]["data"]["state"], "shutdown");
    }

    #[test]
    fn malformed_input_does_not_terminate_worker() {
        let responses = exchange("broken json\n{\"command\":\"status\",\"request_id\":\"s\"}\n");
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["ok"], false);
        assert_eq!(responses[1]["data"]["state"], "unloaded");
    }

    #[test]
    fn infer_without_weights_has_explicit_error() {
        let responses = exchange("{\"command\":\"infer\",\"request_id\":\"1\",\"speaker\":\"aidar\",\"sample_rate\":8000,\"input\":{\"kind\":\"text\",\"text\":\"привет\"}}\n");
        assert_eq!(responses.len(), 1);
        assert!(responses[0]["error"].as_str().unwrap().contains("not loaded"));
    }

    #[test]
    fn missing_model_keeps_state_unloaded() {
        let responses = exchange("{\"command\":\"load\",\"request_id\":\"bad\",\"backend\":\"cpu\",\"model_dir\":\"/does-not-exist\"}\n{\"command\":\"status\",\"request_id\":\"s\"}\n");
        assert_eq!(responses[0]["ok"], false);
        assert_eq!(responses[1]["data"]["state"], "unloaded");
    }

    #[test]
    fn oversized_frame_is_rejected_without_losing_next_request() {
        let huge = "x".repeat(MAX_REQUEST_BYTES + 1);
        let input = format!("{huge}\n{{\"command\":\"status\",\"request_id\":\"next\"}}\n");
        let responses = exchange(&input);
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["ok"], false);
        assert_eq!(responses[1]["request_id"], "next");
        assert_eq!(responses[1]["data"]["state"], "unloaded");
    }

    #[test]
    fn cancel_reports_no_runtime_preemption() {
        let responses = exchange("{\"command\":\"cancel\",\"request_id\":\"cancel\",\"target_request_id\":\"infer\"}\n");
        assert_eq!(responses[0]["data"]["supported"], false);
    }
}
