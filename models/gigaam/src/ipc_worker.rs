use burn::tensor::Device;
use gigaam_rs::{
    AsrOptions, ModelKind,
    zapformer::{HeadSelection, StreamPacket, StreamTranscription, ZapformerStream, language::LanguageTag},
    audio::{self, SpeechRegionSeconds},
    model_registry::{known_models, is_zapformer_model},
    service::{GigaAmService, Segmentation, TranscribeOptions},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
};
use uuid::Uuid;

const PROTOCOL_VERSION: u32 = 1;
const MAX_FRAME: usize = 1024 * 1024;
const MAX_SOURCE: u64 = 64 * 1024 * 1024;
const MAX_STREAM_PACKET: u64 = 4 * 16000 * 10;
const MAX_STREAM_SAMPLES: usize = 16000 * 60 * 5;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u32,
    id: Uuid,
    command: String,
    payload: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Load {
    model_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Infer {
    gigaam_version: u32,
    kind: String,
    input: Snapshot,
    #[serde(default)]
    word_timestamps: bool,
    #[serde(default)]
    segments: Option<Vec<SpeechRegionSeconds>>,
    #[serde(default = "default_group_vad")]
    group_vad: bool,
    #[serde(default)]
    head: Option<HeadSelection>,
    #[serde(default)]
    language: Option<LanguageTag>,
}

fn default_group_vad() -> bool {
    true
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    file_name: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamStart {
    gigaam_version: u32,
    future_context_ms: Option<usize>,
    #[serde(default)]
    language: Option<LanguageTag>,
    #[serde(default)]
    word_timestamps: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamPush {
    stream_id: Uuid,
    packet_index: u64,
    input: Snapshot,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamFinish {
    stream_id: Uuid,
    packet_index: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamCancel {
    stream_id: Uuid,
}

struct ActiveStream {
    id: Uuid,
    next_packet: u64,
    samples: usize,
    state: ZapformerStream,
}

fn stream_transcription(value: StreamTranscription) -> Value {
    json!({"text":value.text,"token_ids":value.token_ids,"frames":value.frames,
        "words":value.words.map(|words| words.into_iter().map(gigaam_rs::service::WordResult::from).collect::<Vec<_>>())})
}

fn stream_reply(id: Uuid, index: u64, samples: usize, frames: usize, finished: bool, packet: StreamPacket) -> Value {
    let heads = match packet {
        StreamPacket::Single(value) => json!({"single":stream_transcription(value)}),
        StreamPacket::Dual { raw,e2e } => json!({"raw":stream_transcription(raw),"e2e":stream_transcription(e2e)}),
    };
    json!({"gigaam_version":PROTOCOL_VERSION,"stream_id":id,"packet_index":index,
        "accepted_samples":samples,"encoded_frames":frames,"finished":finished,"heads":heads})
}

struct Worker {
    model_id: String,
    model_name: String,
    model_kind: ModelKind,
    weights: WeightsSource,
    input_dir: PathBuf,
    output_dir: Option<PathBuf>,
    backend: String,
    variant: &'static str,
    device: Device,
    model: Option<GigaAmService>,
    warm: bool,
    stream: Option<ActiveStream>,
}

enum WeightsSource {
    Directory(PathBuf),
    Explicit {
        checkpoint: PathBuf,
        config: Option<PathBuf>,
        tokenizer: Option<PathBuf>,
        raw_tokenizer: Option<PathBuf>,
        e2e_tokenizer: Option<PathBuf>,
    },
}

impl Worker {
    fn command(&mut self, request: &Request) -> Result<Value, String> {
        if !matches!(request.command.as_str(), "load" | "infer" | "stream_start" | "stream_push" | "stream_finish" | "stream_cancel") && request.payload != json!({}) {
            return Err("unexpected command payload".into());
        }
        match request.command.as_str() {
            "capabilities" => Ok(json!({
                "model_id": self.model_id,
                "backends": [self.backend],
                "gigaam_contract_version": PROTOCOL_VERSION,
                "gigaam_capability": {
                    "formats": ["wav", "flac"],
                    "max_source_bytes": MAX_SOURCE,
                    "kind": self.infer_kind(),
                    "external_vad": true,
                    "streaming": if self.stream_supported() { json!({"pcm_format":"f32le", "sample_rate_hz":16000, "channels":1, "max_packet_bytes":MAX_STREAM_PACKET, "max_utterance_samples":MAX_STREAM_SAMPLES, "max_active_streams":1}) } else { Value::Null },
                },
            })),
            "load" => {
                let load: Load = serde_json::from_value(request.payload.clone())
                    .map_err(|error| error.to_string())?;
                if self.model.is_some() || load.model_id != self.model_id {
                    return Err("model does not match the package or is already loaded".into());
                }
                let service = match &self.weights {
                    WeightsSource::Directory(dir) => {
                        let file = self.checkpoint(dir)?;
                        GigaAmService::load(&file, self.device.clone())
                    }
                    WeightsSource::Explicit { checkpoint, config, tokenizer, raw_tokenizer, e2e_tokenizer } => {
                        let files = gigaam_rs::CheckpointFiles {
                            config: config.as_deref(),
                            model_name: Some(&self.model_name),
                            tokenizers: gigaam_rs::TokenizerFiles {
                                single: tokenizer.as_deref(),
                                raw: raw_tokenizer.as_deref(),
                                e2e: e2e_tokenizer.as_deref(),
                            },
                        };
                        gigaam_rs::GigaAm::load_files(checkpoint, self.device.clone(), files)
                            .map(GigaAmService::from_loaded)
                    }
                }.map_err(|error| format!("{error:#}"))?;
                if service.status().model_kind != self.model_kind {
                    return Err("checkpoint model family differs from package model_id".into());
                }
                if service.is_zapformer() != is_zapformer_model(&self.model_name) {
                    return Err("checkpoint encoder architecture differs from package model_id".into());
                }
                self.model = Some(service);
                self.warm = false;
                Ok(json!({"loaded": true}))
            }
            "warmup" => {
                let service = self.model.as_ref().ok_or("model is not loaded")?;
                let pcm = vec![0.0_f32; audio::SAMPLE_RATE as usize];
                if self.model_kind.is_ctc() || self.model_kind.is_rnnt() {
                    service.transcribe_short_16k(&pcm, false)
                        .map_err(|error| format!("{error:#}"))?;
                } else if self.model_kind.is_ssl() {
                    let embedding = service.embed_16k(&pcm)
                        .map_err(|error| format!("{error:#}"))?;
                    if embedding.data.is_empty() || embedding.data.iter().any(|x| !x.is_finite()) {
                        return Err("model warmup produced invalid embeddings".into());
                    }
                } else {
                    let probabilities = service.emotions_dict_16k(&pcm)
                        .map_err(|error| format!("{error:#}"))?;
                    if probabilities.is_empty() || probabilities.values().any(|x| !x.is_finite()) {
                        return Err("model warmup produced invalid probabilities".into());
                    }
                }
                self.warm = true;
                Ok(json!({"warm": true}))
            }
            "status" => Ok(json!({
                "phase": if self.warm { "ready" } else if self.model.is_some() { "loaded" } else { "started" },
                "model_id": self.model_id,
                "backend": self.backend,
                "variant": self.variant,
                "active_stream": self.stream.as_ref().map(|s| s.id),
            })),
            "infer" => self.infer(request),
            "stream_start" => self.stream_start(request),
            "stream_push" => self.stream_push(request),
            "stream_finish" => self.stream_finish(request),
            "stream_cancel" => {
                let payload: StreamCancel = serde_json::from_value(request.payload.clone()).map_err(|e|e.to_string())?;
                self.require_stream(payload.stream_id, None)?;
                self.stream = None;
                Ok(json!({"cancelled":true}))
            },
            "unload" | "shutdown" => {
                self.stream = None;
                self.model = None;
                self.warm = false;
                Ok(json!({}))
            }
            _ => Err("unknown worker command".into()),
        }
    }

    fn stream_supported(&self) -> bool {
        is_zapformer_model(&self.model_name) && (self.model_kind.is_ctc() || self.model_kind.is_rnnt())
    }

    fn stream_start(&mut self, request: &Request) -> Result<Value, String> {
        let payload: StreamStart = serde_json::from_value(request.payload.clone()).map_err(|e|e.to_string())?;
        if payload.gigaam_version != PROTOCOL_VERSION || !self.stream_supported() {
            return Err("unsupported GigaAM streaming version or model".into());
        }
        if self.stream.is_some() { return Err("an utterance stream is already active".into()); }
        let service = self.model.as_ref().filter(|_|self.warm).ok_or("model is not ready")?;
        let state = service.start_stream(payload.future_context_ms,payload.language.as_ref(),payload.word_timestamps).map_err(|e|format!("{e:#}"))?;
        self.stream = Some(ActiveStream { id:request.id,next_packet:0,samples:0,state });
        Ok(json!({"stream_id":request.id,"next_packet_index":0}))
    }

    fn require_stream(&self, id: Uuid, index: Option<u64>) -> Result<&ActiveStream, String> {
        let state = self.stream.as_ref().filter(|s|s.id==id).ok_or("unknown or inactive stream")?;
        if index.is_some_and(|i|i!=state.next_packet) { return Err("unexpected stream packet index".into()); }
        Ok(state)
    }

    fn stream_push(&mut self, request: &Request) -> Result<Value, String> {
        let payload: StreamPush = serde_json::from_value(request.payload.clone()).map_err(|e|e.to_string())?;
        let previous = self.require_stream(payload.stream_id,Some(payload.packet_index))?.samples;
        let (bytes,_) = read_snapshot_format(&self.input_dir,&payload.input,&["f32"],MAX_STREAM_PACKET)?;
        if bytes.len()%4 != 0 { return Err("PCM snapshot is not aligned to float32".into()); }
        let pcm: Vec<f32> = bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        if pcm.iter().any(|v| !v.is_finite()) { return Err("PCM snapshot contains nonfinite samples".into()); }
        let samples = previous.checked_add(pcm.len()).filter(|&n|n<=MAX_STREAM_SAMPLES).ok_or("utterance sample limit exceeded")?;
        let state = self.stream.as_mut().unwrap();
        let result = state.state.push(&pcm,false);
        match result {
            Ok(packet) => {
                state.samples=samples;
                state.next_packet+=1;
                Ok(stream_reply(state.id,payload.packet_index,samples,state.state.encoded_frames(),false,packet))
            }
            Err(error) => { self.stream=None; Err(format!("stream discarded after inference error: {error:#}")) }
        }
    }

    fn stream_finish(&mut self, request: &Request) -> Result<Value, String> {
        let payload: StreamFinish = serde_json::from_value(request.payload.clone()).map_err(|e|e.to_string())?;
        self.require_stream(payload.stream_id,Some(payload.packet_index))?;
        let mut state=self.stream.take().unwrap();
        let packet=state.state.flush().map_err(|e|format!("stream discarded after finalization error: {e:#}"))?;
        Ok(stream_reply(state.id,payload.packet_index,state.samples,state.state.encoded_frames(),true,packet))
    }

    fn infer_kind(&self) -> &'static str {
        if self.model_kind.is_ssl() { "embeddings" }
        else if self.model_kind.is_emo() { "emotion" }
        else { "asr" }
    }

    fn checkpoint(&self, directory: &Path) -> Result<PathBuf, String> {
        let candidate = directory.join(format!("{}.ckpt", self.model_name));
        if candidate.is_file() {
            require_regular_file(&candidate)?;
            return Ok(candidate);
        }
        for fallback in ["model.ckpt", "model.safetensors"] {
            let path = directory.join(fallback);
            if path.is_file() {
                require_regular_file(&path)?;
                return Ok(path);
            }
        }
        Err(format!("package model checkpoint is missing: {}.ckpt", self.model_name))
    }

    fn infer(&self, request: &Request) -> Result<Value, String> {
        let model = self.model.as_ref().filter(|_| self.warm).ok_or("model is not ready")?;
        let infer: Infer = serde_json::from_value(request.payload.clone()).map_err(|error| error.to_string())?;
        if infer.gigaam_version != PROTOCOL_VERSION || infer.kind != self.infer_kind() {
            return Err("unsupported GigaAM request version or kind".into());
        }
        let (bytes, extension) = read_snapshot(&self.input_dir, &infer.input)?;
        let valid = match extension {
            "wav" => bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE"),
            "flac" => bytes.starts_with(b"fLaC"),
            _ => false,
        };
        if !valid {
            return Err("audio bytes do not match declared WAV or FLAC format".into());
        }
        let pcm = audio::load_audio_16k_bytes(bytes, extension).map_err(|error| format!("{error:#}"))?;
        if pcm.is_empty() || pcm.iter().any(|x| !x.is_finite()) {
            return Err("decoded audio is empty or contains nonfinite samples".into());
        }
        let duration = pcm.len() as f64 / audio::SAMPLE_RATE as f64;
        match infer.kind.as_str() {
            "asr" => {
                let regions = infer.segments.map(|values| {
                    values.into_iter().map(|item| item.into_samples(pcm.len()).map_err(|e| e.to_string()))
                        .collect::<Result<Vec<_>,_>>()
                }).transpose()?;
                let segmentation = match regions.as_deref() {
                    None => Segmentation::FullRecording,
                    Some(regions) if infer.group_vad => Segmentation::GroupedSpeechRegions {
                        regions,
                        grouping: audio::VadGrouping::default(),
                    },
                    Some(regions) => Segmentation::SpeechRegions(regions),
                };
                let answer = model.transcribe_16k_with_asr(&pcm, TranscribeOptions {
                    segmentation,
                    word_timestamps: infer.word_timestamps,
                }, AsrOptions { head: infer.head, language: infer.language.as_ref() }).map_err(|error| format!("{error:#}"))?;
                Ok(json!({
                    "gigaam_version": PROTOCOL_VERSION,
                    "kind": "asr",
                    "duration_seconds": duration,
                    "sample_rate_hz": audio::SAMPLE_RATE,
                    "text": answer.text,
                    "segments": answer.segments,
                }))
            }
            "emotion" => {
                if infer.word_timestamps || infer.segments.is_some() || infer.head.is_some() || infer.language.is_some() {
                    return Err("emotion inference does not accept ASR segmentation options".into());
                }
                let probabilities = model.emotions_dict_16k(&pcm).map_err(|e| format!("{e:#}"))?;
                Ok(json!({
                    "gigaam_version": PROTOCOL_VERSION,
                    "kind": "emotion",
                    "duration_seconds": duration,
                    "sample_rate_hz": audio::SAMPLE_RATE,
                    "probabilities": probabilities,
                }))
            }
            "embeddings" => {
                if infer.word_timestamps || infer.segments.is_some() || infer.head.is_some() || infer.language.is_some() {
                    return Err("embedding inference does not accept ASR segmentation options".into());
                }
                let output_root = self.output_dir.as_ref().ok_or("embedding output directory was not configured")?;
                let embedding = model.embed_16k(&pcm).map_err(|e| format!("{e:#}"))?;
                if embedding.data.is_empty() || embedding.data.iter().any(|x| !x.is_finite()) {
                    return Err("model produced invalid embeddings".into());
                }
                let name = format!("{}.f32", request.id);
                let path = output_root.join(&name);
                let mut bytes = Vec::with_capacity(embedding.data.len() * 4);
                for value in embedding.data {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                let digest = format!("{:x}", Sha256::digest(&bytes));
                let mut target = OpenOptions::new().create_new(true).write(true).open(&path)
                    .map_err(|error| format!("cannot create output snapshot: {error}"))?;
                if let Err(error) = target.write_all(&bytes).and_then(|_| target.sync_all()) {
                    drop(target);
                    let _ = fs::remove_file(path);
                    return Err(format!("writing output snapshot: {error}"));
                }
                Ok(json!({
                    "gigaam_version": PROTOCOL_VERSION,
                    "kind": "embeddings",
                    "duration_seconds": duration,
                    "sample_rate_hz": audio::SAMPLE_RATE,
                    "frames": embedding.frames,
                    "channels": embedding.channels,
                    "output": {"file_name": name, "size_bytes": bytes.len(), "sha256": digest},
                }))
            }
            _ => Err("unsupported inference kind".into()),
        }
    }
}

fn require_regular_file(path: &Path) -> Result<(), String> {
    let meta = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.is_symlink() {
        return Err("expected a regular model or audio file".into());
    }
    Ok(())
}

fn read_snapshot<'a>(root: &Path, snapshot: &'a Snapshot) -> Result<(Vec<u8>, &'a str), String> {
    read_snapshot_format(root,snapshot,&["wav","flac"],MAX_SOURCE)
}

fn read_snapshot_format<'a>(root: &Path, snapshot: &'a Snapshot, formats: &[&str], max_bytes: u64) -> Result<(Vec<u8>, &'a str), String> {
    let name = &snapshot.file_name;
    let ext = name.rsplit_once('.').map(|(_, ext)| ext)
        .filter(|ext| formats.contains(ext))
        .ok_or("unsupported audio snapshot format")?;
    if name.len() > 128 || name.starts_with('.') ||
       !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')) ||
       !(1..=max_bytes).contains(&snapshot.size_bytes) ||
       snapshot.sha256.len() != 64 || !snapshot.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid audio snapshot".into());
    }
    let path = root.join(name);
    require_regular_file(&path)?;
    if path.canonicalize().map_err(|e| e.to_string())?.parent() != Some(root) {
        return Err("audio snapshot must be inside the worker input directory".into());
    }
    let mut bytes = Vec::new();
    File::open(path).map_err(|e| e.to_string())?.take(snapshot.size_bytes + 1)
        .read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() as u64 != snapshot.size_bytes ||
       format!("{:x}", Sha256::digest(&bytes)) != snapshot.sha256 {
        return Err("audio snapshot size or digest does not match".into());
    }
    Ok((bytes, ext))
}

fn absolute_directory(value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() { return Err("worker directory must be absolute".into()); }
    let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.is_symlink() { return Err("worker directory must be a real directory".into()); }
    path.canonicalize().map_err(|e| e.to_string())
}

fn absolute_file(value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("worker file must be absolute: {}", path.display()));
    }
    require_regular_file(&path)?;
    Ok(path)
}

fn runtime_backend(backend: &str) -> Result<(Device, &'static str), String> {
    match backend {
        "cuda" => {
            #[cfg(feature = "cuda")]
            { Ok((Device::cuda(0), "burn_cubecl_cuda_fp32")) }
            #[cfg(not(feature = "cuda"))]
            { Err("CUDA backend not enabled in this build".into()) }
        }
        "cpu" => {
            #[cfg(feature = "flex")]
            { Ok((Device::flex(), "burn_flex_cpu_fp32")) }
            #[cfg(all(not(feature = "flex"), feature = "cpu"))]
            { Ok((Device::cpu(), "burn_cubecl_cpu_fp32")) }
            #[cfg(not(any(feature = "flex", feature = "cpu")))]
            { Err("CPU backend not enabled in this build".into()) }
        }
        "wgpu" => {
            #[cfg(feature = "wgpu")]
            { Ok((Device::wgpu(Default::default()), "burn_cubecl_wgpu_fp32")) }
            #[cfg(not(feature = "wgpu"))]
            { Err("WGPU backend not enabled in this build".into()) }
        }
        _ => Err("unsupported worker backend; expected cpu, cuda or wgpu".into()),
    }
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut model_id = None;
    let mut model_dir = None;
    let mut checkpoint = None;
    let mut config = None;
    let mut tokenizer = None;
    let mut raw_tokenizer = None;
    let mut e2e_tokenizer = None;
    let mut runtime_dir = None;
    let mut input_dir = None;
    let mut output_dir = None;
    let mut backend = None;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("missing worker launch argument")?;
        match flag.as_str() {
            "--model-id" if model_id.is_none() => model_id = Some(value),
            "--model-dir" if model_dir.is_none() => model_dir = Some(absolute_directory(value)?),
            "--checkpoint" if checkpoint.is_none() => checkpoint = Some(absolute_file(value)?),
            "--config" if config.is_none() => config = Some(absolute_file(value)?),
            "--tokenizer" if tokenizer.is_none() => tokenizer = Some(absolute_file(value)?),
            "--raw-tokenizer" if raw_tokenizer.is_none() => raw_tokenizer = Some(absolute_file(value)?),
            "--e2e-tokenizer" if e2e_tokenizer.is_none() => e2e_tokenizer = Some(absolute_file(value)?),
            "--runtime-dir" if runtime_dir.is_none() => runtime_dir = Some(absolute_directory(value)?),
            "--input-dir" if input_dir.is_none() => input_dir = Some(absolute_directory(value)?),
            "--output-dir" if output_dir.is_none() => output_dir = Some(absolute_directory(value)?),
            "--backend" if backend.is_none() => backend = Some(value),
            _ => return Err("unknown or duplicate worker launch argument".into()),
        }
    }
    let model_id = model_id.ok_or("missing --model-id")?;
    let model_name = model_id.strip_prefix("gigaam_").ok_or("model_id must use gigaam_ prefix")?.to_owned();
    if !known_models().iter().any(|(name,_)| *name == model_name) {
        return Err("unsupported GigaAM model_id".into());
    }
    let kind = ModelKind::from_model_name(&model_name).ok_or("unsupported GigaAM model kind")?;
    let backend = backend.ok_or("missing --backend")?;
    let (device, variant) = runtime_backend(&backend)?;
    let weights = match (checkpoint, model_dir) {
        (Some(checkpoint), None) => WeightsSource::Explicit { checkpoint, config, tokenizer, raw_tokenizer, e2e_tokenizer },
        (None, Some(dir)) if [config, tokenizer, raw_tokenizer, e2e_tokenizer].iter().all(Option::is_none) => WeightsSource::Directory(dir),
        _ => return Err("pass --checkpoint with its optional artifact paths or use --model-dir by itself".into()),
    };
    let mut worker = Worker {
        model_id,
        model_name,
        model_kind: kind,
        weights,
        input_dir: input_dir.ok_or("missing --input-dir")?,
        output_dir,
        backend,
        variant,
        device,
        model: None,
        warm: false,
        stream: None,
    };
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut frame = Vec::new();
    loop {
        frame.clear();
        let count = input.by_ref().take((MAX_FRAME + 1) as u64).read_until(b'\n', &mut frame)
            .map_err(|e| e.to_string())?;
        if count == 0 { return Ok(()); }
        if count > MAX_FRAME || frame.last() != Some(&b'\n') {
            return Err("invalid worker request frame".into());
        }
        let request: Request = serde_json::from_slice(&frame).map_err(|e| e.to_string())?;
        if request.version != PROTOCOL_VERSION {
            return Err("unsupported worker request version".into());
        }
        let result = worker.command(&request);
        let reply = match &result {
            Ok(value) => json!({"version": PROTOCOL_VERSION, "id": request.id, "ok": true, "result": value, "error": null}),
            Err(message) => json!({"version": PROTOCOL_VERSION, "id": request.id, "ok": false, "result": null, "error": message}),
        };
        serde_json::to_writer(&mut output, &reply).map_err(|e| e.to_string())?;
        output.write_all(b"\n").and_then(|_| output.flush()).map_err(|e| e.to_string())?;
        if request.command == "shutdown" && result.is_ok() { return Ok(()); }
    }
}
