use beat_this_rs::{audio, checkpoint::Checkpoint, model::BeatThisModel, predict_beats};
use burn::tensor::Device;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
};
use uuid::Uuid;

const MODEL_ID: &str = "beat_this_small1";
const VERSION: u32 = 1;
const MAX_FRAME: usize = 1024 * 1024;
const MAX_SOURCE: u64 = 64 * 1024 * 1024;

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
    beats_version: u32,
    kind: String,
    input: Track,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Track {
    file_name: String,
    size_bytes: u64,
    sha256: String,
}

struct Worker {
    model_dir: PathBuf,
    soxr: PathBuf,
    input_dir: Option<PathBuf>,
    model: Option<BeatThisModel>,
    warm: bool,
}

impl Worker {
    fn command(&mut self, name: &str, payload: Value) -> Result<Value, String> {
        if !matches!(name, "load" | "infer") && payload != json!({}) {
            return Err("unexpected command payload".into());
        }
        match name {
            "capabilities" => Ok(json!({
                "model_id": MODEL_ID,
                "backends": ["cpu"],
                "beats_contract_version": VERSION,
                "beats_capability": {"formats":["wav", "flac"], "max_source_bytes":MAX_SOURCE}
            })),
            "load" => {
                let load: Load = serde_json::from_value(payload).map_err(|e| e.to_string())?;
                if self.model.is_some() || load.model_id != MODEL_ID {
                    return Err("model does not match the package or is already loaded".into());
                }
                let weights = self.model_dir.join("small1.ckpt");
                require_file(&weights)?;
                let checkpoint = Checkpoint::open(weights)?;
                let model = BeatThisModel::from_checkpoint(&checkpoint, &Device::flex())?;
                if model.transformer_dim != 128 {
                    return Err("expected the small1 model architecture".into());
                }
                self.model = Some(model);
                Ok(json!({"loaded":true}))
            }
            "warmup" => {
                let model = self.model.as_ref().ok_or("model is not loaded")?;
                require_file(&self.soxr)?;
                let pcm = vec![0.0; 16_000];
                let resampled = audio::resample_with_packaged_soxr(&pcm, 16_000, &self.soxr)?;
                let spectrogram = audio::log_mel(&resampled)?;
                let prediction = beat_this_rs::predict_logits(model, &spectrogram)?;
                if prediction.beat.is_empty()
                    || prediction
                        .beat
                        .iter()
                        .chain(&prediction.downbeat)
                        .any(|v| !v.is_finite())
                {
                    return Err("model warmup produced invalid logits".into());
                }
                self.warm = true;
                Ok(json!({"warm":true}))
            }
            "status" => Ok(json!({
                "phase": if self.warm { "ready" } else if self.model.is_some() { "loaded" } else { "started" },
                "model_id": MODEL_ID,
                "backend": "cpu",
                "variant": "burn_flex_cpu_fp32"
            })),
            "infer" => self.infer(payload),
            "unload" | "shutdown" => {
                self.model = None;
                self.warm = false;
                Ok(json!({}))
            }
            _ => Err("unknown worker command".into()),
        }
    }

    fn infer(&self, payload: Value) -> Result<Value, String> {
        let model = self
            .model
            .as_ref()
            .filter(|_| self.warm)
            .ok_or("model is not ready")?;
        let request: Infer = serde_json::from_value(payload).map_err(|e| e.to_string())?;
        if request.beats_version != VERSION || request.kind != "beats" {
            return Err("unsupported beats request version or kind".into());
        }
        let root = self
            .input_dir
            .as_deref()
            .ok_or("input directory was not configured")?;
        let (bytes, extension) = read_track(root, &request.input)?;
        let supported = match extension {
            "wav" => bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE"),
            "flac" => bytes.starts_with(b"fLaC"),
            _ => false,
        };
        if !supported {
            return Err("audio bytes do not match the declared WAV or FLAC format".into());
        }
        let audio = audio::decode_snapshot(bytes, extension)?;
        if audio.pcm.iter().any(|sample| !sample.is_finite()) {
            return Err("audio contains a non-finite sample".into());
        }
        let duration = audio.pcm.len() as f64 / f64::from(audio.sample_rate);
        let resampled =
            audio::resample_with_packaged_soxr(&audio.pcm, audio.sample_rate, &self.soxr)?;
        let spectrogram = audio::log_mel(&resampled)?;
        let times = predict_beats(model, &spectrogram)?;
        Ok(json!({
            "beats_version": VERSION,
            "kind": "beats",
            "duration_seconds": duration,
            "sample_rate_hz": audio.sample_rate,
            "beats_seconds": times.beats,
            "downbeats_seconds": times.downbeats
        }))
    }
}

fn require_file(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.is_symlink() {
        return Err("expected a regular model or audio file".into());
    }
    Ok(())
}

fn read_track<'a>(root: &Path, track: &'a Track) -> Result<(Vec<u8>, &'a str), String> {
    let leaf = &track.file_name;
    let extension = leaf
        .rsplit_once('.')
        .map(|(_, ext)| ext)
        .filter(|ext| matches!(*ext, "wav" | "flac"))
        .ok_or("unsupported snapshot format")?;
    if leaf.len() > 128
        || leaf.starts_with('.')
        || !leaf
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        || !(1..=MAX_SOURCE).contains(&track.size_bytes)
        || track.sha256.len() != 64
    {
        return Err("invalid audio snapshot".into());
    }
    let path = root.join(leaf);
    require_file(&path)?;
    if path.canonicalize().map_err(|e| e.to_string())?.parent() != Some(root) {
        return Err("audio snapshot must be inside the worker input directory".into());
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take(track.size_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 != track.size_bytes
        || format!("{:x}", Sha256::digest(&bytes)) != track.sha256
    {
        return Err("audio snapshot size or digest does not match".into());
    }
    Ok((bytes, extension))
}

fn absolute_directory(value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("worker directory must be absolute".into());
    }
    let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !metadata.is_dir() || metadata.is_symlink() {
        return Err("worker directory must be a real directory".into());
    }
    path.canonicalize().map_err(|e| e.to_string())
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut model_dir = None;
    let mut runtime_dir = None;
    let mut input_dir = None;
    let mut backend = None;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("missing worker launch argument")?;
        match flag.as_str() {
            "--model-dir" if model_dir.is_none() => model_dir = Some(absolute_directory(value)?),
            "--runtime-dir" if runtime_dir.is_none() => {
                runtime_dir = Some(absolute_directory(value)?)
            }
            "--input-dir" if input_dir.is_none() => input_dir = Some(absolute_directory(value)?),
            "--backend" if backend.is_none() => backend = Some(value),
            _ => return Err("unknown or duplicate worker launch argument".into()),
        }
    }
    if backend.as_deref() != Some("cpu") {
        return Err("the worker supports the cpu backend with the Burn Flex variant".into());
    }
    #[cfg(target_os = "windows")]
    let soxr_name = "soxr.dll";
    #[cfg(target_os = "macos")]
    let soxr_name = "libsoxr.dylib";
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let soxr_name = "libsoxr.so.0";
    let mut worker = Worker {
        model_dir: model_dir.ok_or("missing --model-dir")?,
        soxr: runtime_dir
            .ok_or("missing --runtime-dir")?
            .join("soxr/0.1.3")
            .join(soxr_name),
        input_dir,
        model: None,
        warm: false,
    };
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut frame = Vec::new();
    loop {
        frame.clear();
        let count = input
            .by_ref()
            .take((MAX_FRAME + 1) as u64)
            .read_until(b'\n', &mut frame)
            .map_err(|e| e.to_string())?;
        if count == 0 {
            return Ok(());
        }
        if count > MAX_FRAME || frame.last() != Some(&b'\n') {
            return Err("invalid worker request frame".into());
        }
        let request: Request = serde_json::from_slice(&frame).map_err(|e| e.to_string())?;
        if request.version != VERSION {
            return Err("unsupported worker request version".into());
        }
        let result = worker.command(&request.command, request.payload);
        let response = match &result {
            Ok(value) => {
                json!({"version":VERSION, "id":request.id, "ok":true, "result":value, "error":null})
            }
            Err(error) => {
                json!({"version":VERSION, "id":request.id, "ok":false, "result":null, "error":error})
            }
        };
        serde_json::to_writer(&mut output, &response).map_err(|e| e.to_string())?;
        output
            .write_all(b"\n")
            .and_then(|_| output.flush())
            .map_err(|e| e.to_string())?;
        if request.command == "shutdown" && result.is_ok() {
            return Ok(());
        }
    }
}
