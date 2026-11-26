use burn::tensor::Device;
use dettato_rs::{
    audio,
    checkpoint::ModelPaths,
    pipeline::{Dettato, DettatoModelPaths},
};
use native_worker_ipc::{MAX_SOURCE_BYTES, Snapshot, absolute_path, serve};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum MidiOutput {
    Combined,
    Tracks,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Infer {
    dettato_version: u32,
    input: Snapshot,
    output: MidiOutput,
    #[serde(default)]
    resample_low_rate_drums: bool,
}

struct Worker {
    paths: DettatoModelPaths,
    device: Device,
    backend: String,
    gpu_id: Option<String>,
    input_dir: PathBuf,
    output_dir: PathBuf,
    cache_mib: usize,
    formats: &'static [&'static str],
    model: Option<Dettato>,
    ready: bool,
}

impl Worker {
    fn command(&mut self, name: &str, payload: Value) -> Result<Value, String> {
        if name != "infer" && payload != json!({}) {
            return Err("unexpected command payload".into());
        }
        match name {
            "capabilities" => Ok(json!({
                "service": "audio_to_midi", "dettato_contract_version": 1,
                "backend": self.backend, "formats": self.formats, "max_source_bytes": MAX_SOURCE_BYTES,
                "outputs": ["combined", "tracks"],
            })),
            "load" => {
                if self.model.is_some() {
                    return Err("model is already loaded".into());
                }
                let mut model = Dettato::from_paths(&self.paths, &self.device)?;
                model
                    .transcriber
                    .set_shared_cache_budget_bytes(self.cache_mib.saturating_mul(1024 * 1024));
                self.device.sync().map_err(|error| error.to_string())?;
                self.model = Some(model);
                Ok(json!({}))
            }
            "warmup" => {
                let model = self.model.as_ref().ok_or("model is not loaded")?;
                model.transcribe(&vec![0.0; 8000], 44_100)?;
                self.device.sync().map_err(|error| error.to_string())?;
                self.ready = true;
                Ok(json!({}))
            }
            "status" => Ok(json!({
                "phase": if self.ready { "ready" } else if self.model.is_some() { "loaded" } else { "started" },
                "backend": self.backend, "gpu_id": self.gpu_id,
            })),
            "infer" => {
                self.infer(serde_json::from_value(payload).map_err(|error| error.to_string())?)
            }
            "unload" | "shutdown" => {
                self.model = None;
                self.ready = false;
                Ok(json!({}))
            }
            _ => Err("unknown worker command".into()),
        }
    }

    fn infer(&mut self, request: Infer) -> Result<Value, String> {
        let model = self
            .model
            .as_mut()
            .filter(|_| self.ready)
            .ok_or("model is not ready")?;
        if request.dettato_version != 1 {
            return Err("unsupported Dettato request version".into());
        }
        let extension = request
            .input
            .file_name
            .rsplit_once('.')
            .map(|(_, extension)| extension)
            .filter(|extension| self.formats.contains(extension))
            .ok_or("unsupported audio format")?;
        let audio = audio::decode_snapshot(request.input.read(&self.input_dir)?, extension)?;
        model.set_resample_low_rate_drums(request.resample_low_rate_drums);
        let result = model.transcribe(&audio.samples, audio.sample_rate)?;
        let song = result.transcription;
        let files = match request.output {
            MidiOutput::Combined => vec![(None, song.to_midi_bytes()?)],
            MidiOutput::Tracks => song
                .tracks
                .iter()
                .enumerate()
                .map(|(index, track)| Ok((Some(track), song.track_midi_bytes(index)?)))
                .collect::<Result<_, String>>()?,
        };
        let mut outputs = Vec::with_capacity(files.len());
        let mut paths = Vec::with_capacity(files.len());
        for (track, bytes) in files {
            match Snapshot::write(&self.output_dir, "mid", &bytes) {
                Ok(file) => {
                    paths.push(self.output_dir.join(&file.file_name));
                    outputs.push(json!({
                        "track": track.map(|t| json!({"name": t.name, "program": t.program, "is_drum": t.is_drum})),
                        "file": file,
                    }));
                }
                Err(error) => {
                    for path in paths {
                        let _ = fs::remove_file(path);
                    }
                    return Err(error);
                }
            }
        }
        Ok(json!({
            "dettato_version": 1, "duration_seconds": song.seconds, "sample_rate_hz": audio.sample_rate,
            "grouped_notes": result.info.grouped_notes, "drum_hits": result.info.drum_hits,
            "drums_skipped_low_sample_rate": result.info.drums_skipped_low_sample_rate, "outputs": outputs,
        }))
    }
}

fn device(backend: &str, gpu: Option<&str>) -> Result<Device, String> {
    if let Some(id) = gpu {
        let target = backend
            .parse()
            .map_err(|error: gpu_device_resolver::ResolveError| error.to_string())?;
        let id = id
            .parse()
            .map_err(|error: gpu_device_resolver::ResolveError| error.to_string())?;
        return gpu_device_resolver::resolve(target, id)
            .map(|gpu| gpu.device)
            .map_err(|error| error.to_string());
    }
    match backend {
        #[cfg(feature = "flex")]
        "cpu" => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(0)),
        #[cfg(feature = "wgpu")]
        "wgpu" => Ok(Device::wgpu(Default::default())),
        _ => Err(format!("unsupported backend: {backend}")),
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    if args.next().as_deref() != Some("--ipc-stdio") {
        return Err("expected --ipc-stdio".into());
    }
    let mut options = HashMap::new();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("missing value for {flag}"))?;
        if options.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate option {flag}"));
        }
    }
    let mut take = |name: &str| {
        options
            .remove(name)
            .ok_or_else(|| format!("missing {name}"))
    };
    let backend = take("--backend")?;
    let input_dir = absolute_path(Path::new(&take("--input-dir")?), true)?;
    let output_dir = absolute_path(Path::new(&take("--output-dir")?), true)?;
    let mut stage = |name: &str| -> Result<ModelPaths, String> {
        Ok(ModelPaths {
            config: absolute_path(Path::new(&take(&format!("--{name}-config"))?), false)?,
            weights: absolute_path(Path::new(&take(&format!("--{name}-weights"))?), false)?,
        })
    };
    let paths = DettatoModelPaths {
        transcriber: stage("transcriber")?,
        evidence: stage("evidence")?,
        grouper: stage("grouper")?,
        drums: stage("drums")?,
    };
    let gpu_id = options.remove("--gpu-id");
    let cache_mib = options
        .remove("--cache-mib")
        .map(|value| value.parse::<usize>().map_err(|_| "invalid --cache-mib"))
        .transpose()?
        .unwrap_or(256);
    if let Some(flag) = options.keys().next() {
        return Err(format!("unknown option {flag}"));
    }
    let device = device(&backend, gpu_id.as_deref())?;
    let mut worker = Worker {
        paths,
        device,
        backend,
        gpu_id,
        input_dir,
        output_dir,
        cache_mib,
        formats: if cfg!(feature = "extended-audio") {
            &["wav", "flac", "mp3"]
        } else {
            &["wav", "flac"]
        },
        model: None,
        ready: false,
    };
    serve(|name, payload| worker.command(name, payload))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
