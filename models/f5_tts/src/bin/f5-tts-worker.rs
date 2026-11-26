use burn::tensor::Device;
use f5_tts_native::{
    cfm::SamplingConfig,
    hyphenation::EnglishHyphenator,
    pipeline::{CrossLingualF5Runtime, CrossLingualGenerateOptions, StandardF5Runtime},
    reference::ReferenceAudio,
    vocos::VocosConfig,
    wav,
};
use native_worker_ipc::{Snapshot, absolute_path, serve};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    env,
    io::Cursor,
    path::{Path, PathBuf},
    process::ExitCode,
};

const RATE: usize = 24_000;

enum Runtime {
    Russian(Box<StandardF5Runtime>),
    CrossLingual(Box<CrossLingualF5Runtime>),
}

impl Runtime {
    fn synthesize(
        &mut self,
        reference: &ReferenceAudio,
        reference_text: &str,
        text: &str,
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Russian(runtime) => {
                runtime.synthesize_text(reference, reference_text, text, 1.0, None, seed, sampling)
            }
            Self::CrossLingual(runtime) => runtime.synthesize_english(
                reference,
                text,
                &EnglishHyphenator::us_english().map_err(|e| e.to_string())?,
                CrossLingualGenerateOptions {
                    speed: 1.0,
                    fixed_seconds: None,
                    cross_fade_seconds: 0.15,
                    seed,
                    sampling,
                },
            ),
        }
        .map_err(|error| error.to_string())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Infer {
    tts_version: u32,
    text: String,
    reference: Snapshot,
    reference_text: Option<String>,
    seed: u64,
    steps: usize,
}

struct Worker {
    device: Device,
    backend: String,
    model_id: String,
    dit: PathBuf,
    vocos: PathBuf,
    vocab: PathBuf,
    speed: Option<PathBuf>,
    input_dir: PathBuf,
    output_dir: PathBuf,
    runtime: Option<Runtime>,
    ready: bool,
}

impl Worker {
    fn command(&mut self, name: &str, payload: Value) -> Result<Value, String> {
        if name != "infer" && payload != json!({}) {
            return Err("unexpected command payload".into());
        }
        match name {
            "capabilities" => Ok(
                json!({"service":"tts","tts_contract_version":1,"backend":self.backend,
                "model_id":self.model_id,"reference_formats":["wav"],"sample_rate_hz":RATE,
                "requires_reference_text":self.speed.is_none()}),
            ),
            "load" => {
                if self.runtime.is_some() {
                    return Err("model is already loaded".into());
                }
                let runtime = if let Some(speed) = &self.speed {
                    CrossLingualF5Runtime::load(
                        &self.dit,
                        &self.vocos,
                        speed,
                        &self.vocab,
                        VocosConfig::MEL_24KHZ,
                        self.device.clone(),
                    )
                    .map(|runtime| Runtime::CrossLingual(Box::new(runtime)))
                } else {
                    StandardF5Runtime::load(
                        &self.dit,
                        &self.vocos,
                        &self.vocab,
                        VocosConfig::MEL_24KHZ,
                        self.device.clone(),
                    )
                    .map(|runtime| Runtime::Russian(Box::new(runtime)))
                }
                .map_err(|error| error.to_string())?;
                self.device.sync().map_err(|error| error.to_string())?;
                self.runtime = Some(runtime);
                Ok(json!({}))
            }
            "warmup" => {
                let runtime = self.runtime.as_mut().ok_or("model is not loaded")?;
                let samples: Vec<_> = (0..RATE)
                    .map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / RATE as f32).sin() * 0.1)
                    .collect();
                let reference =
                    ReferenceAudio::from_channels(&[&samples], RATE).map_err(|e| e.to_string())?;
                let (prompt, text) = if self.speed.is_some() {
                    ("", "Hello.")
                } else {
                    ("Привет.", "Привет.")
                };
                runtime.synthesize(
                    &reference,
                    prompt,
                    text,
                    0,
                    SamplingConfig {
                        steps: 4,
                        ..Default::default()
                    },
                )?;
                self.device.sync().map_err(|error| error.to_string())?;
                self.ready = true;
                Ok(json!({}))
            }
            "status" => Ok(
                json!({"phase": if self.ready {"ready"} else if self.runtime.is_some() {"loaded"} else {"started"},
                "model_id":self.model_id,"backend":self.backend}),
            ),
            "infer" => {
                let runtime = self
                    .runtime
                    .as_mut()
                    .filter(|_| self.ready)
                    .ok_or("model is not ready")?;
                let request: Infer =
                    serde_json::from_value(payload).map_err(|error| error.to_string())?;
                if request.tts_version != 1
                    || !(2..=128).contains(&request.steps)
                    || request.text.trim().is_empty()
                    || !request.reference.file_name.ends_with(".wav")
                {
                    return Err(
                        "invalid F5-TTS request version, steps, text or reference format".into(),
                    );
                }
                let prompt = request.reference_text.as_deref().unwrap_or("").trim();
                if self.speed.is_none() && prompt.is_empty() {
                    return Err("Russian F5-TTS requires reference_text".into());
                }
                let audio = reference_wav(request.reference.read(&self.input_dir)?)?;
                let pcm = runtime.synthesize(
                    &audio,
                    prompt,
                    &request.text,
                    request.seed,
                    SamplingConfig {
                        steps: request.steps,
                        ..Default::default()
                    },
                )?;
                let output = Snapshot::write(
                    &self.output_dir,
                    "wav",
                    &wav::mono_pcm16(&pcm).map_err(|e| e.to_string())?,
                )?;
                Ok(
                    json!({"tts_version":1,"output":output,"sample_rate_hz":RATE,"samples":pcm.len()}),
                )
            }
            "unload" | "shutdown" => {
                self.runtime = None;
                self.ready = false;
                Ok(json!({}))
            }
            _ => Err("unknown worker command".into()),
        }
    }
}

fn reference_wav(bytes: Vec<u8>) -> Result<ReferenceAudio, String> {
    let mut wav = hound::WavReader::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = wav.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 || channels > 8 || !(1..=384_000).contains(&spec.sample_rate) {
        return Err("invalid reference WAV layout".into());
    }
    let limit = spec.sample_rate as usize * 12 * channels;
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => wav
            .samples::<f32>()
            .take(limit)
            .collect::<Result<Vec<_>, _>>(),
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            wav.samples::<i32>()
                .take(limit)
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect()
        }
    }
    .map_err(|error| error.to_string())?;
    if !samples.len().is_multiple_of(channels) {
        return Err("incomplete reference WAV frame".into());
    }
    let mono = samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect::<Vec<_>>();
    ReferenceAudio::from_channels(&[&mono], spec.sample_rate as usize)
        .map_err(|error| error.to_string())
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
    let model_id = take("--model-id")?;
    let speed = match model_id.as_str() {
        "f5_tts_ru" => None,
        "f5_tts_cross_lingual" => Some(absolute_path(Path::new(&take("--speed-weights")?), false)?),
        _ => return Err(format!("unsupported F5-TTS model: {model_id}")),
    };
    let mut worker = Worker {
        device: match backend.as_str() {
            "cuda" => Device::cuda(0),
            "wgpu" => Device::wgpu(Default::default()),
            _ => return Err(format!("unsupported backend: {backend}")),
        },
        backend,
        model_id,
        speed,
        dit: absolute_path(Path::new(&take("--dit-weights")?), false)?,
        vocos: absolute_path(Path::new(&take("--vocos-weights")?), false)?,
        vocab: absolute_path(Path::new(&take("--vocab")?), false)?,
        input_dir: absolute_path(Path::new(&take("--input-dir")?), true)?,
        output_dir: absolute_path(Path::new(&take("--output-dir")?), true)?,
        runtime: None,
        ready: false,
    };
    if let Some(id) = options.remove("--gpu-id") {
        let error = |error: gpu_device_resolver::ResolveError| error.to_string();
        worker.device = gpu_device_resolver::resolve(
            worker.backend.parse().map_err(error)?,
            id.parse().map_err(error)?,
        )
        .map_err(error)?
        .device;
    }
    if let Some(flag) = options.keys().next() {
        return Err(format!("unknown option {flag}"));
    }
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
