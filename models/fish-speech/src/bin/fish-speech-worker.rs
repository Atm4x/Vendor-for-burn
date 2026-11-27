use fish_speech_core::{cpu_weighted_ar::SelectionMode, synthesis::SynthesisRequest};
use fish_speech_vendor::{
    cubecl_backend::CubeClRequestLimits,
    model_runtime::{CubeClFishModel, LoadedFishArtifacts, ModelPaths},
    reference_cache::{CachePolicy, ReferenceBytes},
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Infer {
    tts_version: u32,
    text: String,
    reference: Snapshot,
    reference_text: String,
    max_new_tokens: usize,
}

struct Worker {
    backend: String,
    gpu_index: usize,
    seed: u64,
    paths: ModelPaths,
    semantic_weights: Option<PathBuf>,
    firefly_weights: Option<PathBuf>,
    input_dir: PathBuf,
    output_dir: PathBuf,
    model: Option<CubeClFishModel>,
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
                "model_id":"fish_speech_1_5","reference_formats":["wav"],"requires_reference_text":true}),
            ),
            "load" => {
                if self.model.is_some() {
                    return Err("model is already loaded".into());
                }
                let weights = LoadedFishArtifacts::open_with_weights(
                    &self.paths,
                    self.semantic_weights.as_deref(),
                    self.firefly_weights.as_deref(),
                )
                .map_err(|error| error.to_string())?;
                let device = match self.backend.as_str() {
                    "cuda" => {
                        cubecl::Device::cuda(self.gpu_index).map_err(|error| error.to_string())?
                    }
                    "wgpu" => cubecl::Device::Wgpu(
                        cubecl::wgpu::WgpuDevice::new(cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(
                            self.gpu_index,
                        ))
                        .on(cubecl::wgpu::WgpuBackend::Vulkan),
                    ),
                    _ => return Err("invalid worker backend".into()),
                };
                self.model = Some(
                    weights
                        .into_cubecl_model(
                            device.client(),
                            self.seed,
                            SelectionMode::Sample,
                            CubeClRequestLimits::default(),
                        )
                        .map_err(|error| error.to_string())?,
                );
                Ok(json!({}))
            }
            "warmup" => {
                let model = self.model.as_mut().ok_or("model is not loaded")?;
                synthesize(model, "Hello.", &[], 8)?;
                self.ready = true;
                Ok(json!({}))
            }
            "status" => Ok(
                json!({"phase": if self.ready {"ready"} else if self.model.is_some() {"loaded"} else {"started"},
                "backend":self.backend,"gpu_index":self.gpu_index,"seed":self.seed}),
            ),
            "infer" => {
                let model = self
                    .model
                    .as_mut()
                    .filter(|_| self.ready)
                    .ok_or("model is not ready")?;
                let request: Infer =
                    serde_json::from_value(payload).map_err(|error| error.to_string())?;
                if request.tts_version != 1
                    || !(1..=2048).contains(&request.max_new_tokens)
                    || request.text.trim().is_empty()
                    || request.reference_text.trim().is_empty()
                    || !request.reference.file_name.ends_with(".wav")
                {
                    return Err(
                        "invalid Fish Speech request version, text, length or reference format"
                            .into(),
                    );
                }
                let wav = request.reference.read(&self.input_dir)?;
                if !wav.starts_with(b"RIFF") || wav.get(8..12) != Some(b"WAVE") {
                    return Err("reference is not a WAV file".into());
                }
                let references = model
                    .prepare_references(
                        &[ReferenceBytes {
                            audio: &wav,
                            text: &request.reference_text,
                        }],
                        CachePolicy::On,
                    )
                    .map_err(|error| error.to_string())?;
                let audio = synthesize(model, &request.text, &references, request.max_new_tokens)?;
                if audio.pcm.is_empty() || audio.pcm.iter().any(|sample| !sample.is_finite()) {
                    return Err("invalid generated audio".into());
                }
                let bytes = encode_wav(&audio.pcm, audio.sample_rate as u32)?;
                let file = Snapshot::write(&self.output_dir, "wav", &bytes)?;
                Ok(
                    json!({"tts_version":1,"output":file,"sample_rate_hz":audio.sample_rate,"samples":audio.pcm.len()}),
                )
            }
            "unload" | "shutdown" => {
                self.model = None;
                self.ready = false;
                Ok(json!({}))
            }
            _ => Err("unknown worker command".into()),
        }
    }
}

fn synthesize(
    model: &mut CubeClFishModel,
    text: &str,
    references: &[fish_speech_core::prompt::ReferenceExample],
    max_new_tokens: usize,
) -> Result<fish_speech_core::synthesis::SynthesisSample, String> {
    let config = model.config().map_err(|error| error.to_string())?;
    let request = SynthesisRequest {
        text,
        references,
        num_samples: 1,
        num_codebooks: config.num_codebooks,
        iterative_prompt: false,
        chunk_length: 100,
        max_context_length: 2048,
        max_new_tokens,
        sampling: Default::default(),
    };
    model
        .synthesize(&request, || false, false, |_| Ok(()))
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or("Fish Speech returned no sample".into())
}

fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let mut output = Cursor::new(Vec::new());
    let result = (|| {
        let mut writer = hound::WavWriter::new(
            &mut output,
            hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for sample in samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
        }
        writer.finalize()
    })();
    result.map_err(|error| error.to_string())?;
    Ok(output.into_inner())
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
    let legacy_layout =
        options.contains_key("--semantic-dir") || options.contains_key("--firefly-dir");
    let mut take = |name: &str| {
        options
            .remove(name)
            .ok_or_else(|| format!("missing {name}"))
    };
    let backend = take("--backend")?;
    if !matches!(backend.as_str(), "cuda" | "wgpu") {
        return Err(format!("unsupported backend: {backend}"));
    }
    let firefly_config = absolute_path(Path::new(&take("--firefly-config")?), false)?;
    let (paths, semantic_weights, firefly_weights) = if legacy_layout {
        (
            ModelPaths::new(
                absolute_path(Path::new(&take("--semantic-dir")?), true)?,
                absolute_path(Path::new(&take("--firefly-dir")?), true)?,
                firefly_config,
            ),
            None,
            None,
        )
    } else {
        let semantic_manifest = absolute_path(Path::new(&take("--semantic-manifest")?), false)?;
        let firefly_manifest = absolute_path(Path::new(&take("--firefly-manifest")?), false)?;
        let semantic_dir = semantic_manifest
            .parent()
            .ok_or("invalid semantic manifest location")?
            .to_owned();
        let firefly_dir = firefly_manifest
            .parent()
            .ok_or("invalid Firefly manifest location")?
            .to_owned();
        (
            ModelPaths {
                semantic_config: absolute_path(Path::new(&take("--semantic-config")?), false)?,
                semantic_tokenizer: absolute_path(
                    Path::new(&take("--semantic-tokenizer")?),
                    false,
                )?,
                semantic_dir,
                semantic_manifest,
                firefly_config_source: firefly_config,
                firefly_dir,
                firefly_manifest,
            },
            Some(absolute_path(
                Path::new(&take("--semantic-weights")?),
                false,
            )?),
            Some(absolute_path(
                Path::new(&take("--firefly-weights")?),
                false,
            )?),
        )
    };
    let input_dir = absolute_path(Path::new(&take("--input-dir")?), true)?;
    let output_dir = absolute_path(Path::new(&take("--output-dir")?), true)?;
    let gpu_index = options
        .remove("--gpu-index")
        .map(|value| value.parse::<usize>().map_err(|_| "invalid --gpu-index"))
        .transpose()?
        .unwrap_or(0);
    let seed = options
        .remove("--seed")
        .map(|value| value.parse::<u64>().map_err(|_| "invalid --seed"))
        .transpose()?
        .unwrap_or(42);
    if let Some(flag) = options.keys().next() {
        return Err(format!("unknown option {flag}"));
    }
    let mut worker = Worker {
        backend,
        gpu_index,
        seed,
        paths,
        semantic_weights,
        firefly_weights,
        input_dir,
        output_dir,
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
