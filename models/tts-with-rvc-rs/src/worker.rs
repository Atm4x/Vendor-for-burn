use crate::{
    chunked::ChunkConfig,
    conversion::{ConversionReport, ConversionSettings},
    runtime_backend,
    rvc_inference::{InferenceOptions, RvcV2Inferencer},
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

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME: usize = 1024 * 1024;
pub const MAX_SOURCE: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub name: String,
    pub backend: String,
    pub backend_path: Option<PathBuf>,
    pub runtime_dlls: Vec<PathBuf>,
    pub gpu_id: Option<String>,
    pub voice: PathBuf,
    pub hubert: PathBuf,
    pub rmvpe: PathBuf,
    pub index: Option<PathBuf>,
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub batch_size: usize,
    pub chunk: ChunkConfig,
}

impl LaunchConfig {
    fn validate(&self) -> Result<(), String> {
        if self.name.is_empty()
            || self.name.len() > 80
            || !self
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err("worker name must contain 1..80 ASCII letters, digits, '_' or '-'".into());
        }
        if self.backend == "cpu" && (self.backend_path.is_some() || !self.runtime_dlls.is_empty()) {
            return Err("external GPU runtime DLLs are only supported for GPU backends".into());
        }
        if !matches!(self.backend.as_str(), "cpu" | "cuda" | "wgpu") {
            return Err("unsupported backend".into());
        }
        if let Some(id) = &self.gpu_id {
            if self.backend == "cpu" {
                return Err("--gpu-id is only supported for cuda or wgpu".into());
            }
            id.parse::<gpu_device_resolver::GpuId>()
                .map_err(|e| e.to_string())?;
        }
        if !(1..=32).contains(&self.batch_size) {
            return Err("batch_size must be 1..=32".into());
        }
        self.chunk.validate()?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub version: u32,
    pub id: Uuid,
    pub command: String,
    pub payload: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    file_name: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InferRequest {
    input: Snapshot,
    #[serde(default)]
    index_rate: Option<f32>,
    #[serde(default)]
    pitch_shift_semitones: Option<f32>,
    #[serde(default)]
    rms_mix_rate: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SwitchVoiceRequest {
    voice: String,
    #[serde(default)]
    index: Option<String>,
}

pub trait WorkerEngine: Sized {
    fn load(config: &LaunchConfig) -> Result<Self, String>;
    fn switch_voice(&mut self, voice: &Path, index: Option<&Path>) -> Result<(), String>;
    fn warmup(&mut self) -> Result<(), String>;
    fn infer(
        &mut self,
        input: &[u8],
        settings: &ConversionSettings,
    ) -> Result<(Vec<u8>, ConversionReport), String>;
}

pub struct NativeRvcEngine(RvcV2Inferencer);

impl WorkerEngine for NativeRvcEngine {
    fn load(config: &LaunchConfig) -> Result<Self, String> {
        let device =
            runtime_backend::selected_physical_device(&config.backend, config.gpu_id.as_deref())?;
        let voice = config.voice.to_string_lossy();
        let hubert = config.hubert.to_string_lossy();
        let rmvpe = config.rmvpe.to_string_lossy();
        let index = config.index.as_ref().map(|p| p.to_string_lossy());
        let model =
            RvcV2Inferencer::load(&voice, &hubert, &rmvpe, index.as_deref(), device.clone())?;
        device
            .sync()
            .map_err(|e| format!("model load sync failed: {e}"))?;
        Ok(Self(model))
    }

    fn switch_voice(&mut self, voice: &Path, index: Option<&Path>) -> Result<(), String> {
        self.0.switch_voice(
            &voice.to_string_lossy(),
            index.map(|p| p.to_string_lossy()).as_deref(),
        )
    }

    fn warmup(&mut self) -> Result<(), String> {
        let options = InferenceOptions {
            index_rate: 0.0,
            latent_noise_scale: 0.0,
            ..InferenceOptions::default()
        };
        let output = self.0.infer_16khz(&vec![0.0; 16_000], &options)?;
        if output.is_empty() || output.iter().any(|value| !value.is_finite()) {
            return Err("RVC warmup produced invalid audio".into());
        }
        Ok(())
    }

    fn infer(
        &mut self,
        input: &[u8],
        settings: &ConversionSettings,
    ) -> Result<(Vec<u8>, ConversionReport), String> {
        crate::conversion::convert_wav_bytes(&self.0, input, settings, |_| {})
    }
}

pub struct Worker<E: WorkerEngine> {
    config: LaunchConfig,
    engine: Option<E>,
    warm: bool,
}

impl<E: WorkerEngine> Worker<E> {
    pub fn new(config: LaunchConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            engine: None,
            warm: false,
        })
    }

    fn command(&mut self, request: &Request) -> Result<Value, String> {
        if !matches!(request.command.as_str(), "infer" | "set_voice")
            && request.payload != json!({})
        {
            return Err("unexpected command payload".into());
        }
        match request.command.as_str() {
            "capabilities" => Ok(json!({
                "name": self.config.name,
                "backends": [self.config.backend],
                "rvc_capability": {
                    "kind": "voice_conversion", "formats": ["wav"],
                    "output_formats": ["wav"], "output_sample_rate_hz": 40_000,
                    "max_source_bytes": MAX_SOURCE,
                    "physical_gpu_selectors": ["pci", "windows_luid"],
                    "parameters": ["index_rate", "pitch_shift_semitones", "rms_mix_rate"],
                    "hot_voice_switch": true,
                },
            })),
            "status" => Ok(json!({
                "phase": if self.warm { "ready" } else if self.engine.is_some() { "loaded" } else { "started" },
                "name": self.config.name,
                "backend": self.config.backend,
                "gpu_id": self.config.gpu_id,
                "voice": self.config.voice,
                "index": self.config.index,
                "variant": match self.config.backend.as_str() {
                    "cuda" => "burn_cubecl_cuda_fp32",
                    "wgpu" => "burn_cubecl_wgpu_fp32",
                    _ => "burn_cubecl_cpu_fp32",
                },
            })),
            "load" => {
                if self.engine.is_some() {
                    return Err("model is already loaded; unload before loading again".into());
                }
                let model = E::load(&self.config)?;
                self.engine = Some(model);
                self.warm = false;
                Ok(json!({"loaded": true}))
            }
            "warmup" => {
                let model = self.engine.as_mut().ok_or("model is not loaded")?;
                self.warm = false;
                model.warmup()?;
                self.warm = true;
                Ok(json!({"warm": true}))
            }
            "infer" => self.infer(request),
            "set_voice" => self.set_voice(request),
            "unload" | "shutdown" => {
                self.engine = None;
                self.warm = false;
                Ok(json!({}))
            }
            _ => Err("unknown worker command".into()),
        }
    }

    fn set_voice(&mut self, request: &Request) -> Result<Value, String> {
        let engine = self.engine.as_mut().ok_or("model is not loaded")?;
        let args: SwitchVoiceRequest =
            serde_json::from_value(request.payload.clone()).map_err(|e| e.to_string())?;
        let voice = absolute_regular_file(args.voice)?;
        let index = args.index.map(absolute_regular_file).transpose()?;
        engine.switch_voice(&voice, index.as_deref())?;
        self.config.voice = voice;
        self.config.index = index;
        self.warm = false;
        Ok(
            json!({"loaded": true, "warm": false, "voice": self.config.voice, "index": self.config.index}),
        )
    }

    fn infer(&mut self, request: &Request) -> Result<Value, String> {
        let model = self
            .engine
            .as_mut()
            .filter(|_| self.warm)
            .ok_or("model is not ready")?;
        let args: InferRequest =
            serde_json::from_value(request.payload.clone()).map_err(|e| e.to_string())?;
        let has_index = self.config.index.is_some();
        let index_rate = args
            .index_rate
            .unwrap_or(if has_index { 0.75 } else { 0.0 });
        let pitch_shift_semitones = args.pitch_shift_semitones.unwrap_or(0.0);
        let rms_mix_rate = args.rms_mix_rate.unwrap_or(0.5);
        if !index_rate.is_finite()
            || !(0.0..=1.0).contains(&index_rate)
            || (index_rate > 0.0 && !has_index)
        {
            return Err("index_rate must be 0..=1 and requires index.index when nonzero".into());
        }
        if !pitch_shift_semitones.is_finite()
            || !rms_mix_rate.is_finite()
            || !(0.0..=1.0).contains(&rms_mix_rate)
        {
            return Err("invalid pitch_shift_semitones or rms_mix_rate".into());
        }
        let input = read_snapshot(&self.config.input_dir, &args.input)?;
        if !input.starts_with(b"RIFF") || input.get(8..12) != Some(b"WAVE") {
            return Err("input bytes do not match WAV format".into());
        }
        let settings = ConversionSettings {
            inference: InferenceOptions {
                index_rate,
                pitch_shift_semitones,
                ..InferenceOptions::default()
            },
            chunk: self.config.chunk,
            batch_size: self.config.batch_size,
            rms_mix_rate,
        };
        let (wav, report) = model.infer(&input, &settings)?;
        if !wav.starts_with(b"RIFF") || wav.get(8..12) != Some(b"WAVE") {
            return Err("inference returned invalid WAV output".into());
        }
        let file_name = format!("{}.wav", request.id.hyphenated());
        let path = self.config.output_dir.join(&file_name);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| format!("cannot create output snapshot: {e}"))?;
        if let Err(err) = file.write_all(&wav).and_then(|_| file.sync_all()) {
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(format!("cannot write output snapshot: {err}"));
        }
        Ok(json!({
            "input_sample_rate_hz": report.input_sample_rate,
            "input_samples": report.input_samples,
            "sample_rate_hz": report.output_sample_rate,
            "output_samples": report.output_samples,
            "output": {
                "file_name": file_name,
                "size_bytes": wav.len(),
                "sha256": format!("{:x}", Sha256::digest(&wav)),
            },
        }))
    }
}

fn read_snapshot(root: &Path, info: &Snapshot) -> Result<Vec<u8>, String> {
    let name = info.file_name.as_str();
    if !name.ends_with(".wav")
        || name.len() > 128
        || name.starts_with('.')
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        || !(1..=MAX_SOURCE).contains(&info.size_bytes)
        || info.sha256.len() != 64
        || !info
            .sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("invalid input snapshot".into());
    }
    let path = root.join(name);
    let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err("input snapshot must be a regular file".into());
    }
    if path.canonicalize().map_err(|e| e.to_string())?.parent() != Some(root) {
        return Err("input snapshot escaped input directory".into());
    }
    let mut data = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take(info.size_bytes + 1)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() as u64 != info.size_bytes || format!("{:x}", Sha256::digest(&data)) != info.sha256
    {
        return Err("input snapshot size or SHA-256 mismatch".into());
    }
    Ok(data)
}

fn absolute_directory(value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("worker directory must be absolute".into());
    }
    let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("worker directory must be a real directory".into());
    }
    path.canonicalize().map_err(|e| e.to_string())
}

fn absolute_regular_file(value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("weight file path must be absolute".into());
    }
    let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err("weight file must be a regular non-symlink file".into());
    }
    path.canonicalize().map_err(|e| e.to_string())
}

fn absolute_backend_path(value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("backend runtime path must be absolute".into());
    }
    let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if metadata.file_type().is_symlink() || !(metadata.is_file() || metadata.is_dir()) {
        return Err("backend runtime path must be a regular file or directory".into());
    }
    path.canonicalize().map_err(|e| e.to_string())
}

pub fn parse_launch(mut args: impl Iterator<Item = String>) -> Result<LaunchConfig, String> {
    if args.next().as_deref() != Some("--ipc-stdio") {
        return Err("worker must be launched with --ipc-stdio".into());
    }
    let mut name = None;
    let mut backend = None;
    let mut backend_path = None;
    let mut runtime_dlls = Vec::new();
    let mut gpu_id = None;
    let mut voice = None;
    let mut hubert = None;
    let mut rmvpe = None;
    let mut index = None;
    let mut input_dir = None;
    let mut output_dir = None;
    let mut batch_size = None;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("missing worker launch argument value")?;
        match flag.as_str() {
            "--name" if name.is_none() => name = Some(value),
            "--backend" if backend.is_none() => backend = Some(value),
            "--backend-path" if backend_path.is_none() => {
                backend_path = Some(absolute_backend_path(value)?)
            }
            "--gpu-id" if gpu_id.is_none() => gpu_id = Some(value),
            "--runtime-dll" => runtime_dlls.push(absolute_backend_path(value)?),
            "--voice" if voice.is_none() => voice = Some(absolute_regular_file(value)?),
            "--hubert" if hubert.is_none() => hubert = Some(absolute_regular_file(value)?),
            "--rmvpe" if rmvpe.is_none() => rmvpe = Some(absolute_regular_file(value)?),
            "--index" if index.is_none() => index = Some(absolute_regular_file(value)?),
            "--input-dir" if input_dir.is_none() => input_dir = Some(absolute_directory(value)?),
            "--output-dir" if output_dir.is_none() => output_dir = Some(absolute_directory(value)?),
            "--batch-size" if batch_size.is_none() => {
                batch_size = Some(
                    value
                        .parse::<usize>()
                        .map_err(|e| format!("invalid batch size: {e}"))?,
                )
            }
            _ => return Err("unknown or duplicate worker launch argument".into()),
        }
    }
    let config = LaunchConfig {
        name: name.unwrap_or_else(|| "rvc_v2".into()),
        backend: backend.ok_or("missing --backend")?,
        backend_path,
        runtime_dlls,
        gpu_id,
        voice: voice.ok_or("missing --voice")?,
        hubert: hubert.ok_or("missing --hubert")?,
        rmvpe: rmvpe.ok_or("missing --rmvpe")?,
        index,
        input_dir: input_dir.ok_or("missing --input-dir")?,
        output_dir: output_dir.ok_or("missing --output-dir")?,
        batch_size: batch_size.unwrap_or(2),
        chunk: ChunkConfig::default(),
    };
    config.validate()?;
    Ok(config)
}

pub fn serve<E: WorkerEngine>(
    config: LaunchConfig,
    mut input: impl BufRead,
    mut output: impl Write,
) -> Result<(), String> {
    let mut worker = Worker::<E>::new(config)?;
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
        if request.version != PROTOCOL_VERSION {
            return Err("unsupported worker protocol version".into());
        }
        let result = worker.command(&request);
        let response = match &result {
            Ok(value) => {
                json!({"version": PROTOCOL_VERSION, "id": request.id, "ok": true, "result": value, "error": null})
            }
            Err(error) => {
                json!({"version": PROTOCOL_VERSION, "id": request.id, "ok": false, "result": null, "error": error})
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

pub fn run_native(args: impl Iterator<Item = String>) -> Result<(), String> {
    let config = parse_launch(args)?;
    runtime_backend::configure_native_libraries(
        &config.backend,
        config.backend_path.as_deref(),
        &config.runtime_dlls,
    )?;
    let _ = runtime_backend::selected_physical_device(&config.backend, config.gpu_id.as_deref())?;
    serve::<NativeRvcEngine>(config, io::stdin().lock(), io::stdout().lock())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_io;
    use std::io::Cursor;

    struct MockEngine {
        voice: PathBuf,
        index: Option<PathBuf>,
    }
    impl WorkerEngine for MockEngine {
        fn load(config: &LaunchConfig) -> Result<Self, String> {
            Ok(Self {
                voice: config.voice.clone(),
                index: config.index.clone(),
            })
        }
        fn switch_voice(&mut self, voice: &Path, index: Option<&Path>) -> Result<(), String> {
            if voice.file_name().is_some_and(|name| name == "reject.pth") {
                return Err("invalid voice weights".into());
            }
            self.voice = voice.to_path_buf();
            self.index = index.map(Path::to_path_buf);
            Ok(())
        }
        fn warmup(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn infer(
            &mut self,
            _input: &[u8],
            _settings: &ConversionSettings,
        ) -> Result<(Vec<u8>, ConversionReport), String> {
            Ok((
                audio_io::encode_mono_wav(&[0.0; 400], 40_000)?,
                ConversionReport {
                    input_samples: 160,
                    input_sample_rate: 16_000,
                    output_samples: 400,
                    output_sample_rate: 40_000,
                },
            ))
        }
    }

    #[test]
    fn gpu_id_is_validated_and_cpu_does_not_accept_one() {
        let path = std::env::temp_dir();
        let mut config = config(&path);
        config.gpu_id = Some("pci:0000:01:00.0".into());
        assert!(config.validate().unwrap_err().contains("only supported"));
        config.backend = "cuda".into();
        assert!(config.validate().is_ok());
        config.gpu_id = Some("not-a-physical-gpu".into());
        assert!(config.validate().unwrap_err().contains("invalid physical"));
    }

    const ID: &str = "d693ac39-dd75-48f8-b5fb-5db199110159";
    fn config(root: &Path) -> LaunchConfig {
        LaunchConfig {
            name: "rvc_v2".into(),
            backend: "cpu".into(),
            backend_path: None,
            runtime_dlls: Vec::new(),
            gpu_id: None,
            voice: root.join("voice-any.pth"),
            hubert: root.join("shared-hubert.pt"),
            rmvpe: root.join("shared-rmvpe.pt"),
            index: None,
            input_dir: root.into(),
            output_dir: root.into(),
            batch_size: 2,
            chunk: ChunkConfig::default(),
        }
    }
    fn invoke(
        worker: &mut Worker<MockEngine>,
        command: &str,
        payload: Value,
    ) -> Result<Value, String> {
        worker.command(&Request {
            version: 1,
            id: Uuid::parse_str(ID).unwrap(),
            command: command.into(),
            payload,
        })
    }
    #[test]
    fn lifecycle_one_model_and_ready_gate() {
        let path = std::env::temp_dir();
        let mut w = Worker::<MockEngine>::new(config(&path)).unwrap();
        assert_eq!(
            invoke(&mut w, "status", json!({})).unwrap()["phase"],
            "started"
        );
        assert!(invoke(&mut w, "warmup", json!({})).is_err());
        assert!(invoke(&mut w, "infer", json!({})).is_err());
        invoke(&mut w, "load", json!({})).unwrap();
        assert!(invoke(&mut w, "load", json!({})).is_err());
        assert_eq!(
            invoke(&mut w, "status", json!({})).unwrap()["phase"],
            "loaded"
        );
        invoke(&mut w, "warmup", json!({})).unwrap();
        assert_eq!(
            invoke(&mut w, "status", json!({})).unwrap()["phase"],
            "ready"
        );
        invoke(&mut w, "unload", json!({})).unwrap();
        assert_eq!(
            invoke(&mut w, "status", json!({})).unwrap()["phase"],
            "started"
        );
    }
    #[test]
    fn rejects_bad_snapshot_hash_and_directory_escape() {
        let root = std::env::temp_dir().join(format!("rvc-worker-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("input.wav");
        let wav = audio_io::encode_mono_wav(&[0.0; 160], 16_000).unwrap();
        fs::write(&path, &wav).unwrap();
        let hash = format!("{:x}", Sha256::digest(&wav));
        let valid = Snapshot {
            file_name: "input.wav".into(),
            size_bytes: wav.len() as u64,
            sha256: hash,
        };
        assert_eq!(
            read_snapshot(&root.canonicalize().unwrap(), &valid).unwrap(),
            wav
        );
        assert!(
            read_snapshot(
                &root,
                &Snapshot {
                    sha256: "0".repeat(64),
                    ..valid
                }
            )
            .is_err()
        );
        assert!(
            read_snapshot(
                &root,
                &Snapshot {
                    file_name: "../input.wav".into(),
                    size_bytes: 1,
                    sha256: "0".repeat(64)
                }
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn verified_infer_creates_exclusive_hashed_output() {
        let root = std::env::temp_dir().join(format!("rvc-worker-output-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let input = audio_io::encode_mono_wav(&[0.0; 3200], 16_000).unwrap();
        fs::write(root.join("sample.wav"), &input).unwrap();
        let mut worker = Worker::<MockEngine>::new(config(&root.canonicalize().unwrap())).unwrap();
        invoke(&mut worker, "load", json!({})).unwrap();
        invoke(&mut worker, "warmup", json!({})).unwrap();
        let payload = json!({
            "input":{
                "file_name":"sample.wav", "size_bytes":input.len(),
                "sha256":format!("{:x}",Sha256::digest(&input))
            }
        });
        let result = invoke(&mut worker, "infer", payload.clone()).unwrap();
        assert_eq!(result["sample_rate_hz"], 40_000);
        assert_eq!(result["output_samples"], 400);
        let output = root.join(result["output"]["file_name"].as_str().unwrap());
        let encoded = fs::read(&output).unwrap();
        assert_eq!(
            encoded.len(),
            result["output"]["size_bytes"].as_u64().unwrap() as usize
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&encoded)),
            result["output"]["sha256"]
        );
        assert!(
            invoke(&mut worker, "infer", payload).is_err(),
            "request ID reuse must not overwrite output"
        );
        assert_eq!(fs::read(&output).unwrap(), encoded);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn invalid_frame_length_and_snapshot_are_rejected() {
        let config = config(&std::env::temp_dir());
        let large = vec![b'x'; MAX_FRAME + 1];
        assert!(serve::<MockEngine>(config, Cursor::new(large), Vec::<u8>::new()).is_err());
        let root = std::env::temp_dir().join(format!("rvc-worker-snapshot-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let data = b"RIFFxxxxWAVE";
        fs::write(root.join("sample.wav"), data).unwrap();
        let sample = Snapshot {
            file_name: "sample.wav".into(),
            size_bytes: data.len() as u64,
            sha256: format!("{:x}", Sha256::digest(data)),
        };
        assert_eq!(
            read_snapshot(&root.canonicalize().unwrap(), &sample).unwrap(),
            data
        );
        assert!(
            read_snapshot(
                &root.canonicalize().unwrap(),
                &Snapshot {
                    size_bytes: sample.size_bytes + 1,
                    ..sample
                }
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn frame_and_version_errors_are_fatal_but_command_errors_are_not() {
        let config = config(&std::env::temp_dir());
        let input = format!(
            "{{\"version\":1,\"id\":\"{ID}\",\"command\":\"unknown\",\"payload\":{{}}}}\n{{\"version\":1,\"id\":\"{ID}\",\"command\":\"status\",\"payload\":{{}}}}\n"
        );
        let mut output = Vec::new();
        serve::<MockEngine>(config.clone(), Cursor::new(input), &mut output).unwrap();
        let lines: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["ok"], false);
        assert_eq!(lines[1]["result"]["phase"], "started");
        let wrong =
            format!("{{\"version\":2,\"id\":\"{ID}\",\"command\":\"status\",\"payload\":{{}}}}\n");
        assert!(serve::<MockEngine>(config, Cursor::new(wrong), Vec::<u8>::new()).is_err());
    }
    #[test]
    fn explicit_assets_and_optional_name_are_accepted() {
        let root = std::env::temp_dir().join(format!("rvc-worker-assets-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        for name in [
            "singer.pth",
            "my-hubert.pt",
            "pitch-model.pt",
            "custom.index",
        ] {
            fs::write(root.join(name), b"fixture").unwrap();
        }
        let dir = root.canonicalize().unwrap();
        let args = [
            "--ipc-stdio".to_owned(),
            "--backend".into(),
            "cpu".into(),
            "--voice".into(),
            dir.join("singer.pth").display().to_string(),
            "--hubert".into(),
            dir.join("my-hubert.pt").display().to_string(),
            "--rmvpe".into(),
            dir.join("pitch-model.pt").display().to_string(),
            "--index".into(),
            dir.join("custom.index").display().to_string(),
            "--input-dir".into(),
            dir.display().to_string(),
            "--output-dir".into(),
            dir.display().to_string(),
        ];
        let config = parse_launch(args.into_iter()).unwrap();
        assert_eq!(config.name, "rvc_v2");
        assert_eq!(config.voice.file_name().unwrap(), "singer.pth");
        assert_eq!(
            config.index.as_deref().unwrap().file_name().unwrap(),
            "custom.index"
        );
        assert!(
            parse_launch(
                [
                    "--ipc-stdio".to_string(),
                    "--backend".into(),
                    "cpu".into(),
                    "--model-id".into(),
                    "rvc_v2_x".into(),
                ]
                .into_iter()
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn multiple_external_runtime_dlls_are_accepted_at_launch() {
        let root = std::env::temp_dir().join(format!("rvc-worker-dlls-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        for name in [
            "voice.pth",
            "hubert.pt",
            "rmvpe.pt",
            "nvrtc64_120_0.dll",
            "cudart64_12.dll",
        ] {
            fs::write(root.join(name), b"fixture").unwrap();
        }
        let dir = root.canonicalize().unwrap();
        let mut args = vec!["--ipc-stdio".into(), "--backend".into(), "cuda".into()];
        for name in ["voice", "hubert", "rmvpe"] {
            args.push(format!("--{name}"));
            args.push(
                dir.join(format!(
                    "{name}.{}",
                    if name == "voice" { "pth" } else { "pt" }
                ))
                .display()
                .to_string(),
            );
        }
        for dll in ["nvrtc64_120_0.dll", "cudart64_12.dll"] {
            args.push("--runtime-dll".into());
            args.push(dir.join(dll).display().to_string());
        }
        for flag in ["--input-dir", "--output-dir"] {
            args.push(flag.into());
            args.push(dir.display().to_string());
        }
        let config = parse_launch(args.into_iter()).unwrap();
        assert_eq!(config.runtime_dlls.len(), 2);
        assert_eq!(config.name, "rvc_v2");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn backend_path_rejected_for_cpu() {
        let path = std::env::temp_dir();
        let mut config = config(&path);
        config.backend_path = Some(path);
        assert!(config.validate().unwrap_err().contains("GPU"));
    }

    #[test]
    fn set_voice_preserves_shared_runtime_and_requires_new_warmup() {
        let root = std::env::temp_dir().join(format!("rvc-worker-voices-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        for name in ["first.pth", "second.pth", "second.index"] {
            fs::write(root.join(name), b"mock fixture").unwrap();
        }
        let root = root.canonicalize().unwrap();
        let mut config = config(&root);
        config.voice = root.join("first.pth");
        let mut worker = Worker::<MockEngine>::new(config).unwrap();
        assert!(
            invoke(
                &mut worker,
                "set_voice",
                json!({
                    "voice": root.join("second.pth"), "index": root.join("second.index")
                })
            )
            .is_err()
        );
        invoke(&mut worker, "load", json!({})).unwrap();
        invoke(&mut worker, "warmup", json!({})).unwrap();
        let changed = invoke(
            &mut worker,
            "set_voice",
            json!({
                "voice": root.join("second.pth"), "index": root.join("second.index")
            }),
        )
        .unwrap();
        assert_eq!(changed["warm"], false);
        assert_eq!(
            invoke(&mut worker, "status", json!({})).unwrap()["phase"],
            "loaded"
        );
        assert_eq!(
            worker.engine.as_ref().unwrap().voice,
            root.join("second.pth")
        );
        assert_eq!(
            worker.engine.as_ref().unwrap().index,
            Some(root.join("second.index"))
        );
        assert_eq!(worker.config.hubert, root.join("shared-hubert.pt"));
        assert_eq!(worker.config.rmvpe, root.join("shared-rmvpe.pt"));
        assert!(invoke(&mut worker, "infer", json!({})).is_err());
        invoke(&mut worker, "warmup", json!({})).unwrap();
        assert_eq!(
            invoke(&mut worker, "status", json!({})).unwrap()["phase"],
            "ready"
        );
        let status = invoke(&mut worker, "status", json!({})).unwrap();
        assert_eq!(
            status["voice"],
            root.join("second.pth").display().to_string()
        );
        assert_eq!(
            status["index"],
            root.join("second.index").display().to_string()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn set_voice_failure_keeps_previous_voice_ready_and_index_does_not_leak() {
        let root =
            std::env::temp_dir().join(format!("rvc-worker-voice-failure-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        for name in ["first.pth", "second.pth", "reject.pth", "old.index"] {
            fs::write(root.join(name), b"mock fixture").unwrap();
        }
        let root = root.canonicalize().unwrap();
        let mut config = config(&root);
        config.voice = root.join("first.pth");
        config.index = Some(root.join("old.index"));
        let mut worker = Worker::<MockEngine>::new(config).unwrap();
        invoke(&mut worker, "load", json!({})).unwrap();
        invoke(&mut worker, "warmup", json!({})).unwrap();
        assert!(
            invoke(
                &mut worker,
                "set_voice",
                json!({
                    "voice": root.join("missing.pth")
                })
            )
            .is_err()
        );
        assert!(
            invoke(
                &mut worker,
                "set_voice",
                json!({
                    "voice": root.join("reject.pth")
                })
            )
            .unwrap_err()
            .contains("invalid voice weights")
        );
        assert!(
            invoke(
                &mut worker,
                "set_voice",
                json!({
                    "voice": "relative.pth"
                })
            )
            .is_err()
        );
        assert!(
            invoke(
                &mut worker,
                "set_voice",
                json!({
                    "voice": root.join("second.pth"), "unknown": true
                })
            )
            .is_err()
        );
        assert_eq!(worker.config.voice, root.join("first.pth"));
        assert_eq!(worker.config.index, Some(root.join("old.index")));
        assert_eq!(
            invoke(&mut worker, "status", json!({})).unwrap()["phase"],
            "ready"
        );
        let switched = invoke(
            &mut worker,
            "set_voice",
            json!({
                "voice": root.join("second.pth")
            }),
        )
        .unwrap();
        assert!(switched["index"].is_null());
        assert!(worker.config.index.is_none());
        assert!(worker.engine.as_ref().unwrap().index.is_none());
        invoke(&mut worker, "unload", json!({})).unwrap();
        invoke(&mut worker, "load", json!({})).unwrap();
        assert_eq!(
            worker.engine.as_ref().unwrap().voice,
            root.join("second.pth")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn set_voice_ipc_keeps_stdout_protocol_only() {
        let root =
            std::env::temp_dir().join(format!("rvc-worker-voice-framing-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("second.pth"), b"mock fixture").unwrap();
        let config = config(&root.canonicalize().unwrap());
        let commands = [
            ("load", json!({})),
            ("warmup", json!({})),
            ("set_voice", json!({"voice": root.join("second.pth")})),
            ("status", json!({})),
            ("warmup", json!({})),
            ("status", json!({})),
        ];
        let requests: String = commands
            .iter()
            .map(|(command, payload)| {
                json!({"version": 1, "id": ID, "command": command, "payload": payload}).to_string()
                    + "\n"
            })
            .collect();
        let mut output = Vec::new();
        serve::<MockEngine>(config, Cursor::new(requests), &mut output).unwrap();
        let lines: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), commands.len());
        assert!(
            lines
                .iter()
                .all(|line| line["ok"] == true && line["id"] == ID)
        );
        assert_eq!(lines[3]["result"]["phase"], "loaded");
        assert_eq!(lines[5]["result"]["phase"], "ready");
        fs::remove_dir_all(root).unwrap();
    }
}
