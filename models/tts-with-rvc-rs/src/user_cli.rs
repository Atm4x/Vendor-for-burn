#[derive(serde::Deserialize)]
pub(crate) struct WaveJob {
    pub(crate) input: String,
    pub(crate) output: String,
    #[serde(default)]
    pub(crate) index_rate: Option<f32>,
    #[serde(default)]
    pub(crate) pitch_shift_semitones: Option<f32>,
    #[serde(default)]
    pub(crate) rms_mix_rate: Option<f32>,
}

pub(crate) fn process_wave_job(
    inferencer: &tts_with_rvc_rs::rvc_inference::RvcV2Inferencer,
    job: &WaveJob,
    options: &tts_with_rvc_rs::rvc_inference::InferenceOptions,
    config: tts_with_rvc_rs::chunked::ChunkConfig,
    tensor_batch_size: usize,
) -> Result<(), String> {
    use tts_with_rvc_rs::audio_io;
    if job.input.is_empty() || job.output.is_empty() || job.input == job.output {
        return Err("batch input and output paths must be nonempty and distinct".into());
    }
    let audio = audio_io::read_mono_wav(&job.input)?;
    let pcm = audio_io::resample(&audio.samples, audio.sample_rate, 16_000)?;
    let mut settings = options.clone();
    if let Some(rate) = job.index_rate {
        settings.index_rate = rate;
    }
    if let Some(shift) = job.pitch_shift_semitones {
        settings.pitch_shift_semitones = shift;
    }
    eprintln!(
        "RVC processing {} input_samples={} input_sr={}",
        job.input,
        audio.samples.len(),
        audio.sample_rate
    );
    let filtered = tts_with_rvc_rs::preprocess::highpass_48hz_zero_phase(&pcm)?;
    let mut converted = inferencer.infer_long_filtered_16khz_batched(&filtered, &settings, config, tensor_batch_size, |p| {
        eprintln!(
            "RVC chunks={}/{} output_samples={}",
            p.chunk_index, p.chunk_count, p.output_samples
        );
    })?;
    let mix = job.rms_mix_rate.unwrap_or(0.5);
    audio_io::mix_rms_envelope(&filtered, 16_000, &mut converted, 40_000, mix)?;
    audio_io::normalize_peak(&mut converted)?;
    audio_io::write_mono_wav(&job.output, &converted, 40_000)?;
    println!(
        "RVC WAV→WAV complete: samples={} sr=40000 path={}",
        converted.len(),
        job.output
    );
    Ok(())
}

use std::{collections::{HashMap, HashSet}, fs, process::Command};
use burn::tensor::Device;

pub(crate) fn selected_device(name: &str) -> Result<Device, String> {
    match name {
        #[cfg(feature = "cpu")]
        "cpu" => Ok(Device::cpu()),
        #[cfg(all(not(feature = "cpu"), feature = "flex"))]
        "cpu" => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(0)),
        #[cfg(feature = "wgpu")]
        "wgpu" => Ok(Device::wgpu(Default::default())),
        other if matches!(other, "cpu" | "cuda" | "wgpu") => Err(format!(
            "backend {other} is not compiled into this executable; rebuild with the corresponding Cargo feature"
        )),
        other => Err(format!("unsupported device {other}; expected cpu, cuda or wgpu")),
    }
}

use tts_with_rvc_rs::{chunked::ChunkConfig, rvc_inference::{InferenceOptions, RvcV2Inferencer}};

const USAGE: &str = r#"RVC native Rust/Burn CLI
Usage:
  tts-with-rvc-rs convert --model voice.pth --hubert hubert_base.pt --rmvpe rmvpe.pt --input source.wav --output converted.wav [options]
  tts-with-rvc-rs convert-batch --model voice.pth --hubert hubert_base.pt --rmvpe rmvpe.pt --manifest jobs.json [options]
  tts-with-rvc-rs doctor
Options:
  --index FILE             FAISS IVF index (.index); omit to disable retrieval
  --device cpu|cuda|wgpu  Select a compiled backend (default cpu); never silently falls back
  --batch-size N           Tensor batch size 1..32 (default 2)
  --index-rate R           FAISS mix 0..1 (default 0.75 with index, else 0)
  --pitch N                Pitch shift, semitones (default 0)
  --rms-mix R              Source RMS envelope mix 0..1 (default 0.5)
  --core-frames N          Core chunk length, units 10ms (default 300)
  --overlap-frames N       Overlap, units 10ms (default 12)
  --context-frames N       Context on both sides, units 10ms (default 40)

Batch JSON example: [{"input":"a.wav","output":"a_rvc.wav"}, {"input":"b.wav","output":"b_rvc.wav","pitch_shift_semitones":2}]
All checkpoint paths are read from disk. Models are never downloaded or uploaded.
CUDA runtime installation alone does not enable CUDA in a CPU binary.
"#;

fn parse_options(args: impl Iterator<Item=String>) -> Result<HashMap<String,String>,String> {
    let mut map = HashMap::new();
    let mut args = args;
    while let Some(key) = args.next() {
        if !matches!(key.as_str(), "--model" | "--hubert" | "--rmvpe" | "--input" | "--output" | "--manifest" |
            "--index" | "--device" | "--batch-size" | "--index-rate" | "--pitch" | "--rms-mix" |
            "--core-frames" | "--overlap-frames" | "--context-frames") {
            return Err(format!("unknown argument {key}; run --help"));
        }
        let value = args.next().ok_or_else(||format!("missing value for {key}"))?;
        if value.is_empty() || value.starts_with("--") {
            return Err(format!("empty value for {key}"));
        }
        if map.insert(key.clone(),value).is_some() {return Err(format!("duplicate argument {key}"));}
    }
    Ok(map)
}

fn required<'a>(map: &'a HashMap<String,String>, name: &str) -> Result<&'a str,String> {
    map.get(name).map(String::as_str).ok_or_else(||format!("missing required {name}"))
}

fn number<T: std::str::FromStr>(map: &HashMap<String,String>, name: &str, default: T) -> Result<T,String>
where T::Err: std::fmt::Display {
    map.get(name).map(|v|v.parse().map_err(|e|format!("invalid {name}: {e}"))).transpose().map(|v|v.unwrap_or(default))
}

pub(crate) fn run_named(command: &str, args: impl Iterator<Item=String>) -> Result<(),String> {
    if matches!(command,"help" | "--help" | "-h") {print!("{USAGE}"); return Ok(());}
    if command == "doctor" {
        if args.count() != 0 {return Err("doctor takes no arguments".into());}
        println!("RVC native Rust/Burn; tensor batching supported");
        println!("Compiled backends: cpu={} cuda={} wgpu={}",
            cfg!(any(feature = "flex", feature = "cpu")),
            cfg!(feature = "cuda"), cfg!(feature = "wgpu"));
        match Command::new("nvidia-smi").arg("--query-gpu=name,driver_version").arg("--format=csv,noheader").output() {
            Ok(output) if output.status.success() => print!("NVIDIA GPU (driver): {}", String::from_utf8_lossy(&output.stdout)),
            _ => println!("NVIDIA driver not detected by nvidia-smi"),
        }
        println!("CUDA selection requires a binary built with the cuda feature and a compatible driver/runtime; no CPU fallback");
        return Ok(());
    }
    let map = parse_options(args)?;
    let device = selected_device(map.get("--device").map(String::as_str).unwrap_or("cpu"))?;
    let voice = required(&map,"--model")?;
    let hubert = required(&map,"--hubert")?;
    let rmvpe = required(&map,"--rmvpe")?;
    let index = map.get("--index").map(String::as_str).filter(|v|*v != "-");
    let mut options = InferenceOptions::default();
    options.index_rate = number(&map,"--index-rate",if index.is_some(){0.75}else{0.0})?;
    options.pitch_shift_semitones = number(&map,"--pitch",0.0)?;
    if !options.index_rate.is_finite() || !(0.0..=1.0).contains(&options.index_rate) || (options.index_rate>0.0 && index.is_none()) {
        return Err("--index-rate must be in 0..1, and nonzero retrieval requires --index".into());
    }
    if !options.pitch_shift_semitones.is_finite() {return Err("--pitch must be finite".into());}
    let batch_size: usize = number(&map,"--batch-size",2)?;
    if !(1..=32).contains(&batch_size) {return Err("--batch-size must be 1..32".into());}
    let config = ChunkConfig {
        core_frames:number(&map,"--core-frames",ChunkConfig::default().core_frames)?,
        overlap_frames:number(&map,"--overlap-frames",ChunkConfig::default().overlap_frames)?,
        context_frames:number(&map,"--context-frames",ChunkConfig::default().context_frames)?,
        ..ChunkConfig::default()
    };
    config.validate()?;
    let rms_mix: f32 = number(&map,"--rms-mix",0.5)?;
    if !rms_mix.is_finite() || !(0.0..=1.0).contains(&rms_mix) {return Err("--rms-mix must be in 0..1".into());}
    let jobs = if command == "convert" {
        if map.contains_key("--manifest") {return Err("convert accepts --input/--output, not --manifest".into());}
        vec![WaveJob {input:required(&map,"--input")?.into(),output:required(&map,"--output")?.into(),
            index_rate:None,pitch_shift_semitones:None,rms_mix_rate:Some(rms_mix)}]
    } else {
        if map.contains_key("--input") || map.contains_key("--output") {
            return Err("convert-batch accepts --manifest, not --input/--output".into());
        }
        let manifest = required(&map,"--manifest")?;
        let mut jobs: Vec<WaveJob> = serde_json::from_slice(&fs::read(manifest).map_err(|e|format!("cannot read manifest: {e}"))?)
            .map_err(|e|format!("invalid jobs.json: {e}"))?;
        if jobs.is_empty() {return Err("manifest is empty".into());}
        for job in &mut jobs {if job.rms_mix_rate.is_none() {job.rms_mix_rate = Some(rms_mix);}}
        jobs
    };
    let mut outputs = HashSet::new();
    let inputs: HashSet<_> = jobs.iter().map(|job|std::path::Path::new(&job.input)).collect();
    for job in &jobs {
        if job.input.is_empty() || job.output.is_empty() || inputs.contains(std::path::Path::new(&job.output)) || !outputs.insert(&job.output) {
            return Err("input and output paths must differ; outputs must be unique".into());
        }
    }
    eprintln!("Loading RVC, HuBERT, RMVPE and optional FAISS index...");
    let inferencer = RvcV2Inferencer::load(voice, hubert, rmvpe, index, device)?;
    for (i, job) in jobs.iter().enumerate() {
        eprintln!("RVC {}/{}: {}",i+1,jobs.len(),job.input);
        process_wave_job(&inferencer,job,&options,config,batch_size)?;
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    #[test]
    fn parser_rejects_duplicates_and_unknown_options() {
        assert!(parse_options(["--model","a","--model","b"].into_iter().map(str::to_owned)).is_err());
        assert!(parse_options(["--mystery","a"].into_iter().map(str::to_owned)).is_err());
        assert!(parse_options(["--model","--input"].into_iter().map(str::to_owned)).is_err());
    }
    #[test]
    fn device_selection_does_not_silently_fall_back_to_cpu() {
        assert!(selected_device("unknown").is_err());
        #[cfg(not(feature = "cuda"))]
        assert!(selected_device("cuda").unwrap_err().contains("not compiled"));
        #[cfg(not(feature = "wgpu"))]
        assert!(selected_device("wgpu").unwrap_err().contains("not compiled"));
        #[cfg(feature = "cpu")]
        assert_eq!(selected_device("cpu").unwrap(), Device::cpu());
        #[cfg(all(not(feature = "cpu"), feature = "flex"))]
        assert_eq!(selected_device("cpu").unwrap(), Device::flex());
    }
}
