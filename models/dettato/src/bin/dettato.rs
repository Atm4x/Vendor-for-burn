use std::{env, path::PathBuf, process::ExitCode, time::Instant};
use burn::tensor::Device;
#[cfg(feature = "wgpu")]
use burn::tensor::DeviceKind;
use dettato_rs::{checkpoint::ModelPaths, pipeline::{Dettato, DettatoModelPaths}};

const USAGE: &str = "Usage: dettato --transcriber <config> <weights> --evidence <config> <weights> --grouper <config> <weights> --drums <config> <weights> --input <audio> [--output <midi> | --output-dir <directory>] [--separate-instruments] [--resample-low-rate-drums] [--cache-mib <memory>] [--device <backend>]\n\
    Supported audio formats depend on the enabled Symphonia codecs.\n\
    --output-dir writes one MIDI file per detected track.\n\
    By default, tracks for the same identified instrument are merged.\n\
    Backends: auto, cuda[:index], wgpu[:discrete-index], cpu, flex.\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComputeDevice {
    Auto,
    Cuda(usize),
    Wgpu(Option<usize>),
    Cpu,
    Flex,
}

impl ComputeDevice {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda(0)),
            "wgpu" => Ok(Self::Wgpu(None)),
            "cpu" => Ok(Self::Cpu),
            "flex" => Ok(Self::Flex),
            _ => {
                let (backend, index) = raw.split_once(':')
                    .ok_or_else(|| format!("unknown device: {raw}"))?;
                let index = index.parse::<usize>()
                    .map_err(|_| format!("invalid device index: {raw}"))?;
                match backend {
                    "cuda" => Ok(Self::Cuda(index)),
                    "wgpu" => Ok(Self::Wgpu(Some(index))),
                    _ => Err(format!("unknown device: {raw}")),
                }
            }
        }
    }

    fn open(self) -> Result<Device, String> {
        match self {
            Self::Auto => Ok(Device::default()),
            Self::Cuda(index) => {
                #[cfg(feature = "cuda")]
                { Ok(Device::cuda(index)) }
                #[cfg(not(feature = "cuda"))]
                { let _ = index; Err("CUDA backend not compiled: build with --features cuda".into()) }
            },
            Self::Wgpu(index) => {
                #[cfg(feature = "wgpu")]
                { Ok(Device::wgpu(index.map(DeviceKind::DiscreteGpu).unwrap_or_default())) }
                #[cfg(not(feature = "wgpu"))]
                { let _ = index; Err("WGPU backend not compiled: build with --features wgpu".into()) }
            },
            Self::Cpu => {
                #[cfg(feature = "cpu")]
                { Ok(Device::cpu()) }
                #[cfg(not(feature = "cpu"))]
                { Err("CubeCL CPU backend not compiled: build with --features cpu (requires Tracel LLVM)".into()) }
            },
            Self::Flex => {
                #[cfg(feature = "flex")]
                { Ok(Device::flex()) }
                #[cfg(not(feature = "flex"))]
                { Err("Flex backend not compiled: build with --features flex".into()) }
            },
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Output {
    Midi(PathBuf),
    Directory(PathBuf),
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    models: DettatoModelPaths,
    input: PathBuf,
    output: Output,
    separate_instruments: bool,
    resample_low_rate_drums: bool,
    cache_mib: usize,
    device: ComputeDevice,
}

fn parse_args(arguments: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let mut transcriber = None;
    let mut evidence = None;
    let mut grouper = None;
    let mut drums = None;
    let mut input = None;
    let mut output = None;
    let mut separate_instruments = false;
    let mut resample_low_rate_drums = false;
    let mut cache_mib = 256usize;
    let mut device = ComputeDevice::Auto;
    let mut args = arguments.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--transcriber" | "--evidence" | "--grouper" | "--drums" => {
                let slot = match arg.as_str() {
                    "--transcriber" => &mut transcriber,
                    "--evidence" => &mut evidence,
                    "--grouper" => &mut grouper,
                    _ => &mut drums,
                };
                if slot.is_some() {
                    return Err(format!("{arg} supplied more than once"));
                }
                let mut next_path = || args.next().filter(|v| !v.starts_with('-'))
                    .map(PathBuf::from)
                    .ok_or_else(|| format!("{arg} requires a config path and a weights path"));
                *slot = Some(ModelPaths { config: next_path()?, weights: next_path()? });
            }
            "--input" | "-i" => {
                let path = args.next().filter(|v| !v.starts_with('-'))
                    .ok_or("--input requires an audio file")?;
                if input.replace(PathBuf::from(path)).is_some() {
                    return Err("--input supplied more than once".into());
                }
            }
            "--output" | "-o" | "--output-dir" => {
                let path = args.next().filter(|v| !v.starts_with('-'))
                    .ok_or_else(|| format!("{arg} requires a path"))?;
                if output.is_some() {
                    return Err("use only one --output or --output-dir".into());
                }
                let path = PathBuf::from(path);
                output = Some(if arg == "--output-dir" { Output::Directory(path) } else { Output::Midi(path) });
            }
            "--separate-instruments" => separate_instruments = true,
            "--resample-low-rate-drums" => resample_low_rate_drums = true,
            "--device" | "--backend" => {
                let raw = args.next().ok_or("--device requires a backend")?;
                device = ComputeDevice::parse(&raw)?;
            }
            "--cache-mib" => {
                cache_mib = args.next().ok_or("--cache-mib requires a nonnegative integer")?
                    .parse::<usize>().map_err(|_| "invalid --cache-mib memory limit")?;
            },
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    let models = DettatoModelPaths {
        transcriber: transcriber.ok_or("missing --transcriber config and weights")?,
        evidence: evidence.ok_or("missing --evidence config and weights")?,
        grouper: grouper.ok_or("missing --grouper config and weights")?,
        drums: drums.ok_or("missing --drums config and weights")?,
    };
    let input: PathBuf = input.ok_or("missing --input audio file")?;
    let output = output.unwrap_or_else(|| Output::Midi(input.with_extension("mid")));
    if matches!(&output, Output::Midi(path) | Output::Directory(path) if path == &input) {
        return Err("input and output paths must differ".into());
    }
    Ok(Some(Args { models, input, output, separate_instruments, resample_low_rate_drums, cache_mib, device }))
}

fn run(args: Args) -> Result<(), String> {
    let device = args.device.open()?;
    println!("Compute device: {device:?}");
    let load_started = Instant::now();
    let mut pipeline = Dettato::from_paths(&args.models, &device)?;
    device.sync().map_err(|error| format!("Synchronizing model load: {error}"))?;
    println!("Models loaded: {:.3} s", load_started.elapsed().as_secs_f64());
    pipeline.set_merge_same_instrument(!args.separate_instruments);
    pipeline.set_resample_low_rate_drums(args.resample_low_rate_drums);
    pipeline.transcriber.set_shared_cache_budget_bytes(args.cache_mib.saturating_mul(1024 * 1024));
    let inference_started = Instant::now();
    let result = pipeline.transcribe_file(&args.input)?;
    let inference = inference_started.elapsed();
    let times = &result.timings;
    println!("Preparation: {:.3} s", times.preparation.as_secs_f64());
    if let Some(transcriber) = times.transcriber {
        println!("Transcriber: {:.3} s", transcriber.as_secs_f64());
        println!("Evidence: {:.3} s", times.evidence.as_secs_f64());
        println!("Grouper: {:.3} s", times.grouper.as_secs_f64());
    } else {
        println!("Transcriber: included in Evidence and Grouper (shared cache budget exceeded)");
        println!("Evidence with Transcriber: {:.3} s", times.evidence.as_secs_f64());
        println!("Grouper with Transcriber: {:.3} s", times.grouper.as_secs_f64());
    }
    if let Some(drums) = times.drums {
        println!("Drums: {:.3} s", drums.as_secs_f64());
    }
    println!("Transcription: {:.3} s, RTF {:.3}", inference.as_secs_f64(),
        inference.as_secs_f64() / result.transcription.seconds);
    match args.output {
        Output::Midi(path) => {
            result.transcription.save_midi(&path)?;
            println!("MIDI: {}", path.display());
        }
        Output::Directory(path) => {
            let files = result.transcription.save_midi_tracks(&path)?;
            println!("MIDI files: {} in {}", files.len(), path.display());
            for file in files {
                println!("MIDI: {}", file.display());
            }
        }
    }
    println!("Duration: {:.2} seconds", result.transcription.seconds);
    println!("Instrument tracks: {}", result.transcription.tracks.iter().filter(|t| !t.is_drum).count());
    println!("Grouped notes: {}", result.info.grouped_notes);
    println!("Drum hits: {}", result.info.drum_hits);
    if result.info.drums_skipped_low_sample_rate {
        println!("Drums skipped: original sample rate is below 40 kHz");
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(None) => { print!("{USAGE}"); ExitCode::SUCCESS }
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("Dettato failed: {error}"); ExitCode::FAILURE }
        },
        Err(error) => { eprintln!("{error}\n{USAGE}"); ExitCode::FAILURE }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_ARGS: &[&str] = &[
        "--transcriber", "configs/pitched.json", "checkpoints/a.bin",
        "--evidence", "configs/repair.json", "checkpoints/b.bin",
        "--grouper", "config files/track.json", "checkpoints/c.bin",
        "--drums", "configs/percussion.json", "checkpoints/d.bin",
    ];

    fn parse(values: &[&str]) -> Result<Option<Args>, String> {
        parse_args(values.iter().map(|s| (*s).to_string()))
    }

    fn parse_with_models(values: &[&str]) -> Result<Option<Args>, String> {
        parse_args(MODEL_ARGS.iter().chain(values).map(|s| (*s).to_string()))
    }

    #[test]
    fn defaults_to_same_directory_midi_output() {
        let args = parse_with_models(&["-i", "song.wav"]).unwrap().unwrap();
        assert_eq!(args.input, PathBuf::from("song.wav"));
        assert_eq!(args.models.transcriber.weights, PathBuf::from("checkpoints/a.bin"));
        assert_eq!(args.models.grouper.config, PathBuf::from("config files/track.json"));
        assert_eq!(args.output, Output::Midi(PathBuf::from("song.mid")));
        assert!(!args.separate_instruments);
        assert!(!args.resample_low_rate_drums);
        assert_eq!(args.cache_mib, 256);
        assert_eq!(args.device, ComputeDevice::Auto);
    }

    #[test]
    fn explicit_track_separation_and_output() {
        let args = parse_with_models(&["-i", "music.wav", "-o", "output/notes.mid", "--separate-instruments", "--resample-low-rate-drums"])
            .unwrap().unwrap();
        assert_eq!(args.output, Output::Midi(PathBuf::from("output/notes.mid")));
        assert!(args.separate_instruments);
        assert!(args.resample_low_rate_drums);
    }

    #[test]
    fn directory_output_selects_per_track_export() {
        let args = parse_with_models(&["-i", "song.wav", "--output-dir", "output/tracks"])
            .unwrap().unwrap();
        assert_eq!(args.output, Output::Directory(PathBuf::from("output/tracks")));
        assert!(parse_with_models(&["-i", "song.wav", "--output-dir", "tracks", "-o", "song.mid"]).is_err());
    }

    #[test]
    fn invalid_or_incomplete_arguments_fail() {
        assert!(parse(&["--input", "song.wav"]).is_err());
        assert!(parse(&["--evidence", "repair.json", "--input", "song.wav"]).is_err());
        assert!(parse_with_models(&["--evidence", "repair.json", "duplicate.bin", "--input", "song.wav"]).is_err());
        assert!(parse_with_models(&["--input"]).is_err());
        assert!(parse_with_models(&["--input", "song.mid", "--output", "song.mid"]).is_err());
        assert!(parse_with_models(&["--input", "song.wav", "--unknown"]).is_err());
        assert!(parse(&["--help"]).unwrap().is_none());
        assert!(parse_with_models(&["--input", "song.wav", "--cache-mib", "bogus"]).is_err());
        assert_eq!(parse_with_models(&["--input", "song.wav", "--cache-mib", "4096"]).unwrap().unwrap().cache_mib, 4096);
    }
    #[test]
    fn device_selection_is_explicit_and_validated() {
        let base = ["-i", "song.wav"];
        for (name, expected) in [
            ("auto", ComputeDevice::Auto),
            ("cuda", ComputeDevice::Cuda(0)),
            ("cuda:2", ComputeDevice::Cuda(2)),
            ("wgpu", ComputeDevice::Wgpu(None)),
            ("wgpu:1", ComputeDevice::Wgpu(Some(1))),
            ("cpu", ComputeDevice::Cpu),
            ("flex", ComputeDevice::Flex),
        ] {
            let mut args = base.to_vec();
            args.extend(["--device", name]);
            assert_eq!(parse_with_models(&args).unwrap().unwrap().device, expected);
        }
        for bad in ["cuda:-1", "cuda:abc", "wgpu:-1", "wgpu:", "foo", "flex:0"] {
            let mut args = base.to_vec();
            args.extend(["--device", bad]);
            assert!(parse_with_models(&args).is_err(), "unexpectedly accepted {bad}");
        }
        assert!(parse_with_models(&["-i", "song.wav", "--device"]).is_err());
    }

    #[test]
    fn missing_backend_feature_returns_error_instead_of_silent_fallback() {
        #[cfg(not(feature = "cuda"))]
        assert!(ComputeDevice::Cuda(0).open().unwrap_err().contains("cuda"));
        #[cfg(not(feature = "wgpu"))]
        assert!(ComputeDevice::Wgpu(None).open().unwrap_err().contains("wgpu"));
        #[cfg(not(feature = "cpu"))]
        assert!(ComputeDevice::Cpu.open().unwrap_err().contains("cpu"));
    }

}
