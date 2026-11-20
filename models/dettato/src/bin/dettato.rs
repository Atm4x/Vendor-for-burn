use std::{env, path::PathBuf, process::ExitCode};
use burn::tensor::Device;
#[cfg(feature = "wgpu")]
use burn::tensor::DeviceKind;
use dettato_rs::pipeline::Dettato;

const USAGE: &str = "Usage: dettato --weights <directory> --input <audio> [--output <midi>] [--separate-instruments] [--cache-mib <memory>] [--device <backend>]\n\
    Supported audio formats depend on the enabled Symphonia codecs.\n\
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
struct Args {
    weights: PathBuf,
    input: PathBuf,
    output: PathBuf,
    separate_instruments: bool,
    cache_mib: usize,
    device: ComputeDevice,
}

fn parse_args(arguments: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let mut weights = None;
    let mut input = None;
    let mut output = None;
    let mut separate_instruments = false;
    let mut cache_mib = 256usize;
    let mut device = ComputeDevice::Auto;
    let mut args = arguments.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--weights" | "-w" => {
                let path = args.next().filter(|v| !v.starts_with('-'))
                    .ok_or("--weights requires a directory")?;
                if weights.replace(PathBuf::from(path)).is_some() {
                    return Err("--weights supplied more than once".into());
                }
            }
            "--input" | "-i" => {
                let path = args.next().filter(|v| !v.starts_with('-'))
                    .ok_or("--input requires an audio file")?;
                if input.replace(PathBuf::from(path)).is_some() {
                    return Err("--input supplied more than once".into());
                }
            }
            "--output" | "-o" => {
                let path = args.next().filter(|v| !v.starts_with('-'))
                    .ok_or("--output requires a MIDI file path")?;
                if output.replace(PathBuf::from(path)).is_some() {
                    return Err("--output supplied more than once".into());
                }
            }
            "--separate-instruments" => separate_instruments = true,
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
    let weights = weights.ok_or("missing --weights directory")?;
    let input: PathBuf = input.ok_or("missing --input audio file")?;
    let output = output.unwrap_or_else(|| input.with_extension("mid"));
    if input == output {
        return Err("input and output paths must differ".into());
    }
    Ok(Some(Args { weights, input, output, separate_instruments, cache_mib, device }))
}

fn run(args: Args) -> Result<(), String> {
    let device = args.device.open()?;
    println!("Compute device: {device:?}");
    let mut pipeline = Dettato::from_directory(&args.weights, &device)?;
    pipeline.set_merge_same_instrument(!args.separate_instruments);
    pipeline.transcriber.set_shared_cache_budget_bytes(args.cache_mib.saturating_mul(1024 * 1024));
    let result = pipeline.transcribe_file(&args.input)?;
    result.transcription.save_midi(&args.output)?;
    println!("MIDI: {}", args.output.display());
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

    fn parse(values: &[&str]) -> Result<Option<Args>, String> {
        parse_args(values.iter().map(|s| (*s).to_string()))
    }

    #[test]
    fn defaults_to_same_directory_midi_output() {
        let args = parse(&["--weights", "weights", "-i", "song.wav"]).unwrap().unwrap();
        assert_eq!(args.input, PathBuf::from("song.wav"));
        assert_eq!(args.output, PathBuf::from("song.mid"));
        assert!(!args.separate_instruments);
        assert_eq!(args.cache_mib, 256);
        assert_eq!(args.device, ComputeDevice::Auto);
    }

    #[test]
    fn explicit_track_separation_and_output() {
        let args = parse(&["-w", "weights", "-i", "music.wav", "-o", "output/notes.mid", "--separate-instruments"])
            .unwrap().unwrap();
        assert_eq!(args.output, PathBuf::from("output/notes.mid"));
        assert!(args.separate_instruments);
    }

    #[test]
    fn invalid_or_incomplete_arguments_fail() {
        assert!(parse(&["--input", "song.wav"]).is_err());
        assert!(parse(&["--weights", "weights", "--input"]).is_err());
        assert!(parse(&["--weights", "weights", "--input", "song.mid", "--output", "song.mid"]).is_err());
        assert!(parse(&["--weights", "weights", "--input", "song.wav", "--unknown"]).is_err());
        assert!(parse(&["--help"]).unwrap().is_none());
        assert!(parse(&["--weights", "weights", "--input", "song.wav", "--cache-mib", "bogus"]).is_err());
        assert_eq!(parse(&["--weights", "weights", "--input", "song.wav", "--cache-mib", "4096"]).unwrap().unwrap().cache_mib, 4096);
    }
    #[test]
    fn device_selection_is_explicit_and_validated() {
        let base = ["-w", "weights", "-i", "song.wav"];
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
            assert_eq!(parse(&args).unwrap().unwrap().device, expected);
        }
        for bad in ["cuda:-1", "cuda:abc", "wgpu:-1", "wgpu:", "foo", "flex:0"] {
            let mut args = base.to_vec();
            args.extend(["--device", bad]);
            assert!(parse(&args).is_err(), "unexpectedly accepted {bad}");
        }
        assert!(parse(&["-w", "weights", "-i", "song.wav", "--device"]).is_err());
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
