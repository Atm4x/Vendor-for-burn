use std::{env, path::PathBuf, process::ExitCode};
use burn::tensor::Device;
use dettato_rs::pipeline::Dettato;

const USAGE: &str = "Usage: dettato --weights <directory> --input <audio> [--output <midi>] [--separate-instruments] [--cache-mib <memory>]\n\
    Supported audio formats depend on the enabled Symphonia codecs.\n\
    By default, tracks for the same identified instrument are merged.\n";

#[derive(Debug, PartialEq, Eq)]
struct Args {
    weights: PathBuf,
    input: PathBuf,
    output: PathBuf,
    separate_instruments: bool,
    cache_mib: usize,
}

fn parse_args(arguments: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let mut weights = None;
    let mut input = None;
    let mut output = None;
    let mut separate_instruments = false;
    let mut cache_mib = 256usize;
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
    Ok(Some(Args { weights, input, output, separate_instruments, cache_mib }))
}

fn run(args: Args) -> Result<(), String> {
    let mut pipeline = Dettato::from_directory(&args.weights, &Device::default())?;
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
}
