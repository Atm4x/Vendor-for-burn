use anyhow::{Context, Result, ensure};
use gigaam_rs::{
    audio::{self, SpeechRegionSeconds},
    service::{GigaAmService, Segmentation, TranscribeOptions},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    path::Path,
};

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    id: Value,
    #[serde(flatten)]
    command: Command,
}

#[derive(Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
enum Command {
    Status,
    Transcribe {
        audio_path: String,
        #[serde(default)]
        word_timestamps: bool,
        #[serde(default)]
        segments: Option<Vec<SpeechRegionSeconds>>,
        #[serde(default)]
        group_vad: bool,
    },
    TranscribeLongform {
        audio_path: String,
        segments: Vec<SpeechRegionSeconds>,
        #[serde(default)]
        word_timestamps: bool,
        #[serde(default = "default_true")]
        group_vad: bool,
    },
    TranscribeBatch {
        audio_paths: Vec<String>,
        #[serde(default)]
        word_timestamps: bool,
    },
    Embed {
        audio_path: String,
        output_path: String,
    },
    Emotion {
        audio_path: String,
    },
    Shutdown,
}

fn default_true() -> bool {
    true
}

pub fn run(service: &GigaAmService, python_audio: bool) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.context("reading request from stdin")?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Result<Request> = serde_json::from_str(&line).map_err(Into::into);
        let shutdown = matches!(
            &parsed,
            Ok(Request {
                command: Command::Shutdown,
                ..
            })
        );
        let response = match parsed {
            Ok(request) => {
                let id = request.id;
                match execute(service, request.command, python_audio) {
                    Ok(value) => json!({"id": id, "ok": true, "result": value}),
                    Err(error) => json!({"id": id, "ok": false, "error": format!("{error:#}")}),
                }
            }
            Err(error) => json!({"id": null, "ok": false, "error": error.to_string()}),
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
        if shutdown {
            break;
        }
    }
    Ok(())
}

fn decode_file(path: &str, python_audio: bool) -> Result<Vec<f32>> {
    if python_audio {
        audio::load_audio_ffmpeg_16k(Path::new(path))
    } else {
        audio::load_audio_16k(Path::new(path))
    }
    .with_context(|| format!("loading audio {path}"))
}

fn execute(service: &GigaAmService, command: Command, python_audio: bool) -> Result<Value> {
    match command {
        Command::Status => Ok(serde_json::to_value(service.status())?),
        Command::Transcribe {
            audio_path,
            word_timestamps,
            segments,
            group_vad,
        } => {
            ensure!(
                !group_vad || segments.is_some(),
                "group_vad requires segments"
            );
            let pcm = decode_file(&audio_path, python_audio)?;
            let regions = segments
                .map(|values| {
                    values
                        .into_iter()
                        .map(|v| v.into_samples(pcm.len()))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            let segmentation = match regions.as_deref() {
                None => Segmentation::FullRecording,
                Some(regions) if group_vad => Segmentation::GroupedSpeechRegions {
                    regions,
                    grouping: audio::VadGrouping::default(),
                },
                Some(regions) => Segmentation::SpeechRegions(regions),
            };
            let answer = service.transcribe_16k(
                &pcm,
                TranscribeOptions {
                    segmentation,
                    word_timestamps,
                },
            )?;
            Ok(serde_json::to_value(answer)?)
        }
        Command::TranscribeLongform {
            audio_path,
            segments,
            word_timestamps,
            group_vad,
        } => {
            let pcm = decode_file(&audio_path, python_audio)?;
            let regions = segments
                .into_iter()
                .map(|value| value.into_samples(pcm.len()))
                .collect::<Result<Vec<_>>>()?;
            let segmentation = if group_vad {
                Segmentation::GroupedSpeechRegions {
                    regions: &regions,
                    grouping: audio::VadGrouping::default(),
                }
            } else {
                Segmentation::SpeechRegions(&regions)
            };
            Ok(serde_json::to_value(service.transcribe_16k(
                &pcm,
                TranscribeOptions {
                    segmentation,
                    word_timestamps,
                },
            )?)?)
        }
        Command::TranscribeBatch {
            audio_paths,
            word_timestamps,
        } => {
            let pcm = audio_paths
                .iter()
                .map(|path| decode_file(path, python_audio))
                .collect::<Result<Vec<_>>>()?;
            Ok(serde_json::to_value(
                service.transcribe_batch_16k(&pcm, word_timestamps)?,
            )?)
        }
        Command::Embed {
            audio_path,
            output_path,
        } => {
            let pcm = decode_file(&audio_path, python_audio)?;
            let result = service.embed_16k(&pcm)?;
            let mut bytes = Vec::with_capacity(result.data.len() * 4);
            for value in result.data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(&output_path, bytes)
                .with_context(|| format!("writing embeddings to {}", output_path))?;
            Ok(
                json!({"frames": result.frames, "channels": result.channels, "output_path": output_path}),
            )
        }
        Command::Emotion { audio_path } => {
            let pcm = decode_file(&audio_path, python_audio)?;
            Ok(serde_json::to_value(service.emotions_dict_16k(&pcm)?)?)
        }
        Command::Shutdown => Ok(json!({"stopped": true})),
    }
}
