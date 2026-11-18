use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use fish_speech_core::audio::SAMPLE_RATE;
use fish_speech_core::reference_audio::{ReferenceMelInput, ReferenceMelPreprocessor};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as DecodeError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use thiserror::Error;

const MAX_SOURCE_BYTES: usize = 256 * 1024 * 1024;
const MAX_KERNEL_COEFFICIENTS: usize = 8 * 1024 * 1024;
const MAX_DECODED_FRAMES: usize = 30 * 60 * 384_000;
const MAX_TRANSCODED_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ReferenceLoadError {
    #[error("failed to read reference: {0}")]
    Io(#[from] std::io::Error),
    #[error("reference audio is invalid: {0}")]
    Invalid(&'static str),
    #[error("unsupported or damaged reference audio: {0}")]
    Decode(String),
    #[error("audio resampling failed: {0}")]
    Resample(String),
    #[error("reference LogMel failed: {0}")]
    Mel(#[from] fish_speech_core::error::FishError),
}

pub type Result<T> = std::result::Result<T, ReferenceLoadError>;

#[derive(Clone, Debug)]
pub struct ReferencePcm {
    pub pcm: Vec<f32>,
    pub sample_rate: usize,
    pub source_rate: usize,
    pub source_channels: usize,
}

impl ReferencePcm {
    pub fn prepare_mel(&self) -> Result<ReferenceMelInput> {
        Ok(ReferenceMelPreprocessor::new().prepare_mono(&[&self.pcm], self.sample_rate)?)
    }
}

pub struct ReferenceAudioLoader {
    target_rate: usize,
    external_ffmpeg: Option<PathBuf>,
}

impl Default for ReferenceAudioLoader {
    fn default() -> Self {
        Self::new(SAMPLE_RATE)
    }
}

impl ReferenceAudioLoader {
    pub fn new(target_rate: usize) -> Self {
        Self {
            target_rate,
            external_ffmpeg: None,
        }
    }

    pub fn with_ffmpeg(mut self, executable: impl Into<PathBuf>) -> Self {
        self.external_ffmpeg = Some(executable.into());
        self
    }

    pub fn load_path(&self, path: impl AsRef<Path>) -> Result<ReferencePcm> {
        let metadata = std::fs::metadata(path.as_ref())?;
        if metadata.len() > MAX_SOURCE_BYTES as u64 {
            return Err(ReferenceLoadError::Invalid("source file is too large"));
        }
        let data = std::fs::read(path)?;
        self.load_bytes(&data)
    }

    pub fn load_bytes(&self, bytes: &[u8]) -> Result<ReferencePcm> {
        if bytes.is_empty() || bytes.len() > MAX_SOURCE_BYTES || self.target_rate == 0 {
            return Err(ReferenceLoadError::Invalid(
                "invalid source size or target rate",
            ));
        }
        match self.decode_symphonia(bytes) {
            Ok(pcm) => Ok(pcm),
            Err(original_error) => match &self.external_ffmpeg {
                Some(executable) => {
                    let wav = transcode_to_wav(executable, bytes)?;
                    self.decode_symphonia(&wav)
                }
                None => Err(original_error),
            },
        }
    }

    fn decode_symphonia(&self, bytes: &[u8]) -> Result<ReferencePcm> {
        let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
        let mut format = symphonia::default::get_probe()
            .probe(
                &Hint::new(),
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| ReferenceLoadError::Decode(e.to_string()))?;
        let track = format
            .default_track(TrackType::Audio)
            .ok_or(ReferenceLoadError::Invalid("no audio track"))?;
        let params = track
            .codec_params
            .as_ref()
            .ok_or(ReferenceLoadError::Invalid(
                "audio codec parameters missing",
            ))?
            .audio()
            .ok_or(ReferenceLoadError::Invalid(
                "invalid audio codec parameters",
            ))?;
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .map_err(|e| ReferenceLoadError::Decode(e.to_string()))?;
        let track_id = track.id;
        let mut pcm = Vec::<f32>::new();
        let mut source_rate = None;
        let mut source_channels = None;
        loop {
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                Err(e) => return Err(ReferenceLoadError::Decode(e.to_string())),
            };
            if packet.track_id != track_id {
                continue;
            }
            let audio = match decoder.decode(&packet) {
                Ok(audio) => audio,
                Err(DecodeError::DecodeError(_)) => {
                    return Err(ReferenceLoadError::Invalid("damaged audio packet"));
                }
                Err(e) => return Err(ReferenceLoadError::Decode(e.to_string())),
            };
            let channels = audio.spec().channels().count();
            let rate = audio.spec().rate() as usize;
            if channels == 0 || channels > 32 || rate == 0 || rate > 384_000 {
                return Err(ReferenceLoadError::Invalid(
                    "invalid channel count or sample rate",
                ));
            }
            if source_channels.is_some_and(|previous| previous != channels)
                || source_rate.is_some_and(|previous| previous != rate)
            {
                return Err(ReferenceLoadError::Invalid(
                    "audio format changes during decoding",
                ));
            }
            source_channels = Some(channels);
            source_rate = Some(rate);
            if pcm.len() / channels + audio.frames() > MAX_DECODED_FRAMES {
                return Err(ReferenceLoadError::Invalid(
                    "reference audio duration exceeds limit",
                ));
            }
            let mut interleaved = vec![0f32; audio.samples_interleaved()];
            audio.copy_to_slice_interleaved(&mut interleaved);
            for frame in interleaved.chunks_exact(channels) {
                let avg = frame.iter().map(|&x| x as f64).sum::<f64>() / channels as f64;
                let avg = avg as f32;
                if !avg.is_finite() {
                    return Err(ReferenceLoadError::Invalid("non-finite PCM sample"));
                }
                pcm.push(avg);
            }
        }
        let source_rate = source_rate.ok_or(ReferenceLoadError::Invalid("empty audio stream"))?;
        let source_channels =
            source_channels.ok_or(ReferenceLoadError::Invalid("no audio channels"))?;
        if pcm.is_empty() {
            return Err(ReferenceLoadError::Invalid("empty decoded reference"));
        }
        if source_rate != self.target_rate {
            pcm = resample_mono(&pcm, source_rate, self.target_rate)?;
        }
        Ok(ReferencePcm {
            pcm,
            sample_rate: self.target_rate,
            source_rate,
            source_channels,
        })
    }
}

fn transcode_to_wav(ffmpeg: &Path, source: &[u8]) -> Result<Vec<u8>> {
    let mut child = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| ReferenceLoadError::Decode(format!("FFmpeg unavailable: {e}")))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or(ReferenceLoadError::Invalid("failed to open FFmpeg input"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or(ReferenceLoadError::Invalid("failed to open FFmpeg output"))?;
    let mut converted = Vec::new();
    let output_result = std::thread::scope(|scope| {
        let writer = scope.spawn(move || stdin.write_all(source));
        let read_result = stdout
            .by_ref()
            .take(MAX_TRANSCODED_BYTES + 1)
            .read_to_end(&mut converted);
        if read_result.is_err() || converted.len() as u64 > MAX_TRANSCODED_BYTES {
            let _ = child.kill();
        }
        let write_result = writer.join();
        (read_result, write_result)
    });
    let status = child
        .wait()
        .map_err(|e| ReferenceLoadError::Decode(format!("FFmpeg wait failed: {e}")))?;
    if converted.len() as u64 > MAX_TRANSCODED_BYTES {
        return Err(ReferenceLoadError::Invalid(
            "transcoded reference exceeds 512 MiB",
        ));
    }
    output_result.0.map_err(ReferenceLoadError::Io)?;
    if !status.success() || !matches!(output_result.1, Ok(Ok(()))) || converted.is_empty() {
        return Err(ReferenceLoadError::Decode(
            "FFmpeg could not decode the supplied reference audio".into(),
        ));
    }
    seal_streamed_wav(&mut converted)?;
    Ok(converted)
}

fn seal_streamed_wav(wav: &mut [u8]) -> Result<()> {
    if wav.len() < 44 || &wav[..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err(ReferenceLoadError::Invalid(
            "unexpected transcoded WAV header",
        ));
    }
    let riff_len = u32::try_from(wav.len() - 8)
        .map_err(|_| ReferenceLoadError::Invalid("transcoded WAV too large"))?;
    wav[4..8].copy_from_slice(&riff_len.to_le_bytes());
    let mut cursor = 12usize;
    while cursor + 8 <= wav.len() {
        let chunk_bytes = u32::from_le_bytes(wav[cursor + 4..cursor + 8].try_into().unwrap());
        if &wav[cursor..cursor + 4] == b"data" {
            if chunk_bytes != u32::MAX {
                return Err(ReferenceLoadError::Invalid(
                    "unexpected bounded WAV data chunk",
                ));
            }
            let data_bytes = u32::try_from(wav.len() - cursor - 8)
                .map_err(|_| ReferenceLoadError::Invalid("transcoded PCM too large"))?;
            wav[cursor + 4..cursor + 8].copy_from_slice(&data_bytes.to_le_bytes());
            return Ok(());
        }
        let advance = 8usize
            .checked_add(chunk_bytes as usize)
            .and_then(|n| n.checked_add((chunk_bytes & 1) as usize))
            .ok_or(ReferenceLoadError::Invalid("invalid WAV chunk length"))?;
        cursor = cursor
            .checked_add(advance)
            .ok_or(ReferenceLoadError::Invalid("invalid WAV chunk offset"))?;
    }
    Err(ReferenceLoadError::Invalid(
        "missing transcoded WAV data chunk",
    ))
}

pub fn resample_mono(pcm: &[f32], from: usize, to: usize) -> Result<Vec<f32>> {
    if pcm.is_empty() || from == 0 || to == 0 || from > 384_000 || to > 384_000 {
        return Err(ReferenceLoadError::Invalid("invalid resampling input"));
    }
    if pcm.iter().any(|x| !x.is_finite()) {
        return Err(ReferenceLoadError::Invalid("non-finite resampling sample"));
    }
    if from == to {
        return Ok(pcm.to_vec());
    }
    let gcd = gcd(from, to);
    let old = from / gcd;
    let new = to / gcd;
    let base = old.min(new) as f64 * 0.99;
    let width = (6.0 * old as f64 / base).ceil() as usize;
    let taps = width
        .checked_mul(2)
        .and_then(|v| v.checked_add(old))
        .ok_or(ReferenceLoadError::Invalid("resampling kernel too large"))?;
    let target_frames = pcm
        .len()
        .checked_mul(new)
        .and_then(|v| v.checked_add(old - 1))
        .map(|v| v / old)
        .ok_or(ReferenceLoadError::Invalid("resampling output too large"))?;
    if target_frames > MAX_DECODED_FRAMES {
        return Err(ReferenceLoadError::Invalid(
            "resampled reference exceeds duration limit",
        ));
    }

    // Match torchaudio.transforms.Resample's default sinc_interp_hann kernel.
    let kernel_len = new.checked_mul(taps).ok_or(ReferenceLoadError::Invalid(
        "resampling kernel length overflow",
    ))?;
    if kernel_len > MAX_KERNEL_COEFFICIENTS {
        return Err(ReferenceLoadError::Invalid(
            "unsupported sample-rate ratio: resampling kernel too large",
        ));
    }
    let mut kernel = vec![0.0f32; kernel_len];
    for phase in 0..new {
        for tap in 0..taps {
            let offset = (tap as f64 - width as f64) / old as f64;
            let t = (offset - phase as f64 / new as f64) * base;
            let t = t.clamp(-6.0, 6.0);
            let window = (t * std::f64::consts::PI / 12.0).cos().powi(2);
            let pi_t = t * std::f64::consts::PI;
            let sinc = if pi_t == 0.0 { 1.0 } else { pi_t.sin() / pi_t };
            kernel[phase * taps + tap] = (sinc * window * base / old as f64) as f32;
        }
    }
    let mut result = Vec::with_capacity(target_frames);
    for j in 0..target_frames {
        let phase = j % new;
        let input_start = j / new * old;
        let mut acc = 0.0f32;
        for tap in 0..taps {
            let src = input_start + tap;
            if src >= width && src - width < pcm.len() {
                acc += pcm[src - width] * kernel[phase * taps + tap];
            }
        }
        result.push(acc);
    }
    Ok(result)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(rate: u32, channels: u16, frames: &[[i16; 2]]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(
                &mut cursor,
                hound::WavSpec {
                    channels,
                    sample_rate: rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            for frame in frames {
                for &v in frame.iter().take(channels as usize) {
                    writer.write_sample(v).unwrap();
                }
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn stereo_44100_mixes_channels_as_python() {
        let samples = vec![[16384, -16384]; 4096];
        let output = ReferenceAudioLoader::default()
            .load_bytes(&wav(44100, 2, &samples))
            .unwrap();
        assert_eq!(output.source_rate, 44100);
        assert_eq!(output.source_channels, 2);
        assert_eq!(output.pcm.len(), samples.len());
        assert!(output.pcm.iter().all(|x| x.abs() < 1e-6));
        assert_eq!(output.prepare_mel().unwrap().mels.channels, 160);
    }

    #[test]
    fn mono_16000_resamples_to_firefly_rate() {
        let samples = (0..4800)
            .map(|i| [((i as f32 * 0.08).sin() * 24000.0) as i16, 0])
            .collect::<Vec<_>>();
        let result = ReferenceAudioLoader::default()
            .load_bytes(&wav(16000, 1, &samples))
            .unwrap();
        assert_eq!(result.source_rate, 16000);
        assert_eq!(result.sample_rate, 44100);
        assert!((result.pcm.len() as i64 - 13230).abs() <= 1);
        assert!(result.pcm.iter().all(|x| x.is_finite()));
        assert_eq!(
            result.prepare_mel().unwrap().mels.frames,
            result.pcm.len() / 512
        );
    }

    #[test]
    fn optional_ffmpeg_recovers_mp3_aac_and_asf_in_memory() {
        if Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return;
        }
        let samples = (0..8_192)
            .map(|i| [((i as f32 * 0.07).sin() * 22_000.0) as i16, 0])
            .collect::<Vec<_>>();
        let source = wav(44_100, 1, &samples);
        for format in ["mp3", "adts", "asf", "mp4"] {
            let mut encoder = Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "wav",
                    "-i",
                    "pipe:0",
                ])
                .args(if format == "mp4" {
                    vec!["-c:a", "aac", "-movflags", "frag_keyframe+empty_moov"]
                } else {
                    vec![]
                })
                .args(["-f", format, "pipe:1"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            encoder.stdin.take().unwrap().write_all(&source).unwrap();
            let compressed = encoder.wait_with_output().unwrap();
            assert!(compressed.status.success(), "{format}");
            let native = ReferenceAudioLoader::default().load_bytes(&compressed.stdout);
            assert!(native.is_err(), "{format} should require the opt-in codec");
            let decoded = ReferenceAudioLoader::default()
                .with_ffmpeg("ffmpeg")
                .load_bytes(&compressed.stdout)
                .unwrap();
            assert_eq!(decoded.sample_rate, 44_100, "{format}");
            assert_eq!(decoded.source_channels, 1, "{format}");
            assert!(decoded.pcm.len() >= 4096, "{format}");
            assert!(decoded.pcm.iter().all(|v| v.is_finite()), "{format}");
        }
    }

    #[test]
    fn invalid_and_empty_inputs_fail() {
        assert!(ReferenceAudioLoader::default().load_bytes(&[]).is_err());
        assert!(
            ReferenceAudioLoader::default()
                .load_bytes(b"random nonsense")
                .is_err()
        );
        assert!(resample_mono(&[f32::NAN], 16000, 44100).is_err());
    }

    #[test]
    fn resampler_matches_torchaudio_sinc_hann_golden() {
        let source = (0..2000)
            .map(|i| {
                (i as f64 * 0.028).sin() as f32 * 0.4 + (i as f64 * 0.0018).cos() as f32 * 0.15
            })
            .collect::<Vec<_>>();
        let golden = include_str!("../fixtures/reference_resample_torchaudio.tsv");
        for rate in [16000usize, 48000] {
            let actual = resample_mono(&source, rate, 44100).unwrap();
            for line in golden.lines() {
                if line.starts_with('#') || line.is_empty() {
                    continue;
                }
                let fields = line.split_whitespace().collect::<Vec<_>>();
                if fields[0].parse::<usize>().unwrap() != rate {
                    continue;
                }
                let length = fields[2].parse::<usize>().unwrap();
                let index = fields[3].parse::<usize>().unwrap();
                let expected = fields[4].parse::<f32>().unwrap();
                assert_eq!(actual.len(), length);
                assert!(
                    (actual[index] - expected).abs() < 2e-5,
                    "rate={rate} frame={index}, rust={}, torch={expected}",
                    actual[index]
                );
            }
        }
    }
}
