use std::io::Cursor;
use std::path::Path;

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
}

impl Default for ReferenceAudioLoader {
    fn default() -> Self {
        Self::new(SAMPLE_RATE)
    }
}

impl ReferenceAudioLoader {
    pub fn new(target_rate: usize) -> Self {
        Self { target_rate }
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
