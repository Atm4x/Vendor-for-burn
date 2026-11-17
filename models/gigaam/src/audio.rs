use anyhow::{Context, Result, ensure};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;
use std::f32::consts::PI;
use std::path::Path;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FFT_SIZE: usize = 400;
pub const HOP: usize = 160;

pub fn load_wav_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate >= 8000 && spec.sample_rate <= 192000,
        "unsupported sample rate"
    );
    ensure!(
        spec.channels >= 1 && spec.channels <= 8,
        "unsupported channel count"
    );
    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            ensure!(
                (8..=32).contains(&spec.bits_per_sample),
                "unsupported integer WAV bit depth"
            );
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<std::result::Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
    };
    let mono: Vec<f32> = data
        .chunks_exact(spec.channels as usize)
        .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
        .collect();
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    Ok(resample_sinc(&mono, spec.sample_rate, SAMPLE_RATE))
}

/// Decode PCM WAV, FLAC, or Ogg/Vorbis entirely in Rust and resample to 16 kHz.
/// Compressed audio uses Symphonia; PCM WAV keeps the existing tested Hound path.
pub fn load_audio_16k(path: &Path) -> Result<Vec<f32>> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
    {
        return load_wav_16k(path);
    }

    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as AudioError;
    use symphonia::core::formats::{FormatOptions, TrackType, probe::Hint};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let src =
        std::fs::File::open(path).with_context(|| format!("opening audio {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .with_context(|| format!("detecting audio format {}", path.display()))?;
    let track = format
        .default_track(TrackType::Audio)
        .context("no supported audio track")?;
    let audio_params = track
        .codec_params
        .as_ref()
        .context("missing audio codec parameters")?
        .audio()
        .context("track is not audio")?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .context("audio codec unsupported by native decoder")?;
    let track_id = track.id;
    let mut pcm = Vec::<f32>::new();
    let mut rate = None;
    let mut scratch = Vec::<f32>::new();
    while let Some(packet) = format.next_packet().context("reading audio packet")? {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(AudioError::DecodeError(error)) => anyhow::bail!("corrupt audio packet: {error}"),
            Err(error) => return Err(error.into()),
        };
        let current_rate = decoded.spec().rate();
        let channels = decoded.spec().channels().count();
        ensure!(
            (8_000..=192_000).contains(&current_rate) && (1..=8).contains(&channels),
            "unsupported sample rate or channel count"
        );
        ensure!(
            rate.is_none_or(|existing| existing == current_rate),
            "changing sample rate inside audio stream is unsupported"
        );
        rate = Some(current_rate);
        scratch.resize(decoded.samples_interleaved(), 0.0);
        decoded.copy_to_slice_interleaved(&mut scratch);
        pcm.extend(
            scratch
                .chunks_exact(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32),
        );
    }
    ensure!(!pcm.is_empty(), "audio file contains no decoded samples");
    let rate = rate.context("audio stream has no sample rate")?;
    if rate == SAMPLE_RATE {
        Ok(pcm)
    } else {
        Ok(resample_sinc(&pcm, rate, SAMPLE_RATE))
    }
}

pub fn resample_sinc(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    if input_rate == output_rate {
        return samples.to_vec();
    }
    let output_len = (samples.len() as u64 * output_rate as u64 / input_rate as u64) as usize;
    let ratio = input_rate as f64 / output_rate as f64;
    let cutoff = (0.95f64 * output_rate as f64 / input_rate as f64).min(0.95);
    let width: isize = 32;
    (0..output_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let pivot = pos.floor() as isize;
            let mut sum = 0.0f64;
            let mut weights = 0.0f64;
            for j in (pivot - width + 1)..=(pivot + width) {
                if j < 0 || j >= samples.len() as isize {
                    continue;
                }
                let t = pos - j as f64;
                let x = std::f64::consts::PI * t;
                let sinc = if x.abs() < 1e-10 {
                    cutoff
                } else {
                    (cutoff * x).sin() / x
                };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * t / width as f64).cos();
                let w = sinc * window;
                sum += w * samples[j as usize] as f64;
                weights += w;
            }
            if weights.abs() < 1e-12 {
                0.0
            } else {
                (sum / weights) as f32
            }
        })
        .collect()
}

fn reflect_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let r = index.rem_euclid(period);
    if r < len as isize {
        r as usize
    } else {
        (period - r) as usize
    }
}

fn htk_hz_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}
fn htk_mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct MelConfig {
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub center: bool,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_fft: FFT_SIZE,
            win_length: FFT_SIZE,
            hop_length: HOP,
            center: true,
        }
    }
}

impl MelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.n_fft >= 2 && self.n_fft <= 8192,
            "n_fft outside 2..=8192"
        );
        ensure!(
            self.win_length >= 1 && self.win_length <= self.n_fft,
            "win_length must be between 1 and n_fft"
        );
        ensure!(
            self.hop_length >= 1 && self.hop_length <= 8192,
            "invalid hop_length"
        );
        Ok(())
    }
}

fn mel_weights(n_mels: usize, n_fft: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let low = htk_hz_to_mel(0.0);
    let high = htk_hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| htk_mel_to_hz(low + (high - low) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let mut weights = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        for k in 0..bins {
            let f = k as f32 * SAMPLE_RATE as f32 / n_fft as f32;
            let lo = (f - mel_points[m]) / (mel_points[m + 1] - mel_points[m]);
            let hi = (mel_points[m + 2] - f) / (mel_points[m + 2] - mel_points[m + 1]);
            weights[m * bins + k] = lo.min(hi).max(0.0);
        }
    }
    weights
}

#[derive(Debug, Clone)]
pub struct Features {
    /// [mel_bins, frames], channels-first
    pub values: Vec<f32>,
    pub n_mels: usize,
    pub frames: usize,
}

pub fn log_mel(samples: &[f32], n_mels: usize) -> Result<Features> {
    log_mel_with_config(samples, n_mels, MelConfig::default())
}

pub fn log_mel_with_config(samples: &[f32], n_mels: usize, config: MelConfig) -> Result<Features> {
    config.validate()?;
    let n_fft = config.n_fft;
    let hop = config.hop_length;
    if config.center {
        ensure!(
            samples.len() > n_fft / 2,
            "audio must have more than {} samples for reflect-padded STFT",
            n_fft / 2
        );
    } else {
        ensure!(
            samples.len() >= n_fft,
            "audio must have at least {n_fft} samples for uncentered STFT"
        );
    }
    ensure!(
        samples.iter().all(|sample| sample.is_finite()),
        "audio contains non-finite PCM samples"
    );
    ensure!((1..=256).contains(&n_mels), "n_mels outside 1..=256");
    let frames = if config.center {
        samples.len() / hop + 1
    } else {
        (samples.len() - n_fft) / hop + 1
    };
    let bins = n_fft / 2 + 1;
    let weights = mel_weights(n_mels, n_fft);
    let mut data = vec![0f32; n_mels * frames];
    let window_start = (n_fft - config.win_length) / 2;
    let mut hann = vec![0.0f32; n_fft];
    for i in 0..config.win_length {
        hann[window_start + i] = 0.5 - 0.5 * (2.0 * PI * i as f32 / config.win_length as f32).cos();
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut scratch = vec![Complex::<f32>::default(); fft.get_inplace_scratch_len()];
    let mut buffer = vec![Complex::<f32>::default(); n_fft];
    for t in 0..frames {
        let start = t * hop;
        for i in 0..n_fft {
            let index = if config.center {
                reflect_index(
                    start as isize + i as isize - (n_fft / 2) as isize,
                    samples.len(),
                )
            } else {
                start + i
            };
            buffer[i] = Complex::new(samples[index] * hann[i], 0.0);
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        for m in 0..n_mels {
            let mut mel = 0.0f32;
            for k in 0..bins {
                mel += buffer[k].norm_sqr() * weights[m * bins + k];
            }
            data[m * frames + t] = mel.clamp(1e-9, 1e9).ln();
        }
    }
    Ok(Features {
        values: data,
        n_mels,
        frames,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct AudioChunk<'a> {
    pub start_sample: usize,
    pub samples: &'a [f32],
}

/// Returns non-overlapping chunks and exact offsets in the original PCM.
pub fn speech_chunks_with_offsets(samples: &[f32], max_samples: usize) -> Vec<AudioChunk<'_>> {
    if samples.is_empty() || max_samples == 0 {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut pos = 0;
    while pos < samples.len() {
        let end = pos.saturating_add(max_samples).min(samples.len());
        let mut split = end;
        if end < samples.len() && max_samples > SAMPLE_RATE as usize {
            let begin = end.saturating_sub(SAMPLE_RATE as usize).max(pos + 1);
            let step = 160;
            let mut energy = f64::MAX;
            for i in (begin..end).step_by(step) {
                let e = samples[i..(i + step).min(samples.len())]
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>();
                if e < energy {
                    energy = e;
                    split = i;
                }
            }
        }
        chunks.push(AudioChunk {
            start_sample: pos,
            samples: &samples[pos..split],
        });
        pos = split;
    }
    chunks
}

/// A half-open speech interval in the original 16 kHz PCM, typically from an external VAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechRegion {
    pub start_sample: usize,
    pub end_sample: usize,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct SpeechRegionSeconds {
    pub start: f64,
    pub end: f64,
}

impl SpeechRegionSeconds {
    /// Convert seconds to 16 kHz sample indices, rejecting invalid VAD outputs.
    pub fn into_samples(self, pcm_samples: usize) -> Result<SpeechRegion> {
        ensure!(
            self.start.is_finite() && self.end.is_finite(),
            "VAD boundaries must be finite"
        );
        ensure!(
            self.start >= 0.0 && self.end > self.start,
            "invalid VAD time interval"
        );
        let start = (self.start * SAMPLE_RATE as f64).round();
        let end = (self.end * SAMPLE_RATE as f64).round();
        ensure!(
            start >= 0.0 && end > start && end <= pcm_samples as f64,
            "VAD interval is empty, exceeds PCM length or rounds to zero samples"
        );
        Ok(SpeechRegion {
            start_sample: start as usize,
            end_sample: end as usize,
        })
    }
}

/// Parameters of the grouping stage from GigaAM's Python `segment_audio_file`.
/// All durations are represented in 16 kHz samples to avoid float rounding.
#[derive(Debug, Clone, Copy)]
pub struct VadGrouping {
    pub max_samples: usize,
    pub min_samples: usize,
    pub strict_limit_samples: usize,
    pub new_chunk_threshold_samples: usize,
}

impl Default for VadGrouping {
    fn default() -> Self {
        let rate = SAMPLE_RATE as usize;
        Self {
            max_samples: 22 * rate,
            min_samples: 15 * rate,
            strict_limit_samples: 30 * rate,
            new_chunk_threshold_samples: rate / 5,
        }
    }
}

fn validate_speech_regions(samples: &[f32], regions: &[SpeechRegion]) -> Result<()> {
    let mut previous_end = 0;
    for (index, region) in regions.iter().enumerate() {
        ensure!(
            region.start_sample < region.end_sample,
            "speech region {index} is empty or reversed"
        );
        ensure!(
            region.end_sample <= samples.len(),
            "speech region {index} exceeds PCM length"
        );
        ensure!(
            region.start_sample >= previous_end,
            "speech region {index} overlaps or is unordered"
        );
        previous_end = region.end_sample;
    }
    Ok(())
}

/// Group externally detected speech regions according to GigaAM's Python
/// longform segmenter. Unlike `speech_chunks_in_regions`, neighboring VAD
/// regions may be joined into one contiguous PCM slice including the silence
/// between them. Chunks over `strict_limit_samples` are divided equally.
/// No VAD model is loaded; only the supplied intervals are used.
pub fn speech_chunks_grouped_vad<'a>(
    samples: &'a [f32],
    regions: &[SpeechRegion],
    options: VadGrouping,
) -> Result<Vec<AudioChunk<'a>>> {
    validate_speech_regions(samples, regions)?;
    ensure!(
        options.max_samples > 0 && options.min_samples > 0 && options.strict_limit_samples > 0,
        "VAD grouping durations must be positive"
    );

    fn flush<'a>(
        samples: &'a [f32],
        chunks: &mut Vec<AudioChunk<'a>>,
        start: usize,
        end: usize,
        strict_limit_samples: usize,
    ) {
        let duration = end - start;
        let pieces = duration / strict_limit_samples + 1;
        let pieces = if duration > strict_limit_samples {
            pieces
        } else {
            1
        };
        for i in 0..pieces {
            let left = start + duration * i / pieces;
            let right = start + duration * (i + 1) / pieces;
            chunks.push(AudioChunk {
                start_sample: left,
                samples: &samples[left..right],
            });
        }
    }

    let mut chunks = Vec::new();
    let (mut curr_start, mut curr_end, mut curr_duration) = (0usize, 0usize, 0.0_f64);
    let rate = SAMPLE_RATE as f64;
    for region in regions {
        let start = region.start_sample;
        let end = region.end_sample;
        if curr_duration == 0.0 {
            curr_start = start;
        } else if curr_duration > options.new_chunk_threshold_samples as f64 / rate
            && (curr_duration + (end as f64 / rate - curr_end as f64 / rate)
                > options.max_samples as f64 / rate
                || curr_duration > options.min_samples as f64 / rate)
        {
            flush(
                samples,
                &mut chunks,
                curr_start,
                curr_end,
                options.strict_limit_samples,
            );
            curr_start = start;
        }
        curr_end = end;
        curr_duration = (curr_end as f64 / rate) - (curr_start as f64 / rate);
    }
    if curr_duration > options.new_chunk_threshold_samples as f64 / rate {
        flush(
            samples,
            &mut chunks,
            curr_start,
            curr_end,
            options.strict_limit_samples,
        );
    }
    Ok(chunks)
}

/// Return VAD-aligned chunks without copying PCM. Long regions are subdivided
/// using the same low-energy boundary heuristic as the fallback longform path.
pub fn speech_chunks_in_regions<'a>(
    samples: &'a [f32],
    regions: &[SpeechRegion],
    max_samples: usize,
) -> Result<Vec<AudioChunk<'a>>> {
    ensure!(max_samples > 0, "maximum chunk length must be positive");
    validate_speech_regions(samples, regions)?;

    let mut chunks = Vec::new();
    for region in regions {
        let sub = &samples[region.start_sample..region.end_sample];
        for chunk in speech_chunks_with_offsets(sub, max_samples) {
            chunks.push(AudioChunk {
                start_sample: region.start_sample + chunk.start_sample,
                samples: chunk.samples,
            });
        }
    }
    Ok(chunks)
}

pub fn speech_chunks(samples: &[f32], max_samples: usize) -> Vec<&[f32]> {
    speech_chunks_with_offsets(samples, max_samples)
        .into_iter()
        .map(|chunk| chunk.samples)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn log_mel_rejects_torchaudio_reflect_padding_underflow() {
        for count in [0, 1, 199, 200] {
            assert!(log_mel(&vec![0.0; count], 64).is_err(), "length {count}");
        }
        assert_eq!(log_mel(&vec![0.0; 201], 64).unwrap().frames, 2);
    }

    #[test]
    fn log_mel_rejects_invalid_pcm_values() {
        let mut samples = vec![0.0; 16_000];
        samples[400] = f32::NAN;
        assert!(log_mel(&samples, 64).is_err());
        samples[400] = f32::INFINITY;
        assert!(log_mel(&samples, 64).is_err());
    }

    #[test]
    fn uncentered_stft_has_official_v3_frame_count() {
        let config = MelConfig {
            n_fft: 320,
            win_length: 320,
            hop_length: 160,
            center: false,
        };
        assert!(log_mel_with_config(&vec![0.0; 319], 64, config).is_err());
        assert_eq!(
            log_mel_with_config(&vec![0.0; 320], 64, config)
                .unwrap()
                .frames,
            1
        );
        assert_eq!(
            log_mel_with_config(&vec![0.0; 16_000], 64, config)
                .unwrap()
                .frames,
            99
        );
    }

    #[test]
    fn mel_frame_count() {
        let m = log_mel(&vec![0.0; 16000], 64).unwrap();
        assert_eq!(m.frames, 101);
        assert_eq!(m.values.len(), 64 * 101);
        assert!(m.values.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn vad_seconds_reject_nonfinite_and_zero_sample_segments() {
        let convert = |start, end| SpeechRegionSeconds { start, end }.into_samples(16_000);
        assert_eq!(
            convert(0.25, 0.75).unwrap(),
            SpeechRegion {
                start_sample: 4_000,
                end_sample: 12_000
            }
        );
        for (start, end) in [
            (f64::NAN, 0.4),
            (0.0, f64::INFINITY),
            (-0.1, 0.1),
            (0.4, 0.3),
            (1.0, 1.1),
            (0.0, 0.00001),
        ] {
            assert!(convert(start, end).is_err());
        }
    }

    #[test]
    fn external_vad_regions_keep_absolute_offsets() {
        let pcm = (0..(16_000 * 9)).map(|x| x as f32).collect::<Vec<_>>();
        let regions = [
            SpeechRegion {
                start_sample: 4_000,
                end_sample: 28_000,
            },
            SpeechRegion {
                start_sample: 64_000,
                end_sample: 112_000,
            },
        ];
        let chunks = speech_chunks_in_regions(&pcm, &regions, 16_000).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start_sample, 4_000);
        assert_eq!(
            chunks.last().unwrap().start_sample + chunks.last().unwrap().samples.len(),
            112_000
        );
        assert!(chunks.iter().all(|c| c.samples.len() <= 16_000));
        for chunk in &chunks {
            let base = chunk.start_sample;
            assert_eq!(chunk.samples[0], pcm[base]);
        }
        let total = chunks.iter().map(|c| c.samples.len()).sum::<usize>();
        assert_eq!(total, 24_000 + 48_000);
    }

    #[test]
    fn external_vad_regions_reject_invalid_intervals() {
        let pcm = vec![0.0; 100];
        for regions in [
            vec![SpeechRegion {
                start_sample: 10,
                end_sample: 10,
            }],
            vec![SpeechRegion {
                start_sample: 20,
                end_sample: 10,
            }],
            vec![SpeechRegion {
                start_sample: 0,
                end_sample: 101,
            }],
            vec![
                SpeechRegion {
                    start_sample: 10,
                    end_sample: 30,
                },
                SpeechRegion {
                    start_sample: 29,
                    end_sample: 40,
                },
            ],
            vec![
                SpeechRegion {
                    start_sample: 50,
                    end_sample: 60,
                },
                SpeechRegion {
                    start_sample: 10,
                    end_sample: 20,
                },
            ],
        ] {
            assert!(speech_chunks_in_regions(&pcm, &regions, 24).is_err());
        }
        assert!(speech_chunks_in_regions(&pcm, &[], 0).is_err());
        assert!(speech_chunks_in_regions(&pcm, &[], 24).unwrap().is_empty());
    }

    #[test]
    fn resampling_ratio() {
        assert_eq!(resample_sinc(&vec![0.0; 44100], 44100, 16000).len(), 16000);
    }
    #[test]
    fn speech_chunk_limits() {
        let x = vec![0f32; 16000 * 59];
        let c = speech_chunks(&x, 16000 * 24);
        assert!(c.len() >= 3);
        assert!(c.iter().all(|x| x.len() <= 16000 * 24));
        assert_eq!(c.iter().map(|x| x.len()).sum::<usize>(), x.len());
        let timed = speech_chunks_with_offsets(&x, 16000 * 24);
        assert_eq!(timed.first().unwrap().start_sample, 0);
        for pair in timed.windows(2) {
            assert_eq!(
                pair[0].start_sample + pair[0].samples.len(),
                pair[1].start_sample
            );
        }
        assert_eq!(
            timed.last().unwrap().start_sample + timed.last().unwrap().samples.len(),
            x.len()
        );
    }
}
