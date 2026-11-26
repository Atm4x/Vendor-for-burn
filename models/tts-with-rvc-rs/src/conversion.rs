use crate::{
    audio_io,
    chunked::{ChunkConfig, ChunkProgress},
    preprocess,
    rvc_inference::{InferenceOptions, RvcV2Inferencer},
};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ConversionSettings {
    pub inference: InferenceOptions,
    pub chunk: ChunkConfig,
    pub batch_size: usize,
    pub rms_mix_rate: f32,
}

impl ConversionSettings {
    pub fn validate(&self) -> Result<(), String> {
        self.chunk.validate()?;
        if !(1..=32).contains(&self.batch_size) {
            return Err("tensor batch size must be 1..=32".into());
        }
        if !self.rms_mix_rate.is_finite() || !(0.0..=1.0).contains(&self.rms_mix_rate) {
            return Err("rms mix rate must be finite and in 0..=1".into());
        }
        if !self.inference.index_rate.is_finite()
            || !(0.0..=1.0).contains(&self.inference.index_rate)
        {
            return Err("index rate must be finite and in 0..=1".into());
        }
        if !self.inference.pitch_shift_semitones.is_finite() {
            return Err("pitch shift must be finite".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversionReport {
    pub input_samples: usize,
    pub input_sample_rate: u32,
    pub output_samples: usize,
    pub output_sample_rate: u32,
}

pub fn convert_wav_bytes(
    model: &RvcV2Inferencer,
    input: &[u8],
    settings: &ConversionSettings,
    progress: impl FnMut(ChunkProgress),
) -> Result<(Vec<u8>, ConversionReport), String> {
    settings.validate()?;
    let source = audio_io::read_mono_wav_bytes(input)?;
    let input_samples = source.samples.len();
    let input_sample_rate = source.sample_rate;
    let mono16k = audio_io::resample(&source.samples, source.sample_rate, 16_000)?;
    let filtered = preprocess::highpass_48hz_zero_phase(&mono16k)?;
    let mut converted = model.infer_long_filtered_16khz_batched(
        &filtered,
        &settings.inference,
        settings.chunk,
        settings.batch_size,
        progress,
    )?;
    audio_io::mix_rms_envelope(
        &filtered,
        16_000,
        &mut converted,
        40_000,
        settings.rms_mix_rate,
    )?;
    audio_io::normalize_peak(&mut converted)?;
    let report = ConversionReport {
        input_samples,
        input_sample_rate,
        output_samples: converted.len(),
        output_sample_rate: 40_000,
    };
    let wav = audio_io::encode_mono_wav(&converted, 40_000)?;
    Ok((wav, report))
}

pub fn convert_file(
    model: &RvcV2Inferencer,
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    settings: &ConversionSettings,
    progress: impl FnMut(ChunkProgress),
) -> Result<ConversionReport, String> {
    let input = input.as_ref();
    let output = output.as_ref();
    if input.as_os_str().is_empty() || output.as_os_str().is_empty() || input == output {
        return Err("input and output must be nonempty and distinct".into());
    }
    let bytes = std::fs::read(input).map_err(|e| format!("read WAV: {e}"))?;
    let (converted, report) = convert_wav_bytes(model, &bytes, settings, progress)?;
    std::fs::write(output, converted).map_err(|e| format!("write WAV: {e}"))?;
    Ok(report)
}
