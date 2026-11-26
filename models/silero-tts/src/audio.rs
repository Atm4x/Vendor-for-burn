//! Mono PCM16 WAV output compatible with the Silero package's save_wav interface.

use anyhow::{Result, ensure};
use std::path::Path;

pub fn write_wav_pcm16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    ensure!(sample_rate > 0, "WAV sample rate must be positive");
    ensure!(samples.iter().all(|s| s.is_finite()), "WAV contains non-finite PCM samples");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample(((*sample * 32767.0).trunc() as i32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_matches_original_mono_int16_conversion() {
        let mut path = std::env::temp_dir();
        path.push(format!("silero-wav-{}.wav", std::process::id()));
        let samples = [0.0, 1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 1.25];
        write_wav_pcm16(&path, &samples, 24000).unwrap();
        let mut reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_rate, 24000);
        let got: Vec<i16> = reader.samples::<i16>().map(|value| value.unwrap()).collect();
        assert_eq!(got, [0, 32767, -32767, 16383, -16383, 3276, -3276, -24578]);
        drop(reader);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_audio_is_rejected_before_file_creation() {
        let mut path = std::env::temp_dir();
        path.push(format!("silero-invalid-{}.wav", std::process::id()));
        assert!(write_wav_pcm16(&path, &[f32::NAN], 24000).is_err());
        assert!(!path.exists());
    }
}
