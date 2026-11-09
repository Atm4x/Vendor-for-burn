use std::{fs, path::Path};

pub struct MonoAudio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

/// Read uncompressed little-endian mono/stereo WAV, converting channels to mono.
pub fn read_mono_wav(path: impl AsRef<Path>) -> Result<MonoAudio, String> {
    let bytes = fs::read(path).map_err(|e| format!("read WAV: {e}"))?;
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("input must be a RIFF/WAVE file".into());
    }
    let mut cursor = 12;
    let (mut format, mut data) = (None, None);
    while cursor + 8 <= bytes.len() {
        let chunk_size =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let start = cursor + 8;
        let end = start
            .checked_add(chunk_size)
            .ok_or("WAV chunk size overflow")?;
        if end > bytes.len() {
            return Err("truncated WAV chunk".into());
        }
        if &bytes[cursor..cursor + 4] == b"fmt " {
            format = Some(&bytes[start..end]);
        }
        if &bytes[cursor..cursor + 4] == b"data" {
            data = Some(&bytes[start..end]);
        }
        cursor = end
            .checked_add(chunk_size & 1)
            .ok_or("WAV chunk padding overflow")?;
    }
    let f = format.ok_or("missing WAV format chunk")?;
    let content = data.ok_or("missing WAV audio chunk")?;
    if f.len() < 16 {
        return Err("invalid WAV format chunk".into());
    }
    let codec = u16::from_le_bytes(f[0..2].try_into().unwrap());
    let channels = u16::from_le_bytes(f[2..4].try_into().unwrap()) as usize;
    let sample_rate = u32::from_le_bytes(f[4..8].try_into().unwrap());
    let bits = u16::from_le_bytes(f[14..16].try_into().unwrap());
    if channels == 0 || channels > 2 || sample_rate < 8_000 || sample_rate > 384_000 {
        return Err("WAV requires one or two channels and a supported sample rate".into());
    }
    let bytes_per_sample = match (codec, bits) {
        (1, 16) => 2,
        (1, 24) => 3,
        (1, 32) | (3, 32) => 4,
        _ => return Err(format!("WAV codec {codec}, depth {bits} unsupported")),
    };
    let frame_bytes = channels * bytes_per_sample;
    if content.len() % frame_bytes != 0 {
        return Err("WAV data has incomplete frame".into());
    }
    let mut samples = Vec::with_capacity(content.len() / frame_bytes);
    for frame in content.chunks_exact(frame_bytes) {
        let mut mixed = 0.0;
        for channel in 0..channels {
            let part = &frame[channel * bytes_per_sample..(channel + 1) * bytes_per_sample];
            let sample = match (codec, bits) {
                (1, 16) => i16::from_le_bytes(part.try_into().unwrap()) as f32 / 32768.0,
                (1, 24) => {
                    ((i32::from(part[0]) | (i32::from(part[1]) << 8) | (i32::from(part[2]) << 16))
                        << 8
                        >> 8) as f32
                        / 8388608.0
                }
                (1, 32) => i32::from_le_bytes(part.try_into().unwrap()) as f32 / 2147483648.0,
                (3, 32) => f32::from_le_bytes(part.try_into().unwrap()),
                _ => unreachable!(),
            };
            if !sample.is_finite() {
                return Err("non-finite audio sample".into());
            }
            mixed += sample / channels as f32;
        }
        samples.push(mixed);
    }
    Ok(MonoAudio {
        sample_rate,
        samples,
    })
}

/// Windowed-sinc bandwidth-limited offline resampling, suitable for RVC's 16-kHz content input.
pub fn resample(input: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>, String> {
    if from_hz == 0 || to_hz == 0 {
        return Err("sample rate must be positive".into());
    }
    if from_hz == to_hz {
        return Ok(input.to_vec());
    }
    let output_len = (input.len() as u64)
        .checked_mul(to_hz as u64)
        .ok_or("audio resample size overflow")?
        / from_hz as u64;
    if output_len > usize::MAX as u64 {
        return Err("resampled audio length overflow".into());
    }
    let cutoff = (to_hz as f64 / from_hz as f64).min(1.0) * 0.96;
    let radius = 24_isize;
    let mut output = Vec::with_capacity(output_len as usize);
    for frame in 0..output_len {
        let center = frame as f64 * from_hz as f64 / to_hz as f64;
        let origin = center.floor() as isize;
        let mut weighted = 0.0;
        let mut total = 0.0;
        for index in origin - radius + 1..=origin + radius {
            if index < 0 || index >= input.len() as isize {
                continue;
            }
            let offset = center - index as f64;
            let phase = std::f64::consts::PI * offset * cutoff;
            let sinc = if phase.abs() < 1e-12 {
                1.0
            } else {
                phase.sin() / phase
            };
            let window = 0.5 + 0.5 * (std::f64::consts::PI * offset / radius as f64).cos();
            let weight = cutoff * sinc * window;
            weighted += weight * input[index as usize] as f64;
            total += weight;
        }
        output.push(if total == 0.0 {
            0.0
        } else {
            (weighted / total) as f32
        });
    }
    Ok(output)
}

pub fn write_mono_wav(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), String> {
    let payload_size = samples.len().checked_mul(2).ok_or("WAV length overflow")?;
    let payload = u32::try_from(payload_size).map_err(|e| e.to_string())?;
    let chunk_size = 36_u32
        .checked_add(payload)
        .ok_or("WAV chunk size exceeds RIFF limit")?;
    let mut bytes = Vec::with_capacity(44 + payload_size);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend(chunk_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend(16_u32.to_le_bytes());
    bytes.extend(1_u16.to_le_bytes());
    bytes.extend(1_u16.to_le_bytes());
    bytes.extend(sample_rate.to_le_bytes());
    bytes.extend((sample_rate * 2).to_le_bytes());
    bytes.extend(2_u16.to_le_bytes());
    bytes.extend(16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend(payload.to_le_bytes());
    for &sample in samples {
        let sample = sample.clamp(-1.0, 1.0);
        let value = (sample * 32767.0).round() as i16;
        bytes.extend(value.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|e| format!("write WAV: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_constants_and_identity() {
        let source = vec![0.3; 44_100];
        assert_eq!(resample(&source, 44_100, 44_100).unwrap(), source);
        let result = resample(&source, 44_100, 16_000).unwrap();
        assert_eq!(result.len(), 16_000);
        assert!(result[100..15_900].iter().all(|x| (x - 0.3).abs() < 1e-5));
    }

    #[test]
    fn wav_roundtrip() {
        let path = std::env::temp_dir().join(format!("burn-rvc-wav-{}.wav", std::process::id()));
        write_mono_wav(&path, &[0.0, 0.5, -0.5], 40_000).unwrap();
        let file = read_mono_wav(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(file.sample_rate, 40_000);
        assert!((file.samples[1] - 0.5).abs() < 0.001);
    }
}
