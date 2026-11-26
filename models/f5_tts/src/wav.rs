use crate::{F5Error, Result, mel::SAMPLE_RATE};

pub fn mono_pcm16(samples: &[f32]) -> Result<Vec<u8>> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(F5Error::InvalidShape(
            "WAV requires nonempty finite mono PCM",
        ));
    }
    let converted: Vec<i16> = samples.iter().map(|&sample| {
        (sample.clamp(-1.0, 1.0) * 32768.0).round().clamp(-32768.0, 32767.0) as i16
    }).collect();
    mono_pcm16_from_i16(&converted)
}

pub fn mono_pcm16_from_i16(samples: &[i16]) -> Result<Vec<u8>> {
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .and_then(|length| u32::try_from(length).ok())
        .ok_or(F5Error::InvalidShape("WAV exceeds 4 GiB limit"))?;
    let riff_bytes = data_bytes
        .checked_add(36)
        .ok_or(F5Error::InvalidShape("RIFF exceeds 4 GiB limit"))?;
    let mut result = Vec::with_capacity(44 + data_bytes as usize);
    result.extend_from_slice(b"RIFF");
    result.extend_from_slice(&riff_bytes.to_le_bytes());
    result.extend_from_slice(b"WAVEfmt ");
    result.extend_from_slice(&16_u32.to_le_bytes());
    result.extend_from_slice(&1_u16.to_le_bytes());
    result.extend_from_slice(&1_u16.to_le_bytes());
    result.extend_from_slice(&(SAMPLE_RATE as u32).to_le_bytes());
    result.extend_from_slice(&((SAMPLE_RATE * 2) as u32).to_le_bytes());
    result.extend_from_slice(&2_u16.to_le_bytes());
    result.extend_from_slice(&16_u16.to_le_bytes());
    result.extend_from_slice(b"data");
    result.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        result.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn produces_valid_mono_wave_header_and_clamps_samples() {
        let output = mono_pcm16(&[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]).unwrap();
        assert_eq!(&output[..4], b"RIFF");
        assert_eq!(&output[8..12], b"WAVE");
        assert_eq!(
            u32::from_le_bytes(output[24..28].try_into().unwrap()),
            24_000
        );
        assert_eq!(u16::from_le_bytes(output[34..36].try_into().unwrap()), 16);
        assert_eq!(u32::from_le_bytes(output[40..44].try_into().unwrap()), 12);
        assert_eq!(
            i16::from_le_bytes(output[44..46].try_into().unwrap()),
            -32768
        );
        assert_eq!(
            i16::from_le_bytes(output[52..54].try_into().unwrap()),
            32767
        );
        assert_eq!(output.len(), 56);
    }

    #[test]
    fn rejects_nan_and_empty_pcm() {
        assert!(mono_pcm16(&[]).is_err());
        assert!(mono_pcm16(&[0.0, f32::NAN]).is_err());
    }
}
