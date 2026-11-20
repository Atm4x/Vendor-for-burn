use std::{fs, path::Path};

pub(crate) fn write_batch(
    directory: &Path,
    chunk_indices: &[usize],
    input_samples: usize,
    frames: usize,
    f0: &[f32],
    pitch: &[i64],
) -> Result<(), String> {
    if chunk_indices.is_empty()
        || f0.len() != chunk_indices.len() * frames
        || pitch.len() != f0.len()
        || frames == 0
        || input_samples < 3_200
        || f0.iter().any(|value| !value.is_finite())
        || pitch.iter().any(|&value| !(1..=255).contains(&value))
    {
        return Err("invalid full-RVC F0 audit batch".into());
    }
    fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create F0 audit directory: {error}"))?;
    for (position, &index) in chunk_indices.iter().enumerate() {
        let start = position * frames;
        let end = start + frames;
        let prefix = directory.join(format!("chunk_{index:03}"));
        let f0_bytes = f0[start..end]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let pitch_bytes = pitch[start..end]
            .iter()
            .map(|&value| value as u8)
            .collect::<Vec<_>>();
        fs::write(prefix.with_extension("f0.f32le"), f0_bytes)
            .map_err(|error| format!("cannot write F0 audit values for chunk {index}: {error}"))?;
        fs::write(prefix.with_extension("pitch.u8"), pitch_bytes)
            .map_err(|error| format!("cannot write F0 audit pitch for chunk {index}: {error}"))?;
        let meta = serde_json::json!({
            "chunk_index": index,
            "frames": frames,
            "input_samples_16khz": input_samples,
            "f0_format": "f32le",
            "pitch_format": "u8",
            "source": "post-resample post-highpass exact full-RVC reflected-chunk RMVPE F0",
        });
        let meta = serde_json::to_vec_pretty(&meta)
            .map_err(|error| format!("cannot serialize F0 audit chunk {index}: {error}"))?;
        fs::write(prefix.with_extension("json"), meta).map_err(|error| {
            format!("cannot write F0 audit metadata for chunk {index}: {error}")
        })?;
    }
    Ok(())
}


/// Experimental full-RVC ablation: compare a fast F0 path against a reference
/// recorded from the exact same deterministic chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferenceMode {
    Voicing,
    Full,
}

impl ReferenceMode {
    pub(crate) fn parse(input: &str) -> Result<Self, String> {
        match input {
            "voicing" => Ok(Self::Voicing),
            "full" => Ok(Self::Full),
            other => Err(format!("unknown F0 reference ablation mode: {other}")),
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct ReferenceStats {
    pub(crate) frames: usize,
    pub(crate) voicing_changes: usize,
    pub(crate) replaced_values: usize,
}

pub(crate) fn apply_reference(
    directory: &Path,
    chunk_indices: &[usize],
    input_samples: usize,
    frames: usize,
    candidates: &mut [Vec<f32>],
    mode: ReferenceMode,
) -> Result<ReferenceStats, String> {
    if chunk_indices.is_empty() || candidates.len() != chunk_indices.len() || frames == 0 {
        return Err("invalid reference F0 batch dimensions".into());
    }
    let mut stats = ReferenceStats::default();
    for (&index, candidate) in chunk_indices.iter().zip(candidates.iter_mut()) {
        if candidate.len() < frames {
            return Err(format!("candidate F0 too short for chunk {index}"));
        }
        let prefix = directory.join(format!("chunk_{index:03}"));
        let metadata_bytes = fs::read(prefix.with_extension("json"))
            .map_err(|error| format!("F0 reference metadata missing for chunk {index}: {error}"))?;
        let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes)
            .map_err(|error| format!("invalid F0 reference metadata for chunk {index}: {error}"))?;
        if metadata["chunk_index"].as_u64() != Some(index as u64)
            || metadata["frames"].as_u64() != Some(frames as u64)
            || metadata["input_samples_16khz"].as_u64() != Some(input_samples as u64)
        {
            return Err(format!("F0 reference chunk geometry mismatch for chunk {index}"));
        }
        let bytes = fs::read(prefix.with_extension("f0.f32le"))
            .map_err(|error| format!("F0 reference data missing for chunk {index}: {error}"))?;
        if bytes.len() != frames * std::mem::size_of::<f32>() {
            return Err(format!("F0 reference length mismatch for chunk {index}"));
        }
        for (position, encoded) in bytes.chunks_exact(4).enumerate() {
            let reference = f32::from_le_bytes(encoded.try_into().expect("four bytes"));
            let current = &mut candidate[position];
            if !reference.is_finite()
                || !(0.0..=2_000.0).contains(&reference)
                || !current.is_finite() || *current < 0.0
            {
                return Err(format!("nonfinite or invalid F0 at chunk {index} frame {position}"));
            }
            let voicing_changed = (*current > 0.0) != (reference > 0.0);
            if voicing_changed {
                stats.voicing_changes += 1;
            }
            if (mode == ReferenceMode::Full || voicing_changed)
                && current.to_bits() != reference.to_bits()
            {
                *current = reference;
                stats.replaced_values += 1;
            }
        }
        stats.frames += frames;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_ablation_only_restores_voicing_or_all_values() {
        let path = std::env::temp_dir().join(format!("rvc-f0-reference-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        write_batch(
            &path, &[5, 14], 61_440, 4,
            &[100.0, 0.0, 200.0, 240.0, 0.0, 238.5, 225.0, 0.0],
            &[30, 1, 50, 67, 1, 66, 63, 1],
        ).unwrap();
        let source = [
            vec![100.01, 186.7, 199.99, 240.02],
            vec![0.0, 0.0, 225.01, 0.0],
        ];
        let mut voicing = source.clone();
        let result = apply_reference(
            &path, &[5, 14], 61_440, 4, &mut voicing, ReferenceMode::Voicing,
        ).unwrap();
        assert_eq!(result.frames, 8);
        assert_eq!(result.voicing_changes, 2);
        assert_eq!(result.replaced_values, 2);
        assert_eq!(voicing[0], [100.01, 0.0, 199.99, 240.02]);
        assert_eq!(voicing[1], [0.0, 238.5, 225.01, 0.0]);
        let mut full = source;
        let result = apply_reference(
            &path, &[5, 14], 61_440, 4, &mut full, ReferenceMode::Full,
        ).unwrap();
        assert_eq!(result.voicing_changes, 2);
        assert_eq!(result.replaced_values, 6);
        assert_eq!(full[0], [100.0, 0.0, 200.0, 240.0]);
        assert_eq!(full[1], [0.0, 238.5, 225.0, 0.0]);
        assert!(apply_reference(
            &path, &[5, 14], 60_800, 4, &mut full, ReferenceMode::Full,
        ).is_err());
        fs::remove_dir_all(&path).unwrap();
    }


    #[test]
    fn writes_exact_f0_and_pitch_rows() {
        let path = std::env::temp_dir().join(format!("rvc-f0-audit-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        write_batch(
            &path,
            &[3, 4],
            60_800,
            3,
            &[0.0, 220.0, 440.0, 0.0, 300.0, 330.0],
            &[1, 70, 90, 1, 80, 85],
        )
        .unwrap();
        for (index, expected) in [(3, [0.0f32, 220.0, 440.0]), (4, [0.0, 300.0, 330.0])] {
            let prefix = path.join(format!("chunk_{index:03}"));
            let bytes = fs::read(prefix.with_extension("f0.f32le")).unwrap();
            let values = bytes
                .chunks_exact(4)
                .map(|row| f32::from_le_bytes(row.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values.as_slice(), &expected);
            assert_eq!(
                fs::read(prefix.with_extension("json")).unwrap().len() > 20,
                true
            );
        }
        assert_eq!(
            fs::read(path.join("chunk_003.pitch.u8")).unwrap(),
            [1, 70, 90]
        );
        assert_eq!(
            fs::read(path.join("chunk_004.pitch.u8")).unwrap(),
            [1, 80, 85]
        );
        fs::remove_dir_all(&path).unwrap();
    }
}
