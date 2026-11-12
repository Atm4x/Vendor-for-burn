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

#[cfg(test)]
mod tests {
    use super::*;

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
