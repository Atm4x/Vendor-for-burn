const INPUT_HZ: usize = 16_000;
const OUTPUT_HZ: usize = 40_000;
const HOP: usize = 160;
const OUTPUT_HOP: usize = 400;
const MAX_FRAMES: usize = 1_000;

#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    pub core_frames: usize,
    pub overlap_frames: usize,
    pub context_frames: usize,
    pub trailing_frames: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            core_frames: 300,
            overlap_frames: 12,
            context_frames: 40,
            trailing_frames: 4,
        }
    }
}

impl ChunkConfig {
    pub fn validate(self) -> Result<(), String> {
        if self.core_frames == 0
            || self.overlap_frames.saturating_mul(2) >= self.core_frames
            || self
                .core_frames
                .checked_add(self.context_frames.saturating_mul(2))
                .and_then(|n| n.checked_add(self.trailing_frames))
                .is_none_or(|n| n > MAX_FRAMES)
        {
            return Err("invalid RVC chunk config: core + 2*context + tail must be <= 10 s and overlap < half core".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkProgress {
    pub chunk_index: usize,
    pub chunk_count: usize,
    pub output_samples: usize,
}

pub fn infer_chunked_with(
    pcm: &[f32],
    config: ChunkConfig,
    mut infer: impl FnMut(&[f32], usize) -> Result<Vec<f32>, String>,
) -> Result<Vec<f32>, String> {
    infer_chunked_with_progress(pcm, config, &mut infer, |_| {})
}

pub fn infer_chunked_with_progress(
    pcm: &[f32],
    config: ChunkConfig,
    mut infer: impl FnMut(&[f32], usize) -> Result<Vec<f32>, String>,
    mut progress: impl FnMut(ChunkProgress),
) -> Result<Vec<f32>, String> {
    config.validate()?;
    if pcm.is_empty() || pcm.iter().any(|x| !x.is_finite()) {
        return Err("RVC chunking requires nonempty finite 16-kHz PCM".into());
    }
    let output_length = pcm
        .len()
        .checked_mul(OUTPUT_HZ)
        .ok_or("RVC output length overflow")?
        / INPUT_HZ;
    if output_length == 0 {
        return Err("RVC audio is shorter than one output sample".into());
    }
    let total_frames = pcm.len().div_ceil(HOP);
    let step = config.core_frames - config.overlap_frames;
    let mut chunks = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + config.core_frames).min(total_frames);
        chunks.push((start, end));
        if end == total_frames {
            break;
        }
        start += step;
    }

    let mut output = vec![0.0f32; output_length];
    let mut weight_sum = vec![0.0f32; output_length];
    for (chunk_index, &(start, end)) in chunks.iter().enumerate() {
        let left = config.context_frames * HOP;
        let core = (end - start) * HOP;
        let right = (config.context_frames + config.trailing_frames) * HOP;
        let audio_start = start * HOP;
        let mut segment = Vec::with_capacity(left + core + right);
        let segment_start = audio_start as i64 - left as i64;
        for position in 0..left + core + right {
            let source = reflect_index(segment_start + position as i64, pcm.len());
            segment.push(pcm[source]);
        }
        if segment.len() < 3_200 || segment.len() > INPUT_HZ * 10 {
            return Err(format!(
                "RVC chunk {chunk_index} has invalid model input size {}",
                segment.len()
            ));
        }
        let rendered = infer(&segment, chunk_index)
            .map_err(|err| format!("RVC chunk {chunk_index}/{}: {err}", chunks.len()))?;
        let out_start = start * OUTPUT_HOP;
        let out_end = (end * OUTPUT_HOP).min(output_length);
        let crop_start = config.context_frames * OUTPUT_HOP;
        let crop_end = crop_start + out_end - out_start;
        if rendered.len() < crop_end {
            return Err(format!(
                "RVC chunk {chunk_index} generated {} samples, requires {crop_end} including context",
                rendered.len()
            ));
        }
        let overlap = config.overlap_frames * OUTPUT_HOP;
        for (offset, &sample) in rendered[crop_start..crop_end].iter().enumerate() {
            if !sample.is_finite() {
                return Err(format!(
                    "RVC chunk {chunk_index} generated non-finite audio"
                ));
            }
            let absolute = out_start + offset;
            let fade_in = if start > 0 && overlap > 0 {
                ((absolute - out_start) as f32 / overlap as f32).min(1.0)
            } else {
                1.0
            };
            let fade_out = if end < total_frames && overlap > 0 {
                ((out_end - absolute - 1) as f32 / overlap as f32).min(1.0)
            } else {
                1.0
            };
            let weight = fade_in.min(fade_out);
            output[absolute] += sample * weight;
            weight_sum[absolute] += weight;
        }
        progress(ChunkProgress {
            chunk_index: chunk_index + 1,
            chunk_count: chunks.len(),
            output_samples: out_end,
        });
    }
    for (sample, weight) in output.iter_mut().zip(weight_sum) {
        if weight <= 0.0 || !weight.is_finite() {
            return Err("RVC overlap-add left an uncovered output sample".into());
        }
        *sample /= weight;
    }
    Ok(output)
}

fn reflect_index(mut position: i64, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length as i64 - 1);
    position = position.rem_euclid(period);
    if position >= length as i64 {
        (period - position) as usize
    } else {
        position as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_audio(samples: &[f32], _: usize) -> Result<Vec<f32>, String> {
        Ok((0..samples.len() * 5 / 2)
            .map(|i| samples[i * 2 / 5])
            .collect())
    }

    #[test]
    fn overlap_add_preserves_long_audio_and_exact_sample_count() {
        let pcm: Vec<f32> = (0..20_137)
            .map(|i| ((i * 31 % 509) as f32) / 509.0)
            .collect();
        let cfg = ChunkConfig {
            core_frames: 32,
            overlap_frames: 6,
            context_frames: 15,
            trailing_frames: 4,
        };
        let mut progress = Vec::new();
        let out = infer_chunked_with_progress(&pcm, cfg, mock_audio, |p| progress.push(p)).unwrap();
        assert!(progress.len() > 2);
        assert_eq!(progress.last().unwrap().chunk_count, progress.len());
        assert_eq!(out.len(), pcm.len() * 5 / 2);
        for (i, &sample) in out.iter().enumerate() {
            assert!(
                (sample - pcm[i * 2 / 5]).abs() < 2e-6,
                "output position {i}"
            );
        }
    }

    #[test]
    fn exact_multiple_and_short_last_chunk_are_covered() {
        let cfg = ChunkConfig {
            core_frames: 40,
            overlap_frames: 6,
            context_frames: 12,
            trailing_frames: 4,
        };
        for frames in [1, 40, 41, 74, 75, 80, 115] {
            let input = vec![0.25; frames * HOP];
            let output = infer_chunked_with(&input, cfg, mock_audio).unwrap();
            assert_eq!(output.len(), frames * OUTPUT_HOP);
            assert!(output.iter().all(|v| (*v - 0.25).abs() < 1e-6));
        }
    }

    #[test]
    fn rejects_insufficient_decoder_output_and_bad_config() {
        let cfg = ChunkConfig {
            core_frames: 40,
            overlap_frames: 4,
            context_frames: 12,
            trailing_frames: 4,
        };
        assert!(infer_chunked_with(&[0.1; 3200], cfg, |_, _| Ok(vec![0.0; 2])).is_err());
        assert!(infer_chunked_with(&[f32::NAN], cfg, mock_audio).is_err());
        assert!(
            ChunkConfig {
                core_frames: 20,
                overlap_frames: 10,
                ..cfg
            }
            .validate()
            .is_err()
        );
        assert!(
            ChunkConfig {
                core_frames: 900,
                context_frames: 100,
                ..cfg
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn chunk_index_progress_and_boundary_weights() {
        let cfg = ChunkConfig {
            core_frames: 40,
            overlap_frames: 8,
            context_frames: 12,
            trailing_frames: 4,
        };
        let input = vec![0.0; 120 * HOP];
        let out = infer_chunked_with(&input, cfg, |segment, index| {
            Ok(vec![index as f32; segment.len() * 5 / 2])
        })
        .unwrap();
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[32 * OUTPUT_HOP] - 0.0).abs() < 1e-6);
        assert!(out[36 * OUTPUT_HOP] > 0.4 && out[36 * OUTPUT_HOP] < 0.6);
        assert!((out[40 * OUTPUT_HOP] - 1.0).abs() < 1e-6);
        assert!(out.last().unwrap().is_finite());
    }
}
