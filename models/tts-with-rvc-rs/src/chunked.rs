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
    progress: impl FnMut(ChunkProgress),
) -> Result<Vec<f32>, String> {
    infer_chunked_batched_with_progress(
        pcm,
        config,
        1,
        |segments, indices| {
            segments.iter().zip(indices).map(|(audio, &i)| infer(audio, i)).collect()
        },
        progress,
    )
}

/// Execute equal-length model inputs as a single tensor batch; preserve chunk order.
/// The final partial chunk is kept separate instead of padding its model input.
pub fn infer_chunked_batched_with_progress(
    pcm: &[f32],
    config: ChunkConfig,
    batch_size: usize,
    mut infer: impl FnMut(&[Vec<f32>], &[usize]) -> Result<Vec<Vec<f32>>, String>,
    mut progress: impl FnMut(ChunkProgress),
) -> Result<Vec<f32>, String> {
    config.validate()?;
    if batch_size == 0 || batch_size > 32 {
        return Err("RVC tensor batch size must be in 1..=32".into());
    }
    if pcm.is_empty() || pcm.iter().any(|x| !x.is_finite()) {
        return Err("RVC chunking requires nonempty finite 16-kHz PCM".into());
    }
    let output_length = pcm.len().checked_mul(OUTPUT_HZ)
        .ok_or("RVC output length overflow")? / INPUT_HZ;
    let total_frames = pcm.len().div_ceil(HOP);
    let chunks = chunk_ranges(total_frames, config);
    let mut output = vec![0.0f32; output_length];
    let mut weight_sum = vec![0.0f32; output_length];
    let mut chunk_index = 0;
    while chunk_index < chunks.len() {
        let first_size = (chunks[chunk_index].1 - chunks[chunk_index].0) * HOP;
        let mut audio_batch = Vec::new();
        let mut indices = Vec::new();
        while chunk_index + audio_batch.len() < chunks.len() && audio_batch.len() < batch_size {
            let i = chunk_index + audio_batch.len();
            let (start, end) = chunks[i];
            if (end - start) * HOP != first_size { break; }
            let left = config.context_frames * HOP;
            let right = (config.context_frames + config.trailing_frames) * HOP;
            let core = (end - start) * HOP;
            let mut segment = Vec::with_capacity(left + core + right);
            let segment_start = (start * HOP) as i64 - left as i64;
            for position in 0..left + core + right {
                segment.push(pcm[reflect_index(segment_start + position as i64, pcm.len())]);
            }
            if !(3_200..=INPUT_HZ * 10).contains(&segment.len()) {
                return Err(format!("RVC chunk {i} has invalid model input size {}", segment.len()));
            }
            audio_batch.push(segment);
            indices.push(i);
        }
        let rendered_batch = infer(&audio_batch, &indices)
            .map_err(|err| format!("RVC chunks {}..{}: {err}", chunk_index, chunk_index + audio_batch.len()))?;
        if rendered_batch.len() != audio_batch.len() {
            return Err(format!("RVC batch returned {} waveforms for {} inputs", rendered_batch.len(), audio_batch.len()));
        }
        for (relative, rendered) in rendered_batch.into_iter().enumerate() {
            let i = chunk_index + relative;
            let (start, end) = chunks[i];
            let out_start = start * OUTPUT_HOP;
            let out_end = (end * OUTPUT_HOP).min(output_length);
            let crop_start = config.context_frames * OUTPUT_HOP;
            let crop_end = crop_start + out_end - out_start;
            if rendered.len() < crop_end {
                return Err(format!("RVC chunk {i} generated {} samples, requires {crop_end} including context", rendered.len()));
            }
            let overlap = config.overlap_frames * OUTPUT_HOP;
            for (offset, &sample) in rendered[crop_start..crop_end].iter().enumerate() {
                if !sample.is_finite() { return Err(format!("RVC chunk {i} generated non-finite audio")); }
                let absolute = out_start + offset;
                let fade_in = if start > 0 && overlap > 0 {
                    ((absolute - out_start) as f32 / overlap as f32).min(1.0)
                } else { 1.0 };
                let fade_out = if end < total_frames && overlap > 0 {
                    ((out_end - absolute - 1) as f32 / overlap as f32).min(1.0)
                } else { 1.0 };
                let weight = fade_in.min(fade_out);
                output[absolute] += sample * weight;
                weight_sum[absolute] += weight;
            }
            progress(ChunkProgress { chunk_index: i + 1, chunk_count: chunks.len(), output_samples: out_end });
        }
        chunk_index += audio_batch.len();
    }
    for (sample, weight) in output.iter_mut().zip(weight_sum) {
        if weight <= 0.0 || !weight.is_finite() {
            return Err("RVC overlap-add left an uncovered output sample".into());
        }
        *sample /= weight;
    }
    Ok(output)
}

pub(crate) fn chunk_ranges(total_frames: usize, config: ChunkConfig) -> Vec<(usize, usize)> {
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
    chunks
}

pub(crate) fn reflect_index(mut position: i64, length: usize) -> usize {
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

#[cfg(test)]
mod tensor_batch_tests {
    use super::*;

    #[test]
    fn batched_chunking_matches_scalar_and_preserves_finite_boundaries() {
        let pcm: Vec<f32> = (0..91_377).map(|i| (i as f32 * 0.001).sin()).collect();
        let config = ChunkConfig { core_frames: 60, overlap_frames: 12, context_frames: 20, trailing_frames: 4 };
        let render = |input: &[f32]| (0..input.len() * 5 / 2)
            .map(|i| input[i * 2 / 5]).collect::<Vec<f32>>();
        let expected = infer_chunked_with(&pcm, config, |input, _| Ok(render(input))).unwrap();
        let mut batches = Vec::new();
        let actual = infer_chunked_batched_with_progress(&pcm, config, 3, |inputs, ids| {
            batches.push((inputs.len(), ids.to_vec()));
            Ok(inputs.iter().map(|input| render(input)).collect())
        }, |_|{}).unwrap();
        assert_eq!(expected, actual);
        assert!(batches.iter().any(|(len, _)| *len == 3));
        assert!(batches.iter().all(|(len, ids)| *len == ids.len()));
        assert_eq!(actual.len(), pcm.len() * 5 / 2);
    }

    #[test]
    fn long_audio_batch_boundaries_and_partial_samples_match_scalar() {
        let config = ChunkConfig::default();
        for length in [1, 159, 160, 161, 3_201, 159_997, 1_440_017] {
            let pcm: Vec<f32> = (0..length)
                .map(|sample| (((sample * 103 + 71) % 19_997) as f32 - 10_000.0) / 19_997.0)
                .collect();
            let render = |segment: &[f32]| {
                (0..segment.len() * 5 / 2)
                    .map(|sample| segment[sample * 2 / 5])
                    .collect::<Vec<f32>>()
            };
            let mut scalar_progress = Vec::new();
            let scalar = infer_chunked_with_progress(
                &pcm,
                config,
                |segment, _| Ok(render(segment)),
                |event| scalar_progress.push(event),
            )
            .unwrap();
            let mut batch_progress = Vec::new();
            let mut batches = Vec::new();
            let batched = infer_chunked_batched_with_progress(
                &pcm,
                config,
                4,
                |segments, indices| {
                    batches.push((segments.len(), indices.to_vec()));
                    Ok(segments.iter().map(|segment| render(segment)).collect())
                },
                |event| batch_progress.push(event),
            )
            .unwrap();
            assert_eq!(scalar, batched, "batch/scalar changed PCM at length {length}");
            assert_eq!(scalar_progress, batch_progress);
            assert_eq!(scalar.len(), length * OUTPUT_HZ / INPUT_HZ);
            assert!(scalar.iter().all(|value| value.is_finite()));
            assert!(batches.iter().all(|(size, indices)| *size == indices.len()));
            for (output_sample, &value) in scalar.iter().enumerate() {
                let expected = pcm[output_sample * INPUT_HZ / OUTPUT_HZ];
                assert!(
                    (value - expected).abs() < 2e-6,
                    "unexpected chunk boundary: input={length}, output={output_sample}, actual={value}, expected={expected}",
                );
            }
        }
    }

    #[test]
    fn batch_rejects_missing_outputs_and_oversized_batch() {
        let pcm = vec![0.1; 18_000];
        let cfg = ChunkConfig { core_frames: 30, overlap_frames: 4, context_frames: 12, trailing_frames: 4 };
        assert!(infer_chunked_batched_with_progress(&pcm, cfg, 0, |_, _| Ok(vec![]), |_| {}).is_err());
        assert!(infer_chunked_batched_with_progress(&pcm, cfg, 33, |_, _| Ok(vec![]), |_| {}).is_err());
        assert!(infer_chunked_batched_with_progress(&pcm, cfg, 2, |_, _| Ok(vec![]), |_| {}).is_err());
    }
}
