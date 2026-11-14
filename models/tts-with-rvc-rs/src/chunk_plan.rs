#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    FixedOverlap,
    QuietSplice { extra_chunks: usize },
}

impl ChunkStrategy {
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("RVC_CHUNK_STRATEGY").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("") | Ok("fixed") => Ok(Self::FixedOverlap),
            Ok("quiet-splice") => {
                let extra_chunks = std::env::var("RVC_QUIET_EXTRA_CHUNKS").ok()
                    .map(|value| value.parse::<usize>().map_err(|_| "RVC_QUIET_EXTRA_CHUNKS must be 0..=3".to_owned()))
                    .transpose()?.unwrap_or(0);
                if extra_chunks > 3 {
                    return Err("RVC_QUIET_EXTRA_CHUNKS must be 0..=3".into());
                }
                Ok(Self::QuietSplice { extra_chunks })
            },
            Ok(value) => Err(format!("RVC_CHUNK_STRATEGY must be fixed or quiet-splice, got {value}")),
            Err(error) => Err(format!("invalid RVC_CHUNK_STRATEGY: {error}")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    frame: usize,
    cost: f64,
    previous: usize,
}

pub fn plan_quiet_splices(pcm: &[f32], max_core_frames: usize) -> Result<Vec<(usize, usize)>, String> {
    plan_quiet_splices_with_extra(pcm, max_core_frames, 0)
}

pub fn plan_quiet_splices_with_extra(
    pcm: &[f32], max_core_frames: usize, extra_chunks: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if extra_chunks > 3 {
        return Err("quiet chunk extra split count must be 0..=3".into());
    }
    if max_core_frames == 0 || pcm.is_empty() || pcm.iter().any(|v| !v.is_finite()) {
        return Err("quiet chunk planner requires finite nonempty PCM and positive core size".into());
    }
    let total = pcm.len().div_ceil(160);
    let pieces = total.div_ceil(max_core_frames) + extra_chunks;
    if pieces > total {
        return Err("quiet chunk planner cannot create more segments than frames".into());
    }
    if pieces == 1 {
        return Ok(vec![(0, total)]);
    }
    let radius = (max_core_frames / 8).clamp(1, 60);
    let min_core_frames = (max_core_frames / 3).max(1).min(total / pieces);
    let mut frame_energy = vec![0.0_f64; total];
    for (i, &value) in pcm.iter().enumerate() {
        frame_energy[i / 160] += (value as f64).powi(2);
    }
    let global_energy = (frame_energy.iter().sum::<f64>() / pcm.len() as f64).max(1e-15);
    let mut energies = vec![0.0_f64; total + 1];
    for (frame, score) in energies.iter_mut().enumerate().take(total).skip(1) {
        let left = (frame - 1) * 160;
        let right = (frame + 1).saturating_mul(160).min(pcm.len());
        *score = (frame_energy[frame - 1] + frame_energy[frame])
            / (right - left) as f64 / global_energy;
    }

    let mut levels: Vec<Vec<Candidate>> = Vec::with_capacity(pieces - 1);
    for boundary in 1..pieces {
        let ideal = ((boundary as u128 * total as u128 + (pieces / 2) as u128)
            / pieces as u128) as usize;
        let min_frame = ideal.saturating_sub(radius).max(1);
        let max_frame = ideal.saturating_add(radius).min(total - 1);
        let mut candidates = Vec::new();
        for frame in min_frame..=max_frame {
            let displacement = frame.abs_diff(ideal) as f64 / radius as f64;
            let own_cost = energies[frame] + 0.01 * displacement * displacement;
            let (previous, best) = if boundary == 1 {
                if !(min_core_frames..=max_core_frames).contains(&frame) {
                    continue;
                }
                (0, 0.0)
            } else {
                let mut best = f64::INFINITY;
                let mut best_index = 0;
                for (index, previous) in levels[boundary - 2].iter().enumerate() {
                    if frame <= previous.frame {
                        continue;
                    }
                    let length = frame - previous.frame;
                    if (min_core_frames..=max_core_frames).contains(&length)
                        && previous.cost < best
                    {
                        best = previous.cost;
                        best_index = index;
                    }
                }
                (best_index, best)
            };
            if best.is_finite() {
                candidates.push(Candidate { frame, cost: best + own_cost, previous });
            }
        }
        if candidates.is_empty() {
            return Err(format!("quiet chunk planner cannot place boundary {boundary}/{pieces}"));
        }
        levels.push(candidates);
    }

    let last = levels.last().ok_or("quiet chunk planner has no boundaries")?;
    let (mut index, _) = last.iter().enumerate()
        .filter(|(_, candidate)| (min_core_frames..=max_core_frames).contains(&(total - candidate.frame)))
        .min_by(|(_, a), (_, b)| a.cost.total_cmp(&b.cost))
        .ok_or("quiet chunk planner cannot complete final segment")?;
    let mut boundaries = vec![0; pieces + 1];
    boundaries[pieces] = total;
    for level in (0..levels.len()).rev() {
        let candidate = levels[level][index];
        boundaries[level + 1] = candidate.frame;
        index = candidate.previous;
    }
    Ok(boundaries.windows(2).map(|pair| (pair[0], pair[1])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify(length: usize, core: usize) {
        let pcm: Vec<f32> = (0..length).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let plan = plan_quiet_splices(&pcm, core).unwrap();
        assert_eq!(plan.len(), length.div_ceil(160).div_ceil(core));
        assert_eq!(plan.first().unwrap().0, 0);
        assert_eq!(plan.last().unwrap().1, length.div_ceil(160));
        for (i, &(start, end)) in plan.iter().enumerate() {
            assert!(start < end && end - start <= core);
            if i > 0 { assert_eq!(start, plan[i - 1].1); }
        }
    }

    #[test]
    fn exhaustive_small_lengths_and_partial_frames() {
        for core in [20, 30, 60, 300, 900, 916] {
            for frames in [1, 2, 19, 20, 30, 60, 61, 117, 300, 601, 900, 901, 916, 917, 1199, 1801, 8994] {
                verify(frames * 160, core);
                verify(frames * 160 - 1, core);
            }
        }
    }

    #[test]
    fn short_inputs_and_extra_splits_never_underflow() {
        for core in [1, 2, 5, 20, 60, 900] {
            for frames in 1..=90 {
                let pcm = vec![0.0; frames*160];
                for extra in 0..=3 {
                    if frames.div_ceil(core) + extra > frames { continue; }
                    let plan = plan_quiet_splices_with_extra(&pcm, core, extra).unwrap();
                    assert_eq!(plan.len(), frames.div_ceil(core) + extra);
                    assert!(plan.windows(2).all(|pair| pair[0].1 == pair[1].0));
                }
            }
        }
    }

    #[test]
    fn silent_regions_are_preferred_over_loud_regions() {
        let pcm: Vec<f32> = (0..1_200 * 160).map(|i| {
            if (580..620).contains(&(i / 160)) { 0.0 } else { 0.8 }
        }).collect();
        let plan = plan_quiet_splices(&pcm, 800).unwrap();
        assert_eq!(plan.len(), 2);
        assert!((580..=620).contains(&plan[0].1), "{plan:?}");
    }

    #[test]
    fn deterministic_flat_energy_prefers_even_splits() {
        let pcm = vec![0.0; 1_201 * 160];
        let first = plan_quiet_splices(&pcm, 900).unwrap();
        let second = plan_quiet_splices(&pcm, 900).unwrap();
        assert_eq!(first, second);
        assert!((first[0].1 as isize - 600).abs() <= 1);
    }

    #[test]
    fn optional_extra_chunks_are_contiguous_and_bounded() {
        let pcm: Vec<f32> = (0..9_000*160)
            .map(|i| if i % (750*160) < 40*160 { 0.0 } else { 0.9 }).collect();
        let minimal = plan_quiet_splices(&pcm, 900).unwrap();
        for extra in 1..=3 {
            let plan = plan_quiet_splices_with_extra(&pcm, 900, extra).unwrap();
            assert_eq!(plan.len(), minimal.len() + extra);
            assert_eq!(plan.last().unwrap().1, 9000);
            assert!(plan.windows(2).all(|pair| pair[0].1 == pair[1].0));
            assert!(plan.iter().all(|(start,end)| end > start && end-start <= 900));
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(plan_quiet_splices_with_extra(&[0.0], 900, 3).is_err());
        assert!(plan_quiet_splices_with_extra(&[0.0; 1600], 900, 4).is_err());
        assert!(plan_quiet_splices(&[], 900).is_err());
        assert!(plan_quiet_splices(&[f32::NAN], 900).is_err());
        assert!(plan_quiet_splices(&[1.0], 0).is_err());
    }
}
