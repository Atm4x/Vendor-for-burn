#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    FixedOverlap,
    QuietSplice { extra_chunks: usize },
    CostAware(CostModel),
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
            Ok("cost-aware") => Ok(Self::CostAware(CostModel::from_env()?)),
            Ok(value) => Err(format!("RVC_CHUNK_STRATEGY must be fixed, quiet-splice or cost-aware, got {value}")),
            Err(error) => Err(format!("invalid RVC_CHUNK_STRATEGY: {error}")),
        }
    }

    pub fn requires_contiguous_splice(self) -> bool {
        !matches!(self, Self::FixedOverlap)
    }
}

/// An analytical work proxy, deliberately NOT a latency prediction.
/// Units are dimensionless; calibrate the terms from measured device timings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostModel {
    pub launch: u32,
    pub linear: u32,
    pub attention_milli: u32,
    pub splice: u32,
}

impl Default for CostModel {
    fn default() -> Self {
        Self { launch: 10_000, linear: 10, attention_milli: 10, splice: 100 }
    }
}

impl CostModel {
    pub fn from_env() -> Result<Self, String> {
        fn read(name: &str, default: u32) -> Result<u32, String> {
            match std::env::var(name) {
                Ok(s) => s.parse::<u32>().ok().filter(|&v| v <= 1_000_000)
                    .ok_or_else(|| format!("{name} must be a decimal integer in 0..=1000000")),
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(error) => Err(format!("{name}: {error}")),
            }
        }
        let defaults = Self::default();
        let result = Self {
            launch: read("RVC_COST_LAUNCH", defaults.launch)?,
            linear: read("RVC_COST_LINEAR", defaults.linear)?,
            attention_milli: read("RVC_COST_ATTENTION_MILLI", defaults.attention_milli)?,
            splice: read("RVC_COST_SPLICE", defaults.splice)?,
        };
        if result.launch == 0 && result.linear == 0 && result.attention_milli == 0 && result.splice == 0 {
            return Err("RVC cost model requires at least one positive coefficient".into());
        }
        Ok(result)
    }

    fn work(self, core: usize, context: usize, trailing: usize) -> f64 {
        let model_frames = (core + 2*context + trailing) as f64;
        self.launch as f64
            + self.linear as f64 * model_frames
            + (self.attention_milli as f64 / 1000.0) * model_frames * model_frames
    }
}

/// Global shortest path over all possible non-overlapping chunk boundaries.
/// Each edge represents a complete HuBERT/RMVPE/TextEncoder/NSF invocation.
/// The boundary term measures RMS energy at the cut; the compute terms include
/// model context and quadratic attention. O(T * max_core) time, O(T) memory.
pub fn plan_cost_aware(
    pcm: &[f32], max_core: usize, context: usize, trailing: usize, model: CostModel,
) -> Result<Vec<(usize, usize)>, String> {
    if pcm.is_empty() || max_core == 0 || pcm.iter().any(|v| !v.is_finite()) {
        return Err("cost-aware chunk planner requires finite PCM and max_core > 0".into());
    }
    let total = pcm.len().div_ceil(160);
    let model_max = max_core.checked_add(context.checked_mul(2)
        .ok_or("RVC context overflow")?).and_then(|v| v.checked_add(trailing))
        .ok_or("RVC chunk length overflow")?;
    if model_max > 1000 { return Err("cost-aware model length exceeds 1000 frames".into()); }
    let min_core = max_core.min(20);
    let mut energy = vec![0.0_f64; total];
    for (i, &value) in pcm.iter().enumerate() {
        let v = value as f64;
        energy[i / 160] += v * v;
    }
    let average = (energy.iter().sum::<f64>() / pcm.len() as f64).max(1e-20);
    let mut boundary = vec![0.0_f64; total + 1];
    for (frame, value) in boundary.iter_mut().enumerate().take(total).skip(1) {
        let sample_count = ((frame + 1)*160).min(pcm.len()) - ((frame - 1)*160);
        *value = (((energy[frame-1] + energy[frame]) / sample_count as f64) / average)
            .min(10_000.0) * model.splice as f64;
    }
    let mut costs = vec![f64::INFINITY; total+1];
    let mut predecessors = vec![usize::MAX; total+1];
    let mut counts = vec![usize::MAX; total+1];
    costs[0] = 0.0;
    counts[0] = 0;
    for end in 1..=total {
        for start in end.saturating_sub(max_core)..end {
            let length = end - start;
            if length < min_core && start != 0 && end != total { continue; }
            if !costs[start].is_finite() { continue; }
            let candidate = costs[start] + model.work(length, context, trailing)
                + if start > 0 { boundary[start] } else { 0.0 };
            let candidate_count = counts[start] + 1;
            if candidate < costs[end] - 1e-9
                || (candidate - costs[end]).abs() < 1e-9 && candidate_count < counts[end]
            {
                costs[end] = candidate;
                predecessors[end] = start;
                counts[end] = candidate_count;
            }
        }
    }
    if !costs[total].is_finite() { return Err("cost-aware chunk planner found no feasible partition".into()); }
    let mut chunks = Vec::with_capacity(counts[total]);
    let mut end = total;
    while end > 0 {
        let start = predecessors[end];
        if start == usize::MAX || start >= end {
            return Err("cost-aware chunk planner has invalid backtrace".into());
        }
        chunks.push((start,end));
        end = start;
    }
    chunks.reverse();
    Ok(chunks)
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
    fn cost_aware_prefers_fewer_large_chunks_when_dispatch_dominates() {
        let pcm = vec![0.25; 240 * 160];
        let model = CostModel { launch: 10_000, linear: 0, attention_milli: 0, splice: 0 };
        let result = plan_cost_aware(&pcm, 90, 40, 4, model).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.first().unwrap().0, 0);
        assert_eq!(result.last().unwrap().1, 240);
        assert_eq!(result.iter().map(|(a,b)|b-a).sum::<usize>(), 240);
    }

    #[test]
    fn attention_dominated_model_prefers_shorter_chunks() {
        let pcm = vec![0.25; 180 * 160];
        let launch = CostModel { launch: 10_000, linear: 1, attention_milli: 0, splice: 0 };
        let attention = CostModel { launch: 100, linear: 1, attention_milli: 100, splice: 0 };
        let large = plan_cost_aware(&pcm, 180, 40, 4, launch).unwrap();
        let small = plan_cost_aware(&pcm, 180, 40, 4, attention).unwrap();
        assert!(small.len() > large.len(), "large={large:?}, small={small:?}");
        assert!(small.iter().all(|(a,b)|b-a<=180));
    }

    #[test]
    fn cost_aware_selects_quiet_boundary_in_equal_cost_plans() {
        let mut pcm = vec![1.0; 100 * 160];
        for frame in 46..54 { pcm[frame*160..(frame+1)*160].fill(0.0); }
        let model = CostModel { launch: 10_000, linear: 0, attention_milli: 0, splice: 1_000 };
        let result = plan_cost_aware(&pcm, 60, 40, 4, model).unwrap();
        assert_eq!(result.len(), 2, "{result:?}");
        assert!((47..=53).contains(&result[0].1), "{result:?}");
    }

    #[test]
    fn exhaustive_partition_coverage_and_determinism() {
        let model = CostModel::default();
        for max in [20, 40, 120, 900] {
            for frames in [1, 2, 19, 20, 21, 59, 83, 101, 240, 900, 901, 1401] {
                let pcm = (0..frames*160-1).map(|i|((i*11%31) as f32 - 15.0)*0.025).collect::<Vec<_>>();
                let a = plan_cost_aware(&pcm, max, 40, 4, model).unwrap();
                let b = plan_cost_aware(&pcm, max, 40, 4, model).unwrap();
                assert_eq!(a,b);
                assert_eq!(a.first().unwrap().0,0);
                assert_eq!(a.last().unwrap().1,frames);
                assert!(a.windows(2).all(|pair| pair[0].1==pair[1].0));
                assert!(a.iter().all(|(s,e)|s<e && e-s<=max));
            }
        }
    }

    #[test]
    fn cost_aware_matches_exhaustive_partition_optimum() {
        let frames = 65usize;
        let pcm = (0..frames * 160).map(|i| {
            if (20..26).contains(&(i/160)) || (40..45).contains(&(i/160)) {0.0}
            else {0.4 + ((i*13%43) as f32)/300.0}
        }).collect::<Vec<_>>();
        let frame_energy = (0..frames).map(|t| pcm[t*160..(t+1)*160].iter()
            .map(|&v|(v as f64)*(v as f64)).sum::<f64>()).collect::<Vec<_>>();
        let norm = (frame_energy.iter().sum::<f64>() / pcm.len() as f64).max(1e-20);
        fn solve(
            start:usize, ends:&mut Vec<usize>, best:&mut f64, energy:&[f64], norm:f64,
            model:CostModel,
        ) {
            let total = energy.len();
            if start == total {
                let mut prev = 0usize;
                let mut score = 0.0;
                for &end in ends.iter() {
                    score += model.work(end-prev,8,2);
                    if prev > 0 {
                        score += ((energy[prev-1]+energy[prev]) / 320.0 / norm)
                            .min(10_000.0)*model.splice as f64;
                    }
                    prev = end;
                }
                *best = best.min(score);
                return;
            }
            for end in start+1..=(start+25).min(total) {
                if end-start < 20 && start != 0 && end != total {continue;}
                ends.push(end);
                solve(end,ends,best,energy,norm,model);
                ends.pop();
            }
        }
        for model in [
            CostModel {launch:1000,linear:8,attention_milli:10,splice:10},
            CostModel {launch:200,linear:1,attention_milli:300,splice:1000},
            CostModel {launch:10000,linear:10,attention_milli:1,splice:100000},
        ] {
            let chosen = plan_cost_aware(&pcm,25,8,2,model).unwrap();
            let mut best = f64::INFINITY;
            solve(0,&mut Vec::new(),&mut best,&frame_energy,norm,model);
            let mut got = 0f64;
            for (i,&(start,end)) in chosen.iter().enumerate() {
                got += model.work(end-start,8,2);
                if i>0 {
                    got += ((frame_energy[start-1]+frame_energy[start])/320.0/norm)
                        .min(10_000.0)*model.splice as f64;
                }
            }
            assert!((got-best).abs()<1e-5,"chosen={chosen:?}, got={got}, best={best}");
        }
    }

    #[test]
    fn invalid_cost_aware_geometry_is_rejected() {
        let model=CostModel::default();
        assert!(plan_cost_aware(&[],900,40,4,model).is_err());
        assert!(plan_cost_aware(&[f32::NAN],900,40,4,model).is_err());
        assert!(plan_cost_aware(&[0.0;160],0,40,4,model).is_err());
        assert!(plan_cost_aware(&[0.0;160],950,40,4,model).is_err());
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
