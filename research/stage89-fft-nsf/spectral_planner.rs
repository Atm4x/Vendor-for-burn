//! Stage89 source-only research: checkpoint-wide exact-topology FFT-vs-direct planner.
//! Does not change production inference and makes no GPU performance claims.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layer {
    pub channels_in: usize,
    pub channels_out: usize,
    pub kernel: usize,
    pub dilation: usize,
    pub frames: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub fft_len: usize,
    pub effective_kernel: usize,
    pub valid_frames: usize,
    pub frequency_bins: usize,
    pub windows: usize,
    pub spectral_weight_bytes: u64,
    pub direct_flops: f64,
    pub spectral_flops: f64,
}

pub fn candidates(layer: Layer) -> Vec<Plan> {
    if layer.channels_in == 0 || layer.channels_out == 0 || layer.kernel == 0
        || layer.kernel % 2 == 0 || layer.dilation == 0 || layer.frames == 0
    {
        return Vec::new();
    }
    let Some(k_eff) = (layer.kernel - 1).checked_mul(layer.dilation)
        .and_then(|x| x.checked_add(1)) else { return Vec::new() };
    let mut out = Vec::new();
    for fft_len in [64usize, 128, 256, 512, 1024, 2048] {
        if fft_len < k_eff { continue; }
        let valid_frames = fft_len - k_eff + 1;
        let windows = layer.frames.div_ceil(valid_frames);
        let bins = fft_len / 2 + 1;
        let Some(bytes) = layer.channels_in.checked_mul(layer.channels_out)
            .and_then(|x| x.checked_mul(bins))
            .and_then(|x| x.checked_mul(8)) else { continue };
        let direct = 2.0 * layer.frames as f64 * layer.channels_in as f64
            * layer.channels_out as f64 * layer.kernel as f64;
        let matmul = 6.0 * windows as f64 * bins as f64
            * layer.channels_in as f64 * layer.channels_out as f64;
        let compose = 4.0 * windows as f64 * bins as f64 * layer.channels_out as f64;
        let fft = 5.0 * windows as f64 * fft_len as f64 * fft_len.ilog2() as f64
            * (layer.channels_in + layer.channels_out) as f64;
        out.push(Plan {
            fft_len, effective_kernel: k_eff, valid_frames,
            frequency_bins: bins, windows,
            spectral_weight_bytes: bytes as u64,
            direct_flops: direct, spectral_flops: matmul + compose + fft,
        });
    }
    out
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub fft: Option<Plan>,
}

pub fn optimize(layers: &[Layer], memory_budget_bytes: u64, bucket_bytes: u64) -> Vec<Decision> {
    assert!(bucket_bytes > 0);
    let slots = (memory_budget_bytes / bucket_bytes) as usize;
    let mut dp: Vec<Option<(f64, Vec<Decision>)>> = vec![None; slots + 1];
    dp[0] = Some((0.0, Vec::new()));
    for &layer in layers {
        let layer_plans = candidates(layer);
        let mut next = vec![None::<(f64, Vec<Decision>)>; slots + 1];
        for (used, candidate) in dp.iter().enumerate() {
            let Some((score, choices)) = candidate else { continue };
            let mut offer = |slots_needed: usize, saved: f64, fft: Option<Plan>| {
                if let Some(at) = used.checked_add(slots_needed).filter(|&n| n <= slots) {
                    let score_new = *score + saved;
                    if next[at].as_ref().is_none_or(|(prior, _)| score_new > *prior) {
                        let mut new_choices = choices.clone();
                        new_choices.push(Decision { fft });
                        next[at] = Some((score_new, new_choices));
                    }
                }
            };
            offer(0, 0.0, None);
            for plan in &layer_plans {
                let saved = plan.direct_flops - plan.spectral_flops;
                if saved <= 0.0 { continue; }
                let slots_needed = plan.spectral_weight_bytes.div_ceil(bucket_bytes) as usize;
                offer(slots_needed, saved, Some(*plan));
            }
        }
        dp = next;
    }
    dp.into_iter().flatten().max_by(|(a, _), (b, _)| a.total_cmp(b))
        .map(|(_, choices)| choices).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_geometry_matches_overlap_save() {
        let layer = Layer { channels_in: 128, channels_out: 128, kernel: 11, dilation: 5, frames: 38_400 };
        let p = candidates(layer).into_iter().find(|p| p.fft_len == 512).unwrap();
        assert_eq!(p.effective_kernel, 51);
        assert_eq!(p.valid_frames, 462);
        assert_eq!(p.windows, 84);
        assert_eq!(p.frequency_bins, 257);
        assert!(p.spectral_weight_bytes > 0);
        assert!(p.direct_flops > p.spectral_flops);
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        let invalid = Layer { channels_in: 1, channels_out: 1, kernel: 4, dilation: 3, frames: 50 };
        assert!(candidates(invalid).is_empty());
        assert!(candidates(Layer { kernel: 3, channels_in: 0, ..invalid }).is_empty());
    }

    #[test]
    fn budget_is_never_exceeded() {
        let layers: Vec<_> = (0..12).map(|idx| Layer {
            channels_in: if idx % 2 == 0 { 128 } else { 256 },
            channels_out: if idx % 2 == 0 { 128 } else { 256 },
            kernel: 11, dilation: 5, frames: 38_400,
        }).collect();
        for budget in [64, 128, 256, 512] {
            let cap = budget * 1024 * 1024;
            let selection = optimize(&layers, cap, 2*1024*1024);
            assert_eq!(selection.len(), layers.len());
            assert!(selection.iter().map(|d| d.fft.map_or(0, |p| p.spectral_weight_bytes)).sum::<u64>() <= cap);
        }
    }

    #[test]
    fn global_decision_matches_exhaustive_small_case() {
        let layers = [
            Layer { channels_in: 16, channels_out: 16, kernel: 11, dilation: 3, frames: 2000 },
            Layer { channels_in: 16, channels_out: 16, kernel: 7, dilation: 1, frames: 2000 },
            Layer { channels_in: 32, channels_out: 32, kernel: 3, dilation: 1, frames: 2000 },
        ];
        let bucket = 64 * 1024;
        let cap = 4 * 1024 * 1024;
        let result = optimize(&layers, cap, bucket);
        let chosen: f64 = layers.iter().zip(result.iter())
            .map(|(l, d)| d.fft.map_or(0.0, |p| candidates(*l)[0].direct_flops - p.spectral_flops))
            .sum();
        let options: Vec<_> = layers.iter().map(|l| {
            let mut plans = vec![None];
            plans.extend(candidates(*l).into_iter().filter(|p| p.direct_flops > p.spectral_flops).map(Some));
            plans
        }).collect();
        let mut brute = 0.0f64;
        for a in &options[0] { for b in &options[1] { for c in &options[2] {
            let selected = [*a,*b,*c];
            let cost: u64 = selected.into_iter().flatten()
                .map(|p| p.spectral_weight_bytes.div_ceil(bucket) * bucket).sum();
            if cost > cap { continue; }
            let saved = layers.iter().zip(selected.iter())
                .map(|(l,p)| p.map_or(0.0, |p| candidates(*l)[0].direct_flops - p.spectral_flops))
                .sum::<f64>();
            brute = brute.max(saved);
        }}}
        assert!((chosen - brute).abs() <= 0.001);
    }
}
