fn act(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * (1.0f32 / 10.0) }
}

fn conv(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    time: usize,
    channels: usize,
    ksize: usize,
    dilation: usize,
    already_activated: bool,
    activate_result: bool,
    residual: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0f32; time * channels];
    let pad = ((ksize - 1) / 2) * dilation;
    for t in 0..time {
        for co in 0..channels {
            let mut acc = 0.0f32;
            for k in 0..ksize {
                let src = t as isize + (k * dilation) as isize - pad as isize;
                if src < 0 || src >= time as isize { continue; }
                for ci in 0..channels {
                    let input = x[src as usize * channels + ci];
                    let input = if already_activated { input } else { act(input) };
                    let weight = w[(k * channels + ci) * channels + co];
                    acc = weight.mul_add(input, acc);
                }
            }
            acc += bias[co];
            if let Some(skip) = residual { acc += skip[t * channels + co]; }
            out[t * channels + co] = if activate_result { act(acc) } else { acc };
        }
    }
    out
}

fn weights(channels: usize, kernel: usize, seed: usize) -> Vec<f32> {
    (0..kernel * channels * channels)
        .map(|i| ((i.wrapping_mul(19).wrapping_add(seed) % 137) as i32 - 67) as f32 * 0.00825)
        .collect()
}

fn pair(
    x: &[f32], w1: &[f32], w2: &[f32], b1: &[f32], b2: &[f32],
    time: usize, channels: usize, kernel: usize, dilation: usize, fuse_activation: bool,
) -> Vec<f32> {
    let h = conv(x, w1, b1, time, channels, kernel, dilation, false, fuse_activation, None);
    conv(&h, w2, b2, time, channels, kernel, 1, fuse_activation, false, Some(x))
}

#[test]
fn bitwise_equivalence_full_pair_including_boundaries() {
    for channels in [1usize, 3, 4, 7, 16, 32, 128] {
        for kernel in [3usize, 7, 11] {
            for dilation in [1usize, 3, 5] {
                for time in [1usize, 2, 5, 11, 23] {
                    if channels == 128 && time > 11 { continue; }
                    let x: Vec<f32> = (0..time * channels)
                        .map(|i| ((i * 29 % 151) as f32 - 75.0) * 0.0135).collect();
                    let w1 = weights(channels, kernel, 13);
                    let w2 = weights(channels, kernel, 41);
                    let b1: Vec<f32> = (0..channels).map(|i| (i as f32 - 7.0) * 0.004).collect();
                    let b2: Vec<f32> = (0..channels).map(|i| (i as f32 - 11.0) * 0.002).collect();
                    let control = pair(&x, &w1, &w2, &b1, &b2, time, channels, kernel, dilation, false);
                    let fused = pair(&x, &w1, &w2, &b1, &b2, time, channels, kernel, dilation, true);
                    assert_eq!(control.len(), fused.len());
                    for (i, (&a, &b)) in control.iter().zip(fused.iter()).enumerate() {
                        assert_eq!(a.to_bits(), b.to_bits(), "C={channels},T={time},K={kernel},D={dilation},i={i}");
                    }
                }
            }
        }
    }
}

#[test]
fn double_application_is_not_equivalent() {
    let x = -1f32;
    assert_ne!(act(x).to_bits(), act(act(x)).to_bits());
}

#[test]
fn activation_repetition_lower_bound() {
    let t = 29_800usize;
    let c = 128usize;
    let k = 11usize;
    let output_lane_vec = 4usize;
    let old_evals = t * c * (c / output_lane_vec) * k;
    let new_evals = t * c;
    assert_eq!(old_evals / new_evals, 352);
    assert_eq!(new_evals, 3_814_400);
}
