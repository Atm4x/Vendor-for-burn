fn activation(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * 0.1 }
}

fn conv(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32],
    time: usize,
    cin: usize,
    cout: usize,
    kernel: usize,
    dilation: usize,
    tile: usize,
    prefetch: bool,
) -> Vec<f32> {
    let padding = ((kernel - 1) / 2) * dilation;
    let mut output = vec![0.0f32; time * cout];
    for t0 in (0..time).step_by(tile) {
        for co in 0..cout {
            let mut accum = [0.0f32; 4];
            for k in 0..kernel {
                let mut next_weight = 0.0f32;
                for ci in 0..cin {
                    let wi = (k * cin + ci) * cout + co;
                    let w = if prefetch && ci > 0 { next_weight } else { weights[wi] };
                    if prefetch && ci + 1 < cin {
                        next_weight = weights[wi + cout];
                    }
                    for dt in 0..tile {
                        let t = t0 + dt;
                        if t >= time { continue; }
                        let ti = t as isize + (k * dilation) as isize - padding as isize;
                        if (0..time as isize).contains(&ti) {
                            let x = activation(input[ti as usize * cin + ci]);
                            accum[dt] = w.mul_add(x, accum[dt]);
                        }
                    }
                }
            }
            for dt in 0..tile {
                let t = t0 + dt;
                if t < time { output[t * cout + co] = accum[dt] + bias[co] + residual[t * cout + co]; }
            }
        }
    }
    output
}

#[test]
fn prefetched_weights_preserve_accumulation_for_dilated_resblocks() {
    for &time in &[1usize, 2, 3, 4, 5, 17, 23] {
        for &cin in &[1usize, 2, 3, 7, 19] {
            for &cout in &[1usize, 3, 4, 8] {
                for &kernel in &[3usize, 7, 11] {
                    for &dilation in &[1usize, 3, 5] {
                        let input: Vec<_> = (0..time * cin)
                            .map(|i| ((i * 17 % 47) as f32 - 23.0) * 0.013).collect();
                        let weights: Vec<_> = (0..kernel * cin * cout)
                            .map(|i| ((i * 19 % 53) as f32 - 26.0) * 0.006).collect();
                        let bias: Vec<_> = (0..cout).map(|i| (i as f32 - 2.0) * 0.05).collect();
                        let residual: Vec<_> = (0..time * cout)
                            .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.002).collect();
                        for &tile in &[2usize, 4] {
                            let args = (time, cin, cout, kernel, dilation, tile);
                            let base = conv(&input, &weights, &bias, &residual, args.0, args.1, args.2, args.3, args.4, args.5, false);
                            let candidate = conv(&input, &weights, &bias, &residual, args.0, args.1, args.2, args.3, args.4, args.5, true);
                            assert_eq!(base.len(), candidate.len());
                            for (j, (&a, &b)) in base.iter().zip(candidate.iter()).enumerate() {
                                assert_eq!(a.to_bits(), b.to_bits(), "{args:?} sample {j}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn prefetch_address_tracks_contiguous_input_channels() {
    for &(cin, cout, kernel) in &[(1usize, 128usize, 11usize), (128, 128, 11), (9, 4, 3), (7, 32, 7)] {
        for k in 0..kernel {
            let mut next = None;
            for ci in 0..cin {
                let direct = (k * cin + ci) * cout;
                let current = next.unwrap_or(direct);
                assert_eq!(current, direct);
                next = if ci + 1 < cin { Some(direct + cout) } else { None };
            }
        }
    }
}
