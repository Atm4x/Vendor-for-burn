fn activate(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * 0.1 }
}

fn compute(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    residual: &[f32],
    time: usize,
    cin: usize,
    kernel: usize,
    dilation: usize,
    use_plane_broadcast: bool,
) -> Vec<f32> {
    const COUT: usize = 128;
    const VECTOR_SIZE: usize = 4;
    const PLANE_SIZE: usize = COUT / VECTOR_SIZE;
    const TILE: usize = 4;
    let padding = ((kernel - 1) * dilation) / 2;
    let mut output = vec![0.0f32; time * COUT];
    for t0 in (0..time).step_by(TILE) {
        let mut sums = [[0.0f32; TILE]; COUT];
        for k in 0..kernel {
            let indices: [Option<usize>; TILE] = std::array::from_fn(|dt| {
                let t = t0 + dt;
                (t < time)
                    .then_some(t as isize + (k * dilation) as isize - padding as isize)
                    .filter(|&i| i >= 0 && (i as usize) < time)
                    .map(|i| i as usize)
            });
            for ci in 0..cin {
                let mut warp_inputs = [0.0f32; TILE];
                for dt in 0..TILE {
                    if let Some(i) = indices[dt] {
                        warp_inputs[dt] = input[i * cin + ci];
                    }
                }
                for lane in 0..PLANE_SIZE {
                    for v in 0..VECTOR_SIZE {
                        let co = lane * VECTOR_SIZE + v;
                        let w = weight[(k * cin + ci) * COUT + co];
                        for dt in 0..TILE {
                            if let Some(i) = indices[dt] {
                                let x = if use_plane_broadcast {
                                    warp_inputs[dt]
                                } else {
                                    input[i * cin + ci]
                                };
                                sums[co][dt] = w.mul_add(activate(x), sums[co][dt]);
                            }
                        }
                    }
                }
            }
        }
        for co in 0..COUT {
            for dt in 0..TILE {
                let t = t0 + dt;
                if t < time {
                    output[t * COUT + co] = sums[co][dt] + bias[co] + residual[t * COUT + co];
                }
            }
        }
    }
    output
}

#[test]
fn broadcast_preserves_all_output_channels_and_f32_order() {
    const COUT: usize = 128;
    for &time in &[1usize, 2, 3, 4, 5, 9, 17, 31] {
        for &cin in &[1usize, 2, 3, 16, 128] {
            for &kernel in &[3usize, 7, 11] {
                for &dilation in &[1usize, 3, 5] {
                    let input: Vec<_> = (0..time * cin)
                        .map(|i| ((i * 13 % 41) as f32 - 20.0) * 0.009).collect();
                    let weight: Vec<_> = (0..kernel * cin * COUT)
                        .map(|i| ((i * 19 % 47) as f32 - 23.0) * 0.012).collect();
                    let bias: Vec<_> = (0..COUT)
                        .map(|i| (i as f32 - 64.0) * 0.002).collect();
                    let residual: Vec<_> = (0..time * COUT)
                        .map(|i| ((i * 11 % 29) as f32 - 14.0) * 0.017).collect();
                    let reference = compute(
                        &input, &weight, &bias, &residual, time, cin, kernel, dilation, false,
                    );
                    let broadcast = compute(
                        &input, &weight, &bias, &residual, time, cin, kernel, dilation, true,
                    );
                    for (i, (&a, &b)) in reference.iter().zip(&broadcast).enumerate() {
                        assert_eq!(a.to_bits(), b.to_bits(),
                            "bitwise mismatch t={time} cin={cin} k={kernel} d={dilation} i={i}");
                    }
                }
            }
        }
    }
}

#[test]
fn one_plane_matches_exactly_one_temporal_tile_for_c128_vec4() {
    const COUT: usize = 128;
    const NOUT: usize = 4;
    const WARP: usize = 32;
    const TILE: usize = 4;
    assert_eq!(COUT / NOUT, WARP);
    for &time in &[1usize, 2, 3, 4, 5, 23, 1192] {
        let pairs = time.div_ceil(TILE);
        let total_work = pairs * (COUT / NOUT);
        assert_eq!(total_work % WARP, 0);
        for unit in 0..total_work {
            let output_vector = unit % (COUT / NOUT);
            let temporal_tile = unit / (COUT / NOUT);
            assert_eq!(output_vector, unit % WARP);
            for lane in 0..WARP {
                let neighbor = (unit / WARP) * WARP + lane;
                assert_eq!(neighbor / (COUT / NOUT), temporal_tile);
            }
        }
    }
}
