const TIME_TILE: usize = 32;

fn relu(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        x * 0.1
    }
}

#[derive(Clone)]
struct Case {
    batch: usize,
    channels: usize,
    time: usize,
    kernel1: usize,
    dilation1: usize,
    kernel2: usize,
    input: Vec<f32>,
    weight1: Vec<f32>,
    weight2: Vec<f32>,
    bias1: Vec<f32>,
    bias2: Vec<f32>,
}

fn value(seed: usize) -> f32 {
    (seed.wrapping_mul(43).wrapping_add(19) % 127) as f32 / 80.0 - 0.8
}

fn case(
    batch: usize,
    channels: usize,
    time: usize,
    kernel1: usize,
    dilation1: usize,
    kernel2: usize,
) -> Case {
    let input = (0..batch * channels * time).map(value).collect();
    let weight1 = (0..channels * channels * kernel1)
        .map(|v| value(v + 17) / channels as f32)
        .collect();
    let weight2 = (0..channels * channels * kernel2)
        .map(|v| value(v + 39) / channels as f32)
        .collect();
    let bias1 = (0..channels).map(|v| value(v + 137) / 10.0).collect();
    let bias2 = (0..channels).map(|v| value(v + 313) / 10.0).collect();
    Case {
        batch,
        channels,
        time,
        kernel1,
        dilation1,
        kernel2,
        input,
        weight1,
        weight2,
        bias1,
        bias2,
    }
}

fn baseline(case: &Case) -> Vec<f32> {
    let (b, c, t) = (case.batch, case.channels, case.time);
    let (k1, k2) = (case.kernel1, case.kernel2);
    let p1 = (k1 - 1) / 2 * case.dilation1;
    let p2 = (k2 - 1) / 2;
    let mut h = vec![0f32; b * c * t];
    for batch in 0..b {
        for oc in 0..c {
            for time in 0..t {
                let mut sum = 0f32;
                for ic in 0..c {
                    for tap in 0..k1 {
                        let ix = time + tap * case.dilation1;
                        if ix >= p1 && ix - p1 < t {
                            sum += relu(case.input[(batch * c + ic) * t + ix - p1])
                                * case.weight1[(oc * c + ic) * k1 + tap];
                        }
                    }
                }
                h[(batch * c + oc) * t + time] = sum + case.bias1[oc];
            }
        }
    }
    let mut output = case.input.clone();
    for batch in 0..b {
        for oc in 0..c {
            for time in 0..t {
                let mut sum = 0f32;
                for ic in 0..c {
                    for tap in 0..k2 {
                        let ix = time + tap;
                        if ix >= p2 && ix - p2 < t {
                            sum += relu(h[(batch * c + ic) * t + ix - p2])
                                * case.weight2[(oc * c + ic) * k2 + tap];
                        }
                    }
                }
                output[(batch * c + oc) * t + time] += sum + case.bias2[oc];
            }
        }
    }
    output
}

fn tiled(case: &Case) -> Vec<f32> {
    let (b, c, t) = (case.batch, case.channels, case.time);
    let (k1, k2) = (case.kernel1, case.kernel2);
    let p1 = (k1 - 1) / 2 * case.dilation1;
    let p2 = (k2 - 1) / 2;
    let mut output = vec![0f32; b * c * t];
    for batch in 0..b {
        for start in (0..t).step_by(TIME_TILE) {
            let expanded = TIME_TILE + 2 * p2;
            let mut middle = vec![0f32; expanded * c];
            for offset in 0..expanded {
                let actual = start as isize + offset as isize - p2 as isize;
                if actual < 0 || actual >= t as isize {
                    continue;
                }
                let actual = actual as usize;
                for oc in 0..c {
                    let mut sum = 0f32;
                    for ic in 0..c {
                        for tap in 0..k1 {
                            let ix = actual + tap * case.dilation1;
                            if ix >= p1 && ix - p1 < t {
                                sum += relu(case.input[(batch * c + ic) * t + ix - p1])
                                    * case.weight1[(oc * c + ic) * k1 + tap];
                            }
                        }
                    }
                    middle[offset * c + oc] = sum + case.bias1[oc];
                }
            }
            for local in 0..TIME_TILE {
                let actual = start + local;
                if actual >= t {
                    break;
                }
                for oc in 0..c {
                    let mut sum = 0f32;
                    for ic in 0..c {
                        for tap in 0..k2 {
                            let pos = start as isize + local as isize + tap as isize - p2 as isize;
                            if pos >= 0 && pos < t as isize {
                                sum += relu(middle[(local + tap) * c + ic])
                                    * case.weight2[(oc * c + ic) * k2 + tap];
                            }
                        }
                    }
                    output[(batch * c + oc) * t + actual] =
                        case.input[(batch * c + oc) * t + actual] + (sum + case.bias2[oc]);
                }
            }
        }
    }
    output
}

#[test]
fn complete_two_conv_pair_matches_across_time_tiles() {
    for (b, c, t, k1, d1, k2) in [
        (1, 1, 1, 3, 1, 3),
        (1, 3, 3, 5, 3, 5),
        (2, 4, 15, 7, 9, 7),
        (2, 5, 16, 11, 3, 11),
        (1, 16, 17, 3, 1, 3),
        (2, 32, 31, 5, 9, 5),
        (1, 64, 33, 3, 1, 3),
        (2, 8, 65, 11, 5, 11),
    ] {
        let case = case(b, c, t, k1, d1, k2);
        let reference = baseline(&case);
        let result = tiled(&case);
        assert_eq!(reference.len(), result.len());
        for (i, (a, b)) in reference.iter().zip(&result).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "shape=(B{b},C{c},T{t}) output #{i}: {a} != {b}"
            );
        }
    }
}

#[test]
fn middle_halo_is_bounded_and_coverage_exact() {
    for channels in [1, 4, 16, 32, 64] {
        for k2 in [1, 3, 5, 7, 9, 11] {
            let halo = (k2 - 1) / 2;
            assert!((TIME_TILE + halo * 2) * channels <= (TIME_TILE + 10) * 64);
            for length in [1, 3, 15, 16, 17, 31, 32, 63] {
                let mut coverage = vec![0; length];
                for start in (0..length).step_by(TIME_TILE) {
                    for local in 0..TIME_TILE {
                        if start + local < length {
                            coverage[start + local] += 1;
                        }
                        for tap in 0..k2 {
                            let middle = local + tap;
                            assert!(middle < TIME_TILE + 2 * halo);
                        }
                    }
                }
                assert!(coverage.iter().all(|x| *x == 1));
            }
        }
    }
}
