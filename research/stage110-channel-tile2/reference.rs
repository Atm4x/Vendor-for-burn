#[derive(Clone, Copy)]
struct Case {
    n: usize, cin: usize, cout: usize, groups: usize, t: usize, k: usize, stride: usize, pad: usize,
}

fn evaluate(c: Case, split: bool) {
    let in_group = c.cin / c.groups;
    let out_group = c.cout / c.groups;
    let out_len = (c.t - 1) * c.stride + c.k - 2 * c.pad;
    let x: Vec<f32> = (0..c.n * c.cin * c.t).map(|i| ((i as f32) * 0.031).sin() * 0.2).collect();
    let w: Vec<f32> = (0..c.cin * out_group * c.k).map(|i| ((i as f32) * 0.027).cos() * 0.1).collect();
    let mut actual = vec![0.0f32; c.n * c.cout * out_len];
    let mut baseline = actual.clone();
    for n in 0..c.n {
        for out_channel in 0..c.cout {
            let group = out_channel / out_group;
            let local_oc = out_channel % out_group;
            for time in 0..out_len {
                let base = time + c.pad;
                let phase = base % c.stride;
                let t_source = base / c.stride;
                let taps = (c.k + c.stride - 1 - phase) / c.stride;
                let mut sum = 0.001 * out_channel as f32;
                for tap in 0..taps {
                    if t_source >= tap && t_source - tap < c.t {
                        let source_time = t_source - tap;
                        let ki = phase + tap * c.stride;
                        for ci in 0..in_group {
                            let global_ci = group * in_group + ci;
                            let x0 = x[(n * c.cin + global_ci) * c.t + source_time];
                            let w0 = w[(global_ci * out_group + local_oc) * c.k + ki];
                            sum += x0 * w0;
                        }
                    }
                }
                baseline[(n * c.cout + out_channel) * out_len + time] = sum;
            }
        }
    }
    for n in 0..c.n {
        for group in 0..c.groups {
            for pair in 0..out_group.div_ceil(2) {
                let local_oc = pair * 2;
                let has_second = local_oc + 1 < out_group;
                for time in 0..out_len {
                    let base = time + c.pad;
                    let phase = base % c.stride;
                    let t_source = base / c.stride;
                    let taps = (c.k + c.stride - 1 - phase) / c.stride;
                    let oc0 = group * out_group + local_oc;
                    let oc1 = oc0 + 1;
                    let mut sum0 = if split { 0.0 } else { oc0 as f32 * 0.001 };
                    let mut sum1 = if split { 0.0 } else { oc1 as f32 * 0.001 };
                    let mut sum0b = 0.0f32;
                    let mut sum1b = 0.0f32;
                    for tap in 0..taps {
                        if t_source >= tap && t_source - tap < c.t {
                            let source_time = t_source - tap;
                            let ki = phase + tap * c.stride;
                            for cp in 0..in_group.div_ceil(2) {
                                for half in 0..2 {
                                    let ci = cp * 2 + half;
                                    if ci == in_group { continue; }
                                    let global_ci = group * in_group + ci;
                                    let x0 = x[(n * c.cin + global_ci) * c.t + source_time];
                                    let w0 = w[(global_ci * out_group + local_oc) * c.k + ki];
                                    if split && half == 1 { sum0b += x0 * w0 } else { sum0 += x0 * w0 }
                                    if has_second {
                                        let w1 = w[(global_ci * out_group + local_oc + 1) * c.k + ki];
                                        if split && half == 1 { sum1b += x0 * w1 } else { sum1 += x0 * w1 }
                                    }
                                }
                            }
                        }
                    }
                    if split { sum0 += sum0b + oc0 as f32 * 0.001; sum1 += sum1b + oc1 as f32 * 0.001; }
                    actual[(n * c.cout + oc0) * out_len + time] = sum0;
                    if has_second { actual[(n * c.cout + oc1) * out_len + time] = sum1; }
                }
            }
        }
    }
    let mut max_abs = 0f32;
    for (expected, got) in baseline.iter().zip(actual.iter()) {
        max_abs = max_abs.max((expected - got).abs());
    }
    assert!(max_abs < 1e-4, "max_abs={max_abs} split={split}");
}

#[test]
fn stride8() { let c=Case{n:2,cin:64,cout:32,groups:1,t:21,k:16,stride:8,pad:4};evaluate(c,false);evaluate(c,true); }
#[test]
fn stride10() { let c=Case{n:1,cin:128,cout:64,groups:1,t:13,k:20,stride:10,pad:5};evaluate(c,false);evaluate(c,true); }
#[test]
fn grouped() { let c=Case{n:2,cin:12,cout:20,groups:4,t:17,k:7,stride:3,pad:2};evaluate(c,false);evaluate(c,true); }
#[test]
fn odd_cin() { let c=Case{n:1,cin:7,cout:13,groups:1,t:17,k:9,stride:4,pad:3};evaluate(c,false);evaluate(c,true); }
#[test]
fn single_channel() { let c=Case{n:1,cin:1,cout:3,groups:1,t:11,k:5,stride:2,pad:1};evaluate(c,false);evaluate(c,true); }
#[test]
fn stride_one() { let c=Case{n:1,cin:11,cout:9,groups:1,t:13,k:3,stride:1,pad:1};evaluate(c,false);evaluate(c,true); }
