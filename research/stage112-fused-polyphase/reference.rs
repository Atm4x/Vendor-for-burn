#[derive(Clone, Copy)]
struct Geometry {
    n: usize,
    cin: usize,
    cout: usize,
    groups: usize,
    len: usize,
    stride: usize,
    padding: usize,
}

fn direct(g: Geometry, x: &[f64], w: &[f64], bias: &[f64]) -> Vec<f64> {
    let kernel = g.stride * 2;
    let out_len = (g.len - 1) * g.stride + kernel - 2 * g.padding;
    let mut output = vec![0.0; g.n * g.cout * out_len];
    for batch in 0..g.n {
        for oc in 0..g.cout {
            for t in 0..out_len {
                output[(batch * g.cout + oc) * out_len + t] = bias[oc];
            }
        }
        for ci in 0..g.cin {
            for ti in 0..g.len {
                for k in 0..kernel {
                    let to = ti as isize * g.stride as isize + k as isize - g.padding as isize;
                    if to < 0 || to >= out_len as isize { continue; }
                    let group = ci / (g.cin / g.groups);
                    for local_oc in 0..g.cout / g.groups {
                        let oc = group * (g.cout / g.groups) + local_oc;
                        let value = x[(batch * g.cin + ci) * g.len + ti]
                            * w[(ci * (g.cout / g.groups) + local_oc) * kernel + k];
                        output[(batch * g.cout + oc) * out_len + to as usize] += value;
                    }
                }
            }
        }
    }
    output
}

fn fused_formula(g: Geometry, x: &[f64], w: &[f64], bias: &[f64]) -> Vec<f64> {
    let kernel = 2 * g.stride;
    let out_len = (g.len - 1) * g.stride + kernel - 2 * g.padding;
    let ci_group = g.cin / g.groups;
    let co_group = g.cout / g.groups;
    let mut output = vec![0.0; g.n * g.cout * out_len];
    for batch in 0..g.n {
        for group in 0..g.groups {
            for phase in 0..g.stride {
                let filter_phase = (phase + g.padding) % g.stride;
                let source_shift = (phase + g.padding) / g.stride;
                for q in 0..out_len.div_ceil(g.stride) {
                    let to = phase + q * g.stride;
                    if to >= out_len { continue; }
                    for local_oc in 0..co_group {
                        let oc = group * co_group + local_oc;
                        let mut sum = bias[oc];
                        for ci_local in 0..ci_group {
                            let ci = group * ci_group + ci_local;
                            for tap in 0usize..2 {
                                let source = source_shift as isize + q as isize - tap as isize;
                                if source < 0 || source >= g.len as isize { continue; }
                                let xidx = (batch * g.cin + ci) * g.len + source as usize;
                                let widx = (ci * co_group + local_oc) * kernel
                                    + filter_phase + tap * g.stride;
                                sum += x[xidx] * w[widx];
                            }
                        }
                        output[(batch * g.cout + oc) * out_len + to] = sum;
                    }
                }
            }
        }
    }
    output
}

#[test]
fn two_tap_phase_matches_direct_all_geometries() {
    let cases = [
        Geometry { n: 2, cin: 64, cout: 32, groups: 1, len: 17, stride: 8, padding: 4 },
        Geometry { n: 2, cin: 128, cout: 64, groups: 1, len: 13, stride: 10, padding: 5 },
        Geometry { n: 1, cin: 7, cout: 13, groups: 1, len: 17, stride: 4, padding: 3 },
        Geometry { n: 1, cin: 12, cout: 20, groups: 4, len: 23, stride: 3, padding: 2 },
        Geometry { n: 2, cin: 16, cout: 8, groups: 2, len: 11, stride: 2, padding: 1 },
        Geometry { n: 1, cin: 3, cout: 5, groups: 1, len: 1, stride: 2, padding: 1 },
        Geometry { n: 1, cin: 8, cout: 10, groups: 2, len: 4, stride: 1, padding: 0 },
        Geometry { n: 1, cin: 17, cout: 21, groups: 1, len: 7, stride: 5, padding: 0 },
    ];
    for g in cases {
        let x: Vec<f64> = (0..g.n * g.cin * g.len)
            .map(|i| (i as f64 * 0.031).sin() * 0.2)
            .collect();
        let w: Vec<f64> = (0..g.cin * (g.cout / g.groups) * 2 * g.stride)
            .map(|i| (i as f64 * 0.027).cos() * 0.1)
            .collect();
        let bias: Vec<f64> = (0..g.cout).map(|i| i as f64 * 0.001).collect();
        let actual = fused_formula(g, &x, &w, &bias);
        let expected = direct(g, &x, &w, &bias);
        let err = actual.iter().zip(expected).map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);
        assert!(err < 1e-11, "shape: C{}/{} O{} T{} S{} P{}, diff {err}",
            g.cin,g.groups,g.cout,g.len,g.stride,g.padding);
    }
}
