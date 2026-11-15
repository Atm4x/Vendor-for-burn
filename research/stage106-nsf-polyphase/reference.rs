#[derive(Clone, Copy)]
struct Shape {
    batch: usize,
    ci: usize,
    co: usize,
    tin: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
}

fn ref_scatter(s: Shape, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    let tout = (s.tin - 1) * s.stride + s.kernel - 2 * s.padding;
    let mut out = vec![0.0; s.batch * s.co * tout];
    for b in 0..s.batch {
        for oc in 0..s.co {
            for t in 0..tout {
                out[(b * s.co + oc) * tout + t] = bias[oc];
            }
        }
        for ic in 0..s.ci {
            let g = ic / (s.ci / s.groups);
            for t in 0..s.tin {
                let input = x[(b * s.ci + ic) * s.tin + t];
                for k in 0..s.kernel {
                    let dst = t * s.stride + k;
                    if dst < s.padding || dst - s.padding >= tout { continue; }
                    for oc_local in 0..(s.co / s.groups) {
                        let oc = g * (s.co / s.groups) + oc_local;
                        let wi = ((ic * (s.co / s.groups) + oc_local) * s.kernel) + k;
                        out[(b * s.co + oc) * tout + dst - s.padding] += input * w[wi];
                    }
                }
            }
        }
    }
    out
}

fn polyphase_pair(s: Shape, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    let tout = (s.tin - 1) * s.stride + s.kernel - 2 * s.padding;
    let mut out = vec![0.0; s.batch * s.co * tout];
    let pairs_per_phase = tout.div_ceil(s.stride * 2);
    for b in 0..s.batch {
        for phase in 0..s.stride {
            for pair in 0..pairs_per_phase {
                let t0 = phase + 2 * s.stride * pair;
                let t1 = t0 + s.stride;
                if t0 >= tout { continue; }
                let base = t0 + s.padding;
                let filter_phase = base % s.stride;
                let src_t = base / s.stride;
                let taps = (s.kernel + s.stride - 1 - filter_phase) / s.stride;
                for oc in 0..s.co {
                    let group = oc / (s.co / s.groups);
                    let local_oc = oc % (s.co / s.groups);
                    let mut sum0 = bias[oc];
                    let mut sum1 = bias[oc];
                    for tap in 0..taps {
                        let k = filter_phase + tap * s.stride;
                        for ici in 0..(s.ci / s.groups) {
                            let ic = group * (s.ci / s.groups) + ici;
                            let wi = ((ic * (s.co / s.groups) + local_oc) * s.kernel) + k;
                            if src_t >= tap && src_t - tap < s.tin {
                                sum0 += x[(b * s.ci + ic) * s.tin + src_t - tap] * w[wi];
                            }
                            if t1 < tout && src_t + 1 >= tap && src_t + 1 - tap < s.tin {
                                sum1 += x[(b * s.ci + ic) * s.tin + src_t + 1 - tap] * w[wi];
                            }
                        }
                    }
                    out[(b * s.co + oc) * tout + t0] = sum0;
                    if t1 < tout {out[(b * s.co + oc) * tout + t1] = sum1;}
                }
            }
        }
    }
    out
}

#[test]
fn polyphase_stride_padding_groups_and_odd_lengths() {
    let cases = [
        Shape { batch:1, ci:1, co:1, tin:1, kernel:2, stride:1, padding:0, groups:1 },
        Shape { batch:1, ci:3, co:5, tin:9, kernel:6, stride:2, padding:2, groups:1 },
        Shape { batch:2, ci:4, co:7, tin:8, kernel:8, stride:4, padding:2, groups:1 },
        Shape { batch:1, ci:8, co:16, tin:14, kernel:3, stride:1, padding:1, groups:1 },
        Shape { batch:1, ci:6, co:10, tin:7, kernel:5, stride:8, padding:1, groups:2 },
        Shape { batch:2, ci:2, co:4, tin:2, kernel:10, stride:5, padding:3, groups:2 },
        Shape { batch:1, ci:4, co:4, tin:3, kernel:16, stride:8, padding:4, groups:4 },
    ];
    for shape in cases {
        let x = (0..shape.batch * shape.ci * shape.tin)
            .map(|i| (i as f32 * 0.17).sin() * 0.2).collect::<Vec<_>>();
        let w = (0..shape.ci * (shape.co / shape.groups) * shape.kernel)
            .map(|i| (i as f32 * 0.13).cos() * 0.2).collect::<Vec<_>>();
        let bias = (0..shape.co).map(|i| i as f32 * 0.005).collect::<Vec<_>>();
        let actual = polyphase_pair(shape, &x, &w, &bias);
        let expected = ref_scatter(shape, &x, &w, &bias);
        let max_error = actual.iter().zip(expected.iter())
            .map(|(a,b)| (*a - *b).abs()).fold(0f32, f32::max);
        assert!(max_error < 1e-5, "case stride={} groups={} diff={max_error}", shape.stride, shape.groups);
    }
}
