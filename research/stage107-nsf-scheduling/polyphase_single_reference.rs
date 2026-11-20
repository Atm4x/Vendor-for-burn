include!("../stage106-nsf-polyphase/reference.rs");

fn polyphase_single(s: Shape, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    let tout = (s.tin - 1) * s.stride + s.kernel - 2 * s.padding;
    let samples_per_phase = tout.div_ceil(s.stride);
    let mut out = vec![0.0; s.batch * s.co * tout];
    for b in 0..s.batch {
        for phase in 0..s.stride {
            for phase_time in 0..samples_per_phase {
                let t = phase + phase_time * s.stride;
                if t >= tout {
                    continue;
                }
                let base = t + s.padding;
                let filter_phase = base % s.stride;
                let source_time = base / s.stride;
                let taps = (s.kernel + s.stride - 1 - filter_phase) / s.stride;
                for oc in 0..s.co {
                    let group = oc / (s.co / s.groups);
                    let local_oc = oc % (s.co / s.groups);
                    let mut sum = bias[oc];
                    let inside = taps > 0 && source_time >= taps - 1 && source_time < s.tin;
                    if inside {
                        for tap in 0..taps {
                            let k = filter_phase + tap * s.stride;
                            for ci in 0..s.ci / s.groups {
                                let ic = group * (s.ci / s.groups) + ci;
                                let wi = ((ic * (s.co / s.groups) + local_oc) * s.kernel) + k;
                                sum += x[(b * s.ci + ic) * s.tin + source_time - tap] * w[wi];
                            }
                        }
                    } else {
                        for tap in 0..taps {
                            if source_time < tap || source_time - tap >= s.tin {
                                continue;
                            }
                            let k = filter_phase + tap * s.stride;
                            for ci in 0..s.ci / s.groups {
                                let ic = group * (s.ci / s.groups) + ci;
                                let wi = ((ic * (s.co / s.groups) + local_oc) * s.kernel) + k;
                                sum += x[(b * s.ci + ic) * s.tin + source_time - tap] * w[wi];
                            }
                        }
                    }
                    out[(b * s.co + oc) * tout + t] = sum;
                }
            }
        }
    }
    out
}

#[test]
fn single_phase_matches_direct_across_grouped_and_odd_shapes() {
    let cases = [
        Shape {
            batch: 1,
            ci: 1,
            co: 1,
            tin: 1,
            kernel: 2,
            stride: 1,
            padding: 0,
            groups: 1,
        },
        Shape {
            batch: 1,
            ci: 3,
            co: 5,
            tin: 9,
            kernel: 6,
            stride: 2,
            padding: 2,
            groups: 1,
        },
        Shape {
            batch: 2,
            ci: 4,
            co: 7,
            tin: 8,
            kernel: 8,
            stride: 4,
            padding: 2,
            groups: 1,
        },
        Shape {
            batch: 1,
            ci: 8,
            co: 16,
            tin: 14,
            kernel: 3,
            stride: 1,
            padding: 1,
            groups: 1,
        },
        Shape {
            batch: 1,
            ci: 6,
            co: 10,
            tin: 7,
            kernel: 5,
            stride: 8,
            padding: 1,
            groups: 2,
        },
        Shape {
            batch: 2,
            ci: 2,
            co: 4,
            tin: 2,
            kernel: 10,
            stride: 5,
            padding: 3,
            groups: 2,
        },
        Shape {
            batch: 1,
            ci: 4,
            co: 4,
            tin: 3,
            kernel: 16,
            stride: 8,
            padding: 4,
            groups: 4,
        },
    ];
    for s in cases {
        let x = (0..s.batch * s.ci * s.tin)
            .map(|i| (i as f32 * 0.17).sin() * 0.2)
            .collect::<Vec<_>>();
        let w = (0..s.ci * (s.co / s.groups) * s.kernel)
            .map(|i| (i as f32 * 0.13).cos() * 0.2)
            .collect::<Vec<_>>();
        let bias = (0..s.co).map(|i| i as f32 * 0.005).collect::<Vec<_>>();
        let a = polyphase_single(s, &x, &w, &bias);
        let b = ref_scatter(s, &x, &w, &bias);
        let err = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            err < 1e-5,
            "stride={},groups={},err={err}",
            s.stride,
            s.groups
        );
    }
}
