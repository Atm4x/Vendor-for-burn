include!("../stage106-nsf-polyphase/noise_reference.rs");

fn noise_interior(s: Shape, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let tout = (s.time + 2 * s.padding - s.dilation * (s.kernel - 1) - 1) / s.stride + 1;
    let mut out = vec![0f32; s.batch * s.cout * tout];
    for b in 0..s.batch {
        for pair in 0..tout.div_ceil(2) {
            let t0 = pair * 2;
            let t1 = t0 + 1;
            let base0 = (t0 * s.stride) as isize - s.padding as isize;
            let base1 = base0 + s.stride as isize;
            let last1 = base1 + ((s.kernel - 1) * s.dilation) as isize;
            let interior = t1 < tout && base0 >= 0 && last1 < s.time as isize;
            for oc in 0..s.cout {
                let mut sum0 = bias[oc];
                let mut sum1 = bias[oc];
                for k in 0..s.kernel {
                    let x0 = base0 + (k * s.dilation) as isize;
                    let x1 = x0 + s.stride as isize;
                    let w = weight[oc * s.kernel + k];
                    if interior || (0..s.time as isize).contains(&x0) {
                        sum0 += input[b * s.time + x0 as usize] * w;
                    }
                    if interior || (t1 < tout && (0..s.time as isize).contains(&x1)) {
                        sum1 += input[b * s.time + x1 as usize] * w;
                    }
                }
                out[(b * s.cout + oc) * tout + t0] = sum0;
                if t1 < tout {
                    out[(b * s.cout + oc) * tout + t1] = sum1;
                }
            }
        }
    }
    out
}

#[test]
fn fast_interior_keeps_exact_boundary_results() {
    for s in [
        Shape {
            batch: 1,
            cout: 1,
            time: 10,
            kernel: 3,
            stride: 1,
            dilation: 1,
            padding: 1,
        },
        Shape {
            batch: 1,
            cout: 5,
            time: 33,
            kernel: 8,
            stride: 4,
            dilation: 1,
            padding: 3,
        },
        Shape {
            batch: 2,
            cout: 7,
            time: 101,
            kernel: 16,
            stride: 8,
            dilation: 1,
            padding: 8,
        },
        Shape {
            batch: 1,
            cout: 16,
            time: 39,
            kernel: 5,
            stride: 1,
            dilation: 2,
            padding: 4,
        },
        Shape {
            batch: 2,
            cout: 32,
            time: 77,
            kernel: 20,
            stride: 10,
            dilation: 1,
            padding: 10,
        },
        Shape {
            batch: 1,
            cout: 4,
            time: 81,
            kernel: 6,
            stride: 2,
            dilation: 3,
            padding: 6,
        },
    ] {
        let x = (0..s.batch * s.time)
            .map(|i| (i as f32 * 0.13).cos() * 0.11)
            .collect::<Vec<_>>();
        let w = (0..s.cout * s.kernel)
            .map(|i| (i as f32 * 0.19).sin() * 0.13)
            .collect::<Vec<_>>();
        let bias = (0..s.cout).map(|i| i as f32 * 0.007).collect::<Vec<_>>();
        let a = noise_interior(s, &x, &w, &bias);
        let b = reference(s, &x, &w, &bias);
        let err = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-6, "noise interior parity failed: {err}");
    }
}
