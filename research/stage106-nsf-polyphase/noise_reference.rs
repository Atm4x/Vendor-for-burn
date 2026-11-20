#[derive(Clone, Copy)]
struct Shape { batch: usize, cout: usize, time: usize, kernel: usize, stride: usize, dilation: usize, padding: usize }

fn reference(s: Shape, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let tout = (s.time + 2 * s.padding - s.dilation * (s.kernel - 1) - 1) / s.stride + 1;
    let mut y = vec![0.0f32; s.batch * s.cout * tout];
    for batch in 0..s.batch {
        for oc in 0..s.cout {
            for out_t in 0..tout {
                let mut sum = bias[oc];
                for k in 0..s.kernel {
                    let input_t = (out_t * s.stride + k * s.dilation) as isize - s.padding as isize;
                    if (0..s.time as isize).contains(&input_t) {
                        sum += input[batch * s.time + input_t as usize] * weight[oc * s.kernel + k];
                    }
                }
                y[(batch * s.cout + oc) * tout + out_t] = sum;
            }
        }
    }
    y
}

fn tiled(s: Shape, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let tout = (s.time + 2 * s.padding - s.dilation * (s.kernel - 1) - 1) / s.stride + 1;
    let mut y = vec![0.0f32; s.batch * s.cout * tout];
    for batch in 0..s.batch {
        for pair in 0..tout.div_ceil(2) {
            let t0 = pair * 2;
            let t1 = t0 + 1;
            for oc in 0..s.cout {
                let mut sum0 = bias[oc];
                let mut sum1 = sum0;
                for k in 0..s.kernel {
                    let x0 = (t0 * s.stride + k * s.dilation) as isize - s.padding as isize;
                    let x1 = x0 + s.stride as isize;
                    let w = weight[oc * s.kernel + k];
                    if (0..s.time as isize).contains(&x0) {
                        sum0 += input[batch * s.time + x0 as usize] * w;
                    }
                    if t1 < tout && (0..s.time as isize).contains(&x1) {
                        sum1 += input[batch * s.time + x1 as usize] * w;
                    }
                }
                y[(batch * s.cout + oc) * tout + t0] = sum0;
                if t1 < tout {y[(batch * s.cout + oc) * tout + t1] = sum1;}
            }
        }
    }
    y
}

#[test]
fn compare_noise_source_across_varied_strides_and_dilations() {
    for shape in [
        Shape { batch:1, cout:1, time:10, kernel:3, stride:1, dilation:1, padding:1 },
        Shape { batch:1, cout:5, time:33, kernel:8, stride:4, dilation:1, padding:3 },
        Shape { batch:2, cout:7, time:101, kernel:16, stride:8, dilation:1, padding:8 },
        Shape { batch:1, cout:16, time:39, kernel:5, stride:1, dilation:2, padding:4 },
        Shape { batch:2, cout:32, time:77, kernel:20, stride:10, dilation:1, padding:10 },
    ] {
        let x = (0..shape.batch * shape.time).map(|i| (i as f32 * 0.13).cos()*0.11).collect::<Vec<_>>();
        let w = (0..shape.cout * shape.kernel).map(|i| (i as f32 * 0.19).sin()*0.13).collect::<Vec<_>>();
        let bias = (0..shape.cout).map(|i| i as f32 * 0.007).collect::<Vec<_>>();
        let a = tiled(shape, &x, &w, &bias);
        let b = reference(shape, &x, &w, &bias);
        let err = a.iter().zip(b.iter()).map(|(a,b)|(*a-*b).abs()).fold(0.0f32,f32::max);
        assert!(err < 1e-6, "noise tiled parity failed: {err}");
    }
}
