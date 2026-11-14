struct Shape {
    batch: usize,
    channels: usize,
    time: usize,
    kernel: usize,
    dilation: usize,
}

fn activate(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * 0.1 }
}

fn conv_reference(x: &[f32], w: &[f32], b: &[f32], shape: &Shape) -> Vec<f32> {
    let mut result = vec![0.0; x.len()];
    let padding = (shape.kernel - 1) * shape.dilation / 2;
    for batch in 0..shape.batch {
        for output_channel in 0..shape.channels {
            for t in 0..shape.time {
                let mut acc = 0.0;
                for input_channel in 0..shape.channels {
                    for tap in 0..shape.kernel {
                        let shifted = t + tap * shape.dilation;
                        if shifted < padding || shifted - padding >= shape.time {
                            continue;
                        }
                        let src = (batch * shape.channels + input_channel) * shape.time + shifted
                            - padding;
                        let weight =
                            (output_channel * shape.channels + input_channel) * shape.kernel + tap;
                        acc += x[src] * w[weight];
                    }
                }
                result[(batch * shape.channels + output_channel) * shape.time + t] =
                    acc + b[output_channel];
            }
        }
    }
    result
}

fn fused_preact_conv_reference(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    residual: Option<&[f32]>,
    shape: &Shape,
) -> Vec<f32> {
    let mut result = vec![0.0; x.len()];
    let padding = (shape.kernel - 1) * shape.dilation / 2;
    for flat in 0..result.len() {
        let t = flat % shape.time;
        let out_channel = flat / shape.time % shape.channels;
        let batch = flat / (shape.time * shape.channels);
        let mut value = 0.0;
        for in_channel in 0..shape.channels {
            for tap in 0..shape.kernel {
                let shifted = t + tap * shape.dilation;
                if shifted >= padding && shifted - padding < shape.time {
                    let input_offset =
                        (batch * shape.channels + in_channel) * shape.time + shifted - padding;
                    let weight_offset =
                        (out_channel * shape.channels + in_channel) * shape.kernel + tap;
                    value += activate(x[input_offset]) * w[weight_offset];
                }
            }
        }
        value += b[out_channel];
        if let Some(base) = residual {
            value += base[flat];
        }
        result[flat] = value;
    }
    result
}

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * (seed + 13) + 37) % 701) as f32 - 350.0) / 719.0)
        .collect()
}

#[test]
fn fused_residual_pair_matches_full_nonlinear_reference() {
    for (batch, channels, time, kernel, dilation) in [
        (1, 2, 7, 1, 1),
        (1, 3, 13, 3, 1),
        (2, 4, 31, 3, 2),
        (1, 5, 99, 5, 3),
        (2, 8, 77, 3, 9),
        (1, 16, 65, 3, 1),
    ] {
        let shape = Shape {
            batch,
            channels,
            time,
            kernel,
            dilation,
        };
        let x = values(batch * channels * time, 11);
        let first_weights = values(channels * channels * kernel, 19);
        let second_weights = values(channels * channels * kernel, 23);
        let first_bias = values(channels, 31);
        let second_bias = values(channels, 47);
        let activated: Vec<_> = x.iter().copied().map(activate).collect();
        let h = conv_reference(&activated, &first_weights, &first_bias, &shape);
        let h_activated: Vec<_> = h.iter().copied().map(activate).collect();
        let y = conv_reference(&h_activated, &second_weights, &second_bias, &shape);
        let expected: Vec<_> = x.iter().zip(y).map(|(a, b)| a + b).collect();
        let h_fused = fused_preact_conv_reference(&x, &first_weights, &first_bias, None, &shape);
        let actual =
            fused_preact_conv_reference(&h_fused, &second_weights, &second_bias, Some(&x), &shape);
        assert_eq!(expected.len(), actual.len());
        for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
            assert!(
                (left - right).abs() <= 2e-5,
                "shape=({batch},{channels},{time},{kernel},{dilation}), sample={index}: expected={left}, actual={right}"
            );
        }
    }
}

#[test]
fn fused_operator_preserves_local_zero_padding_semantics() {
    for (time, kernel, dilation) in [(1, 3, 1), (3, 5, 2), (9, 3, 7)] {
        let shape = Shape {
            batch: 1,
            channels: 2,
            time,
            kernel,
            dilation,
        };
        let x = values(shape.channels * time, 7);
        let w = values(shape.channels * shape.channels * kernel, 17);
        let bias = vec![0.123, -0.31];
        let actual = fused_preact_conv_reference(&x, &w, &bias, None, &shape);
        let reference = conv_reference(
            &x.iter().copied().map(activate).collect::<Vec<_>>(),
            &w,
            &bias,
            &shape,
        );
        assert_eq!(actual, reference);
    }
}
