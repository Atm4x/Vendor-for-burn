#[derive(Clone, Copy)]
struct Geometry {
    channels: usize,
    length: usize,
    kernel: usize,
    dilation: usize,
}

fn pack_output_major(w: &[f32], g: Geometry) -> Vec<f32> {
    let mut out = vec![0.0; w.len()];
    for oc in 0..g.channels {
        for ic in 0..g.channels {
            for k in 0..g.kernel {
                out[(k * g.channels + ic) * g.channels + oc] =
                    w[(oc * g.channels + ic) * g.kernel + k];
            }
        }
    }
    out
}

fn reference(x: &[f32], w: &[f32], bias: &[f32], residual: Option<&[f32]>, g: Geometry) -> Vec<f32> {
    let mut out = vec![0.0f32; g.channels * g.length];
    let pad = (g.kernel - 1) * g.dilation / 2;
    for oc in 0..g.channels {
        for t in 0..g.length {
            let mut sum = 0.0f32;
            for ic in 0..g.channels {
                for k in 0..g.kernel {
                    let p = t + k * g.dilation;
                    if p >= pad && p - pad < g.length {
                        let v = x[ic * g.length + p - pad];
                        let activated = if v < 0.0 { v * 0.1 } else { v };
                        sum += activated * w[(oc * g.channels + ic) * g.kernel + k];
                    }
                }
            }
            sum += bias[oc];
            if let Some(r) = residual { sum += r[oc * g.length + t]; }
            out[oc * g.length + t] = sum;
        }
    }
    out
}

fn packed_time_channel(x: &[f32], packed: &[f32], bias: &[f32], residual: Option<&[f32]>, g: Geometry) -> Vec<f32> {
    let mut out = vec![0.0f32; g.channels * g.length];
    let pad = (g.kernel - 1) * g.dilation / 2;
    for t in 0..g.length {
        for oc in 0..g.channels {
            let mut sum = 0.0f32;
            for k in 0..g.kernel {
                let p = t + k * g.dilation;
                if p >= pad && p - pad < g.length {
                    for ic in 0..g.channels {
                        let v = x[ic * g.length + p - pad];
                        let activated = if v < 0.0 { v * 0.1 } else { v };
                        sum += activated * packed[(k * g.channels + ic) * g.channels + oc];
                    }
                }
            }
            sum += bias[oc];
            if let Some(r) = residual { sum += r[oc * g.length + t]; }
            out[oc * g.length + t] = sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_transport_preserves_every_element() {
        for channels in [1, 3, 7, 32] {
            let g = Geometry { channels, length: 11, kernel: 11, dilation: 5 };
            let weights = (0..channels * channels * 11)
                .map(|i| (i as f32 * 0.17).cos()).collect::<Vec<_>>();
            let packed = pack_output_major(&weights, g);
            for k in 0..11 {
                for ic in 0..channels {
                    for oc in 0..channels {
                        assert_eq!(
                            packed[(k * channels + ic) * channels + oc],
                            weights[(oc * channels + ic) * 11 + k],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn two_loop_orders_match_within_fp32_tolerance() {
        for (channels, length, kernel, dilation) in [
            (1, 1, 3, 1), (3, 19, 3, 1), (7, 35, 7, 3),
            (8, 16, 11, 5), (32, 21, 11, 5),
        ] {
            let g = Geometry { channels, length, kernel, dilation };
            let x = (0..channels * length)
                .map(|i| ((i * 17 + 9) as f32 * 0.037).sin() * 0.24 - 0.12)
                .collect::<Vec<_>>();
            let weights = (0..channels * channels * kernel)
                .map(|i| ((i * 5 + 11) as f32 * 0.043).cos() * 0.05)
                .collect::<Vec<_>>();
            let bias = (0..channels).map(|i| i as f32 * 0.007 - 0.02).collect::<Vec<_>>();
            let packed = pack_output_major(&weights, g);
            for residual in [None, Some(x.as_slice())] {
                let expected = reference(&x, &weights, &bias, residual, g);
                let actual = packed_time_channel(&x, &packed, &bias, residual, g);
                let max_abs = expected.iter().zip(&actual)
                    .map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
                assert!(max_abs <= 3e-5,
                    "channels={channels} kernel={kernel} dilation={dilation}, abs={max_abs}");
            }
        }
    }
}
