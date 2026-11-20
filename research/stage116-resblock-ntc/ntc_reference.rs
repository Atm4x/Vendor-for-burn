#[derive(Clone, Copy)]
struct Geometry {
    channels: usize,
    length: usize,
    batch: usize,
    kernel: usize,
    dilation: usize,
}

fn nct_to_ntc(input: &[f32], g: Geometry) -> Vec<f32> {
    let mut result = vec![0.0; input.len()];
    for b in 0..g.batch {
        for c in 0..g.channels {
            for t in 0..g.length {
                result[(b * g.length + t) * g.channels + c] =
                    input[(b * g.channels + c) * g.length + t];
            }
        }
    }
    result
}

fn ntc_to_nct(input: &[f32], g: Geometry) -> Vec<f32> {
    let mut result = vec![0.0; input.len()];
    for b in 0..g.batch {
        for c in 0..g.channels {
            for t in 0..g.length {
                result[(b * g.channels + c) * g.length + t] =
                    input[(b * g.length + t) * g.channels + c];
            }
        }
    }
    result
}

fn weights(g: Geometry, salt: usize) -> Vec<f32> {
    (0..g.channels * g.channels * g.kernel)
        .map(|i| (((i * 31 + salt * 17) % 251) as f32 - 125.0) * 0.00026)
        .collect()
}

fn pack_weights(w: &[f32], g: Geometry) -> Vec<f32> {
    let mut packed = vec![0.0; w.len()];
    for oc in 0..g.channels {
        for ic in 0..g.channels {
            for k in 0..g.kernel {
                packed[(k * g.channels + ic) * g.channels + oc] =
                    w[(oc * g.channels + ic) * g.kernel + k];
            }
        }
    }
    packed
}

fn conv_nct(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    g: Geometry,
) -> Vec<f32> {
    let mut out = vec![0.0; input.len()];
    let pad = (g.kernel - 1) * g.dilation / 2;
    for b in 0..g.batch {
        for oc in 0..g.channels {
            for t in 0..g.length {
                let mut s = 0.0f32;
                for ic in 0..g.channels {
                    for k in 0..g.kernel {
                        let i = (t + k * g.dilation) as isize - pad as isize;
                        if i >= 0 && i < g.length as isize {
                            let x = input[(b * g.channels + ic) * g.length + i as usize];
                            let v = if x >= 0.0 { x } else { x * 0.1 };
                            s += v * weight[(oc * g.channels + ic) * g.kernel + k];
                        }
                    }
                }
                s += bias[oc];
                let index = (b * g.channels + oc) * g.length + t;
                if let Some(r) = residual {
                    s += r[index];
                }
                out[index] = s;
            }
        }
    }
    out
}

fn conv_ntc(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    g: Geometry,
) -> Vec<f32> {
    let mut out = vec![0.0; input.len()];
    let pad = (g.kernel - 1) * g.dilation / 2;
    for b in 0..g.batch {
        for t in 0..g.length {
            for oc in 0..g.channels {
                let mut s = 0.0f32;
                for k in 0..g.kernel {
                    let i = (t + k * g.dilation) as isize - pad as isize;
                    if i >= 0 && i < g.length as isize {
                        for ic in 0..g.channels {
                            let x = input[(b * g.length + i as usize) * g.channels + ic];
                            let v = if x >= 0.0 { x } else { x * 0.1 };
                            s += v * weight[(k * g.channels + ic) * g.channels + oc];
                        }
                    }
                }
                s += bias[oc];
                let index = (b * g.length + t) * g.channels + oc;
                if let Some(r) = residual {
                    s += r[index];
                }
                out[index] = s;
            }
        }
    }
    out
}

fn full_resblocks(start: &[f32], g: Geometry, packed: bool) -> Vec<f32> {
    let shared = if packed {
        nct_to_ntc(start, g)
    } else {
        start.to_vec()
    };
    let mut sum = vec![0.0; start.len()];
    for (branch, kernel) in [3, 7, 11].into_iter().enumerate() {
        let mut x = shared.clone();
        for (pair, dilation) in [1, 3, 5].into_iter().enumerate() {
            let first_g = Geometry {
                kernel,
                dilation,
                ..g
            };
            let second_g = Geometry {
                kernel,
                dilation: 1,
                ..g
            };
            let w1 = weights(first_g, branch * 100 + pair * 2);
            let w2 = weights(second_g, branch * 100 + pair * 2 + 1);
            let b1 = vec![0.0004; g.channels];
            let b2 = vec![-0.0009; g.channels];
            let y = if packed {
                conv_ntc(&x, &pack_weights(&w1, first_g), &b1, None, first_g)
            } else {
                conv_nct(&x, &w1, &b1, None, first_g)
            };
            x = if packed {
                conv_ntc(&y, &pack_weights(&w2, second_g), &b2, Some(&x), second_g)
            } else {
                conv_nct(&y, &w2, &b2, Some(&x), second_g)
            };
        }
        let x = if packed { ntc_to_nct(&x, g) } else { x };
        for (dst, v) in sum.iter_mut().zip(x) {
            *dst += v;
        }
    }
    for v in &mut sum {
        *v /= 3.0;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_ntc_pack_preserves_bit_patterns() {
        for (batch, channels, length) in [(1, 3, 19), (2, 7, 31), (3, 16, 17), (1, 32, 12)] {
            let g = Geometry {
                batch,
                channels,
                length,
                kernel: 11,
                dilation: 5,
            };
            let original = (0..batch * channels * length)
                .map(|i| f32::from_bits(0x3f000000_u32 + (i % 4096) as u32))
                .collect::<Vec<_>>();
            let returned = ntc_to_nct(&nct_to_ntc(&original, g), g);
            assert_eq!(original, returned);
        }
    }

    #[test]
    fn full_three_branch_nonlinear_graph_matches() {
        for (batch, channels, length) in [(1, 3, 19), (2, 7, 31), (1, 16, 22)] {
            let g = Geometry {
                batch,
                channels,
                length,
                kernel: 11,
                dilation: 5,
            };
            let input = (0..batch * channels * length)
                .map(|i| (((i * 41 + 7) % 233) as f32 - 116.0) * 0.004)
                .collect::<Vec<_>>();
            let reference = full_resblocks(&input, g, false);
            let candidate = full_resblocks(&input, g, true);
            let max_abs = reference
                .iter()
                .zip(&candidate)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 1e-5, "{batch}x{channels}x{length}: {max_abs}");
        }
    }
}
