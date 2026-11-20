#[derive(Clone, Copy)]
struct Shape {
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_length: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    groups: usize,
}

impl Shape {
    fn out_length(self) -> usize {
        let effective_kernel = self.dilation * (self.kernel - 1) + 1;
        (self.input_length + self.padding * 2)
            .checked_sub(effective_kernel)
            .map_or(0, |n| n / self.stride + 1)
    }
    fn input_group(self) -> usize {
        self.in_channels / self.groups
    }
    fn output_group(self) -> usize {
        self.out_channels / self.groups
    }
}

fn conv_reference(s: Shape, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    let length = s.out_length();
    let mut out = vec![0.; s.batch * length * s.out_channels];
    for b in 0..s.batch {
        for t in 0..length {
            for oc in 0..s.out_channels {
                let group = oc / s.output_group();
                let mut acc = bias[oc];
                for k in 0..s.kernel {
                    let pos = (t * s.stride + k * s.dilation) as isize - s.padding as isize;
                    if pos >= 0 && (pos as usize) < s.input_length {
                        for ci in 0..s.input_group() {
                            let xv = x[(b * s.input_length + pos as usize) * s.in_channels
                                + group * s.input_group()
                                + ci];
                            let wv = w[(oc * s.kernel + k) * s.input_group() + ci];
                            acc += xv * wv;
                        }
                    }
                }
                out[(b * length + t) * s.out_channels + oc] = acc;
            }
        }
    }
    out
}

fn conv_temporal_pair(s: Shape, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    let length = s.out_length();
    let mut out = vec![0.; s.batch * length * s.out_channels];
    for b in 0..s.batch {
        for p in 0..length.div_ceil(2) {
            let t0 = 2 * p;
            let t1 = t0 + 1;
            for oc in 0..s.out_channels {
                let group = oc / s.output_group();
                let mut acc0 = bias[oc];
                let mut acc1 = bias[oc];
                for k in 0..s.kernel {
                    let ix0 = (t0 * s.stride + k * s.dilation) as isize - s.padding as isize;
                    let ix1 = (t1 * s.stride + k * s.dilation) as isize - s.padding as isize;
                    let valid0 = ix0 >= 0 && (ix0 as usize) < s.input_length;
                    let valid1 = t1 < length && ix1 >= 0 && (ix1 as usize) < s.input_length;
                    if valid0 || valid1 {
                        for ci in 0..s.input_group() {
                            let wv = w[(oc * s.kernel + k) * s.input_group() + ci];
                            if valid0 {
                                let xv = x[(b * s.input_length + ix0 as usize) * s.in_channels
                                    + group * s.input_group()
                                    + ci];
                                acc0 += xv * wv;
                            }
                            if valid1 {
                                let xv = x[(b * s.input_length + ix1 as usize) * s.in_channels
                                    + group * s.input_group()
                                    + ci];
                                acc1 += xv * wv;
                            }
                        }
                    }
                }
                out[(b * length + t0) * s.out_channels + oc] = acc0;
                if t1 < length {
                    out[(b * length + t1) * s.out_channels + oc] = acc1;
                }
            }
        }
    }
    out
}

fn deterministic_data(n: usize, salt: u64) -> Vec<f32> {
    let mut state = salt;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 10007) as f32 - 5003.0) / 4096.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_convolution_matches_reference_across_geometries() {
        let mut tried = 0;
        for batch in [1, 2] {
            for groups in [1, 2, 3] {
                for channels_per_group in [1, 2, 5] {
                    for outputs_per_group in [1, 2, 4] {
                        for input_length in [1, 2, 7, 19] {
                            for kernel in [1, 3, 7] {
                                for (stride, dilation, padding) in
                                    [(1, 1, 0), (1, 1, 3), (2, 3, 2), (3, 2, 7)]
                                {
                                    let s = Shape {
                                        batch,
                                        groups,
                                        in_channels: groups * channels_per_group,
                                        out_channels: groups * outputs_per_group,
                                        input_length,
                                        kernel,
                                        stride,
                                        dilation,
                                        padding,
                                    };
                                    let x = deterministic_data(
                                        batch * s.in_channels * input_length,
                                        17713 + tried,
                                    );
                                    let w = deterministic_data(
                                        s.out_channels * kernel * channels_per_group,
                                        611 + tried,
                                    );
                                    let bias = deterministic_data(s.out_channels, 221 + tried);
                                    let r = conv_reference(s, &x, &w, &bias);
                                    let p = conv_temporal_pair(s, &x, &w, &bias);
                                    assert_eq!(r.len(), p.len());
                                    for (index, (a, b)) in r.iter().zip(&p).enumerate() {
                                        assert_eq!(
                                            a.to_bits(),
                                            b.to_bits(),
                                            "case {tried}, output {index}"
                                        );
                                    }
                                    tried += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(tried, 2592);
    }

    #[test]
    fn temporal_pair_weight_loads_are_halved_for_even_lengths() {
        for length in 1usize..32 {
            let old = length * 128 * 11 * 64;
            let pair = length.div_ceil(2) * 128 * 11 * 64;
            assert!(pair <= old);
            assert_eq!(pair, old / 2 + (length % 2) * (128 * 11 * 64 / 2));
        }
    }
}
