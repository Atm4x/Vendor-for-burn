#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub start: i128,
    pub end: i128,
}

impl Interval {
    fn new(start: i128, end: i128) -> Result<Self, String> {
        if start >= end {
            return Err("NSF tiling interval must be nonempty".into());
        }
        Ok(Self { start, end })
    }

    fn extend(&self, radius: i128) -> Self {
        Self {
            start: self.start - radius,
            end: self.end + radius,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConvGeometry {
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
}

impl ConvGeometry {
    fn check(&self) -> Result<(), String> {
        if self.kernel == 0 || self.stride == 0 || self.dilation == 0 {
            return Err("NSF tiling encountered a zero convolution dimension".into());
        }
        Ok(())
    }

    fn required_input(&self, outputs: Interval) -> Result<Interval, String> {
        self.check()?;
        let s = self.stride as i128;
        let p = self.padding as i128;
        let reach = (self.kernel as i128 - 1) * self.dilation as i128;
        Interval::new(outputs.start * s - p, (outputs.end - 1) * s - p + reach + 1)
    }

    fn required_transpose_input(&self, outputs: Interval) -> Result<Interval, String> {
        self.check()?;
        if self.dilation != 1 {
            return Err("NSF tiled transposed convolution requires dilation=1".into());
        }
        let s = self.stride as i128;
        let p = self.padding as i128;
        let kernel = self.kernel as i128;
        let first = (outputs.start + p - (kernel - 1)).div_euclid(s);
        let first = if (outputs.start + p - (kernel - 1)).rem_euclid(s) == 0 {
            first
        } else {
            first + 1
        };
        let last_exclusive = (outputs.end - 1 + p).div_euclid(s) + 1;
        Interval::new(first, last_exclusive)
    }
}

#[derive(Clone, Debug)]
pub struct StageGeometry {
    pub upsample: ConvGeometry,
    pub noise: ConvGeometry,
    pub max_residual_radius: usize,
}

#[derive(Clone, Debug)]
pub struct NsfTileGeometry {
    pub pre: ConvGeometry,
    pub post: ConvGeometry,
    pub stages: Vec<StageGeometry>,
    pub total_upsample: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePlan {
    pub core_start: usize,
    pub core_end: usize,
    pub input_start: usize,
    pub input_end: usize,
    pub output_start: usize,
    pub output_end: usize,
}

impl NsfTileGeometry {
    pub fn plan(&self, total_frames: usize, core_frames: usize) -> Result<Vec<TilePlan>, String> {
        if total_frames == 0
            || core_frames == 0
            || self.stages.is_empty()
            || self.total_upsample == 0
        {
            return Err("NSF tiled decoder requires positive input, tile size and upsample".into());
        }
        self.pre.check()?;
        self.post.check()?;
        for stage in &self.stages {
            stage.upsample.check()?;
            stage.noise.check()?;
        }
        if self.pre.stride != 1 || self.post.stride != 1 {
            return Err("NSF tile geometry only supports stride-one pre/post convolutions".into());
        }
        let calculated_rate = self
            .stages
            .iter()
            .try_fold(1usize, |value, stage| {
                value.checked_mul(stage.upsample.stride)
            })
            .ok_or("NSF tile upsample product overflow")?;
        if calculated_rate != self.total_upsample {
            return Err("NSF tile upsample rates do not match decoder".into());
        }
        let count = total_frames as i128;
        let rate = self.total_upsample as i128;
        let mut result = Vec::with_capacity(total_frames.div_ceil(core_frames));
        for core_start in (0..total_frames).step_by(core_frames) {
            let core_end = core_start.saturating_add(core_frames).min(total_frames);
            let mut needed = Interval::new(core_start as i128 * rate, core_end as i128 * rate)?;
            needed = self.post.required_input(needed)?;
            let mut harmonic: Option<Interval> = None;
            for stage in self.stages.iter().rev() {
                needed = needed.extend(stage.max_residual_radius as i128);
                let noise_range = stage.noise.required_input(needed)?;
                harmonic = Some(match harmonic {
                    Some(previous) => Interval::new(
                        previous.start.min(noise_range.start),
                        previous.end.max(noise_range.end),
                    )?,
                    None => noise_range,
                });
                needed = stage.upsample.required_transpose_input(needed)?;
            }
            needed = self.pre.required_input(needed)?;
            if let Some(harmonic) = harmonic {
                let first_harmonic_frame = harmonic.start.div_euclid(rate);
                let last_harmonic_frame = (harmonic.end - 1).div_euclid(rate) + 1;
                needed = Interval::new(
                    needed.start.min(first_harmonic_frame),
                    needed.end.max(last_harmonic_frame),
                )?;
            }
            let input_start = needed.start.max(0) as usize;
            let input_end = needed.end.min(count) as usize;
            if input_start > core_start || input_end < core_end {
                return Err("NSF tile planner excluded requested core".into());
            }
            result.push(TilePlan {
                core_start,
                core_end,
                input_start,
                input_end,
                output_start: (core_start - input_start) * self.total_upsample,
                output_end: (core_end - input_start) * self.total_upsample,
            });
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> NsfTileGeometry {
        NsfTileGeometry {
            pre: ConvGeometry {
                kernel: 7,
                stride: 1,
                dilation: 1,
                padding: 3,
            },
            post: ConvGeometry {
                kernel: 7,
                stride: 1,
                dilation: 1,
                padding: 3,
            },
            stages: [2, 2]
                .into_iter()
                .enumerate()
                .map(|(i, stride)| StageGeometry {
                    upsample: ConvGeometry {
                        kernel: 2 * stride,
                        stride,
                        dilation: 1,
                        padding: stride / 2,
                    },
                    noise: ConvGeometry {
                        kernel: if i == 0 { 3 } else { 1 },
                        stride: if i == 0 { 2 } else { 1 },
                        dilation: 1,
                        padding: if i == 0 { 1 } else { 0 },
                    },
                    max_residual_radius: 60,
                })
                .collect(),
            total_upsample: 4,
        }
    }

    #[test]
    fn tiled_nsfs_cover_all_output_exactly_once() {
        let g = geometry();
        for length in [1, 2, 6, 33, 96, 301] {
            for tile in [1, 4, 8, 16, 32, 100] {
                let parts = g.plan(length, tile).unwrap();
                assert_eq!(parts.first().unwrap().core_start, 0);
                assert_eq!(parts.last().unwrap().core_end, length);
                for (index, part) in parts.iter().enumerate() {
                    if index > 0 {
                        assert_eq!(parts[index - 1].core_end, part.core_start);
                    }
                    assert!(part.input_start <= part.core_start);
                    assert!(part.input_end >= part.core_end);
                    assert_eq!(
                        part.output_end - part.output_start,
                        (part.core_end - part.core_start) * 4
                    );
                }
            }
        }
    }

    #[test]
    fn transpose_reverse_interval_matches_brute_force() {
        for stride in [1, 2, 3, 5] {
            let spec = ConvGeometry {
                kernel: 2 * stride,
                stride,
                dilation: 1,
                padding: stride / 2,
            };
            for first in 0..35 {
                for last in first + 1..40 {
                    let needed = spec
                        .required_transpose_input(Interval::new(first, last).unwrap())
                        .unwrap();
                    for input in -8_i128..35 {
                        let active = (0..spec.kernel).any(|k| {
                            let output = input * stride as i128 - spec.padding as i128 + k as i128;
                            output >= first && output < last
                        });
                        assert_eq!(active, input >= needed.start && input < needed.end,
                            "stride={stride}, interval={first}..{last}, input={input}, got={needed:?}");
                    }
                }
            }
        }
    }

    fn toy_conv(
        x: &[f32],
        stride: usize,
        dilation: usize,
        padding: usize,
        kernel: &[f32],
    ) -> Vec<f32> {
        let len = (x.len() + 2 * padding - (kernel.len() - 1) * dilation - 1) / stride + 1;
        (0..len)
            .map(|index| {
                kernel
                    .iter()
                    .enumerate()
                    .map(|(j, &weight)| {
                        let source = index as isize * stride as isize
                            + j as isize * dilation as isize
                            - padding as isize;
                        if source < 0 {
                            0.0
                        } else {
                            x.get(source as usize).copied().unwrap_or(0.0) * weight
                        }
                    })
                    .sum()
            })
            .collect()
    }

    fn toy_transpose(x: &[f32], stride: usize, padding: usize, kernel: &[f32]) -> Vec<f32> {
        let length = (x.len() - 1) * stride + kernel.len() - 2 * padding;
        let mut out = vec![0.0; length];
        for (i, &value) in x.iter().enumerate() {
            for (k, &weight) in kernel.iter().enumerate() {
                let dest = i as isize * stride as isize + k as isize - padding as isize;
                if dest >= 0 && (dest as usize) < length {
                    out[dest as usize] += value * weight;
                }
            }
        }
        out
    }

    fn toy_decoder(input: &[f32], harmonic: &[f32]) -> Vec<f32> {
        let weights = [0.13, -0.27, 0.31, 0.2, -0.07, 0.09, -0.15];
        let mut x = toy_conv(input, 1, 1, 3, &weights);
        for (stride, future) in [(2, 2), (2, 1)] {
            let activated: Vec<f32> = x
                .iter()
                .map(|v| if *v < 0.0 { *v * 0.1 } else { *v })
                .collect();
            let up = toy_transpose(&activated, stride, stride / 2, &[0.1, -0.2, 0.3, 0.2]);
            let noise = if future == 2 {
                toy_conv(harmonic, 2, 1, 1, &[0.2, -0.1, 0.07])
            } else {
                toy_conv(harmonic, 1, 1, 0, &[0.19])
            };
            assert_eq!(up.len(), noise.len());
            let prep: Vec<f32> = up.iter().zip(noise.iter()).map(|(a, b)| a + b).collect();
            let mut result = vec![0.0; prep.len()];
            for kernel in [3, 5] {
                let w = &weights[..kernel];
                let mut y = prep.clone();
                for dilation in [1, 3, 5] {
                    let a: Vec<_> = y
                        .iter()
                        .map(|v| if *v < 0.0 { *v * 0.1 } else { *v })
                        .collect();
                    let b = toy_conv(&a, 1, dilation, (kernel - 1) * dilation / 2, w);
                    let b: Vec<_> = b
                        .iter()
                        .map(|v| if *v < 0.0 { *v * 0.1 } else { *v })
                        .collect();
                    let b = toy_conv(&b, 1, 1, (kernel - 1) / 2, w);
                    for (o, b) in y.iter_mut().zip(b.iter()) {
                        *o += b;
                    }
                }
                for (o, y) in result.iter_mut().zip(y.iter()) {
                    *o += *y * 0.5;
                }
            }
            x = result;
        }
        let x: Vec<_> = x
            .iter()
            .map(|v| if *v < 0.0 { *v * 0.01 } else { *v })
            .collect();
        toy_conv(&x, 1, 1, 3, &weights)
            .into_iter()
            .map(f32::tanh)
            .collect()
    }

    #[test]
    fn tiled_two_stage_nonlinear_waveform_matches_whole_sequence() {
        let geom = NsfTileGeometry {
            pre: ConvGeometry {
                kernel: 7,
                stride: 1,
                dilation: 1,
                padding: 3,
            },
            post: ConvGeometry {
                kernel: 7,
                stride: 1,
                dilation: 1,
                padding: 3,
            },
            stages: [(2, 2), (2, 1)]
                .into_iter()
                .map(|(s, f)| StageGeometry {
                    upsample: ConvGeometry {
                        kernel: 4,
                        stride: s,
                        dilation: 1,
                        padding: 1,
                    },
                    noise: ConvGeometry {
                        kernel: if f == 2 { 3 } else { 1 },
                        stride: f,
                        dilation: 1,
                        padding: if f == 2 { 1 } else { 0 },
                    },
                    max_residual_radius: ((3 - 1) / 2 + (5 - 1) / 2) * (2 + 4 + 6),
                })
                .collect(),
            total_upsample: 4,
        };
        let n = 160;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.113).sin() * 0.1).collect();
        let harmonic: Vec<f32> = (0..n * 4).map(|i| (i as f32 * 0.07).cos() * 0.05).collect();
        let baseline = toy_decoder(&input, &harmonic);
        for core in [1, 3, 7, 17, 31, 64] {
            let mut output = Vec::new();
            let plan = geom.plan(n, core).unwrap();
            for tile in plan {
                let full = toy_decoder(
                    &input[tile.input_start..tile.input_end],
                    &harmonic[tile.input_start * 4..tile.input_end * 4],
                );
                output.extend_from_slice(&full[tile.output_start..tile.output_end]);
            }
            assert_eq!(output.len(), baseline.len());
            let max_abs = output
                .iter()
                .zip(&baseline)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_abs <= 1e-7, "core={core} max_abs={max_abs}");
        }
    }
}
