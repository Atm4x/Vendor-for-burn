//! Device-executed grouped residual FSQ, independent of TensorRT and CPU FSQ execution.

use cubecl::prelude::*;
use fish_speech_core::{
    error::{FishError, Result},
    firefly_cpu::AudioTensor,
    fsq_cpu::{GroupedResidualFsqDecoder, GroupedResidualFsqEncoder},
};

const MAX_TENSOR_BYTES: usize = 512 * 1024 * 1024;

fn checked_len(factors: &[usize]) -> Result<usize> {
    let len = factors
        .iter()
        .try_fold(1usize, |acc, &next| acc.checked_mul(next))
        .ok_or(FishError::InvalidShape("CubeCL FSQ tensor extent overflow"))?;
    if len == 0 || len > MAX_TENSOR_BYTES / size_of::<f32>() {
        return Err(FishError::InvalidShape(
            "CubeCL FSQ tensor extent outside budget",
        ));
    }
    Ok(len)
}

fn launch_geometry(n: usize) -> Result<CubeCount> {
    let grid = u32::try_from(n.div_ceil(128))
        .map_err(|_| FishError::InvalidShape("CubeCL FSQ launch grid overflow"))?;
    Ok(CubeCount::Static(grid, 1, 1))
}

fn read_f32(client: &Client, handle: cubecl::server::Handle, n: usize) -> Result<Vec<f32>> {
    let bytes = client
        .read_one(handle)
        .map_err(|_| FishError::InvalidTransition("CubeCL FSQ output readback failed"))?;
    if bytes.len() != n * 4 {
        return Err(FishError::InvalidShape("CubeCL FSQ output size mismatch"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn read_i32(client: &Client, handle: cubecl::server::Handle, n: usize) -> Result<Vec<i32>> {
    let bytes = client
        .read_one(handle)
        .map_err(|_| FishError::InvalidTransition("CubeCL FSQ output readback failed"))?;
    if bytes.len() != n * 4 {
        return Err(FishError::InvalidShape("CubeCL FSQ output size mismatch"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

#[cube(launch)]
fn fsq_decode_kernel(
    codes: &[i32],
    weights: &[f32],
    bias: &[f32],
    levels: &[i32],
    bases: &[i32],
    scales: &[f32],
    out: &mut [f32],
    #[comptime] frames: usize,
    #[comptime] groups: usize,
    #[comptime] quantizers: usize,
    #[comptime] width: usize,
    #[comptime] dims: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let frame = ABSOLUTE_POS % frames;
        let channel = ABSOLUTE_POS / frames;
        let batch = channel / (groups * width);
        let group = (channel / width) % groups;
        let feature = channel % width;
        let mut value = bias[group * width + feature];
        for dim in 0..dims {
            let mut residual = 0.0f32;
            for quantizer in 0..quantizers {
                let index =
                    codes[((batch * groups + group) * quantizers + quantizer) * frames + frame];
                if index >= 0 {
                    let digit = (index / bases[dim]) % levels[dim];
                    let half = f32::cast_from(levels[dim] / 2);
                    residual +=
                        (f32::cast_from(digit) - half) / half * scales[quantizer * dims + dim];
                }
            }
            value += weights[(group * width + feature) * dims + dim] * residual;
        }
        out[ABSOLUTE_POS] = value;
    }
}

#[cube(launch)]
fn fsq_project_kernel(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    half_width: &[f32],
    offset: &[f32],
    shift: &[f32],
    out: &mut [f32],
    invalid: &mut [i32],
    #[comptime] frames: usize,
    #[comptime] groups: usize,
    #[comptime] width: usize,
    #[comptime] dims: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let dim = ABSOLUTE_POS % dims;
        let frame = (ABSOLUTE_POS / dims) % frames;
        let group = (ABSOLUTE_POS / (dims * frames)) % groups;
        let batch = ABSOLUTE_POS / (dims * frames * groups);
        let mut total = bias[group * dims + dim];
        for feature in 0..width {
            total += weights[(group * dims + dim) * width + feature]
                * input[(batch * groups * width + group * width + feature) * frames + frame];
        }
        let bounded = (total + shift[dim]).tanh() * half_width[dim] - offset[dim];
        out[ABSOLUTE_POS] = bounded;
        invalid[ABSOLUTE_POS] =
            if total != total || total > 3.402823e38f32 || total < -3.402823e38f32 {
                1
            } else {
                0
            };
    }
}

#[cube(launch)]
fn fsq_quantize_kernel(
    projected: &[f32],
    levels: &[i32],
    half_width: &[f32],
    offset: &[f32],
    shift: &[f32],
    scales: &[f32],
    digits: &mut [i32],
    invalid: &mut [i32],
    #[comptime] quantizers: usize,
    #[comptime] dims: usize,
) {
    if ABSOLUTE_POS < projected.len() {
        let dim = ABSOLUTE_POS % dims;
        let slot = ABSOLUTE_POS / dims;
        let mut residual = projected[ABSOLUTE_POS];
        for quantizer in 0..quantizers {
            let scale = scales[quantizer * dims + dim];
            let bounded = (residual / scale + shift[dim]).tanh() * half_width[dim] - offset[dim];
            let rounded = bounded.round();
            let digit = i32::cast_from(rounded + f32::cast_from(levels[dim] / 2));
            digits[(slot * quantizers + quantizer) * dims + dim] = digit;
            if digit < 0 || digit >= levels[dim] || rounded != rounded {
                invalid[ABSOLUTE_POS] = 1;
            }
            residual -= (rounded / f32::cast_from(levels[dim] / 2)) * scale;
        }
    }
}

#[cube(launch)]
fn fsq_pack_kernel(
    digits: &[i32],
    bases: &[i32],
    packed: &mut [i32],
    #[comptime] frames: usize,
    #[comptime] groups: usize,
    #[comptime] quantizers: usize,
    #[comptime] dims: usize,
) {
    if ABSOLUTE_POS < packed.len() {
        let frame = ABSOLUTE_POS % frames;
        let quantizer = (ABSOLUTE_POS / frames) % quantizers;
        let group = (ABSOLUTE_POS / (frames * quantizers)) % groups;
        let batch = ABSOLUTE_POS / (frames * quantizers * groups);
        let slot = (batch * groups + group) * frames + frame;
        let mut index = 0i32;
        for dim in 0..dims {
            index += digits[(slot * quantizers + quantizer) * dims + dim] * bases[dim];
        }
        packed[ABSOLUTE_POS] = index;
    }
}

/// Stateless FSQ backend. Data stays on the device across projection, quantization and packing;
/// only the public result and error status are read to the host.
pub struct CubeClFsqExecutor {
    client: Client,
}

impl CubeClFsqExecutor {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    pub fn decode(
        &self,
        decoder: &GroupedResidualFsqDecoder,
        indices: &[i32],
        batch: usize,
        frames: usize,
    ) -> Result<AudioTensor> {
        decoder.validate()?;
        let groups = decoder.projections.len();
        let dims = decoder.codebook.levels.len();
        let quantizers = decoder.num_quantizers;
        let width = decoder.projections[0].bias.len();
        let input_len = checked_len(&[batch, groups, quantizers, frames])?;
        let output_len = checked_len(&[batch, groups, width, frames])?;
        if indices.len() != input_len
            || indices
                .iter()
                .any(|&c| c < -1 || c >= decoder.codebook.codebook_size as i32)
        {
            return Err(FishError::InvalidShape(
                "CubeCL FSQ index outside codebook or wrong extent",
            ));
        }
        let weights = decoder
            .projections
            .iter()
            .flat_map(|p| p.weights.iter().copied())
            .collect::<Vec<_>>();
        let bias = decoder
            .projections
            .iter()
            .flat_map(|p| p.bias.iter().copied())
            .collect::<Vec<_>>();
        let levels = decoder
            .codebook
            .levels
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let bases = decoder
            .codebook
            .bases
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let scales = (0..quantizers)
            .flat_map(|q| {
                decoder
                    .codebook
                    .levels
                    .iter()
                    .map(move |&l| ((l - 1) as f32).powi(-(q as i32)))
            })
            .collect::<Vec<_>>();
        let client = &self.client;
        let codes = client.create_from_slice(bytemuck::cast_slice(indices));
        let w = client.create_from_slice(bytemuck::cast_slice(&weights));
        let b = client.create_from_slice(bytemuck::cast_slice(&bias));
        let l = client.create_from_slice(bytemuck::cast_slice(&levels));
        let radix = client.create_from_slice(bytemuck::cast_slice(&bases));
        let scale = client.create_from_slice(bytemuck::cast_slice(&scales));
        let output = client.empty(output_len * 4);
        fsq_decode_kernel::launch(
            client,
            launch_geometry(output_len)?,
            CubeDim::new_1d(128),
            unsafe { BufferArg::from_raw_parts(codes, input_len) },
            unsafe { BufferArg::from_raw_parts(w, weights.len()) },
            unsafe { BufferArg::from_raw_parts(b, bias.len()) },
            unsafe { BufferArg::from_raw_parts(l, dims) },
            unsafe { BufferArg::from_raw_parts(radix, dims) },
            unsafe { BufferArg::from_raw_parts(scale, scales.len()) },
            unsafe { BufferArg::from_raw_parts(output.clone(), output_len) },
            frames,
            groups,
            quantizers,
            width,
            dims,
        );
        AudioTensor::new(
            batch,
            groups * width,
            frames,
            read_f32(client, output, output_len)?,
        )
    }

    pub fn encode(
        &self,
        encoder: &GroupedResidualFsqEncoder,
        features: &AudioTensor,
    ) -> Result<Vec<i32>> {
        let width = encoder.validate()?;
        let groups = encoder.projections.len();
        let dims = encoder.codebook.levels.len();
        let quantizers = encoder.num_quantizers;
        if features.channels != groups * width || features.data.iter().any(|v| !v.is_finite()) {
            return Err(FishError::InvalidShape(
                "CubeCL FSQ feature extent or values invalid",
            ));
        }
        let projected_len = checked_len(&[features.batch, groups, features.frames, dims])?;
        let packed_len = checked_len(&[features.batch, groups, quantizers, features.frames])?;
        let digits_len = checked_len(&[projected_len, quantizers])?;
        let weights = encoder
            .projections
            .iter()
            .flat_map(|p| p.weights.iter().copied())
            .collect::<Vec<_>>();
        let bias = encoder
            .projections
            .iter()
            .flat_map(|p| p.bias.iter().copied())
            .collect::<Vec<_>>();
        let levels = encoder
            .codebook
            .levels
            .iter()
            .map(|&l| l as i32)
            .collect::<Vec<_>>();
        let bases = encoder
            .codebook
            .bases
            .iter()
            .map(|&l| l as i32)
            .collect::<Vec<_>>();
        let half_width = encoder
            .codebook
            .levels
            .iter()
            .map(|&l| (l - 1) as f32 * 1.001 / 2.0)
            .collect::<Vec<_>>();
        let offset = encoder
            .codebook
            .levels
            .iter()
            .map(|&l| if l % 2 == 0 { 0.5f32 } else { 0.0f32 })
            .collect::<Vec<_>>();
        let shift = offset
            .iter()
            .zip(&half_width)
            .map(|(&a, &b)| (a / b).atanh())
            .collect::<Vec<_>>();
        let scales = (0..quantizers)
            .flat_map(|q| {
                encoder
                    .codebook
                    .levels
                    .iter()
                    .map(move |&l| ((l - 1) as f32).powi(-(q as i32)))
            })
            .collect::<Vec<_>>();
        let client = &self.client;
        let input = client.create_from_slice(bytemuck::cast_slice(&features.data));
        let w = client.create_from_slice(bytemuck::cast_slice(&weights));
        let b = client.create_from_slice(bytemuck::cast_slice(&bias));
        let l = client.create_from_slice(bytemuck::cast_slice(&levels));
        let radix = client.create_from_slice(bytemuck::cast_slice(&bases));
        let half = client.create_from_slice(bytemuck::cast_slice(&half_width));
        let off = client.create_from_slice(bytemuck::cast_slice(&offset));
        let sh = client.create_from_slice(bytemuck::cast_slice(&shift));
        let sc = client.create_from_slice(bytemuck::cast_slice(&scales));
        let projected = client.empty(projected_len * 4);
        let invalid = client.create_from_slice(&vec![0u8; projected_len * 4]);
        fsq_project_kernel::launch(
            client,
            launch_geometry(projected_len)?,
            CubeDim::new_1d(128),
            unsafe { BufferArg::from_raw_parts(input, features.data.len()) },
            unsafe { BufferArg::from_raw_parts(w, weights.len()) },
            unsafe { BufferArg::from_raw_parts(b, bias.len()) },
            unsafe { BufferArg::from_raw_parts(half.clone(), dims) },
            unsafe { BufferArg::from_raw_parts(off.clone(), dims) },
            unsafe { BufferArg::from_raw_parts(sh.clone(), dims) },
            unsafe { BufferArg::from_raw_parts(projected.clone(), projected_len) },
            unsafe { BufferArg::from_raw_parts(invalid.clone(), projected_len) },
            features.frames,
            groups,
            width,
            dims,
        );
        let digits = client.empty(digits_len * 4);
        fsq_quantize_kernel::launch(
            client,
            launch_geometry(projected_len)?,
            CubeDim::new_1d(128),
            unsafe { BufferArg::from_raw_parts(projected, projected_len) },
            unsafe { BufferArg::from_raw_parts(l, dims) },
            unsafe { BufferArg::from_raw_parts(half, dims) },
            unsafe { BufferArg::from_raw_parts(off, dims) },
            unsafe { BufferArg::from_raw_parts(sh, dims) },
            unsafe { BufferArg::from_raw_parts(sc, scales.len()) },
            unsafe { BufferArg::from_raw_parts(digits.clone(), digits_len) },
            unsafe { BufferArg::from_raw_parts(invalid.clone(), projected_len) },
            quantizers,
            dims,
        );
        let packed = client.empty(packed_len * 4);
        fsq_pack_kernel::launch(
            client,
            launch_geometry(packed_len)?,
            CubeDim::new_1d(128),
            unsafe { BufferArg::from_raw_parts(digits, digits_len) },
            unsafe { BufferArg::from_raw_parts(radix, dims) },
            unsafe { BufferArg::from_raw_parts(packed.clone(), packed_len) },
            features.frames,
            groups,
            quantizers,
            dims,
        );
        if read_i32(client, invalid, projected_len)?
            .iter()
            .any(|&v| v != 0)
        {
            return Err(FishError::InvalidShape(
                "CubeCL FSQ projected value or digit invalid",
            ));
        }
        read_i32(client, packed, packed_len)
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use cubecl::Device;
    use fish_speech_core::fsq_cpu::{FsqCodebook, FsqGroupProjection};

    fn weights(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 13 + seed) % 37) as f32 / 40.0 - 0.45)
            .collect()
    }
    fn bias(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7 + seed) % 13) as f32 / 30.0 - 0.2)
            .collect()
    }
    fn gpu() -> CubeClFsqExecutor {
        CubeClFsqExecutor::new(Device::Cpu(Default::default()).client())
    }

    #[test]
    fn decode_matches_original_python_golden_and_rust_reference() {
        let fixtures = include_str!("../../core/fixtures/generate_fsq_golden.txt");
        for line in fixtures
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        {
            let mut words = line.split_whitespace();
            let name = words.next().unwrap();
            let dims = words
                .next()
                .unwrap()
                .split(',')
                .map(|x| x.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            let (batch, groups, quantizers, frames, width, seed) =
                (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
            let codes = words
                .next()
                .unwrap()
                .split(',')
                .map(|x| x.parse::<i32>().unwrap())
                .collect::<Vec<_>>();
            let expected = words.map(|x| x.parse::<f32>().unwrap()).collect::<Vec<_>>();
            let model = GroupedResidualFsqDecoder {
                codebook: FsqCodebook::new(&[8, 5, 5, 5]).unwrap(),
                num_quantizers: quantizers,
                projections: (0..groups)
                    .map(|g| FsqGroupProjection {
                        weights: weights(width * 4, seed + g),
                        bias: bias(width, seed + g + 1),
                    })
                    .collect(),
            };
            let actual = gpu().decode(&model, &codes, batch, frames).unwrap();
            assert_eq!(
                (actual.batch, actual.channels, actual.frames),
                (batch, groups * width, frames)
            );
            let rust = model.decode(&codes, batch, frames).unwrap();
            for (i, (&a, (&b, &c))) in actual
                .data
                .iter()
                .zip(expected.iter().zip(&rust.data))
                .enumerate()
            {
                assert!(
                    (a - b).abs() <= 3e-5 + b.abs() * 1e-5,
                    "{name} python mismatch at {i}: {a} vs {b}"
                );
                assert!(
                    (a - c).abs() <= 3e-5 + c.abs() * 1e-5,
                    "{name} cpu mismatch at {i}: {a} vs {c}"
                );
            }
        }
    }

    #[test]
    fn encode_matches_original_python_golden_and_rust_reference() {
        let fixtures = include_str!("../../core/fixtures/generate_fsq_encode_golden.txt");
        for line in fixtures
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        {
            let mut words = line.split_whitespace();
            let name = words.next().unwrap();
            let dims = words
                .next()
                .unwrap()
                .split(',')
                .map(|x| x.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            let (batch, groups, quantizers, frames, width, seed) =
                (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
            let expected = words
                .next()
                .unwrap()
                .split(',')
                .map(|x| x.parse::<i32>().unwrap())
                .collect::<Vec<_>>();
            let features = AudioTensor::new(
                batch,
                groups * width,
                frames,
                (0..batch * groups * width * frames)
                    .map(|i| ((i * 11 + seed) % 103) as f32 / 22.0 - 2.2)
                    .collect(),
            )
            .unwrap();
            let model = GroupedResidualFsqEncoder {
                codebook: FsqCodebook::new(&[8, 5, 5, 5]).unwrap(),
                num_quantizers: quantizers,
                projections: (0..groups)
                    .map(|g| FsqGroupProjection {
                        weights: weights(width * 4, seed + g),
                        bias: bias(4, seed + g + 1),
                    })
                    .collect(),
            };
            let actual = gpu().encode(&model, &features).unwrap();
            assert_eq!(actual, expected, "{name}: CubeCL disagrees with Python");
            assert_eq!(
                actual,
                model.encode(&features).unwrap(),
                "{name}: CubeCL disagrees with Rust CPU"
            );
        }
    }

    #[test]
    fn multi_batch_residual_fsq_stress_matches_cpu_on_device() {
        let backend = gpu();
        for &groups in &[1, 2, 8] {
            for &quantizers in &[1, 2, 4] {
                for &frames in &[1, 3, 7] {
                    let batch = 2;
                    let width = 5;
                    let codebook = FsqCodebook::new(&[8, 5, 5, 5]).unwrap();
                    let encoder = GroupedResidualFsqEncoder {
                        codebook: codebook.clone(),
                        num_quantizers: quantizers,
                        projections: (0..groups)
                            .map(|g| FsqGroupProjection {
                                weights: weights(width * 4, g + frames * 3 + quantizers),
                                bias: bias(4, g * 5 + frames),
                            })
                            .collect(),
                    };
                    let features = AudioTensor::new(
                        batch,
                        groups * width,
                        frames,
                        (0..batch * groups * width * frames)
                            .map(|i| {
                                ((i * 43 + frames * 17 + quantizers) % 131) as f32 / 36.0 - 1.65
                            })
                            .collect(),
                    )
                    .unwrap();
                    let expected_codes = encoder.encode(&features).unwrap();
                    let codes = backend.encode(&encoder, &features).unwrap();
                    assert_eq!(
                        codes, expected_codes,
                        "encode groups={groups} q={quantizers} frames={frames}"
                    );
                    let decoder = GroupedResidualFsqDecoder {
                        codebook,
                        num_quantizers: quantizers,
                        projections: (0..groups)
                            .map(|g| FsqGroupProjection {
                                weights: weights(width * 4, g * 11 + quantizers),
                                bias: bias(width, g * 7 + frames),
                            })
                            .collect(),
                    };
                    let mut indices = codes;
                    for (i, code) in indices.iter_mut().enumerate() {
                        if i % 7 == 0 {
                            *code = -1;
                        }
                        if i % 11 == 0 {
                            *code = 999;
                        }
                        if i % 13 == 0 {
                            *code = 0;
                        }
                    }
                    let actual = backend.decode(&decoder, &indices, batch, frames).unwrap();
                    let expected = decoder.decode(&indices, batch, frames).unwrap();
                    for (i, (a, b)) in actual.data.iter().zip(&expected.data).enumerate() {
                        assert!(
                            (a - b).abs() < 3e-5 + b.abs() * 1e-5,
                            "decode groups={groups} q={quantizers} frames={frames}, index={i}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_indices_and_nonfinite_features_rejected() {
        let decoder = GroupedResidualFsqDecoder {
            codebook: FsqCodebook::new(&[8, 5, 5, 5]).unwrap(),
            num_quantizers: 1,
            projections: vec![FsqGroupProjection {
                weights: vec![0.5; 8],
                bias: vec![0.0; 2],
            }],
        };
        for bad in [-2, 1000, i32::MAX] {
            assert!(gpu().decode(&decoder, &[bad], 1, 1).is_err());
        }
        assert!(gpu().decode(&decoder, &[1, 2], 1, 1).is_err());
        assert!(gpu().decode(&decoder, &[-1], 1, 1).is_ok());
        let encoder = GroupedResidualFsqEncoder {
            codebook: decoder.codebook.clone(),
            num_quantizers: 1,
            projections: vec![FsqGroupProjection {
                weights: vec![0.25; 8],
                bias: vec![0.0; 4],
            }],
        };
        let bad = AudioTensor::new(1, 2, 1, vec![f32::NAN, 0.0]).unwrap();
        assert!(gpu().encode(&encoder, &bad).is_err());
        let bad = AudioTensor::new(1, 1, 1, vec![0.0]).unwrap();
        assert!(gpu().encode(&encoder, &bad).is_err());
    }
}
