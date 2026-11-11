use burn::cubecl;
use burn::cubecl::prelude::*;
use burn::tensor::{DType, Tensor};

#[cfg(feature = "cuda")]
type CubeBackend = burn::backend::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "cpu"))]
type CubeBackend = burn::backend::Cpu;
#[cfg(all(not(feature = "cuda"), not(feature = "cpu"), feature = "wgpu"))]
type CubeBackend = burn::backend::Wgpu;

const BINS: usize = 360;
const WIDTH: usize = 10;

#[cube(launch_unchecked)]
fn fused_salience_kernel(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    frames: usize,
    stride_batch: usize,
    stride_frame: usize,
    stride_bin: usize,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let batch = row / frames;
        let frame = row % frames;
        let start = batch * stride_batch + frame * stride_frame;
        let mut peak = input[start];
        let mut center = 0usize;
        let mut finite = true;
        let mut bin = 0usize;
        while bin < 360 {
            let value = input[start + bin * stride_bin];
            if !(value == value && value <= f32::MAX && value >= -f32::MAX) {
                finite = false;
            }
            if value > peak {
                peak = value;
                center = bin;
            }
            bin += 1;
        }
        let out = row * 10;
        if finite {
            output[out] = center as f32;
            let mut offset = 0usize;
            while offset < 9 {
                let index = center as i32 + offset as i32 - 4;
                let clamped = if index < 0 {
                    0usize
                } else if index >= 360 {
                    359usize
                } else {
                    index as usize
                };
                output[out + 1 + offset] = input[start + clamped * stride_bin];
                offset += 1;
            }
        } else {
            output[out] = -1.0f32;
            let mut offset = 1usize;
            while offset < 10 {
                output[out + offset] = 0.0f32;
                offset += 1;
            }
        }
    }
}

/// One logical plane per salience frame. Lane-reduced max and index preserve
/// the first-maximum tie rule, including negative-only finite inputs.
#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn plane_salience_kernel(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    frames: usize,
    stride_batch: usize,
    stride_frame: usize,
    stride_bin: usize,
) {
    let row = CUBE_POS_X as usize;
    if row < rows && PLANE_POS == 0 {
        let lane = UNIT_POS_PLANE as usize;
        let plane_width = PLANE_DIM as usize;
        let batch = row / frames;
        let frame = row % frames;
        let start = batch * stride_batch + frame * stride_frame;
        let mut peak = input[start + lane * stride_bin];
        let mut center = lane as u32;
        let mut invalid = !(peak == peak && peak <= f32::MAX && peak >= -f32::MAX);
        let mut bin = lane + plane_width;
        while bin < 360 {
            let value = input[start + bin * stride_bin];
            let finite = value == value && value <= f32::MAX && value >= -f32::MAX;
            if !finite {
                invalid = true;
            } else if value > peak {
                peak = value;
                center = bin as u32;
            }
            bin += plane_width;
        }
        let has_nonfinite = plane_any(invalid);
        let global_peak = plane_max(peak);
        let candidate = if peak == global_peak {
            center
        } else {
            u32::MAX
        };
        let first_peak = plane_min(candidate);
        let out = row * 10;
        if lane == 0 {
            output[out] = if has_nonfinite {
                -1.0f32
            } else {
                first_peak as f32
            };
        }
        let mut offset = lane;
        while offset < 9 {
            let index = first_peak as i32 + offset as i32 - 4;
            let clamped = if index < 0 {
                0usize
            } else if index >= 360 {
                359usize
            } else {
                index as usize
            };
            output[out + offset + 1] = if has_nonfinite {
                0.0f32
            } else {
                input[start + clamped * stride_bin]
            };
            offset += plane_width;
        }
    }
}

#[derive(Clone, Copy)]
enum SalienceDispatch {
    Sequential,
    #[cfg(feature = "cuda")]
    Plane,
}

pub(crate) fn plane_compact_salience(salience: Tensor<3>) -> Result<Vec<f32>, String> {
    #[cfg(feature = "cuda")]
    {
        compact_salience_with_dispatch(salience, SalienceDispatch::Plane)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = salience;
        Err("RMVPE plane reductions require the CUDA backend; CubeCL CPU/LLVM has no plane operations".into())
    }
}

pub(crate) fn fused_compact_salience(salience: Tensor<3>) -> Result<Vec<f32>, String> {
    compact_salience_with_dispatch(salience, SalienceDispatch::Sequential)
}

fn compact_salience_with_dispatch(
    salience: Tensor<3>,
    dispatch: SalienceDispatch,
) -> Result<Vec<f32>, String> {
    let [batch, frames, bins] = salience.dims();
    if batch == 0 || frames == 0 || bins != BINS {
        return Err("RMVPE fused salience requires nonempty [B,T,360]".into());
    }
    let cube = salience
        .try_into_primitive::<CubeBackend>()
        .map_err(|error| format!("RMVPE fused requires a direct CubeCL backend: {error:?}"))?;
    if cube.dtype != DType::F32 || cube.meta.is_tiled() {
        return Err("RMVPE fused requires a non-tiled FP32 tensor".into());
    }
    let strides = cube.meta.strides();
    let required = (batch - 1)
        .checked_mul(strides[0])
        .and_then(|x| x.checked_add((frames - 1).checked_mul(strides[1])?))
        .and_then(|x| x.checked_add((BINS - 1).checked_mul(strides[2])?))
        .and_then(|x| x.checked_add(1))
        .ok_or("RMVPE fused input stride overflow")?;
    if required
        .checked_mul(size_of::<f32>())
        .ok_or("RMVPE fused byte size overflow")?
        > cube.handle.size_in_used() as usize
    {
        return Err("RMVPE fused input allocation shorter than tensor striding".into());
    }
    let rows = batch
        .checked_mul(frames)
        .ok_or("RMVPE fused row overflow")?;
    let max_rows = match dispatch {
        SalienceDispatch::Sequential => (u32::MAX as usize).saturating_mul(64),
        #[cfg(feature = "cuda")]
        SalienceDispatch::Plane => u32::MAX as usize,
    };
    if rows > max_rows {
        return Err("RMVPE fused launch grid exceeds CubeCL limits".into());
    }
    let bytes = rows
        .checked_mul(WIDTH * size_of::<f32>())
        .ok_or("RMVPE fused output size overflow")?;
    let output = cube.client.empty(bytes);
    unsafe {
        let input_arg =
            BufferArg::from_raw_parts(cube.handle.clone(), cube.handle.size_in_used() as usize / 4);
        let output_arg = BufferArg::from_raw_parts(output.clone(), rows * WIDTH);
        match dispatch {
            SalienceDispatch::Sequential => fused_salience_kernel::launch_unchecked(
                &cube.client,
                CubeCount::Static(rows.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                input_arg,
                output_arg,
                rows,
                frames,
                strides[0],
                strides[1],
                strides[2],
            ),
            #[cfg(feature = "cuda")]
            SalienceDispatch::Plane => plane_salience_kernel::launch_unchecked(
                &cube.client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(32),
                input_arg,
                output_arg,
                rows,
                frames,
                strides[0],
                strides[1],
                strides[2],
            ),
        }
    }
    let bytes = cube
        .client
        .read_one(output)
        .map_err(|error| error.to_string())?;
    let values = bytes[..]
        .chunks_exact(4)
        .map(|chunk: &[u8]| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<f32>>();
    if values.len() != rows * WIDTH {
        return Err("RMVPE fused readback returned an invalid output length".into());
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    use super::*;
    #[cfg(feature = "cpu")]
    use crate::rmvpe::{compact_salience, compact_salience_to_f0, salience_to_f0};
    #[cfg(feature = "cpu")]
    use burn::tensor::{Device, TensorData};

    #[cfg(feature = "cpu")]
    #[test]
    fn cube_cpu_fused_matches_dense_and_burn_compact() {
        let device = Device::cpu();
        for batch in [1, 2, 4] {
            for frames in [1, 7, 33] {
                let mut values = Vec::new();
                for row in 0..batch * frames {
                    let mut bins = (0..BINS)
                        .map(|i| (((i * 37 + row * 97) % 359) as f32 + 1.0) / 360.0)
                        .collect::<Vec<_>>();
                    match row % 5 {
                        0 => bins[0] = 2.0,
                        1 => bins[359] = 2.0,
                        2 => {
                            bins[1] = 2.0;
                            bins[5] = 2.0;
                        }
                        3 => bins.fill(0.0),
                        _ => bins[178] = 2.0,
                    }
                    values.extend(bins);
                }
                let tensor = Tensor::<3>::from_data(
                    TensorData::new(values.clone(), [batch, frames, BINS]),
                    &device,
                );
                let fused = fused_compact_salience(tensor.clone()).unwrap();
                let burn = compact_salience(tensor)
                    .into_data()
                    .try_into_vec::<f32>()
                    .unwrap();
                assert_eq!(fused, burn);
                for threshold in [0.0, 0.03, 0.5, 1.0, 2.0] {
                    let reference = salience_to_f0(&values, threshold).unwrap();
                    let output = compact_salience_to_f0(&fused, threshold).unwrap();
                    assert_eq!(
                        reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        output.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cube_cpu_fused_sliced_batch_and_nonfinite() {
        let device = Device::cpu();
        let batch = 2;
        let frames = 7;
        let padded = 32;
        let mut padded_values = vec![0.0f32; batch * padded * BINS];
        let mut compact_values = Vec::new();
        for b in 0..batch {
            for t in 0..frames {
                for i in 0..BINS {
                    let value = (((i * 17 + b * 23 + t * 31) % 353) as f32) / 353.0;
                    padded_values[(b * padded + t) * BINS + i] = value;
                    compact_values.push(value);
                }
            }
        }
        let view = Tensor::<3>::from_data(
            TensorData::new(padded_values, [batch, padded, BINS]),
            &device,
        )
        .slice([0..batch, 0..frames, 0..BINS]);
        let fused = fused_compact_salience(view.clone()).unwrap();
        let dense = salience_to_f0(&compact_values, 0.03).unwrap();
        let compact = compact_salience_to_f0(&fused, 0.03).unwrap();
        assert_eq!(
            dense.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            compact.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut data = vec![0.0f32; BINS * 2];
            data[300] = 2.0;
            data[100] = bad;
            let input = Tensor::<3>::from_data(TensorData::new(data, [1, 2, BINS]), &device);
            let fused = fused_compact_salience(input).unwrap();
            assert_eq!(fused[0], -1.0);
            assert!(compact_salience_to_f0(&fused, 0.03).is_err());
        }
    }

    #[cfg(feature = "cpu")]
    fn simulated_lane_compact(values: &[f32]) -> Vec<f32> {
        assert_eq!(values.len() % BINS, 0);
        let mut output = Vec::with_capacity(values.len() / BINS * WIDTH);
        for row in values.chunks_exact(BINS) {
            let mut candidates = [(f32::NEG_INFINITY, 360usize); 32];
            let mut has_nonfinite = false;
            for (lane, candidate) in candidates.iter_mut().enumerate() {
                let mut peak = row[lane];
                let mut center = lane;
                for bin in (lane..BINS).step_by(32).skip(1) {
                    let value = row[bin];
                    if value > peak {
                        peak = value;
                        center = bin;
                    }
                }
                *candidate = (peak, center);
            }
            for value in row {
                has_nonfinite |= !value.is_finite();
            }
            if has_nonfinite {
                output.extend([-1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                continue;
            }
            let global_peak = candidates
                .iter()
                .map(|(v, _)| *v)
                .max_by(f32::total_cmp)
                .unwrap();
            let first_peak = candidates
                .iter()
                .filter(|(v, _)| *v == global_peak)
                .map(|(_, center)| *center)
                .min()
                .unwrap();
            output.push(first_peak as f32);
            for delta in -4..=4 {
                output.push(row[(first_peak as isize + delta).clamp(0, 359) as usize]);
            }
        }
        output
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn plane_reduction_math_matches_dense_and_sequential_kernel() {
        let device = Device::cpu();
        for batch in [1, 2, 4] {
            for frames in [1, 7, 33, 70] {
                let mut values = Vec::with_capacity(batch * frames * BINS);
                for row in 0..batch * frames {
                    let mut bins = (0..BINS)
                        .map(|i| (((i * 37 + row * 19) % 359) as f32 - 180.0) / 360.0)
                        .collect::<Vec<_>>();
                    match row % 6 {
                        0 => bins[0] = 2.0,
                        1 => bins[359] = 2.0,
                        2 => {
                            bins[1] = 2.0;
                            bins[5] = 2.0;
                        }
                        3 => bins.fill(0.0),
                        4 => bins.fill(-f32::MAX),
                        _ => bins[178] = 2.0,
                    }
                    values.extend(bins);
                }
                let tensor = Tensor::<3>::from_data(
                    TensorData::new(values.clone(), [batch, frames, BINS]),
                    &device,
                );
                let plane_math = simulated_lane_compact(&values);
                let sequential = fused_compact_salience(tensor).unwrap();
                assert_eq!(plane_math, sequential, "B={batch} T={frames}");
                for threshold in [0.0, 0.03, 0.5, 1.0] {
                    let dense_f0 = salience_to_f0(&values, threshold).unwrap();
                    let plane_f0 = compact_salience_to_f0(&plane_math, threshold).unwrap();
                    assert_eq!(
                        dense_f0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        plane_f0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "B={batch} T={frames} threshold={threshold}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn plane_reduction_math_nonfinite_and_unsupported_backend() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = vec![0.0f32; BINS * 2];
            values[300] = 2.0;
            values[100] = bad;
            let plane_math = simulated_lane_compact(&values);
            assert_eq!(plane_math[0], -1.0);
            assert!(compact_salience_to_f0(&plane_math, 0.03).is_err());
        }
        let device = Device::cpu();
        let input = Tensor::<3>::zeros([1, 2, BINS], &device);
        assert!(plane_compact_salience(input).is_err());
    }

    #[cfg(all(feature = "flex", feature = "cpu"))]
    #[test]
    fn fused_requires_explicit_cube_backend() {
        let device = Device::flex();
        let input = Tensor::<3>::zeros([1, 2, BINS], &device);
        assert!(fused_compact_salience(input).is_err());
    }
    #[cfg(feature = "cpu")]
    #[test]
    #[ignore = "opt-in CPU microbenchmark; not evidence of CUDA speedup"]
    fn benchmark_fused_vs_dense_and_burn_compact_cpu() {
        use std::hint::black_box;
        use std::time::Instant;

        let device = Device::cpu();
        for (batch, frames) in [(1, 35), (1, 70), (1, 200), (4, 35), (4, 200)] {
            let input: Vec<f32> = (0..batch * frames * BINS)
                .map(|i| (((i.wrapping_mul(619) % 991) as f32) + 1.0) / 992.0)
                .collect();
            let tensor =
                Tensor::<3>::from_data(TensorData::new(input, [batch, frames, BINS]), &device);
            let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for round in 0..15 {
                let order = match round % 3 {
                    0 => [0, 1, 2],
                    1 => [2, 0, 1],
                    _ => [1, 2, 0],
                };
                for mode in order {
                    let now = Instant::now();
                    let values = match mode {
                        0 => tensor.clone().into_data().try_into_vec::<f32>().unwrap(),
                        1 => compact_salience(tensor.clone())
                            .into_data()
                            .try_into_vec::<f32>()
                            .unwrap(),
                        _ => fused_compact_salience(tensor.clone()).unwrap(),
                    };
                    black_box(values);
                    if round >= 3 {
                        samples[mode].push(now.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
            for samples in &mut samples {
                samples.sort_by(f64::total_cmp);
            }
            eprintln!(
                "RMVPE CubeCL CPU B={batch} T={frames} dense_ms={:.5} burn_compact_ms={:.5} fused_ms={:.5}",
                samples[0][6], samples[1][6], samples[2][6]
            );
        }
    }
    #[cfg(all(feature = "cpu", feature = "flex"))]
    #[test]
    #[ignore = "requires private RMVPE checkpoint and WAV supplied by the operator"]
    fn real_rmvpe_salience_matches_fused_cube_cpu() {
        let checkpoint_path = std::env::var("RVC_REAL_RMVPE_CHECKPOINT").unwrap();
        let wav_path = std::env::var("RVC_REAL_INPUT_WAV").unwrap();
        let checkpoint = crate::checkpoint::RvcCheckpoint::open_root(checkpoint_path).unwrap();
        let flex_device = Device::flex();
        let cube_device = Device::cpu();
        let model = crate::rmvpe_mel::RmvpeAudioModel::load(&checkpoint, &flex_device).unwrap();
        let audio = crate::audio_io::read_mono_wav(wav_path).unwrap();
        for (start_secs, duration_secs) in [(0.0f64, 0.35f64), (22.4, 0.7)] {
            let begin = (start_secs * audio.sample_rate as f64).round() as usize;
            let end = ((start_secs + duration_secs) * audio.sample_rate as f64).round() as usize;
            let pcm16 =
                crate::audio_io::resample(&audio.samples[begin..end], audio.sample_rate, 16000)
                    .unwrap();
            let (salience, frames) = model.forward_pcm(&pcm16, &flex_device).unwrap();
            let values = salience.into_data().try_into_vec::<f32>().unwrap();
            assert_eq!(values.len(), frames * BINS);
            let input = Tensor::<3>::from_data(
                TensorData::new(values.clone(), [1, frames, BINS]),
                &cube_device,
            );
            let compact = fused_compact_salience(input).unwrap();
            let plane_reference = simulated_lane_compact(&values);
            assert_eq!(plane_reference, compact);
            for threshold in [0.0, 0.03, 0.5, 1.0] {
                let dense_f0 = salience_to_f0(&values, threshold).unwrap();
                let fused_f0 = compact_salience_to_f0(&compact, threshold).unwrap();
                assert_eq!(
                    dense_f0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    fused_f0.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
            eprintln!(
                "real RMVPE fused CubeCPU parity PASS: start={start_secs} length={duration_secs} frames={frames}"
            );
        }
    }
}
