//! Experimental resident FP32 RMVPE salience-to-F0 conversion.
//! The FP64 host reference remains the compatibility oracle until real-GPU validation.
use burn::cubecl;
use burn::cubecl::prelude::*;
use burn::cubecl::{bytes::Bytes, client::Client, server::Handle};
use burn::tensor::{DType, Tensor};
use burn_cubecl::tensor::CubeTensor;

type CubeBackend = burn_cubecl::Cube;
const BINS: usize = 360;
const BLOCK: u32 = 64;

#[cube(launch_unchecked)]
fn f0_kernel(
    salience: &[f32],
    thresholds: &[f32],
    output: &mut [f32],
    rows: usize,
    frames: usize,
    stride_batch: usize,
    stride_frame: usize,
    stride_bin: usize,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let offset = row / frames * stride_batch + row % frames * stride_frame;
        let mut maximum = salience[offset];
        let mut center = 0usize;
        let mut valid = true;
        for bin in 0..360 {
            let value = salience[offset + bin * stride_bin];
            if !(value == value && value <= f32::MAX && value >= -f32::MAX) {
                valid = false;
            }
            if value > maximum {
                maximum = value;
                center = bin;
            }
        }
        if !valid {
            // Explicit negative sentinel, checked by opt-in numeric validation.
            output[row] = -1.0f32;
        } else if maximum <= thresholds[row / frames] {
            output[row] = 0.0f32;
        } else {
            let first = if center < 4 { 0usize } else { center - 4 };
            let last = if center + 4 >= BINS { BINS - 1 } else { center + 4 };
            let mut numerator = 0.0f32;
            let mut denominator = 0.0f32;
            for bin in first..=last {
                let value = salience[offset + bin * stride_bin];
                numerator += value * (1997.3794_f32 + 20.0f32 * bin as f32);
                denominator += value;
            }
            let cents = if denominator > 0.0f32 {
                numerator / denominator
            } else {
                0.0f32
            };
            output[row] = if cents == 0.0f32 { 0.0f32 } else {
                cents
            };
        }
    }
}

fn f32_bytes(values: &[f32]) -> Bytes {
    Bytes::from_bytes_vec(values.iter().copied().flat_map(f32::to_ne_bytes).collect())
}

/// Returns GPU-resident f32 F0, with no host readback of the 360-bin salience.
pub(crate) fn resident_f0(salience: Tensor<3>, thresholds: &[f32]) -> Result<Tensor<2>, String> {
    let [batch, frames, bins] = salience.dims();
    if batch == 0 || frames == 0 || bins != BINS || thresholds.len() != batch
        || thresholds.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("invalid GPU RMVPE salience or threshold dimensions".into());
    }
    let input = salience.try_into_primitive::<CubeBackend>()
        .map_err(|e| format!("GPU RMVPE F0 requires a direct CubeCL backend: {e:?}"))?;
    if input.dtype != DType::F32 || input.meta.is_tiled() {
        return Err("GPU RMVPE F0 requires non-tiled FP32 salience".into());
    }
    let strides = input.meta.strides();
    let required = (batch - 1).checked_mul(strides[0])
        .and_then(|v| v.checked_add((frames - 1).checked_mul(strides[1])?))
        .and_then(|v| v.checked_add((bins - 1).checked_mul(strides[2])?))
        .and_then(|v| v.checked_add(1)).ok_or("RMVPE F0 stride overflow")?;
    if required.checked_mul(4).ok_or("RMVPE F0 size overflow")?
        > input.handle.size_in_used() as usize
    {
        return Err("RMVPE F0 input exceeds backing GPU allocation".into());
    }
    let rows = batch.checked_mul(frames).ok_or("RMVPE F0 rows overflow")?;
    if rows.div_ceil(BLOCK as usize) > u32::MAX as usize {
        return Err("RMVPE F0 grid exceeds CubeCL launch limits".into());
    }
    let client: Client = input.client.clone();
    let threshold_buffer = client.create(f32_bytes(thresholds));
    let output: Handle = client.empty(rows * 4);
    let arg = |h: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(h.clone(), n) };
    unsafe {
        f0_kernel::launch_unchecked(
            &client,
            CubeCount::Static(rows.div_ceil(BLOCK as usize) as u32, 1, 1),
            CubeDim::new_1d(BLOCK),
            arg(&input.handle, input.handle.size_in_used() as usize / 4),
            arg(&threshold_buffer, batch),
            arg(&output, rows),
            rows, frames, strides[0], strides[1], strides[2],
        );
    }
    let cents = Tensor::from_primitive::<CubeBackend>(CubeTensor::new_contiguous(
        client,
        input.device,
        [batch, frames].into(),
        output,
        DType::F32,
    ));
    let unvoiced = cents.clone().equal_elem(0.0);
    let invalid = cents.clone().lower_elem(0.0);
    let hz = (cents * (std::f32::consts::LN_2 / 1200.0)).exp() * 10.0;
    Ok(hz.mask_fill(unvoiced, 0.0).mask_fill(invalid, -1.0))
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};
    use crate::rmvpe::salience_to_f0;

    #[test]
    fn resident_fp32_f0_near_fp64_reference() {
        let device = Device::cpu();
        for (batch, frames) in [(1, 1), (1, 35), (2, 71)] {
            let values = (0..batch * frames * BINS)
                .map(|i| ((i * 271 + 31) % 1021) as f32 / 1021.0)
                .collect::<Vec<_>>();
            let thresholds = (0..batch).map(|i| if i == 0 { 0.03 } else { 0.5 }).collect::<Vec<_>>();
            let input = Tensor::from_data(TensorData::new(values.clone(), [batch, frames, BINS]), &device);
            let actual = resident_f0(input, &thresholds).unwrap()
                .into_data().try_into_vec::<f32>().unwrap();
            for b in 0..batch {
                let expected = salience_to_f0(&values[b * frames * BINS..(b + 1) * frames * BINS], thresholds[b]).unwrap();
                for (got, want) in actual[b * frames..(b + 1) * frames].iter().zip(expected) {
                    assert!((got - want).abs() < 0.03, "FP32 F0 got {got} reference {want}");
                }
            }
        }
    }
}
