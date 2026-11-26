use burn::cubecl;
use burn::cubecl::prelude::*;
use burn::tensor::{DType, Tensor};
use burn_cubecl::tensor::CubeTensor;

type CubeBackend = burn_cubecl::Cube;
const BLOCK: u32 = 128;

#[cube(launch_unchecked)]
fn split_tf32_kernel(
    input: &[f32],
    high: &mut [f32],
    low: &mut [f32],
    elements: usize,
    channels: usize,
    height: usize,
    width: usize,
    stride_batch: usize,
    stride_channel: usize,
    stride_height: usize,
    stride_width: usize,
) {
    let index = ABSOLUTE_POS as usize;
    if index < elements {
        let channel_plane = height * width;
        let batch_plane = channel_plane * channels;
        let batch = index / batch_plane;
        let channel = index / channel_plane % channels;
        let row = index / width % height;
        let column = index % width;
        let offset = batch * stride_batch
            + channel * stride_channel
            + row * stride_height
            + column * stride_width;
        let value = input[offset];
        let bits = u32::reinterpret(value);
        let rounded = (bits + 0xfff_u32 + ((bits >> 13) & 1_u32)) & 0xffff_e000_u32;
        let upper = f32::reinterpret(rounded);
        high[index] = upper;
        low[index] = value - upper;
    }
}

pub(crate) fn split_tf32(input: Tensor<4>) -> Result<(Tensor<4>, Tensor<4>), String> {
    let dimensions = input.dims();
    if dimensions.contains(&0) {
        return Err("TF32 compensated convolution requires nonempty dimensions".into());
    }
    let input = input.try_into_primitive::<CubeBackend>()
        .map_err(|error| format!("TF32 compensated convolution needs CubeCL: {error:?}"))?;
    if input.dtype != DType::F32 || input.meta.is_tiled() {
        return Err("TF32 compensated convolution requires non-tiled FP32 input".into());
    }
    let elements = dimensions.iter().try_fold(1_usize, |size, next| size.checked_mul(*next))
        .ok_or("TF32 input shape overflow")?;
    let bytes = elements.checked_mul(4).ok_or("TF32 input byte count overflow")?;
    if elements.div_ceil(BLOCK as usize) > u32::MAX as usize {
        return Err("TF32 input exceeds CUDA grid limits".into());
    }
    let strides = input.meta.strides();
    let maximum = (dimensions[0] - 1).checked_mul(strides[0])
        .and_then(|v| v.checked_add((dimensions[1] - 1).checked_mul(strides[1])?))
        .and_then(|v| v.checked_add((dimensions[2] - 1).checked_mul(strides[2])?))
        .and_then(|v| v.checked_add((dimensions[3] - 1).checked_mul(strides[3])?))
        .and_then(|v| v.checked_add(1))
        .ok_or("TF32 tensor stride overflow")?;
    if maximum.checked_mul(4).ok_or("TF32 input storage overflow")?
        > input.handle.size_in_used() as usize
    {
        return Err("TF32 tensor strides exceed allocated storage".into());
    }
    let high = input.client.empty(bytes);
    let low = input.client.empty(bytes);
    let arg = |handle: &cubecl::server::Handle, length: usize| unsafe {
        BufferArg::from_raw_parts(handle.clone(), length)
    };
    unsafe {
        split_tf32_kernel::launch_unchecked(
            &input.client,
            CubeCount::Static(elements.div_ceil(BLOCK as usize) as u32, 1, 1),
            CubeDim::new_1d(BLOCK),
            arg(&input.handle, input.handle.size_in_used() as usize / 4),
            arg(&high, elements),
            arg(&low, elements),
            elements, dimensions[1], dimensions[2], dimensions[3],
            strides[0], strides[1], strides[2], strides[3],
        );
    }
    let shape = dimensions.into();
    let upper = Tensor::from_primitive::<CubeBackend>(CubeTensor::new_contiguous(
        input.client.clone(), input.device.clone(), shape, high, DType::F32,
    ));
    let lower = Tensor::from_primitive::<CubeBackend>(CubeTensor::new_contiguous(
        input.client, input.device, dimensions.into(), low, DType::F32,
    ));
    Ok((upper, lower))
}


#[cube(launch_unchecked)]
fn split_tf32_packed_kernel(
    input: &[f32],
    packed: &mut [f32],
    elements: usize,
    channels: usize,
    height: usize,
    width: usize,
    stride_batch: usize,
    stride_channel: usize,
    stride_height: usize,
    stride_width: usize,
) {
    let index = ABSOLUTE_POS as usize;
    if index < elements {
        let plane = height * width;
        let batch_plane = plane * channels;
        let batch = index / batch_plane;
        let channel = index / plane % channels;
        let row = index / width % height;
        let column = index % width;
        let offset = batch * stride_batch
            + channel * stride_channel
            + row * stride_height
            + column * stride_width;
        let value = input[offset];
        let bits = u32::reinterpret(value);
        let rounded = (bits + 0xfff_u32 + ((bits >> 13) & 1_u32)) & 0xffff_e000_u32;
        let upper = f32::reinterpret(rounded);
        packed[index] = upper;
        packed[elements + index] = value - upper;
    }
}

pub(crate) fn split_tf32_packed(input: Tensor<4>) -> Result<Tensor<4>, String> {
    let dimensions = input.dims();
    if dimensions.contains(&0) {
        return Err("TF32 packed convolution requires nonempty dimensions".into());
    }
    let input = input.try_into_primitive::<CubeBackend>()
        .map_err(|error| format!("TF32 packed convolution needs CubeCL: {error:?}"))?;
    if input.dtype != DType::F32 || input.meta.is_tiled() {
        return Err("TF32 packed convolution requires non-tiled FP32 input".into());
    }
    let elements = dimensions.iter().try_fold(1_usize, |size, next| size.checked_mul(*next))
        .ok_or("TF32 packed input shape overflow")?;
    let packed_elements = elements.checked_mul(2).ok_or("TF32 packed allocation overflow")?;
    let bytes = packed_elements.checked_mul(4).ok_or("TF32 packed byte count overflow")?;
    if elements.div_ceil(BLOCK as usize) > u32::MAX as usize {
        return Err("TF32 packed input exceeds CUDA grid limits".into());
    }
    let strides = input.meta.strides();
    let maximum = (dimensions[0] - 1).checked_mul(strides[0])
        .and_then(|v| v.checked_add((dimensions[1] - 1).checked_mul(strides[1])?))
        .and_then(|v| v.checked_add((dimensions[2] - 1).checked_mul(strides[2])?))
        .and_then(|v| v.checked_add((dimensions[3] - 1).checked_mul(strides[3])?))
        .and_then(|v| v.checked_add(1))
        .ok_or("TF32 packed tensor stride overflow")?;
    if maximum.checked_mul(4).ok_or("TF32 packed tensor storage overflow")?
        > input.handle.size_in_used() as usize
    {
        return Err("TF32 packed tensor strides exceed allocated storage".into());
    }
    let packed = input.client.empty(bytes);
    let arg = |handle: &cubecl::server::Handle, length: usize| unsafe {
        BufferArg::from_raw_parts(handle.clone(), length)
    };
    unsafe {
        split_tf32_packed_kernel::launch_unchecked(
            &input.client,
            CubeCount::Static(elements.div_ceil(BLOCK as usize) as u32, 1, 1),
            CubeDim::new_1d(BLOCK),
            arg(&input.handle, input.handle.size_in_used() as usize / 4),
            arg(&packed, packed_elements),
            elements, dimensions[1], dimensions[2], dimensions[3],
            strides[0], strides[1], strides[2], strides[3],
        );
    }
    let mut shape = dimensions;
    shape[0] = shape[0].checked_mul(2).ok_or("TF32 packed batch overflow")?;
    Ok(Tensor::from_primitive::<CubeBackend>(CubeTensor::new_contiguous(
        input.client, input.device, shape.into(), packed, DType::F32,
    )))
}

#[cfg(test)]
mod tests {
    #[test]
    fn tf32_packed_batch_preserves_hi_lo_ordering() {
        for (batch, channels, height, width) in [(1, 3, 9, 7), (2, 5, 4, 11)] {
            let elements = batch * channels * height * width;
            let mut packed = vec![0.0f32; elements * 2];
            for index in 0..elements {
                let value = ((index * 197 + 17) % 1009) as f32 / 79.0 - 6.0;
                let bits = value.to_bits();
                let upper = f32::from_bits((bits + 0xfff + ((bits >> 13) & 1)) & 0xffff_e000);
                packed[index] = upper;
                packed[elements + index] = value - upper;
            }
            assert_eq!(packed.len(), 2 * elements);
            for index in 0..elements {
                let restored = packed[index] + packed[elements + index];
                let original = ((index * 197 + 17) % 1009) as f32 / 79.0 - 6.0;
                assert!((restored - original).abs() < 2e-6);
            }
        }
    }

    #[test]
    fn tf32_split_recovers_fp32_to_squared_rounding_error() {
        for raw in 1..100_000_u32 {
            let value = ((raw * 977 % 250_001) as f32 - 125_000.0) / 7913.0;
            let bits = value.to_bits();
            let upper = f32::from_bits((bits + 0xfff + ((bits >> 13) & 1)) & 0xffff_e000);
            let delta = value - upper;
            let residual_bits = delta.to_bits();
            let lower = f32::from_bits(
                (residual_bits + 0xfff + ((residual_bits >> 13) & 1)) & 0xffff_e000,
            );
            let error = ((upper + lower) - value).abs();
            assert!(error <= value.abs() * 1e-6 + 1e-6, "value={value} error={error}");
        }
    }
}
