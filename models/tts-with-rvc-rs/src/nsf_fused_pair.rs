use burn::cubecl;
use burn::cubecl::prelude::*;
use burn::tensor::{DType, Tensor};
use burn_cubecl::tensor::CubeTensor;

const BLOCK: u32 = 128;

#[cube(launch_unchecked)]
fn fused_preact_conv_kernel(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32],
    output: &mut [f32],
    elements: usize,
    in_channels: usize,
    out_channels: usize,
    time: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
    input_batch_stride: usize,
    input_channel_stride: usize,
    input_time_stride: usize,
    weight_out_stride: usize,
    weight_in_stride: usize,
    weight_kernel_stride: usize,
    bias_stride: usize,
    residual_batch_stride: usize,
    residual_channel_stride: usize,
    residual_time_stride: usize,
    has_bias: u32,
    add_residual: u32,
) {
    let index = ABSOLUTE_POS as usize;
    if index < elements {
        let position = index % time;
        let output_channel = (index / time) % out_channels;
        let batch = index / (out_channels * time);
        let mut sum = 0.0_f32;
        let mut input_channel = 0usize;
        while input_channel < in_channels {
            let mut tap = 0usize;
            while tap < kernel {
                let padded_position = position + tap * dilation;
                if padded_position >= padding {
                    let source_position = padded_position - padding;
                    if source_position < time {
                        let input_offset = batch * input_batch_stride
                            + input_channel * input_channel_stride
                            + source_position * input_time_stride;
                        let value = input[input_offset];
                        let value = if value >= 0.0_f32 {
                            value
                        } else {
                            0.1_f32 * value
                        };
                        let weight_offset = output_channel * weight_out_stride
                            + input_channel * weight_in_stride
                            + tap * weight_kernel_stride;
                        sum += value * weights[weight_offset];
                    }
                }
                tap += 1;
            }
            input_channel += 1;
        }
        if has_bias != 0_u32 {
            sum += bias[output_channel * bias_stride];
        }
        if add_residual != 0_u32 {
            sum += residual[batch * residual_batch_stride
                + output_channel * residual_channel_stride
                + position * residual_time_stride];
        }
        output[index] = sum;
    }
}

fn checked_extent(shape: [usize; 3], strides: &[usize], buffer_len: usize) -> Result<(), String> {
    let mut last = 1usize;
    for (dim, stride) in shape.iter().zip(strides) {
        last = last
            .checked_add(
                (dim - 1)
                    .checked_mul(*stride)
                    .ok_or("NSF fused tensor extent overflow")?,
            )
            .ok_or("NSF fused tensor extent overflow")?;
    }
    if last > buffer_len {
        return Err("NSF fused tensor exceeds its storage".into());
    }
    Ok(())
}

fn checked_bias_extent(length: usize, stride: usize, buffer_len: usize) -> Result<(), String> {
    let last = (length - 1)
        .checked_mul(stride)
        .and_then(|x| x.checked_add(1))
        .ok_or("NSF fused bias extent overflow")?;
    if last > buffer_len {
        return Err("NSF fused bias exceeds its storage".into());
    }
    Ok(())
}

pub(crate) fn preact_conv(
    input: Tensor<3>,
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    residual: Option<Tensor<3>>,
    dilation: usize,
    padding: usize,
) -> Result<Tensor<3>, String> {
    let [batch, in_channels, time] = input.dims();
    let [out_channels, weight_channels, kernel] = weight.dims();
    if [batch, in_channels, time, out_channels, kernel, dilation]
        .iter()
        .any(|x| *x == 0)
        || in_channels != weight_channels
    {
        return Err("NSF fused convolution: invalid dimensions".into());
    }
    if kernel % 2 != 1 || padding != ((kernel - 1) / 2) * dilation {
        return Err(
            "NSF fused convolution requires same-length symmetric odd-kernel padding".into(),
        );
    }
    if bias.as_ref().is_some_and(|x| x.dims() != [out_channels]) {
        return Err("NSF fused convolution bias mismatch".into());
    }
    if residual
        .as_ref()
        .is_some_and(|x| x.dims() != [batch, out_channels, time])
    {
        return Err("NSF fused convolution residual mismatch".into());
    }
    let input = input
        .try_into_primitive::<burn_cubecl::Cube>()
        .map_err(|x| format!("NSF fused requires CubeCL input: {x:?}"))?;
    let weight = weight
        .try_into_primitive::<burn_cubecl::Cube>()
        .map_err(|x| format!("NSF fused requires CubeCL weight: {x:?}"))?;
    let bias = bias
        .map(|x| x.try_into_primitive::<burn_cubecl::Cube>())
        .transpose()
        .map_err(|x| format!("NSF fused requires CubeCL bias: {x:?}"))?;
    let residual = residual
        .map(|x| x.try_into_primitive::<burn_cubecl::Cube>())
        .transpose()
        .map_err(|x| format!("NSF fused requires CubeCL residual: {x:?}"))?;
    if input.dtype != DType::F32
        || weight.dtype != DType::F32
        || bias.as_ref().is_some_and(|x| x.dtype != DType::F32)
        || residual.as_ref().is_some_and(|x| x.dtype != DType::F32)
        || input.meta.is_tiled()
        || weight.meta.is_tiled()
        || bias.as_ref().is_some_and(|x| x.meta.is_tiled())
        || residual.as_ref().is_some_and(|x| x.meta.is_tiled())
    {
        return Err("NSF fused requires non-tiled FP32 CubeCL tensors".into());
    }
    let input_strides = input.meta.strides();
    let weight_strides = weight.meta.strides();
    checked_extent(
        [batch, in_channels, time],
        input_strides,
        input.handle.size_in_used() as usize / 4,
    )?;
    checked_extent(
        [out_channels, in_channels, kernel],
        weight_strides,
        weight.handle.size_in_used() as usize / 4,
    )?;
    if let Some(bias) = &bias {
        checked_bias_extent(
            out_channels,
            bias.meta.strides()[0],
            bias.handle.size_in_used() as usize / 4,
        )?;
    }
    if let Some(residual) = &residual {
        checked_extent(
            [batch, out_channels, time],
            residual.meta.strides(),
            residual.handle.size_in_used() as usize / 4,
        )?;
    }
    let elements = batch
        .checked_mul(out_channels)
        .and_then(|x| x.checked_mul(time))
        .ok_or("NSF fused output size overflow")?;
    if elements.div_ceil(BLOCK as usize) > u32::MAX as usize {
        return Err("NSF fused output exceeds CubeCL grid".into());
    }
    let bytes = elements
        .checked_mul(4)
        .ok_or("NSF fused output bytes overflow")?;
    let output = input.client.empty(bytes);
    let bias_stride = bias.as_ref().map(|x| x.meta.strides()[0]).unwrap_or(0);
    let residual_strides = residual
        .as_ref()
        .map(|x| x.meta.strides().to_vec())
        .unwrap_or_else(|| vec![0, 0, 0]);
    let has_bias = u32::from(bias.is_some());
    let add_residual = u32::from(residual.is_some());
    let arg = |handle: &cubecl::server::Handle| unsafe {
        BufferArg::from_raw_parts(handle.clone(), handle.size_in_used() as usize / 4)
    };
    unsafe {
        fused_preact_conv_kernel::launch_unchecked(
            &input.client,
            CubeCount::Static(elements.div_ceil(BLOCK as usize) as u32, 1, 1),
            CubeDim::new_1d(BLOCK),
            arg(&input.handle),
            arg(&weight.handle),
            arg(bias.as_ref().map(|x| &x.handle).unwrap_or(&input.handle)),
            arg(residual
                .as_ref()
                .map(|x| &x.handle)
                .unwrap_or(&input.handle)),
            BufferArg::from_raw_parts(output.clone(), elements),
            elements,
            in_channels,
            out_channels,
            time,
            kernel,
            dilation,
            padding,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            weight_strides[0],
            weight_strides[1],
            weight_strides[2],
            bias_stride,
            residual_strides[0],
            residual_strides[1],
            residual_strides[2],
            has_bias,
            add_residual,
        );
    }
    Ok(Tensor::from_primitive::<burn_cubecl::Cube>(
        CubeTensor::new_contiguous(
            input.client,
            input.device,
            [batch, out_channels, time].into(),
            output,
            DType::F32,
        ),
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn checked_extent_detects_out_of_storage_views() {
        assert!(super::checked_extent([2, 4, 17], &[68, 17, 1], 136).is_ok());
        assert!(super::checked_extent([2, 4, 17], &[68, 17, 1], 135).is_err());
        assert!(super::checked_extent([2, 4, 17], &[1, 1, 100], 100).is_err());
        assert!(super::checked_bias_extent(7, 2, 13).is_ok());
        assert!(super::checked_bias_extent(7, 2, 12).is_err());
    }
}
