use burn::cubecl;
use burn::cubecl::prelude::*;
use burn::tensor::{DType, Tensor};
use burn_cubecl::tensor::CubeTensor;

const LANES: u32 = 128;
const TILE_TIME: usize = 32;
const MAX_CHANNELS: usize = 64;
const MAX_SECOND_KERNEL: usize = 11;
const SHARED_ELEMENTS: usize = (TILE_TIME + MAX_SECOND_KERNEL - 1) * MAX_CHANNELS;

#[cube(launch_unchecked)]
fn pair_tile_kernel(
    input: &[f32],
    first_weight: &[f32],
    first_bias: &[f32],
    second_weight: &[f32],
    second_bias: &[f32],
    output: &mut [f32],
    channels: usize,
    time: usize,
    first_kernel: usize,
    first_dilation: usize,
    first_padding: usize,
    second_kernel: usize,
    second_padding: usize,
    input_batch_stride: usize,
    input_channel_stride: usize,
    input_time_stride: usize,
    first_out_stride: usize,
    first_in_stride: usize,
    first_kernel_stride: usize,
    first_bias_stride: usize,
    second_out_stride: usize,
    second_in_stride: usize,
    second_kernel_stride: usize,
    second_bias_stride: usize,
    has_first_bias: u32,
    has_second_bias: u32,
) {
    let lane = UNIT_POS as usize;
    let tile = CUBE_POS_X as usize;
    let batch = CUBE_POS_Y as usize;
    let start = tile * TILE_TIME;
    let halo = second_padding;
    let expanded = TILE_TIME + 2 * halo;
    let mut middle = Shared::<[f32]>::new_slice(SHARED_ELEMENTS);

    let mut item = lane;
    while item < expanded * channels {
        let mid_channel = item % channels;
        let local_time = item / channels;
        let global_signed = (start + local_time) as i64 - halo as i64;
        let mut sum = 0.0_f32;
        if global_signed >= 0 && (global_signed as usize) < time {
            let global_time = global_signed as usize;
            let mut in_channel = 0usize;
            while in_channel < channels {
                let mut tap = 0usize;
                while tap < first_kernel {
                    let source_offset = global_time + tap * first_dilation;
                    if source_offset >= first_padding {
                        let source_time = source_offset - first_padding;
                        if source_time < time {
                            let value = input[batch * input_batch_stride
                                + in_channel * input_channel_stride
                                + source_time * input_time_stride];
                            let activated = if value >= 0.0_f32 {
                                value
                            } else {
                                value * 0.1_f32
                            };
                            sum += activated
                                * first_weight[mid_channel * first_out_stride
                                    + in_channel * first_in_stride
                                    + tap * first_kernel_stride];
                        }
                    }
                    tap += 1;
                }
                in_channel += 1;
            }
            if has_first_bias != 0_u32 {
                sum += first_bias[mid_channel * first_bias_stride];
            }
        }
        middle[item] = sum;
        item += LANES as usize;
    }
    sync_cube();

    let mut out_item = lane;
    while out_item < TILE_TIME * channels {
        let channel = out_item % channels;
        let local_time = out_item / channels;
        let global_time = start + local_time;
        if global_time < time {
            let mut sum = 0.0_f32;
            let mut in_channel = 0usize;
            while in_channel < channels {
                let mut tap = 0usize;
                while tap < second_kernel {
                    let mid_local_time = local_time + tap;
                    let mid_global = (start + mid_local_time) as i64 - halo as i64;
                    if mid_global >= 0 && (mid_global as usize) < time {
                        let value = middle[mid_local_time * channels + in_channel];
                        let activated = if value >= 0.0_f32 {
                            value
                        } else {
                            value * 0.1_f32
                        };
                        sum += activated
                            * second_weight[channel * second_out_stride
                                + in_channel * second_in_stride
                                + tap * second_kernel_stride];
                    }
                    tap += 1;
                }
                in_channel += 1;
            }
            if has_second_bias != 0_u32 {
                sum += second_bias[channel * second_bias_stride];
            }
            let residual = input[batch * input_batch_stride
                + channel * input_channel_stride
                + global_time * input_time_stride];
            output[(batch * channels + channel) * time + global_time] = residual + sum;
        }
        out_item += LANES as usize;
    }
}

fn check_storage(shape: [usize; 3], strides: &[usize], bytes: usize) -> Result<(), String> {
    let mut last = 1usize;
    for (dim, stride) in shape.into_iter().zip(strides.iter().copied()) {
        last = last
            .checked_add(
                (dim - 1)
                    .checked_mul(stride)
                    .ok_or("NSF tile stride overflow")?,
            )
            .ok_or("NSF tile stride overflow")?;
    }
    if last.checked_mul(4).ok_or("NSF tile storage overflow")? > bytes {
        return Err("NSF tile strides exceed underlying storage".into());
    }
    Ok(())
}

fn check_bias(tensor: &CubeTensor, channels: usize) -> Result<(), String> {
    if tensor.dtype != DType::F32
        || tensor.meta.is_tiled()
        || tensor.meta.shape.dims() != [channels]
    {
        return Err("NSF tile bias must be contiguous-rank FP32 CubeCL tensor".into());
    }
    let extent = (channels - 1)
        .checked_mul(tensor.meta.strides()[0])
        .and_then(|v| v.checked_add(1))
        .and_then(|v| v.checked_mul(4))
        .ok_or("NSF tile bias storage overflow")?;
    if extent > tensor.handle.size_in_used() as usize {
        return Err("NSF tile bias exceeds underlying storage".into());
    }
    Ok(())
}

pub(crate) fn supports(channels: usize, second_kernel: usize, second_dilation: usize) -> bool {
    (1..=MAX_CHANNELS).contains(&channels)
        && second_kernel > 0
        && second_kernel <= MAX_SECOND_KERNEL
        && second_kernel % 2 == 1
        && second_dilation == 1
}

pub(crate) fn forward(
    input: Tensor<3>,
    first_weight: Tensor<3>,
    first_bias: Option<Tensor<1>>,
    first_dilation: usize,
    first_padding: usize,
    second_weight: Tensor<3>,
    second_bias: Option<Tensor<1>>,
    second_dilation: usize,
    second_padding: usize,
) -> Result<Tensor<3>, String> {
    let [batch, channels, time] = input.dims();
    let [first_out, first_in, first_kernel] = first_weight.dims();
    let [second_out, second_in, second_kernel] = second_weight.dims();
    if batch == 0
        || time == 0
        || first_kernel == 0
        || first_dilation == 0
        || first_out != channels
        || first_in != channels
        || second_out != channels
        || second_in != channels
        || !supports(channels, second_kernel, second_dilation)
        || first_kernel % 2 == 0
        || second_padding != (second_kernel - 1) / 2
    {
        return Err("NSF tile pair requires same-channel symmetric odd-kernel convolutions (second dilation=1)".into());
    }
    let required_first_padding = ((first_kernel - 1) / 2)
        .checked_mul(first_dilation)
        .ok_or("NSF tile first padding overflow")?;
    let max_offset = time
        .checked_add(TILE_TIME + MAX_SECOND_KERNEL)
        .and_then(|value| value.checked_add((first_kernel - 1).checked_mul(first_dilation)?))
        .ok_or("NSF tile convolution indexing overflow")?;
    if first_padding != required_first_padding || max_offset > i64::MAX as usize {
        return Err("NSF tile convolution indexing is not representable".into());
    }
    if first_bias.as_ref().is_some_and(|x| x.dims() != [channels])
        || second_bias.as_ref().is_some_and(|x| x.dims() != [channels])
    {
        return Err("NSF tile pair bias shape mismatch".into());
    }
    let input = input
        .try_into_primitive::<burn_cubecl::Cube>()
        .map_err(|e| format!("NSF tile CubeCL input required: {e:?}"))?;
    let first_weight = first_weight
        .try_into_primitive::<burn_cubecl::Cube>()
        .map_err(|e| format!("NSF tile CubeCL first weights required: {e:?}"))?;
    let second_weight = second_weight
        .try_into_primitive::<burn_cubecl::Cube>()
        .map_err(|e| format!("NSF tile CubeCL second weights required: {e:?}"))?;
    let first_bias = first_bias
        .map(|x| x.try_into_primitive::<burn_cubecl::Cube>())
        .transpose()
        .map_err(|e| format!("NSF tile first bias requires CubeCL: {e:?}"))?;
    let second_bias = second_bias
        .map(|x| x.try_into_primitive::<burn_cubecl::Cube>())
        .transpose()
        .map_err(|e| format!("NSF tile second bias requires CubeCL: {e:?}"))?;
    for tensor in [&input, &first_weight, &second_weight] {
        if tensor.dtype != DType::F32 || tensor.meta.is_tiled() {
            return Err("NSF tile requires non-tiled FP32 CubeCL tensors".into());
        }
    }
    check_storage(
        [batch, channels, time],
        input.meta.strides(),
        input.handle.size_in_used() as usize,
    )?;
    check_storage(
        [channels, channels, first_kernel],
        first_weight.meta.strides(),
        first_weight.handle.size_in_used() as usize,
    )?;
    check_storage(
        [channels, channels, second_kernel],
        second_weight.meta.strides(),
        second_weight.handle.size_in_used() as usize,
    )?;
    if let Some(bias) = &first_bias {
        check_bias(bias, channels)?;
    }
    if let Some(bias) = &second_bias {
        check_bias(bias, channels)?;
    }
    let elements = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(time))
        .ok_or("NSF tile output elements overflow")?;
    let bytes = elements.checked_mul(4).ok_or("NSF tile bytes overflow")?;
    let tiles = time.div_ceil(TILE_TIME);
    if tiles > u32::MAX as usize || batch > u32::MAX as usize {
        return Err("NSF tile grid exceeds CUDA limits".into());
    }
    let output = input.client.empty(bytes);
    let bs1 = first_bias
        .as_ref()
        .map(|x| x.meta.strides()[0])
        .unwrap_or(0);
    let bs2 = second_bias
        .as_ref()
        .map(|x| x.meta.strides()[0])
        .unwrap_or(0);
    let arg = |handle: &cubecl::server::Handle| unsafe {
        BufferArg::from_raw_parts(handle.clone(), handle.size_in_used() as usize / 4)
    };
    unsafe {
        pair_tile_kernel::launch_unchecked(
            &input.client,
            CubeCount::Static(tiles as u32, batch as u32, 1),
            CubeDim::new_1d(LANES),
            arg(&input.handle),
            arg(&first_weight.handle),
            arg(first_bias
                .as_ref()
                .map(|b| &b.handle)
                .unwrap_or(&input.handle)),
            arg(&second_weight.handle),
            arg(second_bias
                .as_ref()
                .map(|b| &b.handle)
                .unwrap_or(&input.handle)),
            BufferArg::from_raw_parts(output.clone(), elements),
            channels,
            time,
            first_kernel,
            first_dilation,
            first_padding,
            second_kernel,
            second_padding,
            input.meta.strides()[0],
            input.meta.strides()[1],
            input.meta.strides()[2],
            first_weight.meta.strides()[0],
            first_weight.meta.strides()[1],
            first_weight.meta.strides()[2],
            bs1,
            second_weight.meta.strides()[0],
            second_weight.meta.strides()[1],
            second_weight.meta.strides()[2],
            bs2,
            u32::from(first_bias.is_some()),
            u32::from(second_bias.is_some()),
        );
    }
    Ok(Tensor::from_primitive::<burn_cubecl::Cube>(
        CubeTensor::new_contiguous(
            input.client,
            input.device,
            [batch, channels, time].into(),
            output,
            DType::F32,
        ),
    ))
}
