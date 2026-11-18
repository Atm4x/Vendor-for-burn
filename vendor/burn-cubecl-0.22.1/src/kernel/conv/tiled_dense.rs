use crate::tensor::CubeTensor;
use burn_backend::{DType, ops::ConvOptions};
use cubecl::prelude::*;

#[cube]
fn input_offset(
    pos: usize,
    tap: usize,
    input: &Tensor<f32>,
    width: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    #[comptime] check_padding: bool,
    #[comptime] spatial: bool,
) -> (bool, usize) {
    if comptime![spatial] {
        let h = pos / width + tap / 3;
        let w = pos % width + tap % 3;
        let valid = h > 0 && h <= input.shape(1) && w > 0 && w <= input.shape(2);
        (valid, (h - 1) * input.stride(1) + (w - 1) * input.stride(2))
    } else {
        let shifted = pos * stride + tap * dilation;
        if comptime![check_padding] {
            (
                shifted >= padding && shifted - padding < input.shape(1),
                (shifted - padding) * input.stride(1),
            )
        } else {
            (true, shifted * input.stride(1))
        }
    }
}

/// Each workgroup shares a square tile of inputs and filters across 64 lanes.
/// The reduction visits taps and input channels in the same order as direct convolution.
#[cube(launch, address_type = "dynamic")]
fn dense_tiled_conv(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: ComptimeOption<&[f32]>,
    output: &mut Tensor<f32>,
    stride: usize,
    dilation: usize,
    padding: usize,
    #[comptime] check_padding: bool,
    out_per_group: usize,
    #[comptime] block: usize,
    #[comptime] grouped: bool,
    #[comptime] spatial: bool,
) {
    let x = UNIT_POS_X as usize;
    let y = UNIT_POS_Y as usize;
    let row = CUBE_POS_Y as usize * block;
    let col = CUBE_POS_X as usize * block;
    let batch = CUBE_POS_Z as usize;
    let mut length = output.shape(1);
    let mut width = length;
    let mut taps = weight.shape(1);
    let dim_c = comptime![if spatial { 3 } else { 2 }];
    let mut input_base = batch * input.stride(0);
    if comptime![grouped] {
        input_base += (col / out_per_group) * weight.shape(dim_c);
    }
    let lanes = comptime![block / 8];
    let items = comptime![lanes * lanes];
    if comptime![spatial] {
        width = output.shape(2);
        length *= width;
        taps *= weight.shape(2);
    }
    let mut features = Shared::<[f32]>::new_slice(comptime![block * (block + 1)]);
    let mut filters = Shared::<[f32]>::new_slice(comptime![block * (block + 1)]);
    let mut acc = Array::<f32>::new(items);
    #[unroll]
    for i in 0..items {
        #[comptime]
        match bias {
            ComptimeOption::Some(b) => acc[i] = b[col + x + (i % lanes) * 8],
            ComptimeOption::None => acc[i] = 0.0_f32,
        }
    }
    for tap in 0..taps {
        let mut use_tap = Array::<bool>::new(lanes);
        #[unroll]
        for i in 0..lanes {
            let (valid, _) = input_offset(
                row + y + i * 8,
                tap,
                input,
                width,
                stride,
                dilation,
                padding,
                check_padding,
                spatial,
            );
            use_tap[i] = valid;
        }
        let mut weight_offset = tap * weight.stride(1);
        if comptime![spatial] {
            weight_offset = (tap / 3) * weight.stride(1) + (tap % 3) * weight.stride(2);
        }
        for tile in 0..weight.shape(dim_c) / block {
            #[unroll]
            for i in 0..items {
                let index = i * 64 + y * 8 + x;
                let line = index / block;
                let k = index % block;
                let ci = tile * block + k;
                let mut value = 0.0_f32;
                let (valid, offset) = input_offset(
                    row + line,
                    tap,
                    input,
                    width,
                    stride,
                    dilation,
                    padding,
                    check_padding,
                    spatial,
                );
                if row + line < length && valid {
                    value = input[input_base + offset + ci];
                }
                features[line * (block + 1) + k] = value;
                filters[line * (block + 1) + k] =
                    weight[(col + line) * weight.stride(0) + weight_offset + ci];
            }
            sync_cube();
            #[unroll]
            for k in 0..block {
                #[unroll]
                for i in 0..lanes {
                    let a = features[(y + i * 8) * (block + 1) + k];
                    #[unroll]
                    for j in 0..lanes {
                        if !check_padding || use_tap[i] {
                            acc[i * lanes + j] += a * filters[(x + j * 8) * (block + 1) + k];
                        }
                    }
                }
            }
            sync_cube();
        }

    }
    #[unroll]
    for i in 0..lanes {
        let pos = row + y + i * 8;
        if pos < length {
            #[unroll]
            for j in 0..lanes {
                let mut offset = pos * output.stride(1);
                if comptime![spatial] {
                    offset = (pos / width) * output.stride(1) + (pos % width) * output.stride(2);
                }
                let dst = batch * output.stride(0) + offset + col + x + j * 8;
                output[dst] = acc[i * lanes + j];
            }
        }
    }
}

pub(super) fn try_launch<const N: usize>(
    input: &CubeTensor,
    weight: &CubeTensor,
    bias: Option<&CubeTensor>,
    out: &CubeTensor,
    options: &ConvOptions<N>,
) -> bool {
    let shape = weight.meta.shape();
    let spatial = N == 2;
    let eligible = match N {
        1 => (1..=128).contains(&shape[1]),
        2 => {
            shape[1..3] == [3, 3]
                && options.padding.iter().all(|&p| p == (1, 1))
                && options
                    .stride
                    .iter()
                    .chain(&options.dilation)
                    .all(|&v| v == 1)
        }
        _ => false,
    };
    if !eligible || bias.is_some_and(|v| v.dtype != DType::F32 || v.meta.strides()[0] != 1) {
        return false;
    }
    let out_shape = out.meta.shape();
    let batch = out_shape[0];
    let cout = shape[0];
    let cin = shape[N + 1];
    let out_per_group = cout / options.groups;
    let length = out_shape[1..N + 1].iter().product::<usize>();
    let block = if cin % 32 == 0 && out_per_group % 32 == 0 {
        32
    } else {
        16
    };
    let hw = &input.client.properties().hardware;
    if input.dtype != DType::F32
        || weight.dtype != DType::F32
        || hw.plane_size_max <= 1
        || hw.max_units_per_cube < 64
        || hw.max_shared_memory_size < 2 * block * (block + 1) * size_of::<f32>()
        || hw.max_cube_dim.0 < 8
        || hw.max_cube_dim.1 < 8
        || cin < block
        || cin % block != 0
        || out_per_group < block
        || out_per_group % block != 0
        || input.meta.strides()[N + 1] != 1
        || weight.meta.strides()[N + 1] != 1
        || length < block
        || batch == 0
        || cout / block > hw.max_cube_count.0 as usize
        || length.div_ceil(block) > hw.max_cube_count.1 as usize
        || batch > hw.max_cube_count.2 as usize
    {
        return false;
    }
    let grid = CubeCount::Static(
        (cout / block) as u32,
        length.div_ceil(block) as u32,
        batch as u32,
    );
    dense_tiled_conv::launch(
        &input.client,
        grid,
        CubeDim::new_2d(8, 8),
        input
            .required_address_type()
            .max(weight.required_address_type())
            .max(out.required_address_type())
            .max(
                bias.map(CubeTensor::required_address_type)
                    .unwrap_or_default(),
            ),
        input.clone().into_tensor_arg(),
        weight.clone().into_tensor_arg(),
        bias.cloned().map(|b| b.into_buffer_arg()).into(),
        out.clone().into_tensor_arg(),
        options.stride[0],
        options.dilation[0],
        options.padding[0].0,
        spatial || options.padding[0].0 != 0 || options.padding[0].1 != 0,
        out_per_group,
        block,
        options.groups > 1,
        spatial,
    );
    true
}
