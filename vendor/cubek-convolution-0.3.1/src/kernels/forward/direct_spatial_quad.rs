use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    prelude::*,
    std::tensor::layout::linear::{LinearViewMut, linear_view},
    tensor_vector_size_parallel,
};

use super::direct::DirectTensors;
use crate::{components::ConvSetupError, launch::ConvolutionArgs};

#[cube(launch_unchecked, address_type = "dynamic")]
fn spatial_quad_kernel<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    bias: ComptimeOption<&[Vector<E, NOut>]>,
    mut output: LinearViewMut<'_, Vector<E, NOut>>,
    out_h: u32,
    out_w: u32,
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: i32,
    padding_w: i32,
    out_channels_per_group: u32,
    #[define(E)] _dtype: ElemType,
) {
    let vector_in = input.vector_size();
    let vector_out = output.vector_size();
    let out_channel_vectors = weight.shape(0) as u32 / vector_out as u32;
    let tiles_h = (out_h + 1) / 2;
    let tiles_w = (out_w + 1) / 2;
    let unit = ABSOLUTE_POS as u32;
    let total = input.shape(0) as u32 * tiles_h * tiles_w * out_channel_vectors;
    if unit >= total {
        terminate!();
    }

    let channel_vector = unit % out_channel_vectors;
    let tile = unit / out_channel_vectors;
    let batch = tile / (tiles_h * tiles_w);
    let tile_y = tile / tiles_w % tiles_h;
    let tile_x = tile % tiles_w;
    let y0 = tile_y * 2;
    let y1 = y0 + 1;
    let x0 = tile_x * 2;
    let x1 = x0 + 1;
    let oc_start = channel_vector * vector_out as u32;
    let group = oc_start / out_channels_per_group;
    let channels_in_group = weight.shape(3) as u32;
    let channel_start = group * channels_in_group;
    let input_h = input.shape(1) as u32;
    let input_w = input.shape(2) as u32;
    let weight_h = weight.shape(1) as u32;
    let weight_w = weight.shape(2) as u32;
    let initial: ComptimeOption<Vector<E, NOut>> = bias.map(|b| b[channel_vector as usize]);
    let mut sum00 = initial.unwrap_or_else(|| Vector::zero());
    let mut sum01 = sum00;
    let mut sum10 = sum00;
    let mut sum11 = sum00;

    for kh in 0..weight_h {
        let iy0 = (y0 * stride_h + kh * dilation_h) as i32 - padding_h;
        let iy1 = (y1 * stride_h + kh * dilation_h) as i32 - padding_h;
        let vy0 = iy0 >= 0 && (iy0 as u32) < input_h;
        let vy1 = y1 < out_h && iy1 >= 0 && (iy1 as u32) < input_h;

        for kw in 0..weight_w {
            let ix0 = (x0 * stride_w + kw * dilation_w) as i32 - padding_w;
            let ix1 = (x1 * stride_w + kw * dilation_w) as i32 - padding_w;
            let vx0 = ix0 >= 0 && (ix0 as u32) < input_w;
            let vx1 = x1 < out_w && ix1 >= 0 && (ix1 as u32) < input_w;
            let v00 = vy0 && vx0;
            let v01 = vy0 && vx1;
            let v10 = vy1 && vx0;
            let v11 = vy1 && vx1;
            if v00 || v01 || v10 || v11 {
                for ci in range_stepped(0, channels_in_group, vector_in as u32) {
                    let mut a00 = Vector::<E, NIn>::zero();
                    let mut a01 = Vector::<E, NIn>::zero();
                    let mut a10 = Vector::<E, NIn>::zero();
                    let mut a11 = Vector::<E, NIn>::zero();
                    let channel_index = (channel_start + ci) as usize;
                    let base = batch as usize * input.stride(0) + channel_index;
                    if v00 {
                        let i =
                            base + iy0 as usize * input.stride(1) + ix0 as usize * input.stride(2);
                        a00 = input[i / vector_in];
                    }
                    if v01 {
                        let i =
                            base + iy0 as usize * input.stride(1) + ix1 as usize * input.stride(2);
                        a01 = input[i / vector_in];
                    }
                    if v10 {
                        let i =
                            base + iy1 as usize * input.stride(1) + ix0 as usize * input.stride(2);
                        a10 = input[i / vector_in];
                    }
                    if v11 {
                        let i =
                            base + iy1 as usize * input.stride(1) + ix1 as usize * input.stride(2);
                        a11 = input[i / vector_in];
                    }
                    #[unroll]
                    for lane in 0..vector_out {
                        let oc = oc_start as usize + lane;
                        let wi = oc * weight.stride(0)
                            + kh as usize * weight.stride(1)
                            + kw as usize * weight.stride(2)
                            + ci as usize;
                        let w = weight[wi / vector_in];
                        let mut r00 = sum00.extract(lane);
                        let mut r01 = sum01.extract(lane);
                        let mut r10 = sum10.extract(lane);
                        let mut r11 = sum11.extract(lane);
                        #[unroll]
                        for j in 0..vector_in {
                            let wj = w.extract(j);
                            if v00 {
                                r00 += a00.extract(j) * wj;
                            }
                            if v01 {
                                r01 += a01.extract(j) * wj;
                            }
                            if v10 {
                                r10 += a10.extract(j) * wj;
                            }
                            if v11 {
                                r11 += a11.extract(j) * wj;
                            }
                        }
                        sum00.insert(lane, r00);
                        sum01.insert(lane, r01);
                        sum10.insert(lane, r10);
                        sum11.insert(lane, r11);
                    }
                }
            }
        }
    }
    let plane = out_w as usize * out_channel_vectors as usize;
    let base = batch as usize * out_h as usize * plane
        + y0 as usize * plane
        + x0 as usize * out_channel_vectors as usize
        + channel_vector as usize;
    output.write(base, sum00);
    if x1 < out_w {
        output.write(base + out_channel_vectors as usize, sum01);
    }
    if y1 < out_h {
        output.write(base + plane, sum10);
        if x1 < out_w {
            output.write(base + plane + out_channel_vectors as usize, sum11);
        }
    }
}

pub fn launch_spatial_quad<const N: usize>(
    client: &Client,
    tensors: DirectTensors,
    args: ConvolutionArgs<N>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    let DirectTensors {
        input,
        weight,
        bias,
        out,
    } = tensors;
    if N != 2
        || groups == 0
        || input.shape.len() != 4
        || weight.shape.len() != 4
        || out.shape.len() != 4
        || weight.shape[0] % groups != 0
        || input.shape[3] != weight.shape[3] * groups
        || out.shape[3] != weight.shape[0]
    {
        return Err(ConvSetupError::Groups(groups));
    }
    if out.shape.iter().any(|&n| n == 0) {
        return Ok(());
    }
    if input.strides[3] != 1 || weight.strides[3] != 1 || out.strides[3] != 1 {
        return Err(ConvSetupError::Unknown);
    }
    let channels_out_group = weight.shape[0] / groups;
    let mut grouped_shape = out.shape.clone();
    grouped_shape[3] = channels_out_group;
    let vector_out = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &grouped_shape,
        &out.strides,
        3,
    );
    let vector_in = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &weight.shape,
        &weight.strides,
        3,
    );
    if channels_out_group % vector_out != 0 || weight.shape[3] % vector_in != 0 {
        return Err(ConvSetupError::Unknown);
    }
    let work_items = out.shape[0]
        * out.shape[1].div_ceil(2)
        * out.shape[2].div_ceil(2)
        * (out.shape[3] / vector_out);
    let cube_dim = CubeDim::new(client, work_items);
    let cube_count = calculate_cube_count_elemwise(client, work_items, cube_dim);
    let address_type = input
        .required_address_type(dtype.size())
        .max(weight.required_address_type(dtype.size()))
        .max(out.required_address_type(dtype.size()));
    let out_h = out.shape[1] as u32;
    let out_w = out.shape[2] as u32;
    unsafe {
        spatial_quad_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_in,
            vector_out,
            input.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.map(|b| b.into_buffer_arg()).into(),
            linear_view(out),
            out_h,
            out_w,
            args.stride[0] as u32,
            args.stride[1] as u32,
            args.dilation[0] as u32,
            args.dilation[1] as u32,
            args.padding[0] as i32,
            args.padding[1] as i32,
            channels_out_group as u32,
            dtype,
        );
    }
    Ok(())
}
