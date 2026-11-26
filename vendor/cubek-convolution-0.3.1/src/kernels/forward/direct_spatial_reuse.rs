//! Opt-in exact 3x3 convolution research kernel: a 4x4 input footprint for four 2x2 outputs.
//! For each input channel vector, loads 16 input vectors rather than 36 load instructions;
//! weights are consumed once per 3x3 position for all four output accumulators.
//! The transformed loop order may alter the least significant FP32 bits.

use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    ir::FloatKind,
    prelude::*,
    std::tensor::layout::linear::{LinearViewMut, linear_view},
    tensor_vector_size_parallel,
};

use super::direct::DirectTensors;
use crate::{components::ConvSetupError, launch::ConvolutionArgs};

#[cube(launch_unchecked, address_type = "dynamic")]
fn spatial_reuse_3x3_kernel<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    bias: ComptimeOption<&[Vector<E, NOut>]>,
    mut output: LinearViewMut<'_, Vector<E, NOut>>,
    out_h: u32,
    out_w: u32,
    #[define(E)] _dtype: ElemType,
) {
    let vin = input.vector_size();
    let vout = output.vector_size();
    let channel_vectors = weight.shape(0) as u32 / vout as u32;
    let tiles_h = (out_h + 1) / 2;
    let tiles_w = (out_w + 1) / 2;
    let index = ABSOLUTE_POS as u32;
    let total = input.shape(0) as u32 * tiles_h * tiles_w * channel_vectors;
    if index >= total { terminate!(); }

    let cvec = index % channel_vectors;
    let tile = index / channel_vectors;
    let batch = tile / (tiles_h * tiles_w);
    let y = (tile / tiles_w % tiles_h) * 2;
    let x = (tile % tiles_w) * 2;
    let oc = cvec * vout as u32;
    let input_h = input.shape(1) as u32;
    let input_w = input.shape(2) as u32;
    let cin = weight.shape(3) as u32;
    let b: ComptimeOption<Vector<E, NOut>> = bias.map(|v| v[cvec as usize]);
    let mut s00 = b.unwrap_or_else(|| Vector::zero());
    let mut s01 = s00;
    let mut s10 = s00;
    let mut s11 = s00;

    // The entire 4x4 footprint lives within one thread's register array.
    // Loop interchange trades bitwise accumulation order for fewer L1/TEX requests.
    for ci in range_stepped(0, cin, vin as u32) {
        let mut footprint = Array::<Vector<E, NIn>>::new(16usize);
        #[unroll]
        for iy in 0..4u32 {
            #[unroll]
            for ix in 0..4u32 {
                let sy = y as i32 + iy as i32 - 1;
                let sx = x as i32 + ix as i32 - 1;
                let mut value = Vector::<E, NIn>::zero();
                if sy >= 0 && sx >= 0 && (sy as u32) < input_h && (sx as u32) < input_w {
                    let addr = batch as usize * input.stride(0)
                        + sy as usize * input.stride(1)
                        + sx as usize * input.stride(2)
                        + ci as usize;
                    value = input[addr / vin];
                }
                footprint[(iy * 4 + ix) as usize] = value;
            }
        }

        #[unroll]
        for kh in 0..3u32 {
            #[unroll]
            for kw in 0..3u32 {
                let a00 = footprint[(kh * 4 + kw) as usize];
                let a01 = footprint[(kh * 4 + kw + 1) as usize];
                let a10 = footprint[((kh + 1) * 4 + kw) as usize];
                let a11 = footprint[((kh + 1) * 4 + kw + 1) as usize];
                #[unroll]
                for lane in 0..vout {
                    let wi = (oc as usize + lane) * weight.stride(0)
                        + kh as usize * weight.stride(1)
                        + kw as usize * weight.stride(2)
                        + ci as usize;
                    let w = weight[wi / vin];
                    let mut r00 = s00.extract(lane);
                    let mut r01 = s01.extract(lane);
                    let mut r10 = s10.extract(lane);
                    let mut r11 = s11.extract(lane);
                    #[unroll]
                    for j in 0..vin {
                        let wc = w.extract(j);
                        r00 += a00.extract(j) * wc;
                        r01 += a01.extract(j) * wc;
                        r10 += a10.extract(j) * wc;
                        r11 += a11.extract(j) * wc;
                    }
                    s00.insert(lane, r00);
                    s01.insert(lane, r01);
                    s10.insert(lane, r10);
                    s11.insert(lane, r11);
                }
            }
        }
    }
    let pitch = out_w as usize * channel_vectors as usize;
    let dst = batch as usize * out_h as usize * pitch
        + y as usize * pitch + x as usize * channel_vectors as usize + cvec as usize;
    output.write(dst, s00);
    if x + 1 < out_w { output.write(dst + channel_vectors as usize, s01); }
    if y + 1 < out_h {
        output.write(dst + pitch, s10);
        if x + 1 < out_w { output.write(dst + pitch + channel_vectors as usize, s11); }
    }
}

pub fn launch_spatial_reuse<const N: usize>(
    client: &Client,
    tensors: DirectTensors,
    args: ConvolutionArgs<N>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    let DirectTensors { input, weight, bias, out } = tensors;
    if !matches!(dtype, ElemType::Float(FloatKind::F32)) || N != 2 || groups != 1 || input.shape.len() != 4 || weight.shape.len() != 4
        || out.shape.len() != 4 || weight.shape[1] != 3 || weight.shape[2] != 3
        || args.stride[0] != 1 || args.stride[1] != 1
        || args.dilation[0] != 1 || args.dilation[1] != 1
        || args.padding[0] != 1 || args.padding[1] != 1
        || input.shape[3] != weight.shape[3]
        || input.shape[1] != out.shape[1] || input.shape[2] != out.shape[2]
        || out.shape[3] != weight.shape[0]
        || input.strides[3] != 1 || weight.strides[3] != 1 || out.strides[3] != 1
    { return Err(ConvSetupError::Unknown); }
    if out.shape.iter().any(|&n| n == 0) { return Ok(()); }
    let vout = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &out.shape, &out.strides, 3,
    );
    let vin = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &weight.shape, &weight.strides, 3,
    );
    if weight.shape[0] % vout != 0 || weight.shape[3] % vin != 0 {
        return Err(ConvSetupError::Unknown);
    }
    let count = out.shape[0] * out.shape[1].div_ceil(2)
        * out.shape[2].div_ceil(2) * (out.shape[3] / vout);
    let dim = CubeDim::new(client, count);
    let cubes = calculate_cube_count_elemwise(client, count, dim);
    let addr = input.required_address_type(dtype.size())
        .max(weight.required_address_type(dtype.size()))
        .max(out.required_address_type(dtype.size()));
    let out_h = out.shape[1] as u32;
    let out_w = out.shape[2] as u32;
    unsafe {
        spatial_reuse_3x3_kernel::launch_unchecked(
            client, cubes, dim, addr, vin, vout,
            input.into_tensor_arg(), weight.into_tensor_arg(),
            bias.map(|b| b.into_buffer_arg()).into(), linear_view(out),
            out_h, out_w, dtype,
        );
    }
    Ok(())
}
