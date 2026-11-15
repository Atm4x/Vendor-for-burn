use crate::{ops::numeric::empty_device_contiguous_dtype, tensor::CubeTensor};
use burn_backend::{DType, Shape, TensorMetadata};
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube]
fn g_coeff<F: Float>(r: u32, k: u32) -> F {
    if r == 0 {
        if k == 0 { F::new(1.0) } else { F::new(0.0) }
    } else if r == 3 {
        if k == 2 { F::new(1.0) } else { F::new(0.0) }
    } else if r == 1 {
        F::new(0.5)
    } else if k == 1 {
        F::new(-0.5)
    } else {
        F::new(0.5)
    }
}

#[cube]
fn bt_coeff<F: Float>(r: u32, k: u32) -> F {
    if r == 0 {
        if k == 0 { F::new(1.0) } else if k == 2 { F::new(-1.0) } else { F::new(0.0) }
    } else if r == 1 {
        if k == 1 || k == 2 { F::new(1.0) } else { F::new(0.0) }
    } else if r == 2 {
        if k == 1 { F::new(-1.0) } else if k == 2 { F::new(1.0) } else { F::new(0.0) }
    } else if k == 1 { F::new(1.0) } else if k == 3 { F::new(-1.0) } else { F::new(0.0) }
}

#[cube]
fn at_coeff<F: Float>(r: u32, k: u32) -> F {
    if r == 0 {
        if k == 3 { F::new(0.0) } else { F::new(1.0) }
    } else if k == 0 {
        F::new(0.0)
    } else if k == 1 {
        F::new(1.0)
    } else {
        F::new(-1.0)
    }
}

#[cube(launch, address_type = "dynamic")]
fn prepack_weight_kernel<F: Float>(
    weight: &Tensor<F>,
    packed: &mut Tensor<F>,
    cin_group: u32,
    cout_group: u32,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS as u32;
    let count = weight.shape(0) as u32 * cin_group;
    if pos >= count { terminate!(); }
    let ic = pos % cin_group;
    let oc = pos / cin_group;
    let group = oc / cout_group;
    let local_oc = oc % cout_group;

    let mut g = Array::<F>::new(9usize);
    for y in 0..3u32 {
        for x in 0..3u32 {
            let idx = oc as usize * weight.stride(0) + ic as usize * weight.stride(1)
                + y as usize * weight.stride(2) + x as usize * weight.stride(3);
            g[(y * 3 + x) as usize] = weight[idx];
        }
    }
    let mut tmp = Array::<F>::new(12usize);
    for y in 0..4u32 {
        for x in 0..3u32 {
            let mut sum = F::new(0.0);
            for k in 0..3u32 {
                sum += g_coeff::<F>(y, k) * g[(k * 3 + x) as usize];
            }
            tmp[(y * 3 + x) as usize] = sum;
        }
    }
    for y in 0..4u32 {
        for x in 0..4u32 {
            let mut sum = F::new(0.0);
            for k in 0..3u32 {
                sum += tmp[(y * 3 + k) as usize] * g_coeff::<F>(x, k);
            }
            let p = (group * 16 + y * 4 + x) as usize * packed.stride(0)
                + ic as usize * packed.stride(1) + local_oc as usize * packed.stride(2);
            packed[p] = sum;
        }
    }
}

#[cube(launch, address_type = "dynamic")]
fn transform_input_kernel<F: Float>(
    input: &Tensor<F>,
    transformed: &mut Tensor<F>,
    groups: u32,
    cin_group: u32,
    tiles_h: u32,
    tiles_w: u32,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS as u32;
    let tile_count = tiles_h * tiles_w;
    let count = input.shape(0) as u32 * groups * cin_group * tile_count;
    if pos >= count { terminate!(); }
    let ic = pos % cin_group;
    let t = pos / cin_group % tile_count;
    let group = pos / (cin_group * tile_count) % groups;
    let batch = pos / (cin_group * tile_count * groups);
    let ty = t / tiles_w;
    let tx = t % tiles_w;
    let h = input.shape(2) as u32;
    let w = input.shape(3) as u32;
    let mut tile = Array::<F>::new(16usize);
    for y in 0..4u32 {
        let iy = (ty * 2 + y) as i32 - 1;
        for x in 0..4u32 {
            let ix = (tx * 2 + x) as i32 - 1;
            let mut value = F::new(0.0);
            if iy >= 0 && (iy as u32) < h && ix >= 0 && (ix as u32) < w {
                let offset = batch as usize * input.stride(0)
                    + (group * cin_group + ic) as usize * input.stride(1)
                    + iy as usize * input.stride(2) + ix as usize * input.stride(3);
                value = input[offset];
            }
            tile[(y * 4 + x) as usize] = value;
        }
    }
    let mut tmp = Array::<F>::new(16usize);
    for y in 0..4u32 {
        for x in 0..4u32 {
            let mut sum = F::new(0.0);
            for k in 0..4u32 {
                sum += bt_coeff::<F>(y, k) * tile[(k * 4 + x) as usize];
            }
            tmp[(y * 4 + x) as usize] = sum;
        }
    }
    for y in 0..4u32 {
        for x in 0..4u32 {
            let mut sum = F::new(0.0);
            for k in 0..4u32 {
                sum += tmp[(y * 4 + k) as usize] * bt_coeff::<F>(x, k);
            }
            let off = (group * 16 + y * 4 + x) as usize * transformed.stride(0)
                + (batch * tile_count + t) as usize * transformed.stride(1)
                + ic as usize * transformed.stride(2);
            transformed[off] = sum;
        }
    }
}

#[cube(launch, address_type = "dynamic")]
fn inverse_kernel<F: Float>(
    multiplied: &Tensor<F>,
    bias: ComptimeOption<&Tensor<F>>,
    output: &mut Tensor<F>,
    groups: u32,
    cout_group: u32,
    tiles_h: u32,
    tiles_w: u32,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS as u32;
    let tile_count = tiles_h * tiles_w;
    let count = output.shape(0) as u32 * groups * cout_group * tile_count;
    if pos >= count { terminate!(); }
    let oc = pos % cout_group;
    let t = pos / cout_group % tile_count;
    let group = pos / (cout_group * tile_count) % groups;
    let batch = pos / (cout_group * tile_count * groups);
    let ty = t / tiles_w;
    let tx = t % tiles_w;
    let oc_global = group * cout_group + oc;
    let mut m = Array::<F>::new(16usize);
    for y in 0..4u32 {
        for x in 0..4u32 {
            let off = (group * 16 + y * 4 + x) as usize * multiplied.stride(0)
                + (batch * tile_count + t) as usize * multiplied.stride(1)
                + oc as usize * multiplied.stride(2);
            m[(y * 4 + x) as usize] = multiplied[off];
        }
    }
    let mut tmp = Array::<F>::new(8usize);
    for y in 0..2u32 {
        for x in 0..4u32 {
            let mut sum = F::new(0.0);
            for k in 0..4u32 { sum += at_coeff::<F>(y, k) * m[(k * 4 + x) as usize]; }
            tmp[(y * 4 + x) as usize] = sum;
        }
    }
    let initial: ComptimeOption<F> = bias.map(|b| b[oc_global as usize]);
    let initial = initial.unwrap_or_else(|| F::new(0.0));
    for y in 0..2u32 {
        for x in 0..2u32 {
            let oy = ty * 2 + y;
            let ox = tx * 2 + x;
            if oy < output.shape(2) as u32 && ox < output.shape(3) as u32 {
                let mut sum = initial;
                for k in 0..4u32 {
                    sum += tmp[(y * 4 + k) as usize] * at_coeff::<F>(x, k);
                }
                let off = batch as usize * output.stride(0) + oc_global as usize * output.stride(1)
                    + oy as usize * output.stride(2) + ox as usize * output.stride(3);
                output[off] = sum;
            }
        }
    }
}

pub fn prepack_winograd_weights(weight: CubeTensor, groups: usize) -> CubeTensor {
    let [cout, cin_g, kh, kw] = weight.shape().dims();
    assert_eq!((kh, kw), (3, 3));
    assert!(groups > 0 && cout.is_multiple_of(groups));
    assert_eq!(weight.dtype, DType::F32);
    let cout_g = cout / groups;
    let out = empty_device_contiguous_dtype(weight.client.clone(), weight.device.clone(), Shape::new([groups * 16, cin_g, cout_g]), weight.dtype);
    let count = cout * cin_g;
    let cube_dim = CubeDim::new(&weight.client, count);
    prepack_weight_kernel::launch(&weight.client.clone(), calculate_cube_count_elemwise(&weight.client, count, cube_dim), cube_dim,
        weight.required_address_type().max(out.required_address_type()),
        weight.into_tensor_arg(), out.clone().into_tensor_arg(), cin_g as u32, cout_g as u32, dtype_to_storage_type(DType::F32));
    out
}

pub fn transform_winograd_input(input: CubeTensor, groups: usize) -> CubeTensor {
    let [batch, cin, h, w] = input.shape().dims();
    assert!(groups > 0 && cin.is_multiple_of(groups));
    assert_eq!(input.dtype, DType::F32);
    let cin_g = cin / groups;
    let (tiles_h, tiles_w) = (h.div_ceil(2), w.div_ceil(2));
    let out = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), Shape::new([groups * 16, batch * tiles_h * tiles_w, cin_g]), input.dtype);
    let count = batch * groups * tiles_h * tiles_w * cin_g;
    let cube_dim = CubeDim::new(&input.client, count);
    transform_input_kernel::launch(&input.client.clone(), calculate_cube_count_elemwise(&input.client, count, cube_dim), cube_dim,
        input.required_address_type().max(out.required_address_type()),
        input.into_tensor_arg(), out.clone().into_tensor_arg(), groups as u32, cin_g as u32, tiles_h as u32, tiles_w as u32, dtype_to_storage_type(DType::F32));
    out
}

pub fn inverse_winograd_output(
    multiplied: CubeTensor,
    bias: Option<CubeTensor>,
    shape: [usize; 4],
    groups: usize,
) -> CubeTensor {
    let [batch, cout, h, w] = shape;
    assert!(groups > 0 && cout.is_multiple_of(groups));
    assert_eq!(multiplied.dtype, DType::F32);
    let (tiles_h, tiles_w) = (h.div_ceil(2), w.div_ceil(2));
    assert_eq!(multiplied.shape().dims::<3>(), [groups * 16, batch * tiles_h * tiles_w, cout / groups]);
    let out = empty_device_contiguous_dtype(multiplied.client.clone(), multiplied.device.clone(), Shape::new(shape), multiplied.dtype);
    let count = batch * cout * tiles_h * tiles_w;
    let cube_dim = CubeDim::new(&multiplied.client, count);
    inverse_kernel::launch(&multiplied.client.clone(), calculate_cube_count_elemwise(&multiplied.client, count, cube_dim), cube_dim,
        multiplied.required_address_type().max(out.required_address_type()),
        multiplied.into_tensor_arg(), bias.map(|b| b.into_tensor_arg()).into(), out.clone().into_tensor_arg(),
        groups as u32, (cout / groups) as u32, tiles_h as u32, tiles_w as u32, dtype_to_storage_type(DType::F32));
    out
}
