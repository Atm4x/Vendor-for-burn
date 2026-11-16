use crate::{kernel::into_contiguous, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{DType, cubecl::dtype_to_storage_type};
use cubek::convolution::{components::ConvSetupError, forward::launch_preact_residual_temporal};

use crate::ops::{permute_nchw_to_nhwc, permute_nhwc_to_nchw};

pub fn prepare_resblock_input_ntc(input: CubeTensor) -> CubeTensor {
    let input = permute_nchw_to_nhwc(input);
    let input = into_contiguous(input);
    permute_nhwc_to_nchw(input)
}

pub fn conv1d_preact_residual_temporal(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    residual: Option<CubeTensor>,
    dilation: usize,
    padding: usize,
) -> Result<CubeTensor, ConvSetupError> {
    if input.dtype != DType::F32
        || weight.dtype != DType::F32
        || bias.as_ref().is_some_and(|b| b.dtype != DType::F32)
        || residual.as_ref().is_some_and(|r| r.dtype != DType::F32)
    {
        return Err(ConvSetupError::Unknown);
    }
    let input = permute_nchw_to_nhwc(input);
    let weight = permute_nchw_to_nhwc(weight);
    let [batch, time, cin]: [usize; 3] = input.meta.shape().dims();
    let [cout, kernel, weight_cin]: [usize; 3] = weight.meta.shape().dims();
    if batch == 0
        || time == 0
        || cin == 0
        || cout == 0
        || kernel == 0
        || cin != weight_cin
        || kernel % 2 == 0
        || padding != ((kernel - 1) / 2) * dilation
        || dilation == 0
        || weight.meta.strides()[0] != 1
        || weight.meta.strides()[2] != cout
        || weight.meta.strides()[1] != cin * cout
        || bias.as_ref().is_some_and(|b| b.meta.shape()[0] != cout)
    {
        return Err(ConvSetupError::Unknown);
    }
    let residual = residual.map(|r| {
        let r = permute_nchw_to_nhwc(r);
        if r.meta.strides()[2] == 1
            && r.meta.strides()[1] == cout
            && r.meta.strides()[0] == time * cout
        {
            r
        } else {
            into_contiguous(r)
        }
    });
    if residual
        .as_ref()
        .is_some_and(|r| r.meta.shape().as_slice() != [batch, time, cout])
    {
        return Err(ConvSetupError::Unknown);
    }
    let out = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        [batch, time, cout].into(),
        DType::F32,
    );
    let mut packed = weight.binding();
    packed.shape = vec![1, kernel, cin, cout].into();
    packed.strides = vec![kernel * cin * cout, cin * cout, cout, 1].into();
    let client = input.client.clone();
    launch_preact_residual_temporal(
        &client,
        input.binding(),
        packed,
        bias.map(|b| b.binding()),
        residual.map(|r| r.binding()),
        out.clone().binding(),
        1,
        dilation,
        padding,
        dtype_to_storage_type(DType::F32),
    )?;
    Ok(permute_nhwc_to_nchw(out))
}
