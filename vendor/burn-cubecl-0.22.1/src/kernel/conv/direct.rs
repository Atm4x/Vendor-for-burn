use crate::{
    kernel::into_contiguous_aligned, ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubek::convolution::{
    ConvolutionArgs, DirectTensors, components::ConvSetupError, launch_direct,
    forward::{launch_prepacked_temporal_pair, launch_prepacked_temporal_quad,
        launch_prepacked_temporal_quad_interior},
};

/// Perform a convolution using the direct convolution algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_direct<const N: usize>(
    mut input: CubeTensor,
    mut weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let out_dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let dim_c = rank - 1;

    // We only care about the channels here, everything else can be permuted
    if input.meta.strides()[dim_c] != 1 {
        input = into_contiguous_aligned(input);
    }
    let selected = std::env::var("RVC_NSF_CONV_KERNEL").unwrap_or_default();
    let is_temporal = N == 1
        && options.groups == 1
        && matches!(selected.as_str(), "packed" | "quad" | "quad_interior")
        && weight.meta.shape().len() == 3
        && weight.meta.strides()[0] == 1
        && weight.meta.strides()[2] == weight.meta.shape()[0]
        && weight.meta.strides()[1]
            == weight.meta.shape()[0] * weight.meta.shape()[2]
        && weight.meta.shape().iter().all(|&dim| dim > 0);
    if !is_temporal && weight.meta.strides()[dim_c] != 1 {
        weight = into_contiguous_aligned(weight);
    }

    let batch_size = input.meta.shape()[0];
    let in_shape = &input.meta.shape()[1..dim_c];
    let out_channels = weight.meta.shape()[0];
    let kernel_shape = &weight.meta.shape()[1..dim_c];

    let out_size = calculate_conv_output_sizes(
        kernel_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        in_shape,
    );

    let mut shape_out = vec![batch_size];
    shape_out.extend(out_size.iter().copied());
    shape_out.push(out_channels);

    let out = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out.into(),
        out_dtype,
    );

    let padding = options.padding_begin();
    let args = ConvolutionArgs::<N> {
        stride: options.stride,
        padding: core::array::from_fn(|i| padding[i]),
        dilation: options.dilation,
    };

    let client = input.client.clone();
    let dtype = dtype_to_storage_type(out_dtype);

    if is_temporal {
        let cout = weight.meta.shape()[0];
        let kernel = weight.meta.shape()[1];
        let cin = weight.meta.shape()[2];
        let mut weight_binding = weight.binding();
        weight_binding.shape = vec![1, kernel, cin, cout].into();
        weight_binding.strides = vec![kernel * cin * cout, cin * cout, cout, 1].into();
        let tensors = DirectTensors {
            input: input.binding(),
            weight: weight_binding,
            bias: bias.map(|bias| bias.binding()),
            out: out.clone().binding(),
        };
        let temporal_args = ConvolutionArgs::<1> {
            stride: [args.stride[0]],
            dilation: [args.dilation[0]],
            padding: [args.padding[0]],
        };
        match selected.as_str() {
            "packed" => launch_prepacked_temporal_pair(&client, tensors, temporal_args, 1, dtype)?,
            "quad" => launch_prepacked_temporal_quad(&client, tensors, temporal_args, 1, dtype)?,
            "quad_interior" => launch_prepacked_temporal_quad_interior(&client, tensors, temporal_args, 1, dtype)?,
            _ => unreachable!(),
        }
    } else {
        let tensors = DirectTensors {
            input: input.binding(),
            weight: weight.binding(),
            bias: bias.map(|bias| bias.binding()),
            out: out.clone().binding(),
        };
        launch_direct::<N>(&client, tensors, args, options.groups, dtype)?;
    }

    Ok(out)
}
