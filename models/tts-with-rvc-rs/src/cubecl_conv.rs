use burn::tensor::{
    Tensor,
    ops::{ConvOptions, ConvTransposeOptions},
};
use burn_cubecl::{
    Cube,
    kernel::conv::{ConvStrategy, ConvTranspose2dStrategy, conv_forward, conv_transpose2d},
};

pub(crate) fn conv1d(
    input: Tensor<3>,
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    options: ConvOptions<1>,
) -> Tensor<3> {
    let input = input
        .try_into_primitive::<Cube>()
        .expect("CubeCL GEMM requires a direct CubeCL input");
    let weight = weight
        .try_into_primitive::<Cube>()
        .expect("CubeCL GEMM requires a direct CubeCL weight");
    let bias = bias.map(|tensor| {
        tensor
            .try_into_primitive::<Cube>()
            .expect("CubeCL GEMM requires a direct CubeCL bias")
    });
    let output = conv_forward::<1>(input, weight, bias, options, ConvStrategy::ImplicitGemm)
        .expect("CubeCL GEMM convolution is unsupported on this backend or shape");
    Tensor::from_primitive::<Cube>(output)
}

pub(crate) fn conv_transpose1d(
    input: Tensor<3>,
    weight: Tensor<3>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
) -> Tensor<3> {
    let [batch, channels, time] = input.dims();
    let [weight_channels, output_channels, kernel] = weight.dims();
    assert_eq!(weight_channels, channels);
    let output_time = (time - 1) * stride + kernel - 2 * padding;
    let input = input
        .reshape([batch, channels, time, 1])
        .try_into_primitive::<Cube>()
        .expect("CubeCL GEMM transpose requires a direct CubeCL input");
    let weight = weight
        .reshape([channels, output_channels, kernel, 1])
        .try_into_primitive::<Cube>()
        .expect("CubeCL GEMM transpose requires a direct CubeCL weight");
    let bias = bias
        .try_into_primitive::<Cube>()
        .expect("CubeCL GEMM transpose requires a direct CubeCL bias");
    let output = conv_transpose2d(
        input,
        weight,
        Some(bias),
        ConvTransposeOptions::new([stride, 1], [padding, 0], [0, 0], [1, 1], 1),
        ConvTranspose2dStrategy::Gemm,
    )
    .expect("CubeCL GEMM transposed convolution is unsupported on this backend or shape");
    Tensor::<4>::from_primitive::<Cube>(output).reshape([batch, output_channels, output_time])
}


pub(crate) fn conv2d(
    input: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: ConvOptions<2>,
) -> Tensor<4> {
    let input = input
        .try_into_primitive::<Cube>()
        .expect("CubeCL 2D GEMM requires a direct CubeCL input");
    let weight = weight
        .try_into_primitive::<Cube>()
        .expect("CubeCL 2D GEMM requires a direct CubeCL weight");
    let bias = bias.map(|tensor| {
        tensor
            .try_into_primitive::<Cube>()
            .expect("CubeCL 2D GEMM requires a direct CubeCL bias")
    });
    let result = conv_forward::<2>(input, weight, bias, options, ConvStrategy::ImplicitGemm)
        .expect("CubeCL 2D implicit GEMM is unsupported on this backend or shape");
    Tensor::from_primitive::<Cube>(result)
}

pub(crate) fn conv_transpose2d_gemm(
    input: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: ConvTransposeOptions<2>,
) -> Tensor<4> {
    let input = input
        .try_into_primitive::<Cube>()
        .expect("CubeCL 2D GEMM transpose requires a direct CubeCL input");
    let weight = weight
        .try_into_primitive::<Cube>()
        .expect("CubeCL 2D GEMM transpose requires a direct CubeCL weight");
    let bias = bias.map(|tensor| {
        tensor
            .try_into_primitive::<Cube>()
            .expect("CubeCL 2D GEMM transpose requires a direct CubeCL bias")
    });
    let result = conv_transpose2d(input, weight, bias, options, ConvTranspose2dStrategy::Gemm)
        .expect("CubeCL 2D GEMM transpose is unsupported on this backend or shape");
    Tensor::from_primitive::<Cube>(result)
}


pub(crate) fn conv2d_compensated_tf32(
    input: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: ConvOptions<2>,
) -> Tensor<4> {
    let (high, low) = crate::tf32_compensated::split_tf32(input)
        .expect("TF32 compensated convolution requires a supported CubeCL tensor");
    let high_result = conv2d(high, weight.clone(), bias, options.clone());
    let low_result = conv2d(low, weight, None, options);
    high_result + low_result
}
