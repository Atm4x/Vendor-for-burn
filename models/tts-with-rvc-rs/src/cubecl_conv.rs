use burn::tensor::{
    Tensor,
    ops::{ConvOptions, ConvTransposeOptions},
};
use burn_cubecl::{
    Cube,
    kernel::conv::{ConvStrategy, ConvTranspose2dStrategy, conv_forward, conv_forward_spatial_quad, conv_forward_spatial_reuse, conv_forward_spatial_pair, conv_transpose2d, prepack_winograd_weights, transform_winograd_input, inverse_winograd_output},
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


pub(crate) fn conv2d_spatial_pair(
    input: Tensor<4>, weight: Tensor<4>, bias: Option<Tensor<1>>,
    options: ConvOptions<2>,
) -> Tensor<4> {
    let input = input.try_into_primitive::<Cube>().expect("CubeCL spatial pair requires CubeCL input");
    let weight = weight.try_into_primitive::<Cube>().expect("CubeCL spatial pair requires CubeCL weight");
    let bias = bias.map(|v| v.try_into_primitive::<Cube>().expect("CubeCL spatial pair requires CubeCL bias"));
    Tensor::from_primitive::<Cube>(conv_forward_spatial_pair(input, weight, bias, options)
        .expect("Stage105 spatial pair unsupported on this shape"))
}

pub(crate) fn conv2d_spatial_reuse(
    input: Tensor<4>, weight: Tensor<4>, bias: Option<Tensor<1>>,
    options: ConvOptions<2>,
) -> Tensor<4> {
    let input = input.try_into_primitive::<Cube>()
        .expect("CubeCL spatial reuse needs CubeCL input");
    let weight = weight.try_into_primitive::<Cube>()
        .expect("CubeCL spatial reuse needs CubeCL weight");
    let bias = bias.map(|v| v.try_into_primitive::<Cube>()
        .expect("CubeCL spatial reuse needs CubeCL bias"));
    let out = conv_forward_spatial_reuse(input, weight, bias, options)
        .expect("CubeCL spatial reuse requires dense 3x3 stride 1 padding 1");
    Tensor::from_primitive::<Cube>(out)
}

pub(crate) fn conv2d_spatial_quad(
    input: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: ConvOptions<2>,
) -> Tensor<4> {
    let input = input.try_into_primitive::<Cube>()
        .expect("CubeCL spatial quad requires a CubeCL input");
    let weight = weight.try_into_primitive::<Cube>()
        .expect("CubeCL spatial quad requires a CubeCL weight");
    let bias = bias.map(|tensor| tensor.try_into_primitive::<Cube>()
        .expect("CubeCL spatial quad requires a CubeCL bias"));
    let result = conv_forward_spatial_quad(input, weight, bias, options)
        .expect("CubeCL spatial quad kernel is unsupported on this shape");
    Tensor::from_primitive::<Cube>(result)
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

pub(crate) fn conv2d_compensated_tf32_batched(
    input: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: ConvOptions<2>,
) -> Tensor<4> {
    let batch = input.dims()[0];
    let packed = crate::tf32_compensated::split_tf32_packed(input)
        .expect("TF32 packed convolution requires supported CubeCL tensor");
    let output = conv2d(packed, weight, None, options);
    let dims = output.dims();
    assert_eq!(dims[0], 2 * batch, "TF32 packed convolution doubled batch mismatch");
    let high = output.clone().slice([0..batch, 0..dims[1], 0..dims[2], 0..dims[3]]);
    let low = output.slice([batch..dims[0], 0..dims[1], 0..dims[2], 0..dims[3]]);
    let high = match bias {
        Some(bias) => high + bias.reshape([1, dims[1], 1, 1]),
        None => high,
    };
    high + low
}


pub(crate) fn prepack_winograd(weight: Tensor<4>) -> Tensor<3> {
    let primitive = weight.try_into_primitive::<Cube>()
        .expect("Winograd requires direct CubeCL weights");
    Tensor::from_primitive::<Cube>(prepack_winograd_weights(primitive, 1))
}

pub(crate) fn conv2d_winograd(
    input: Tensor<4>,
    packed_weights: Tensor<3>,
    bias: Option<Tensor<1>>,
    out_channels: usize,
) -> Tensor<4> {
    let [batch, _, h, w] = input.dims();
    let input = input.try_into_primitive::<Cube>()
        .expect("Winograd requires direct CubeCL input");
    let transformed = Tensor::<3>::from_primitive::<Cube>(transform_winograd_input(input, 1));
    let multiplied = transformed.matmul(packed_weights);
    let primitive = multiplied.try_into_primitive::<Cube>()
        .expect("Winograd product requires a direct CubeCL tensor");
    let bias = bias.map(|tensor| tensor.try_into_primitive::<Cube>()
        .expect("Winograd bias requires a direct CubeCL tensor"));
    Tensor::from_primitive::<Cube>(inverse_winograd_output(primitive, bias, [batch, out_channels, h, w], 1))
}

#[cfg(all(test, feature = "cpu"))]
mod winograd_tests {
    use super::*;
    use burn::tensor::{Device, TensorData, module};

    #[test]
    fn winograd_f2x2_3x3_matches_direct_on_odd_and_even_shapes() {
        for (n, cin, cout, h, w) in [
            (1, 1, 1, 1, 1),
            (1, 3, 7, 2, 3),
            (2, 4, 5, 7, 9),
            (1, 8, 8, 8, 6),
            (1, 16, 32, 13, 15),
        ] {
            let device = Device::default();
            let x: Vec<f32> = (0..n * cin * h * w)
                .map(|i| (i as f32 * 0.379).sin() * 0.21)
                .collect();
            let weight: Vec<f32> = (0..cout * cin * 9)
                .map(|i| (i as f32 * 0.317).cos() * 0.11)
                .collect();
            let bias: Vec<f32> = (0..cout).map(|i| i as f32 * 0.007).collect();
            let input = Tensor::<4>::from_data(TensorData::new(x, [n, cin, h, w]), &device);
            let weight = Tensor::<4>::from_data(TensorData::new(weight, [cout, cin, 3, 3]), &device);
            let bias = Tensor::<1>::from_data(TensorData::new(bias, [cout]), &device);
            let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
            let expected = module::conv2d(input.clone(), weight.clone(), Some(bias.clone()), options);
            let packed = prepack_winograd(weight);
            let output = conv2d_winograd(input, packed, Some(bias), cout);
            let actual: Vec<f32> = output.into_data().to_vec().unwrap();
            let reference: Vec<f32> = expected.into_data().to_vec().unwrap();
            let max_abs = actual.iter().zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 0.0005, "shape {n} {cin} {cout} {h} {w}: max_abs={max_abs}");
        }
    }
}
