use burn::tensor::{
    Tensor,
    ops::{ConvOptions, ConvTransposeOptions},
};
use burn_cubecl::{
    Cube,
    kernel::conv::{ConvStrategy, ConvTranspose2dStrategy, conv1d_preact_residual_temporal, prepare_resblock_input_ntc, conv1d_single_input_packed, conv1d_single_input_prepack_weight, conv_transpose1d_polyphase, conv_transpose1d_polyphase_packed, conv_transpose1d_prepack_weight, conv_transpose1d_prepack_weight_kic, conv_transpose1d_polyphase_single_packed_kic, prepack_conv_transpose1d_phase_gemm_weight, prepack_conv_transpose1d_phase_gemm_weight_compensated, conv_transpose1d_phase_gemm_packed, conv_transpose1d_phase_gemm_packed_compensated, conv_forward, conv_forward_spatial_quad, conv_forward_spatial_reuse, conv_forward_spatial_pair, conv_transpose2d, prepack_winograd_weights, transform_winograd_input, inverse_winograd_output},
};

pub(crate) fn prepare_resblock_ntc(input: Tensor<3>) -> Tensor<3> {
    let raw = input.try_into_primitive::<Cube>().expect("ResBlock NTC prepack requires CubeCL");
    Tensor::from_primitive::<Cube>(prepare_resblock_input_ntc(raw))
}

pub(crate) fn preact_resblock_conv(
    input: Tensor<3>, weight: Tensor<3>, bias: Option<Tensor<1>>,
    residual: Option<Tensor<3>>, dilation: usize, padding: usize,
) -> Tensor<3> {
    let input = input.try_into_primitive::<Cube>().expect("ResBlock input requires CubeCL");
    let weight = weight.try_into_primitive::<Cube>().expect("ResBlock weight requires CubeCL");
    let bias = bias.map(|b| b.try_into_primitive::<Cube>().expect("ResBlock bias requires CubeCL"));
    let residual = residual.map(|r| r.try_into_primitive::<Cube>().expect("ResBlock residual requires CubeCL"));
    Tensor::from_primitive::<Cube>(
        conv1d_preact_residual_temporal(input, weight, bias, residual, dilation, padding)
            .expect("ResBlock temporal transport unsupported shape"),
    )
}

pub(crate) fn prepack_single_input_conv1d_weight(weight: Tensor<3>) -> Tensor<3> {
    let raw = weight.try_into_primitive::<Cube>().expect("NSF noise prepack requires CubeCL");
    Tensor::from_primitive::<Cube>(conv1d_single_input_prepack_weight(raw))
}

pub(crate) fn conv1d_single_input(
    input: Tensor<3>, packed_weight: Tensor<3>, bias: Option<Tensor<1>>,
    options: ConvOptions<1>,
) -> Tensor<3> {
    let raw_input = input.try_into_primitive::<Cube>().expect("NSF noise requires CubeCL input");
    let raw_weight = packed_weight.try_into_primitive::<Cube>().expect("NSF noise requires packed CubeCL weight");
    let raw_bias = bias.map(|v| v.try_into_primitive::<Cube>().expect("NSF noise requires CubeCL bias"));
    Tensor::from_primitive::<Cube>(
        conv1d_single_input_packed(raw_input, raw_weight, raw_bias, options)
            .expect("NSF single-input-channel Conv1d unsupported shape")
    )
}

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


pub(crate) fn prepack_transpose1d_phase_gemm(
    weight: Tensor<3>, stride: usize, padding: usize, half: bool,
) -> Tensor<4> {
    let weight = if half { weight.cast(burn::tensor::DType::F16) } else { weight };
    let raw = weight.try_into_primitive::<Cube>().expect("phase GEMM weight requires CubeCL");
    Tensor::from_primitive::<Cube>(
        prepack_conv_transpose1d_phase_gemm_weight(raw, stride, padding, 1)
            .expect("phase GEMM prepack requires supported geometry"),
    )
}

pub(crate) fn prepack_transpose1d_phase_gemm_compensated(
    weight: Tensor<3>, stride: usize, padding: usize,
) -> (Tensor<4>, Tensor<4>) {
    let raw = weight.try_into_primitive::<Cube>().expect("phase GEMM weight requires CubeCL");
    let (high, low) = prepack_conv_transpose1d_phase_gemm_weight_compensated(raw, stride, padding, 1)
        .expect("phase GEMM TF32 compensated prepack requires FP32 weights");
    (Tensor::from_primitive::<Cube>(high), Tensor::from_primitive::<Cube>(low))
}

pub(crate) fn conv_transpose1d_phase_gemm(
    input: Tensor<3>, weight: Tensor<4>, weight_low: Option<Tensor<4>>, bias: Tensor<1>,
    stride: usize, padding: usize, kernel: usize, half: bool,
) -> Tensor<3> {
    let input = if half { input.cast(burn::tensor::DType::F16) } else { input };
    let bias = if half { bias.cast(burn::tensor::DType::F16) } else { bias };
    let raw_input = input.try_into_primitive::<Cube>().expect("phase GEMM input requires CubeCL");
    let raw_weight = weight.try_into_primitive::<Cube>().expect("phase GEMM weight requires CubeCL");
    let raw_bias = bias.try_into_primitive::<Cube>().expect("phase GEMM bias requires CubeCL");
    let result = Tensor::<3>::from_primitive::<Cube>(
        if let Some(weight_low) = weight_low {
            let raw_low = weight_low.try_into_primitive::<Cube>()
                .expect("phase GEMM low weight requires CubeCL");
            conv_transpose1d_phase_gemm_packed_compensated(
                raw_input, raw_weight, raw_low, Some(raw_bias), stride, padding, kernel, 1, 0, 1,
            )
        } else {
            conv_transpose1d_phase_gemm_packed(
                raw_input, raw_weight, Some(raw_bias), stride, padding, kernel, 1, 0, 1,
            )
        }.expect("phase GEMM unsupported geometry or backend"),
    );
    if half { result.cast(burn::tensor::DType::F32) } else { result }
}

pub(crate) fn prepack_transpose1d_weight(weight: Tensor<3>) -> Tensor<3> {
    let weight = weight.try_into_primitive::<Cube>().expect("polyphase prepack requires CubeCL");
    Tensor::from_primitive::<Cube>(conv_transpose1d_prepack_weight(weight))
}

pub(crate) fn prepack_transpose1d_weight_kic(weight: Tensor<3>) -> Tensor<3> {
    let weight = weight.try_into_primitive::<Cube>().expect("KIC prepack requires CubeCL");
    Tensor::from_primitive::<Cube>(conv_transpose1d_prepack_weight_kic(weight))
}

pub(crate) fn conv_transpose1d_phase_packed_kic(
    input: Tensor<3>, packed_weight: Tensor<3>, bias: Tensor<1>,
    stride: usize, padding: usize, contiguous_ntc: bool,
) -> Tensor<3> {
    let input = input.try_into_primitive::<Cube>().expect("KIC input requires CubeCL");
    let weight = packed_weight.try_into_primitive::<Cube>().expect("KIC packed weight requires CubeCL");
    let bias = bias.try_into_primitive::<Cube>().expect("KIC bias requires CubeCL");
    Tensor::from_primitive::<Cube>(
        conv_transpose1d_polyphase_single_packed_kic(
            input, weight, Some(bias), stride, padding, 1, 0, 1, contiguous_ntc,
        ).expect("KIC polyphase convolution unsupported on this shape")
    )
}

pub(crate) fn conv_transpose1d_phase_packed(
    input: Tensor<3>, packed_weight: Tensor<3>, bias: Tensor<1>,
    stride: usize, padding: usize,
) -> Tensor<3> {
    let input = input.try_into_primitive::<Cube>().expect("polyphase input requires CubeCL");
    let weight = packed_weight.try_into_primitive::<Cube>().expect("polyphase packed weight requires CubeCL");
    let bias = bias.try_into_primitive::<Cube>().expect("polyphase bias requires CubeCL");
    Tensor::from_primitive::<Cube>(
        conv_transpose1d_polyphase_packed(input, weight, Some(bias), stride, padding, 1, 0, 1)
            .expect("packed polyphase convolution unsupported on this shape")
    )
}

pub(crate) fn conv_transpose1d_phase(
    input: Tensor<3>, weight: Tensor<3>, bias: Tensor<1>,
    stride: usize, padding: usize,
) -> Tensor<3> {
    let input = input.try_into_primitive::<Cube>().expect("polyphase input requires CubeCL");
    let weight = weight.try_into_primitive::<Cube>().expect("polyphase weight requires CubeCL");
    let bias = bias.try_into_primitive::<Cube>().expect("polyphase bias requires CubeCL");
    Tensor::from_primitive::<Cube>(
        conv_transpose1d_polyphase(input, weight, Some(bias), stride, padding, 1, 0, 1)
            .expect("polyphase transposed convolution unsupported on this shape")
    )
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

#[cfg(all(test, feature = "cpu"))]
mod polyphase_tests {
    use super::*;
    use burn::tensor::{module, Device, TensorData};

    #[test]
    fn noise_single_input_tiled_matches_burn_cpu_reference() {
        for (batch, cout, time, kernel, stride, dilation, pad) in [
            (1, 1, 10, 3, 1, 1, 1),
            (1, 5, 33, 8, 4, 1, 3),
            (2, 7, 101, 16, 8, 1, 8),
            (1, 16, 39, 5, 1, 2, 4),
            (2, 32, 77, 20, 10, 1, 10),
        ] {
            let device = Device::cpu();
            let input = Tensor::<3>::from_data(TensorData::new(
                (0..batch * time).map(|i| (i as f32 * 0.17).cos() * 0.21).collect::<Vec<_>>(),
                [batch, 1, time],
            ), &device);
            let weight = Tensor::<3>::from_data(TensorData::new(
                (0..cout * kernel).map(|i| (i as f32 * 0.19).sin() * 0.13).collect::<Vec<_>>(),
                [cout, 1, kernel],
            ), &device);
            let bias = Tensor::<1>::from_data(TensorData::new(
                (0..cout).map(|i| i as f32 * 0.003).collect::<Vec<_>>(),
                [cout],
            ), &device);
            let options = ConvOptions::new_with_padding(
                [stride], [(pad, pad)], [dilation], 1,
            );
            let reference = module::conv1d(
                input.clone(), weight.clone(), Some(bias.clone()), options.clone(),
            );
            let packed = prepack_single_input_conv1d_weight(weight);
            let result = conv1d_single_input(input, packed, Some(bias), options);
            assert_eq!(result.dims(), reference.dims());
            let actual: Vec<f32> = result.into_data().try_into_vec::<f32>().unwrap();
            let expected: Vec<f32> = reference.into_data().try_into_vec::<f32>().unwrap();
            let max_abs = actual.iter().zip(expected.iter())
                .map(|(x,y)| (*x-*y).abs()).fold(0f32, f32::max);
            assert!(max_abs < 3e-5, "noise stride={stride} dilation={dilation} K={kernel} error={max_abs}");
        }
    }

    #[test]
    fn polyphase_transpose1d_matches_burn_cpu_reference() {
        for (batch, cin, cout, time, kernel, stride, padding) in [
            (1, 1, 1, 1, 2, 1, 0),
            (1, 3, 5, 9, 6, 2, 2),
            (2, 4, 7, 8, 8, 4, 2),
            (1, 5, 8, 3, 16, 8, 4),
            (1, 8, 16, 14, 3, 1, 1),
            (2, 1, 3, 2, 10, 5, 3),
        ] {
            let device = Device::default();
            let xs = (0..batch * cin * time)
                .map(|i| (i as f32 * 0.173).sin() * 0.4)
                .collect::<Vec<_>>();
            let ws = (0..cin * cout * kernel)
                .map(|i| (i as f32 * 0.137).cos() * 0.2)
                .collect::<Vec<_>>();
            let bs = (0..cout).map(|i| i as f32 * 0.01).collect::<Vec<_>>();
            let x = Tensor::<3>::from_data(
                TensorData::new(xs, [batch, cin, time]), &device,
            );
            let w = Tensor::<3>::from_data(
                TensorData::new(ws, [cin, cout, kernel]), &device,
            );
            let b = Tensor::<1>::from_data(TensorData::new(bs, [cout]), &device);
            let expected = module::conv_transpose1d(
                x.clone(), w.clone(), Some(b.clone()),
                ConvTransposeOptions::new([stride], [padding], [0], [1], 1),
            );
            let actual = conv_transpose1d_phase(x, w, b, stride, padding);
            assert_eq!(actual.dims(), expected.dims());
            let a: Vec<f32> = actual.into_data().to_vec().unwrap();
            let e: Vec<f32> = expected.into_data().to_vec().unwrap();
            let max_abs = a.iter().zip(e.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 3e-5, "N={batch} Cin={cin} Cout={cout} K={kernel} stride={stride}: error {max_abs}");
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod resblock_transport_tests {
    use super::*;
    use burn::tensor::{activation, module, Device, TensorData};
    use crate::conv1d_layout::{from_host_values, Conv1dWeightLayout};

    #[test]
    fn packed_preact_residual_matches_two_resblock_convs() {
        let device = Device::cpu();
        for (batch, channels, length, kernel, dilation) in [
            (1, 3, 19, 3, 1),
            (2, 7, 35, 7, 3),
            (1, 8, 16, 11, 5),
            (1, 32, 21, 11, 5),
        ] {
            let x_values = (0..batch * channels * length)
                .map(|i| ((i * 17 + 19) as f32 * 0.037).sin() * 0.38 - 0.12)
                .collect::<Vec<_>>();
            let w_values = (0..channels * channels * kernel)
                .map(|i| ((i * 5 + 11) as f32 * 0.043).cos() * 0.05)
                .collect::<Vec<_>>();
            let b_values = (0..channels)
                .map(|i| (i as f32 * 0.017).cos() * 0.011)
                .collect::<Vec<_>>();
            let x = Tensor::<3>::from_data(TensorData::new(
                x_values, [batch, channels, length],
            ), &device);
            let w = Tensor::<3>::from_data(TensorData::new(
                w_values.clone(), [channels, channels, kernel],
            ), &device);
            let packed = from_host_values(
                [channels, channels, kernel], w_values, &device, Conv1dWeightLayout::Temporal,
            );
            let b = Tensor::<1>::from_data(TensorData::new(b_values, [channels]), &device);
            let pad = (kernel - 1) * dilation / 2;
            let options = ConvOptions::new_with_padding([1], [(pad, pad)], [dilation], 1);

            let reference = module::conv1d(
                activation::leaky_relu(x.clone(), 0.1), w, Some(b.clone()), options,
            ) + x.clone();
            let transport = preact_resblock_conv(x.clone(), packed.clone(), Some(b.clone()),
                Some(x.clone()), dilation, pad);
            assert_eq!(transport.dims(), reference.dims());
            let expected = reference.into_data().try_into_vec::<f32>().unwrap();
            let actual = transport.into_data().try_into_vec::<f32>().unwrap();
            let max_abs = actual.iter().zip(expected.iter())
                .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(max_abs < 7e-5,
                "channels={channels} kernel={kernel} dilation={dilation} error={max_abs}");

            let shared_ntc = prepare_resblock_ntc(x.clone());
            let shared_logical = shared_ntc.clone().into_data().try_into_vec::<f32>().unwrap();
            let original_logical = x.clone().into_data().try_into_vec::<f32>().unwrap();
            assert_eq!(shared_logical, original_logical);
            let shared_result = preact_resblock_conv(
                shared_ntc.clone(), packed.clone(), Some(b.clone()), Some(shared_ntc),
                dilation, pad,
            );
            let shared_actual = shared_result.into_data().try_into_vec::<f32>().unwrap();
            let shared_max_abs = shared_actual.iter().zip(&expected)
                .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(shared_max_abs < 7e-5,
                "shared NTC channels={channels} kernel={kernel} dilation={dilation} error={shared_max_abs}");
            let first = preact_resblock_conv(x, packed, Some(b), None, dilation, pad);
            assert_eq!(first.dims(), [batch, channels, length]);
        }
    }
}
