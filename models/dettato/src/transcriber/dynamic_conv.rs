use burn::cubecl::{self, prelude::*};
use burn::tensor::{DType, Tensor};
use burn_cubecl::{Cube, CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use burn_fusion::custom::{
    CustomOpIr, HandleContainer, OperationFn, OperationIr, OperationOutput, StreamId, TensorIr,
};

#[cube(launch_unchecked)]
fn extract_windows(
    input: &[f32],
    output: &mut [f32],
    first: u32,
    elements: u32,
    input_h: u32,
    input_w: u32,
    channels: u32,
    out_h: u32,
    out_w: u32,
    kernel_w: u32,
    patch_width: u32,
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    pad_h: u32,
    pad_w: u32,
) {
    let index = ABSOLUTE_POS as u32;
    if index < elements {
        let pos = first + index / patch_width;
        let tap = index % patch_width;
        let channel = tap % channels;
        let tap_w = (tap / channels) % kernel_w;
        let tap_h = tap / (channels * kernel_w);
        let batch = pos / (out_h * out_w);
        let h = ((pos / out_w) % out_h) * stride_h + tap_h * dilation_h;
        let w = (pos % out_w) * stride_w + tap_w * dilation_w;
        let mut value = 0.0_f32;
        if h >= pad_h && w >= pad_w && h - pad_h < input_h && w - pad_w < input_w {
            let offset =
                (((batch * input_h + h - pad_h) * input_w + w - pad_w) * channels) + channel;
            value = input[offset as usize];
        }
        output[index as usize] = value;
    }
}

pub(super) struct Geometry {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
}

impl Geometry {
    fn output_dim(&self, dim: usize, input: usize) -> Option<usize> {
        let reach = self.dilation[dim].checked_mul(self.kernel[dim].checked_sub(1)?)?;
        input
            .checked_add(self.padding[dim].checked_mul(2)?)?
            .checked_sub(reach.checked_add(1)?)?
            .checked_div(self.stride[dim])?
            .checked_add(1)
    }
}

pub(super) fn forward(
    input: Tensor<4>,
    weights: &Tensor<2>,
    bias: Option<&Tensor<1>>,
    geometry: &Geometry,
) -> Option<Tensor<4>> {
    let [batch, channels, height, width] = input.dims();
    let out_h = geometry.output_dim(0, height)?;
    let out_w = geometry.output_dim(1, width)?;
    let rows = batch.checked_mul(out_h)?.checked_mul(out_w)?;
    let patch_width = channels
        .checked_mul(geometry.kernel[0])?
        .checked_mul(geometry.kernel[1])?;
    let [inner, out_channels] = weights.dims();
    if rows == 0
        || patch_width == 0
        || inner != patch_width
        || input.dtype() != DType::F32
        || weights.dtype() != DType::F32
        || batch
            .checked_mul(channels)?
            .checked_mul(height)?
            .checked_mul(width)?
            > u32::MAX as usize
        || rows > u32::MAX as usize
        || height.checked_add(geometry.padding[0].checked_mul(2)?)? > u32::MAX as usize
        || width.checked_add(geometry.padding[1].checked_mul(2)?)? > u32::MAX as usize
    {
        return None;
    }

    let input = input
        .permute([0, 2, 3, 1])
        .try_into_primitive::<Cube>()
        .ok()?;
    let client = input.client.clone();
    if !matches!(
        client.device(),
        cubecl::Device::Cuda(_) | cubecl::Device::Wgpu(_)
    ) {
        return None;
    }
    let [stride_h, stride_w] = geometry.stride.map(|n| n as u32);
    let [pad_h, pad_w] = geometry.padding.map(|n| n as u32);
    let [dilation_h, dilation_w] = geometry.dilation.map(|n| n as u32);
    let kernel_w = geometry.kernel[1] as u32;
    // Keep each lowered matrix below 16 MiB on either runtime.
    let tile = (4 * 1024 * 1024usize).checked_div(patch_width)?;
    if tile == 0 {
        return None;
    }
    let mut output = Vec::with_capacity(rows.div_ceil(tile));
    for first in (0..rows).step_by(tile) {
        let count = tile.min(rows - first);
        let elements = count * patch_width;
        let desc = CustomOpIr::new(
            "dettato_extract_conv_windows",
            &[input.clone().into_ir()],
            &[TensorIr::uninit(
                client.create_empty_handle(),
                [count, patch_width].into(),
                DType::F32,
            )],
        );
        let exec = desc.clone();
        let op = OperationFn(move |handles: &mut HandleContainer<_>| {
            let ([input], [output]) = exec.as_fixed::<1, 1>();
            let input = into_contiguous(handles.get_float_tensor::<CubeBackend>(input));
            let result = input.client.empty(elements * size_of::<f32>());
            // The matrix shape is bounded above, and out-of-range input coordinates are zero padded.
            unsafe {
                extract_windows::launch_unchecked(
                    &input.client,
                    CubeCount::Static((elements as u32).div_ceil(128), 1, 1),
                    CubeDim::new_1d(128),
                    BufferArg::from_raw_parts(
                        input.handle.clone(),
                        batch * height * width * channels,
                    ),
                    BufferArg::from_raw_parts(result.clone(), elements),
                    first as u32,
                    elements as u32,
                    height as u32,
                    width as u32,
                    channels as u32,
                    out_h as u32,
                    out_w as u32,
                    kernel_w,
                    patch_width as u32,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    pad_h,
                    pad_w,
                );
            }
            let matrix = CubeTensor::new_contiguous(
                input.client,
                input.device,
                output.shape.clone(),
                result,
                DType::F32,
            );
            handles.register_float_tensor::<CubeBackend>(&output.id, matrix);
            Ok(())
        });
        let handle = client
            .register(StreamId::current(), OperationIr::Custom(desc), op)
            .output();
        output.push(Tensor::<2>::from_primitive::<Cube>(handle).matmul(weights.clone()));
    }
    let output = Tensor::cat(output, 0);
    let output = if let Some(bias) = bias {
        output + bias.clone().reshape([1, out_channels])
    } else {
        output
    };
    Some(
        output
            .reshape([batch, out_h, out_w, out_channels])
            .permute([0, 3, 1, 2]),
    )
}

#[cfg(all(test, feature = "cuda", feature = "wgpu"))]
mod tests {
    use super::*;
    use burn::tensor::{Device, DeviceKind, TensorData, module, ops::ConvOptions};

    #[test]
    #[ignore = "requires CUDA and a discrete WGPU adapter"]
    fn strided_dilated_convolution_matches_burn_at_multiple_lengths() {
        let geometry = Geometry {
            kernel: [3, 2],
            stride: [2, 1],
            padding: [2, 1],
            dilation: [2, 1],
        };
        for device in [Device::cuda(0), Device::wgpu(DeviceKind::DiscreteGpu(0))] {
            for height in [9, 13, 27] {
                let input = (0..2 * 3 * height * 7)
                    .map(|i| (i as f32 * 0.071).sin())
                    .collect::<Vec<_>>();
                let filter = (0..5 * 3 * 3 * 2)
                    .map(|i| (i as f32 * 0.091).cos() * 0.1)
                    .collect::<Vec<_>>();
                let input = Tensor::from_data(TensorData::new(input, [2, 3, height, 7]), &device);
                let filter = Tensor::from_data(TensorData::new(filter, [5, 3, 3, 2]), &device);
                let bias = Tensor::from_data(TensorData::new(vec![0.2; 5], [5]), &device);
                let weights = filter.clone().permute([2, 3, 1, 0]).reshape([18, 5]);
                let expected = module::conv2d(
                    input.clone(),
                    filter,
                    Some(bias.clone()),
                    ConvOptions::new(geometry.stride, geometry.padding, geometry.dilation, 1),
                );
                let actual = forward(input, &weights, Some(&bias), &geometry).unwrap();
                assert_eq!(expected.dims(), actual.dims());
                let max_error = (expected - actual)
                    .abs()
                    .max()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()[0];
                assert!(max_error < 5e-4, "{device:?} length {height}: {max_error}");
            }
        }
    }
}
