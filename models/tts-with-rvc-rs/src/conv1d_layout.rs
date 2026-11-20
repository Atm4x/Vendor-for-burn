use burn::tensor::{Device, Tensor, TensorData};

/// Conv1d weight storage layouts that preserve the logical OIK tensor shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Conv1dWeightLayout {
    Standard,
    Prepacked,
    Temporal,
}

pub(crate) fn load_conv1d_weight(
    checkpoint: &crate::checkpoint::RvcCheckpoint,
    prefix: &str,
    normalized: bool,
    layout: Conv1dWeightLayout,
    device: &Device,
) -> Result<Tensor<3>, String> {
    let (shape, values) = if normalized {
        checkpoint.weight_norm_conv1d_values(prefix)?
    } else {
        let (shape, values) = checkpoint.tensor_f32(&format!("{prefix}.weight"))?;
        let shape: [usize; 3] = shape.try_into().map_err(|s: Vec<usize>| format!(
            "{prefix}: expected Conv1d rank 3, found {s:?}"
        ))?;
        (shape, values)
    };
    Ok(from_host_values(shape, values, device, layout))
}

pub(crate) fn from_host_values(
    [out_channels, in_channels, kernel]: [usize; 3],
    original: Vec<f32>,
    device: &Device,
    layout: Conv1dWeightLayout,
) -> Tensor<3> {
    match layout {
        Conv1dWeightLayout::Standard => Tensor::from_data(
            TensorData::new(original, [out_channels, in_channels, kernel]), device,
        ),
        Conv1dWeightLayout::Prepacked => {
            let mut packed = vec![0.0f32; original.len()];
            for oc in 0..out_channels {
                for ic in 0..in_channels {
                    for k in 0..kernel {
                        packed[(oc * kernel + k) * in_channels + ic] =
                            original[(oc * in_channels + ic) * kernel + k];
                    }
                }
            }
            Tensor::from_data(
                TensorData::new(packed, [out_channels, kernel, in_channels]), device,
            ).swap_dims(1, 2)
        }
        Conv1dWeightLayout::Temporal => {
            let mut packed = vec![0.0f32; original.len()];
            for oc in 0..out_channels {
                for ic in 0..in_channels {
                    for k in 0..kernel {
                        packed[(k * in_channels + ic) * out_channels + oc] =
                            original[(oc * in_channels + ic) * kernel + k];
                    }
                }
            }
            Tensor::from_data(
                TensorData::new(packed, [kernel, in_channels, out_channels]), device,
            ).swap_dims(0, 2)
        }
    }
}

pub(crate) fn default_conv1d_layout() -> Conv1dWeightLayout {
    if std::env::var_os("RVC_CONV_WEIGHT_PREPACK").is_some() {
        Conv1dWeightLayout::Prepacked
    } else {
        Conv1dWeightLayout::Standard
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;
    use burn::tensor::{module, ops::ConvOptions};

    #[test]
    fn prepacked_weights_preserve_conv1d_values_for_all_kernels_and_dilations() {
        let device = Device::flex();
        for kernel in [1, 3, 5] {
            for dilation in [1, 2, 4] {
                let input = Tensor::from_data(TensorData::new(
                    (0..2 * 3 * 29).map(|i| (i % 13) as f32 / 7.0).collect(),
                    [2, 3, 29],
                ), &device);
                let weights = Tensor::from_data(TensorData::new(
                    (0..4 * 3 * kernel).map(|i| (i % 11) as f32 / 13.0).collect(),
                    [4, 3, kernel],
                ), &device);
                let pad = (kernel - 1) * dilation / 2;
                let options = ConvOptions::new_with_padding(
                    [1], [(pad, pad)], [dilation], 1,
                );
                let reference = module::conv1d(input.clone(), weights.clone(), None, options.clone())
                    .into_data().try_into_vec::<f32>().unwrap();
                let transformed = from_host_values(
                    [4, 3, kernel],
                    weights.into_data().try_into_vec::<f32>().unwrap(),
                    &device, Conv1dWeightLayout::Prepacked,
                );
                let actual = module::conv1d(input, transformed, None, options)
                    .into_data().try_into_vec::<f32>().unwrap();
                assert_eq!(reference.len(), actual.len());
                for (lhs, rhs) in reference.into_iter().zip(actual) {
                    assert!((lhs - rhs).abs() <= 1e-5, "kernel={kernel} dilation={dilation}: {lhs} vs {rhs}");
                }
            }
        }
    }
}
