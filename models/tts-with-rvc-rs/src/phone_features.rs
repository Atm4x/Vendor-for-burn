use burn::tensor::Tensor;

/// Double HuBERT's 50 Hz content frames to 100 Hz without a host-side readback.
/// The frame truncation matches the existing RVC indexing rule: source[frame / 2].
pub(crate) fn upsample_without_retrieval(content: Tensor<3>, frames: usize) -> Tensor<3> {
    let [batch, content_time, channels] = content.dims();
    assert!(frames > 0 && frames <= content_time * 2);
    content
        .unsqueeze_dim::<4>(2)
        .repeat_dim(2, 2)
        .reshape([batch, content_time * 2, channels])
        .slice([0..batch, 0..frames, 0..channels])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn native_upsampling_preserves_every_fp32_bit_and_batch_boundary() {
        let device = Device::flex();
        for batch in [1, 2, 3] {
            for time in [1, 2, 7, 21] {
                for channels in [1, 3, 17, 768] {
                    let values: Vec<f32> = (0..batch * time * channels)
                        .map(|i| f32::from_bits(0x3f000000_u32 + (i % 100_000) as u32))
                        .collect();
                    for frames in [1, time * 2 - 1, time * 2] {
                        let original = Tensor::<3>::from_data(
                            TensorData::new(values.clone(), [batch, time, channels]),
                            &device,
                        );
                        let output = upsample_without_retrieval(original, frames);
                        assert_eq!(output.dims(), [batch, frames, channels]);
                        let actual = output.into_data().try_to_vec::<f32>().unwrap();
                        let mut expected = Vec::with_capacity(batch * frames * channels);
                        for b in 0..batch {
                            for frame in 0..frames {
                                let start = (b * time + frame / 2) * channels;
                                expected.extend_from_slice(&values[start..start + channels]);
                            }
                        }
                        assert_eq!(actual.len(), expected.len());
                        assert!(
                            actual
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "batch={batch}, time={time}, channels={channels}, frames={frames}"
                        );
                    }
                }
            }
        }
    }
}
