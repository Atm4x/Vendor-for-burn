use burn::tensor::{Int, Tensor, TensorData, activation, module};
use burn::tensor::Device;
use serde::Deserialize;

pub struct TextEncoderFront {
    phone_weight: Tensor<2>,
    phone_bias: Tensor<1>,
    pitch_weight: Option<Tensor<2>>,
}

pub struct TextEncoderFrontOutput {
    pub features: Tensor<3>,
    pub mask: Tensor<3>,
}

impl TextEncoderFront {
    pub fn new(
        phone_weight: Tensor<2>,
        phone_bias: Tensor<1>,
        pitch_weight: Option<Tensor<2>>,
    ) -> Self {
        let [hidden, _] = phone_weight.dims();
        assert_eq!(phone_bias.dims(), [hidden]);
        if let Some(weights) = &pitch_weight {
            assert_eq!(weights.dims(), [256, hidden]);
        }
        Self {
            phone_weight,
            phone_bias,
            pitch_weight,
        }
    }

    pub fn forward(
        &self,
        phone: Tensor<3>,
        pitch: Option<Tensor<2, Int>>,
        lengths: &[usize],
    ) -> TextEncoderFrontOutput {
        let [batch, time, channels] = phone.dims();
        let [hidden, in_channels] = self.phone_weight.dims();
        assert_eq!(channels, in_channels);
        assert_eq!(lengths.len(), batch);
        assert!(lengths.iter().all(|&length| length <= time));

        let device = phone.device();
        let mut x = phone
            .reshape([batch * time, channels])
            .matmul(self.phone_weight.clone().transpose())
            .reshape([batch, time, hidden]);
        x = x + self.phone_bias.clone().reshape([1, 1, hidden]);

        match (pitch, &self.pitch_weight) {
            (Some(indices), Some(weights)) => {
                assert_eq!(indices.dims(), [batch, time]);
                x = x + module::embedding(weights.clone(), indices);
            }
            (None, None) => {}
            (None, Some(_)) => {}
            (Some(_), None) => panic!("pitch indices require a pitch embedding"),
        }

        let features = activation::leaky_relu(x * (hidden as f64).sqrt(), 0.1)
            .permute([0, 2, 1]);
        let mask_values: Vec<f32> = lengths
            .iter()
            .flat_map(|&length| (0..time).map(move |t| if t < length { 1.0 } else { 0.0 }))
            .collect();
        let mask = Tensor::<3>::from_data(
            TensorData::new(mask_values, [batch, 1, time]),
            &device,
        );
        TextEncoderFrontOutput { features, mask }
    }
}

pub fn fused_add_tanh_sigmoid_multiply(
    input_a: Tensor<3>,
    input_b: Tensor<3>,
    n_channels: usize,
) -> Tensor<3> {
    let [batch, channels, time] = input_a.dims();
    assert_eq!(input_b.dims(), [batch, channels, time]);
    assert_eq!(channels, n_channels * 2);
    let combined = input_a + input_b;
    let tanh = activation::tanh(combined.clone().slice([0..batch, 0..n_channels, 0..time]));
    let sigmoid = activation::sigmoid(combined.slice([0..batch, n_channels..channels, 0..time]));
    tanh * sigmoid
}

#[derive(Debug, Deserialize)]
pub struct TextFrontFixture {
    pub batch: usize,
    pub time: usize,
    pub in_channels: usize,
    pub hidden_channels: usize,
    pub phone: Vec<f32>,
    pub pitch: Vec<i64>,
    pub lengths: Vec<usize>,
    pub phone_weight: Vec<f32>,
    pub phone_bias: Vec<f32>,
    pub pitch_weight: Vec<f32>,
    pub expected_features: Vec<f32>,
    pub expected_mask: Vec<f32>,
}

impl TextFrontFixture {
    pub fn model(&self, device: &Device) -> TextEncoderFront {
        let (b, t, i, h) = (self.batch, self.time, self.in_channels, self.hidden_channels);
        assert_eq!(self.phone.len(), b * t * i);
        assert_eq!(self.pitch.len(), b * t);
        assert_eq!(self.phone_weight.len(), h * i);
        assert_eq!(self.phone_bias.len(), h);
        assert_eq!(self.pitch_weight.len(), 256 * h);
        TextEncoderFront::new(
            Tensor::from_data(TensorData::new(self.phone_weight.clone(), [h, i]), device),
            Tensor::from_data(TensorData::new(self.phone_bias.clone(), [h]), device),
            Some(Tensor::from_data(TensorData::new(self.pitch_weight.clone(), [256, h]), device)),
        )
    }

    pub fn forward(&self, device: &Device) -> TextEncoderFrontOutput {
        let (b, t, i) = (self.batch, self.time, self.in_channels);
        let front = self.model(device);
        let phone = Tensor::from_data(TensorData::new(self.phone.clone(), [b, t, i]), device);
        let pitch = Tensor::<2, Int>::from_data(TensorData::new(self.pitch.clone(), [b, t]), device);
        front.forward(phone, Some(pitch), &self.lengths)
    }
}

pub fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}

pub mod attention;
pub mod encoder;

pub mod text_encoder;

pub mod checkpoint;
pub mod flow;

pub mod decoder;

pub mod rmvpe;
pub mod rmvpe_gru;
pub mod rmvpe_mel;
pub mod hubert;

pub mod hubert_transformer;

pub mod hubert_position;

pub mod hubert_model;

pub mod rmvpe_unet;

pub mod nsf_source;
