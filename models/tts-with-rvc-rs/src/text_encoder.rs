use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::Deserialize;

use crate::TextEncoderFront;
use crate::TextFrontFixture;
use crate::attention::PointwiseProjection;
use crate::encoder::{EncoderLayerFixture, TextEncoderLayer};

pub struct TextEncoder {
    front: TextEncoderFront,
    layers: Vec<TextEncoderLayer>,
    stats_projection: PointwiseProjection,
    output_channels: usize,
}

pub struct TextEncoderOutput {
    pub mean: Tensor<3>,
    pub log_scale: Tensor<3>,
    pub mask: Tensor<3>,
}

impl TextEncoder {
    pub fn new(
        front: TextEncoderFront,
        layers: Vec<TextEncoderLayer>,
        stats_projection: PointwiseProjection,
        output_channels: usize,
    ) -> Self {
        assert!(!layers.is_empty());
        assert!(output_channels > 0);
        Self { front, layers, stats_projection, output_channels }
    }

    pub fn forward(
        &self,
        phone: Tensor<3>,
        pitch: Option<Tensor<2, Int>>,
        lengths: &[usize],
        skip_head: Option<usize>,
    ) -> TextEncoderOutput {
        let front = self.front.forward(phone, pitch, lengths);
        let mut mask = front.mask;
        let mut x = front.features * mask.clone();
        for layer in &self.layers {
            x = layer.forward(x, mask.clone());
        }
        x = x * mask.clone();
        if let Some(skip) = skip_head {
            let [batch, channels, time] = x.dims();
            assert!(skip < time, "skip_head must leave a nonempty sequence");
            x = x.slice([0..batch, 0..channels, skip..time]);
            mask = mask.slice([0..batch, 0..1, skip..time]);
        }
        let stats = self.stats_projection.forward(x) * mask.clone();
        let [batch, channels, time] = stats.dims();
        assert_eq!(channels, self.output_channels * 2);
        let mean = stats.clone().slice([0..batch, 0..self.output_channels, 0..time]);
        let log_scale = stats.slice([0..batch, self.output_channels..channels, 0..time]);
        TextEncoderOutput { mean, log_scale, mask }
    }
}

#[derive(Debug, Deserialize)]
pub struct TextEncoderFixture {
    pub front: TextFrontFixture,
    pub layers: Vec<EncoderLayerFixture>,
    pub output_channels: usize,
    pub stats_weight: Vec<f32>,
    pub stats_bias: Vec<f32>,
    pub expected_mean: Vec<f32>,
    pub expected_log_scale: Vec<f32>,
    pub expected_mask: Vec<f32>,
    pub skip_head: usize,
    pub expected_skip_mean: Vec<f32>,
    pub expected_skip_log_scale: Vec<f32>,
    pub expected_skip_mask: Vec<f32>,
}

impl TextEncoderFixture {
    pub fn model(&self, device: &Device) -> TextEncoder {
        let channels = self.front.hidden_channels;
        assert_eq!(self.stats_weight.len(), 2 * self.output_channels * channels);
        assert_eq!(self.stats_bias.len(), 2 * self.output_channels);
        let projection = PointwiseProjection::new(
            Tensor::from_data(
                TensorData::new(self.stats_weight.clone(), [2 * self.output_channels, channels]),
                device,
            ),
            Tensor::from_data(
                TensorData::new(self.stats_bias.clone(), [2 * self.output_channels]),
                device,
            ),
        );
        TextEncoder::new(
            self.front.model(device),
            self.layers.iter().map(|layer| layer.model(device)).collect(),
            projection,
            self.output_channels,
        )
    }

    pub fn forward(&self, device: &Device, skip_head: Option<usize>) -> TextEncoderOutput {
        let (batch, time, channels) =
            (self.front.batch, self.front.time, self.front.in_channels);
        let phone = Tensor::from_data(
            TensorData::new(self.front.phone.clone(), [batch, time, channels]),
            device,
        );
        let pitch = Tensor::<2, Int>::from_data(
            TensorData::new(self.front.pitch.clone(), [batch, time]),
            device,
        );
        self.model(device).forward(phone, Some(pitch), &self.front.lengths, skip_head)
    }
}
