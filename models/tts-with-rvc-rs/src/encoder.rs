use burn::tensor::{activation, module, ops::ConvOptions, Device, Tensor, TensorData};
use serde::Deserialize;

use crate::attention::{AttentionFixture, RelativeSelfAttention};

pub struct ChannelLayerNorm {
    gamma: Tensor<1>,
    beta: Tensor<1>,
}

impl ChannelLayerNorm {
    pub fn new(gamma: Tensor<1>, beta: Tensor<1>) -> Self {
        assert_eq!(gamma.dims(), beta.dims());
        Self { gamma, beta }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, channels, time] = x.dims();
        assert_eq!(self.gamma.dims(), [channels]);
        module::layer_norm(
            x.permute([0, 2, 1]),
            self.gamma.clone(),
            Some(self.beta.clone()),
            1e-5,
        ).permute([0, 2, 1]).reshape([batch, channels, time])
    }
}

pub struct Conv1dLayer {
    weight: Tensor<3>,
    bias: Tensor<1>,
}

impl Conv1dLayer {
    pub fn new(weight: Tensor<3>, bias: Tensor<1>) -> Self {
        assert_eq!(weight.dims()[0], bias.dims()[0]);
        Self { weight, bias }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let kernel = self.weight.dims()[2];
        let options = ConvOptions::new_with_padding(
            [1],
            [((kernel - 1) / 2, kernel / 2)],
            [1],
            1,
        );
        module::conv1d(x, self.weight.clone(), Some(self.bias.clone()), options)
    }
}

pub struct FeedForward {
    first: Conv1dLayer,
    second: Conv1dLayer,
}

impl FeedForward {
    pub fn new(first: Conv1dLayer, second: Conv1dLayer) -> Self {
        assert_eq!(first.weight.dims()[0], second.weight.dims()[1]);
        Self { first, second }
    }

    pub fn forward(&self, x: Tensor<3>, mask: Tensor<3>) -> Tensor<3> {
        let first = self.first.forward(x * mask.clone());
        let second = self.second.forward(activation::relu(first) * mask.clone());
        second * mask
    }
}

pub struct TextEncoderLayer {
    attention: RelativeSelfAttention,
    norm_attention: ChannelLayerNorm,
    feed_forward: FeedForward,
    norm_feed_forward: ChannelLayerNorm,
}

impl TextEncoderLayer {
    pub fn new(
        attention: RelativeSelfAttention,
        norm_attention: ChannelLayerNorm,
        feed_forward: FeedForward,
        norm_feed_forward: ChannelLayerNorm,
    ) -> Self {
        Self { attention, norm_attention, feed_forward, norm_feed_forward }
    }

    pub fn forward(&self, x: Tensor<3>, mask: Tensor<3>) -> Tensor<3> {
        let attn = self.attention.forward(x.clone(), mask.clone());
        let x = self.norm_attention.forward(x + attn);
        let feed_forward = self.feed_forward.forward(x.clone(), mask);
        self.norm_feed_forward.forward(x + feed_forward)
    }
}

#[derive(Debug, Deserialize)]
pub struct EncoderLayerFixture {
    #[serde(flatten)]
    pub attention: AttentionFixture,
    pub filter_channels: usize,
    pub kernel: usize,
    pub norm_attention_gamma: Vec<f32>,
    pub norm_attention_beta: Vec<f32>,
    pub norm_feed_forward_gamma: Vec<f32>,
    pub norm_feed_forward_beta: Vec<f32>,
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub expected_encoder: Vec<f32>,
}

impl EncoderLayerFixture {
    pub fn forward(&self, device: &Device) -> Tensor<3> {
        let channels = self.attention.channels;
        let (inputs, mask) = self.attention.inputs(device);
        assert!(self.kernel > 0);
        assert_eq!(self.expected_encoder.len(), self.attention.input.len());
        let norm = |gamma: &Vec<f32>, beta: &Vec<f32>| {
            assert_eq!(gamma.len(), channels);
            assert_eq!(beta.len(), channels);
            ChannelLayerNorm::new(
                Tensor::from_data(TensorData::new(gamma.clone(), [channels]), device),
                Tensor::from_data(TensorData::new(beta.clone(), [channels]), device),
            )
        };
        let conv = |weight: &Vec<f32>, bias: &Vec<f32>, input: usize, output: usize| {
            assert_eq!(weight.len(), output * input * self.kernel);
            assert_eq!(bias.len(), output);
            Conv1dLayer::new(
                Tensor::from_data(TensorData::new(weight.clone(), [output, input, self.kernel]), device),
                Tensor::from_data(TensorData::new(bias.clone(), [output]), device),
            )
        };
        let layer = TextEncoderLayer::new(
            self.attention.model(device),
            norm(&self.norm_attention_gamma, &self.norm_attention_beta),
            FeedForward::new(
                conv(&self.conv1_weight, &self.conv1_bias, channels, self.filter_channels),
                conv(&self.conv2_weight, &self.conv2_bias, self.filter_channels, channels),
            ),
            norm(&self.norm_feed_forward_gamma, &self.norm_feed_forward_beta),
        );
        layer.forward(inputs, mask.clone()) * mask
    }
}
