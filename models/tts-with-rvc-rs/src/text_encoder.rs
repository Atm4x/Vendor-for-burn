use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::Deserialize;

use crate::TextEncoderFront;
use crate::TextFrontFixture;
use crate::attention::{PointwiseProjection, RelativeSelfAttention};
use crate::encoder::{ChannelLayerNorm, Conv1dLayer, EncoderLayerFixture, FeedForward, TextEncoderLayer};
use crate::checkpoint::RvcCheckpoint;

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

    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        fn projection(checkpoint: &RvcCheckpoint, name: &str, device: &Device) -> Result<PointwiseProjection, String> {
            let (shape, values) = checkpoint.tensor_f32(&format!("{name}.weight"))?;
            let (out_channels, in_channels) = match shape.as_slice() {
                [out_channels, in_channels, 1] => (*out_channels, *in_channels),
                _ => return Err(format!("{name}: unsupported projection shape {shape:?}")),
            };
            let bias = checkpoint.tensor::<1>(&format!("{name}.bias"), device)?;
            if bias.dims() != [out_channels] {
                return Err(format!("{name}: invalid projection bias shape"));
            }
            Ok(PointwiseProjection::new(
                Tensor::from_data(TensorData::new(values, [out_channels, in_channels]), device),
                bias,
            ))
        }
        fn normalization(checkpoint: &RvcCheckpoint, name: &str, device: &Device) -> Result<ChannelLayerNorm, String> {
            let gamma = checkpoint.tensor::<1>(&format!("{name}.gamma"), device)?;
            let beta = checkpoint.tensor::<1>(&format!("{name}.beta"), device)?;
            if gamma.dims() != beta.dims() {
                return Err(format!("{name}: inconsistent normalization channels"));
            }
            Ok(ChannelLayerNorm::new(gamma, beta))
        }
        fn conv(checkpoint: &RvcCheckpoint, name: &str, device: &Device) -> Result<Conv1dLayer, String> {
            let weight = checkpoint.tensor::<3>(&format!("{name}.weight"), device)?;
            let bias = checkpoint.tensor::<1>(&format!("{name}.bias"), device)?;
            if weight.dims()[0] != bias.dims()[0] {
                return Err(format!("{name}: invalid convolution dimensions"));
            }
            Ok(Conv1dLayer::new(weight, bias))
        }
        fn rel(checkpoint: &RvcCheckpoint, name: &str, device: &Device) -> Result<(Tensor<2>, usize, usize), String> {
            let (shape, values) = checkpoint.tensor_f32(name)?;
            let (window, head_channels) = match shape.as_slice() {
                [1, positions, head_channels] if *positions > 0 && *positions % 2 == 1 =>
                    ((positions - 1) / 2, *head_channels),
                _ => return Err(format!("{name}: invalid relative position shape {shape:?}")),
            };
            Ok((Tensor::from_data(TensorData::new(values, [2 * window + 1, head_channels]), device), window, head_channels))
        }

        let phone_weight = checkpoint.tensor::<2>("enc_p.emb_phone.weight", device)?;
        let phone_bias = checkpoint.tensor::<1>("enc_p.emb_phone.bias", device)?;
        let pitch_weight = checkpoint.contains("enc_p.emb_pitch.weight")
            .then(|| checkpoint.tensor::<2>("enc_p.emb_pitch.weight", device))
            .transpose()?;
        let front = TextEncoderFront::new(phone_weight.clone(), phone_bias, pitch_weight);
        let hidden_channels = phone_weight.dims()[0];

        let mut layers = Vec::new();
        while checkpoint.contains(&format!("enc_p.encoder.attn_layers.{}.conv_q.weight", layers.len())) {
            let index = layers.len();
            let prefix = format!("enc_p.encoder.attn_layers.{index}");
            let (relative_keys, window, head_channels) = rel(checkpoint, &format!("{prefix}.emb_rel_k"), device)?;
            let (relative_values, value_window, value_channels) = rel(checkpoint, &format!("{prefix}.emb_rel_v"), device)?;
            if window != value_window || head_channels != value_channels || hidden_channels % head_channels != 0 {
                return Err(format!("{prefix}: incompatible relative position embeddings"));
            }
            let attention = RelativeSelfAttention::new(
                projection(checkpoint, &format!("{prefix}.conv_q"), device)?,
                projection(checkpoint, &format!("{prefix}.conv_k"), device)?,
                projection(checkpoint, &format!("{prefix}.conv_v"), device)?,
                projection(checkpoint, &format!("{prefix}.conv_o"), device)?,
                relative_keys, relative_values, hidden_channels / head_channels, window,
            );
            let feed_forward = FeedForward::new(
                conv(checkpoint, &format!("enc_p.encoder.ffn_layers.{index}.conv_1"), device)?,
                conv(checkpoint, &format!("enc_p.encoder.ffn_layers.{index}.conv_2"), device)?,
            );
            let norm_attention = normalization(checkpoint, &format!("enc_p.encoder.norm_layers_1.{index}"), device)?;
            let norm_feed_forward = normalization(checkpoint, &format!("enc_p.encoder.norm_layers_2.{index}"), device)?;
            layers.push(TextEncoderLayer::new(attention, norm_attention, feed_forward, norm_feed_forward));
        }
        if layers.is_empty() {
            return Err("checkpoint has no RVC TextEncoder layers".into());
        }
        let stats_projection = projection(checkpoint, "enc_p.proj", device)?;
        let stats_channels = checkpoint.tensor_f32("enc_p.proj.bias")?.0[0];
        if stats_channels % 2 != 0 {
            return Err("RVC stats projection must have even output channels".into());
        }
        Ok(Self::new(front, layers, stats_projection, stats_channels / 2))
    }

    pub fn forward(
        &self,
        phone: Tensor<3>,
        pitch: Option<Tensor<2, Int>>,
        lengths: &[usize],
        skip_head: Option<usize>,
    ) -> TextEncoderOutput {
        let device = phone.device();
        let started = std::time::Instant::now();
        let front = self.front.forward(phone, pitch, lengths);
        let mut mask = front.mask;
        let mut x = front.features * mask.clone();
        crate::diagnostic::measure(&device, "text_encoder.front", started, serde_json::json!({"output_shape":x.dims()}));
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let started = std::time::Instant::now();
            x = layer.forward(x, mask.clone());
            crate::diagnostic::measure(&device, "text_encoder.attention_ffn_layer", started,
                serde_json::json!({"layer":layer_index,"output_shape":x.dims()}));
        }
        x = x * mask.clone();
        if let Some(skip) = skip_head {
            let [batch, channels, time] = x.dims();
            assert!(skip < time, "skip_head must leave a nonempty sequence");
            x = x.slice([0..batch, 0..channels, skip..time]);
            mask = mask.slice([0..batch, 0..1, skip..time]);
        }
        let started = std::time::Instant::now();
        let stats = self.stats_projection.forward(x) * mask.clone();
        crate::diagnostic::measure(&device, "text_encoder.stats_projection", started, serde_json::json!({"output_shape":stats.dims()}));
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
