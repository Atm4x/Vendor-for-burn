use burn::tensor::{activation, Device, Tensor};

use crate::checkpoint::RvcCheckpoint;

struct Linear {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl Linear {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let weight = checkpoint.tensor::<2>(&format!("{prefix}.weight"), device)?;
        let bias = checkpoint.tensor::<1>(&format!("{prefix}.bias"), device)?;
        if weight.dims()[0] != bias.dims()[0] {
            return Err(format!("{prefix}: linear weight/bias mismatch"));
        }
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, time, features] = x.dims();
        let [output, input] = self.weight.dims();
        assert_eq!(features, input);
        (x.reshape([batch * time, features]).matmul(self.weight.clone().transpose())
            + self.bias.clone().reshape([1, output]))
            .reshape([batch, time, output])
    }
}

struct LayerNorm {
    weight: Tensor<1>,
    bias: Tensor<1>,
}

impl LayerNorm {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, channels: usize, device: &Device) -> Result<Self, String> {
        let weight = checkpoint.tensor::<1>(&format!("{prefix}.weight"), device)?;
        let bias = checkpoint.tensor::<1>(&format!("{prefix}.bias"), device)?;
        if weight.dims() != [channels] || bias.dims() != [channels] {
            return Err(format!("{prefix}: invalid layer norm dimensions"));
        }
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let channels = x.dims()[2];
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / (variance + 1e-5).sqrt()
            * self.weight.clone().reshape([1, 1, channels])
            + self.bias.clone().reshape([1, 1, channels])
    }
}

/// One inference-mode Fairseq HuBERT Base Transformer layer (post-layer-norm).
/// Key padding mask uses 1.0 for valid frames and 0.0 for padding.
pub struct HubertTransformerLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
    heads: usize,
    channels: usize,
}

impl HubertTransformerLayer {
    pub fn load(checkpoint: &RvcCheckpoint, index: usize, heads: usize, device: &Device) -> Result<Self, String> {
        let prefix = format!("encoder.layers.{index}");
        let q_proj = Linear::load(checkpoint, &format!("{prefix}.self_attn.q_proj"), device)?;
        let k_proj = Linear::load(checkpoint, &format!("{prefix}.self_attn.k_proj"), device)?;
        let v_proj = Linear::load(checkpoint, &format!("{prefix}.self_attn.v_proj"), device)?;
        let out_proj = Linear::load(checkpoint, &format!("{prefix}.self_attn.out_proj"), device)?;
        let channels = q_proj.weight.dims()[0];
        if heads == 0 || channels % heads != 0
            || [k_proj.weight.dims(), v_proj.weight.dims(), out_proj.weight.dims()]
                .iter().any(|&dims| dims != [channels, channels])
            || q_proj.weight.dims() != [channels, channels]
        {
            return Err(format!("{prefix}: invalid attention dimensions or head count"));
        }
        let fc1 = Linear::load(checkpoint, &format!("{prefix}.fc1"), device)?;
        let fc2 = Linear::load(checkpoint, &format!("{prefix}.fc2"), device)?;
        if fc1.weight.dims()[1] != channels
            || fc2.weight.dims() != [channels, fc1.weight.dims()[0]]
        {
            return Err(format!("{prefix}: invalid FFN dimensions"));
        }
        Ok(Self {
            q_proj, k_proj, v_proj, out_proj,
            attention_norm: LayerNorm::load(checkpoint, &format!("{prefix}.self_attn_layer_norm"), channels, device)?,
            fc1, fc2,
            final_norm: LayerNorm::load(checkpoint, &format!("{prefix}.final_layer_norm"), channels, device)?,
            heads, channels,
        })
    }

    pub fn forward(&self, x: Tensor<3>, valid_keys: Option<Tensor<2>>) -> Tensor<3> {
        let [batch, time, channels] = x.dims();
        assert_eq!(channels, self.channels);
        let depth = channels / self.heads;
        let split = |tensor: Tensor<3>| {
            tensor.reshape([batch, time, self.heads, depth]).permute([0, 2, 1, 3])
        };
        let q = split(self.q_proj.forward(x.clone())) * (1.0 / (depth as f64).sqrt());
        let k = split(self.k_proj.forward(x.clone()));
        let v = split(self.v_proj.forward(x.clone()));
        let mut scores = q.matmul(k.transpose());
        if let Some(mask) = valid_keys {
            assert_eq!(mask.dims(), [batch, time]);
            scores = scores.mask_fill(
                mask.reshape([batch, 1, 1, time])
                    .expand([batch, self.heads, time, time])
                    .equal_elem(0.0),
                -1e4,
            );
        }
        let attention = activation::softmax(scores, 3)
            .matmul(v)
            .permute([0, 2, 1, 3])
            .reshape([batch, time, channels]);
        let after_attention = self.attention_norm.forward(x + self.out_proj.forward(attention));
        let hidden = activation::gelu(self.fc1.forward(after_attention.clone()));
        self.final_norm.forward(after_attention + self.fc2.forward(hidden))
    }
}
