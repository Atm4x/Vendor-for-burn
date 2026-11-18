use anyhow::Result;
use burn::tensor::{Device, Tensor, activation::{softmax, relu}, module::{conv1d, layer_norm, linear}, ops::ConvOptions};

use crate::store::TensorStore;

pub struct MultiHeadAttention {
    qkv_weight: Tensor<2>,
    qkv_bias: Tensor<1>,
    out_weight: Tensor<2>,
    out_bias: Tensor<1>,
    heads: usize,
}

impl MultiHeadAttention {
    pub fn load(weights: &TensorStore, prefix: &str, heads: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            qkv_weight: weights.float::<2>(&format!("{prefix}.in_proj_weight"), device)?.transpose(),
            qkv_bias: weights.float(&format!("{prefix}.in_proj_bias"), device)?,
            out_weight: weights.float::<2>(&format!("{prefix}.out_proj.weight"), device)?.transpose(),
            out_bias: weights.float(&format!("{prefix}.out_proj.bias"), device)?,
            heads,
        })
    }

    pub fn forward(&self, input: Tensor<3>, key_padding_mask: Option<Tensor<2>>) -> Tensor<3> {
        let [batch, len, hidden] = input.dims();
        assert_eq!(hidden % self.heads, 0);
        let head_dim = hidden / self.heads;
        let projected = linear(input, self.qkv_weight.clone(), Some(self.qkv_bias.clone()));
        let q = projected.clone().slice([0..batch, 0..len, 0..hidden]);
        let k = projected.clone().slice([0..batch, 0..len, hidden..2 * hidden]);
        let v = projected.slice([0..batch, 0..len, 2 * hidden..3 * hidden]);
        let q = q.reshape([batch, len, self.heads, head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch, len, self.heads, head_dim]).swap_dims(1, 2);
        let v = v.reshape([batch, len, self.heads, head_dim]).swap_dims(1, 2);
        let mut scores = q.matmul(k.swap_dims(2, 3)) / (head_dim as f32).sqrt();
        if let Some(mask) = key_padding_mask {
            scores = scores + mask.reshape([batch, 1, 1, len]) * -1.0e9;
        }
        let attn = softmax(scores, 3);
        let output = attn.matmul(v).swap_dims(1, 2).reshape([batch, len, hidden]);
        linear(output, self.out_weight.clone(), Some(self.out_bias.clone()))
    }
}

pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1_weight: Tensor<1>,
    norm1_bias: Tensor<1>,
    conv1_weight: Tensor<3>,
    conv1_bias: Tensor<1>,
    conv2_weight: Tensor<3>,
    conv2_bias: Tensor<1>,
    norm2_weight: Tensor<1>,
    norm2_bias: Tensor<1>,
}

impl TransformerBlock {
    pub fn load(weights: &TensorStore, prefix: &str, device: &Device) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        Ok(Self {
            attention: MultiHeadAttention::load(weights, &attn, 2, device)?,
            norm1_weight: weights.float(&format!("{prefix}.norm1.weight"), device)?,
            norm1_bias: weights.float(&format!("{prefix}.norm1.bias"), device)?,
            conv1_weight: weights.float(&format!("{prefix}.conv1.weight"), device)?,
            conv1_bias: weights.float(&format!("{prefix}.conv1.bias"), device)?,
            conv2_weight: weights.float(&format!("{prefix}.conv2.weight"), device)?,
            conv2_bias: weights.float(&format!("{prefix}.conv2.bias"), device)?,
            norm2_weight: weights.float(&format!("{prefix}.norm2.weight"), device)?,
            norm2_bias: weights.float(&format!("{prefix}.norm2.bias"), device)?,
        })
    }

    pub fn forward(&self, input: Tensor<3>, mask: Option<Tensor<2>>) -> Tensor<3> {
        let attended = self.attention.forward(input.clone(), mask);
        let x = layer_norm(input + attended, self.norm1_weight.clone(), Some(self.norm1_bias.clone()), 1e-5);
        let conv = conv1d(x.clone().swap_dims(1, 2), self.conv1_weight.clone(), Some(self.conv1_bias.clone()),
            ConvOptions::new([1], [4], [1], 1));
        let conv = conv1d(relu(conv), self.conv2_weight.clone(), Some(self.conv2_bias.clone()),
            ConvOptions::new([1], [0], [1], 1)).swap_dims(1, 2);
        layer_norm(x + conv, self.norm2_weight.clone(), Some(self.norm2_bias.clone()), 1e-5)
    }
}

pub struct TransformerStack {
    pe: Tensor<3>,
    pe_scale: Tensor<1>,
    blocks: Vec<TransformerBlock>,
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
}

impl TransformerStack {
    pub fn load(weights: &TensorStore, prefix: &str, blocks: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            pe: weights.float(&format!("{prefix}.pos_encoder.pe"), device)?,
            pe_scale: weights.float(&format!("{prefix}.pos_encoder.scale"), device)?,
            blocks: (0..blocks).map(|i| TransformerBlock::load(weights, &format!("{prefix}.layers.{i}"), device)).collect::<Result<_>>()?,
            norm_weight: weights.float(&format!("{prefix}.norm.weight"), device)?,
            norm_bias: weights.float(&format!("{prefix}.norm.bias"), device)?,
        })
    }

    pub fn positional(&self, input: Tensor<3>) -> Tensor<3> {
        let [_batch, len, hidden] = input.dims();
        let positions = self.pe.clone().slice([0..len, 0..1, 0..hidden]).swap_dims(0, 1);
        input + positions * self.pe_scale.clone().reshape([1, 1, 1])
    }

    pub fn forward(&self, input: Tensor<3>, mask: Option<Tensor<2>>) -> Tensor<3> {
        let mut hidden = self.positional(input);
        for block in &self.blocks {
            hidden = block.forward(hidden, mask.clone());
        }
        layer_norm(hidden, self.norm_weight.clone(), Some(self.norm_bias.clone()), 1e-5)
    }
}
