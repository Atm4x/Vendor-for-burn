use burn::tensor::{activation, Device, Int, Tensor, TensorData};
use serde::Deserialize;

pub struct PointwiseProjection {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl PointwiseProjection {
    pub fn new(weight: Tensor<2>, bias: Tensor<1>) -> Self {
        assert_eq!(weight.dims()[0], bias.dims()[0]);
        Self { weight, bias }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, channels, time] = x.dims();
        let [out_channels, in_channels] = self.weight.dims();
        assert_eq!(channels, in_channels);
        let y = x
            .permute([0, 2, 1])
            .reshape([batch * time, channels])
            .matmul(self.weight.clone().transpose())
            .reshape([batch, time, out_channels]);
        (y + self.bias.clone().reshape([1, 1, out_channels])).permute([0, 2, 1])
    }
}

pub struct RelativeSelfAttention {
    query: PointwiseProjection,
    key: PointwiseProjection,
    value: PointwiseProjection,
    output: PointwiseProjection,
    relative_keys: Tensor<2>,
    relative_values: Tensor<2>,
    heads: usize,
    window: usize,
}

impl RelativeSelfAttention {
    pub fn new(
        query: PointwiseProjection,
        key: PointwiseProjection,
        value: PointwiseProjection,
        output: PointwiseProjection,
        relative_keys: Tensor<2>,
        relative_values: Tensor<2>,
        heads: usize,
        window: usize,
    ) -> Self {
        let channels = query.weight.dims()[0];
        assert!(heads > 0 && channels % heads == 0);
        assert_eq!(query.weight.dims(), [channels, channels]);
        assert_eq!(key.weight.dims(), [channels, channels]);
        assert_eq!(value.weight.dims(), [channels, channels]);
        assert_eq!(output.weight.dims()[1], channels);
        assert_eq!(relative_keys.dims(), [2 * window + 1, channels / heads]);
        assert_eq!(relative_values.dims(), [2 * window + 1, channels / heads]);
        Self { query, key, value, output, relative_keys, relative_values, heads, window }
    }

    pub fn forward(&self, x: Tensor<3>, x_mask: Tensor<3>) -> Tensor<3> {
        self.forward_with_online(x, x_mask, std::env::var("RVC_ATTENTION_ONLINE").as_deref() == Ok("1"))
    }

    pub fn forward_online(&self, x: Tensor<3>, x_mask: Tensor<3>) -> Tensor<3> {
        self.forward_with_online(x, x_mask, true)
    }

    fn forward_with_online(&self, x: Tensor<3>, x_mask: Tensor<3>, use_online: bool) -> Tensor<3> {
        let [batch, channels, time] = x.dims();
        assert_eq!(x_mask.dims(), [batch, 1, time]);
        let head_channels = channels / self.heads;
        let head_shape = [batch, self.heads, time, head_channels];

        let q = self.query.forward(x.clone())
            .reshape([batch, self.heads, head_channels, time])
            .permute([0, 1, 3, 2]);
        let k = self.key.forward(x.clone())
            .reshape([batch, self.heads, head_channels, time])
            .permute([0, 1, 3, 2]);
        let v = self.value.forward(x)
            .reshape([batch, self.heads, head_channels, time])
            .permute([0, 1, 3, 2]);

        let scaled_q = q * (1.0 / (head_channels as f64).sqrt());
        if use_online {
            let valid = x_mask.reshape([batch, time]);
            let attended = crate::online_attention::forward(
                scaled_q, k, v, Some(valid.clone()), Some(valid),
                Some(crate::online_attention::RelativeAttention {
                    keys: self.relative_keys.clone(),
                    values: self.relative_values.clone(),
                    window: self.window,
                }), 32, 64,
            );
            let combined = attended.permute([0, 1, 3, 2]).reshape([batch, channels, time]);
            return self.output.forward(combined);
        }
        let band = RelativeBand::new(time, self.window, &scaled_q.device());
        let content_scores = scaled_q.clone().matmul(k.transpose());
        let relative_scores = band.scores(scaled_q, self.relative_keys.clone());

        let attention_mask = x_mask.clone().reshape([batch, 1, time, 1])
            * x_mask.reshape([batch, 1, 1, time]);
        let attention_mask = attention_mask.expand([batch, self.heads, time, time]);
        let scores = (content_scores + relative_scores)
            .mask_fill(attention_mask.equal_elem(0.0), -1e4);
        let weights = activation::softmax(scores, 3);

        let content = weights.clone().matmul(v);
        let relative = band.values(weights, self.relative_values.clone()).reshape(head_shape);
        let combined = (content + relative)
            .permute([0, 1, 3, 2])
            .reshape([batch, channels, time]);
        self.output.forward(combined)
    }
}

struct RelativeBand {
    time: usize,
    width: usize,
    embedding_start: usize,
    score_index: Tensor<4, Int>,
    value_index: Tensor<4, Int>,
}

impl RelativeBand {
    fn new(time: usize, window: usize, device: &Device) -> Self {
        let radius = window.min(time.saturating_sub(1));
        let width = radius * 2 + 1;
        let mut score_index = Vec::with_capacity(time * time);
        let mut value_index = Vec::with_capacity(time * width);
        for query in 0..time {
            for key in 0..time {
                let index = if query.abs_diff(key) <= radius { key + radius - query } else { width };
                score_index.push(index as i64);
            }
            for offset in 0..width {
                let key = (query + offset).checked_sub(radius).filter(|&key| key < time);
                value_index.push(key.unwrap_or(time) as i64);
            }
        }
        Self {
            time, width, embedding_start: window - radius,
            score_index: Tensor::from_data(TensorData::new(score_index, [1, 1, time, time]), device),
            value_index: Tensor::from_data(TensorData::new(value_index, [1, 1, time, width]), device),
        }
    }

    fn embeddings(&self, table: Tensor<2>, depth: usize) -> Tensor<5> {
        table.narrow(0, self.embedding_start, self.width).reshape([1, 1, 1, self.width, depth])
    }

    fn scores(&self, query: Tensor<4>, table: Tensor<2>) -> Tensor<4> {
        let [batch, heads, time, depth] = query.dims();
        let zero = Tensor::zeros([batch, heads, time, 1], &query.device());
        let compact = (query.unsqueeze_dim::<5>(3) * self.embeddings(table, depth))
            .sum_dim(4).reshape([batch, heads, time, self.width]);
        Tensor::cat(vec![compact, zero], 3)
            .gather(3, self.score_index.clone().expand([batch, heads, time, time]))
    }

    fn values(&self, weights: Tensor<4>, table: Tensor<2>) -> Tensor<4> {
        let [batch, heads, time, _] = weights.dims();
        let depth = table.dims()[1];
        let zero = Tensor::zeros([batch, heads, time, 1], &weights.device());
        let local = Tensor::cat(vec![weights, zero], 3)
            .gather(3, self.value_index.clone().expand([batch, heads, self.time, self.width]));
        (local.unsqueeze_dim::<5>(4) * self.embeddings(table, depth))
            .sum_dim(3).reshape([batch, heads, time, depth])
    }
}

#[derive(Debug, Deserialize)]
pub struct AttentionFixture {
    pub batch: usize,
    pub time: usize,
    pub channels: usize,
    pub heads: usize,
    pub window: usize,
    pub input: Vec<f32>,
    pub lengths: Vec<usize>,
    pub query_weight: Vec<f32>,
    pub query_bias: Vec<f32>,
    pub key_weight: Vec<f32>,
    pub key_bias: Vec<f32>,
    pub value_weight: Vec<f32>,
    pub value_bias: Vec<f32>,
    pub output_weight: Vec<f32>,
    pub output_bias: Vec<f32>,
    pub relative_keys: Vec<f32>,
    pub relative_values: Vec<f32>,
    pub expected: Vec<f32>,
}

impl AttentionFixture {
    pub fn model(&self, device: &Device) -> RelativeSelfAttention {
        let channels = self.channels;
        let projection = |weight: &Vec<f32>, bias: &Vec<f32>| {
            assert_eq!(weight.len(), channels * channels);
            assert_eq!(bias.len(), channels);
            PointwiseProjection::new(
                Tensor::from_data(TensorData::new(weight.clone(), [channels, channels]), device),
                Tensor::from_data(TensorData::new(bias.clone(), [channels]), device),
            )
        };
        let relative = |values: &Vec<f32>| {
            let dimension = channels / self.heads;
            assert_eq!(values.len(), (self.window * 2 + 1) * dimension);
            Tensor::from_data(
                TensorData::new(values.clone(), [self.window * 2 + 1, dimension]), device,
            )
        };
        RelativeSelfAttention::new(
            projection(&self.query_weight, &self.query_bias),
            projection(&self.key_weight, &self.key_bias),
            projection(&self.value_weight, &self.value_bias),
            projection(&self.output_weight, &self.output_bias),
            relative(&self.relative_keys),
            relative(&self.relative_values),
            self.heads,
            self.window,
        )
    }

    pub fn inputs(&self, device: &Device) -> (Tensor<3>, Tensor<3>) {
        let (batch, channels, time) = (self.batch, self.channels, self.time);
        assert_eq!(self.input.len(), batch * channels * time);
        assert_eq!(self.lengths.len(), batch);
        assert!(self.lengths.iter().all(|&length| length <= time));
        assert_eq!(self.expected.len(), self.input.len());
        let inputs = Tensor::from_data(
            TensorData::new(self.input.clone(), [batch, channels, time]), device,
        );
        let mask_values: Vec<f32> = self.lengths.iter()
            .flat_map(|&length| (0..time).map(move |step| if step < length { 1.0 } else { 0.0 }))
            .collect();
        let mask = Tensor::from_data(TensorData::new(mask_values, [batch, 1, time]), device);
        (inputs, mask)
    }

    pub fn forward(&self, device: &Device) -> Tensor<3> {
        let (inputs, mask) = self.inputs(device);
        self.model(device).forward(inputs, mask)
    }
}
