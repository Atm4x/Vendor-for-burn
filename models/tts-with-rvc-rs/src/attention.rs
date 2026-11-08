use burn::tensor::{activation, module, Device, Int, Tensor, TensorData};
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
        let relative_k = relative_positions(self.relative_keys.clone(), time, self.window);
        let relative_v = relative_positions(self.relative_values.clone(), time, self.window);

        let content_scores = scaled_q.clone().matmul(k.transpose());
        let relative_scores = (scaled_q.reshape([batch, self.heads, time, 1, head_channels])
            * relative_k)
            .sum_dim(4)
            .reshape([batch, self.heads, time, time]);

        let attention_mask = x_mask.clone().reshape([batch, 1, time, 1])
            * x_mask.reshape([batch, 1, 1, time]);
        let attention_mask = attention_mask.expand([batch, self.heads, time, time]);
        let scores = (content_scores + relative_scores)
            .mask_fill(attention_mask.equal_elem(0.0), -1e4);
        let weights = activation::softmax(scores, 3);

        let content = weights.clone().matmul(v);
        let relative = (weights.reshape([batch, self.heads, time, time, 1]) * relative_v)
            .sum_dim(3)
            .reshape(head_shape);
        let combined = (content + relative)
            .permute([0, 1, 3, 2])
            .reshape([batch, channels, time]);
        self.output.forward(combined)
    }
}

fn relative_positions(embeddings: Tensor<2>, time: usize, window: usize) -> Tensor<5> {
    let device = embeddings.device();
    let dimension = embeddings.dims()[1];
    let mut indices = Vec::with_capacity(time * time);
    let mut valid = Vec::with_capacity(time * time);
    for query in 0..time {
        for key in 0..time {
            let delta = key as isize - query as isize;
            let in_window = delta.unsigned_abs() <= window;
            indices.push(if in_window { (window as isize + delta) as i64 } else { 0 });
            valid.push(if in_window { 1.0f32 } else { 0.0f32 });
        }
    }
    let indices = Tensor::<2, Int>::from_data(TensorData::new(indices, [time, time]), &device);
    let valid = Tensor::<5>::from_data(TensorData::new(valid, [1, 1, time, time, 1]), &device);
    module::embedding(embeddings, indices)
        .reshape([1, 1, time, time, dimension]) * valid
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
    pub fn forward(&self, device: &Device) -> Tensor<3> {
        let (batch, channels, time) = (self.batch, self.channels, self.time);
        assert_eq!(self.input.len(), batch * channels * time);
        assert_eq!(self.lengths.len(), batch);
        assert!(self.lengths.iter().all(|&length| length <= time));
        assert_eq!(self.expected.len(), self.input.len());
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
        let attention = RelativeSelfAttention::new(
            projection(&self.query_weight, &self.query_bias),
            projection(&self.key_weight, &self.key_bias),
            projection(&self.value_weight, &self.value_bias),
            projection(&self.output_weight, &self.output_bias),
            relative(&self.relative_keys),
            relative(&self.relative_values),
            self.heads,
            self.window,
        );
        let inputs = Tensor::from_data(
            TensorData::new(self.input.clone(), [batch, channels, time]), device,
        );
        let mask_values: Vec<f32> = self.lengths.iter()
            .flat_map(|&length| (0..time).map(move |step| if step < length { 1.0 } else { 0.0 }))
            .collect();
        let mask = Tensor::from_data(TensorData::new(mask_values, [batch, 1, time]), device);
        attention.forward(inputs, mask)
    }
}
