use burn::tensor::{module, Int, Tensor, TensorData};

pub(crate) struct RelativeAttention {
    pub keys: Tensor<2>,
    pub values: Tensor<2>,
    pub window: usize,
}

/// Blockwise exact softmax reduction (up to floating-point reassociation).
/// A complete [batch, heads, time, time] attention matrix is never constructed.
pub(crate) fn forward(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    valid_keys: Option<Tensor<2>>,
    valid_queries: Option<Tensor<2>>,
    relative: Option<RelativeAttention>,
    query_tile: usize,
    key_tile: usize,
) -> Tensor<4> {
    let [batch, heads, time, depth] = q.dims();
    assert_eq!(k.dims(), [batch, heads, time, depth]);
    assert_eq!(v.dims(), [batch, heads, time, depth]);
    assert!(query_tile > 0 && key_tile > 0 && time > 0);
    if let Some(mask) = &valid_keys { assert_eq!(mask.dims(), [batch, time]); }
    if let Some(mask) = &valid_queries { assert_eq!(mask.dims(), [batch, time]); }
    if let Some(rel) = &relative {
        assert_eq!(rel.keys.dims(), [rel.window * 2 + 1, depth]);
        assert_eq!(rel.values.dims(), [rel.window * 2 + 1, depth]);
    }
    let device = q.device();
    let mut outputs = Vec::with_capacity(time.div_ceil(query_tile));
    for q_start in (0..time).step_by(query_tile) {
        let q_end = (q_start + query_tile).min(time);
        let q_length = q_end - q_start;
        let query = q.clone().slice([0..batch, 0..heads, q_start..q_end, 0..depth]);
        let mut maximum = Tensor::<4>::ones([batch, heads, q_length, 1], &device) * -1e30;
        let mut denominator = Tensor::<4>::zeros([batch, heads, q_length, 1], &device);
        let mut numerator = Tensor::<4>::zeros([batch, heads, q_length, depth], &device);
        for k_start in (0..time).step_by(key_tile) {
            let k_end = (k_start + key_tile).min(time);
            let k_length = k_end - k_start;
            let keys = k.clone().slice([0..batch, 0..heads, k_start..k_end, 0..depth]);
            let values = v.clone().slice([0..batch, 0..heads, k_start..k_end, 0..depth]);
            let mut scores = query.clone().matmul(keys.transpose());
            let relative_tile = relative.as_ref().filter(|rel| {
                let first_key = q_start.saturating_sub(rel.window);
                let last_key = q_end.saturating_sub(1).saturating_add(rel.window);
                k_end > first_key && k_start <= last_key
            }).map(|rel| {
                let mut indices = Vec::with_capacity(q_length * k_length);
                let mut valid = Vec::with_capacity(q_length * k_length);
                for i in q_start..q_end {
                    for j in k_start..k_end {
                        let delta = j as i64 - i as i64;
                        let inside = delta.unsigned_abs() <= rel.window as u64;
                        indices.push(if inside { rel.window as i64 + delta } else { 0 });
                        valid.push(if inside { 1.0f32 } else { 0.0f32 });
                    }
                }
                let index = Tensor::<2, Int>::from_data(TensorData::new(indices, [q_length, k_length]), &device);
                let mask = Tensor::<5>::from_data(TensorData::new(valid, [1, 1, q_length, k_length, 1]), &device);
                let keys = module::embedding(rel.keys.clone(), index.clone())
                    .reshape([1, 1, q_length, k_length, depth]) * mask.clone();
                let values = module::embedding(rel.values.clone(), index)
                    .reshape([1, 1, q_length, k_length, depth]) * mask;
                (keys, values)
            });
            if let Some((ref key_rel, _)) = relative_tile {
                let rel_scores = (query.clone().reshape([batch, heads, q_length, 1, depth]) * key_rel.clone())
                    .sum_dim(4).reshape([batch, heads, q_length, k_length]);
                scores = scores + rel_scores;
            }
            if let Some(key_mask) = &valid_keys {
                let key_valid = key_mask.clone().slice([0..batch, k_start..k_end])
                    .reshape([batch, 1, 1, k_length]);
                let attention_valid = if let Some(query_mask) = &valid_queries {
                    let query_valid = query_mask.clone().slice([0..batch, q_start..q_end])
                        .reshape([batch, 1, q_length, 1]);
                    query_valid * key_valid
                } else { key_valid };
                scores = scores.mask_fill(
                    attention_valid.expand([batch, heads, q_length, k_length]).equal_elem(0.0), -1e4,
                );
            }
            let next_maximum = maximum.clone().max_pair(scores.clone().max_dim(3));
            let scaling = (maximum - next_maximum.clone()).exp();
            let probabilities = (scores - next_maximum.clone()).exp();
            let mut block_value = probabilities.clone().matmul(values);
            if let Some((_, relative_values)) = relative_tile {
                block_value = block_value +
                    (probabilities.clone().reshape([batch, heads, q_length, k_length, 1])
                        * relative_values)
                    .sum_dim(3).reshape([batch, heads, q_length, depth]);
            }
            numerator = numerator * scaling.clone() + block_value;
            denominator = denominator * scaling + probabilities.sum_dim(3);
            maximum = next_maximum;
        }
        outputs.push(numerator / denominator);
    }
    Tensor::cat(outputs, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn online_relative_attention_matches_dense_for_distant_tiles_and_padding() {
        let device = Device::cpu();
        let (batch, heads, time, depth, window) = (1, 2, 83, 4, 3);
        let values = |offset: f32, count: usize| {
            (0..count).map(|index| ((index as f32 * 0.137) + offset).sin()).collect::<Vec<_>>()
        };
        let tensor = |offset| Tensor::from_data(
            TensorData::new(values(offset, batch * heads * time * depth), [batch, heads, time, depth]),
            &device,
        );
        let q = tensor(0.1);
        let k = tensor(1.3);
        let v = tensor(2.1);
        let rkeys = Tensor::from_data(TensorData::new(values(0.6, (2 * window + 1) * depth),
            [2 * window + 1, depth]), &device);
        let rvalues = Tensor::from_data(TensorData::new(values(0.9, (2 * window + 1) * depth),
            [2 * window + 1, depth]), &device);
        let mask = Tensor::from_data(TensorData::new((0..time)
            .map(|i| if i < 73 {1.0f32} else {0.0}).collect::<Vec<_>>(), [1,time]), &device);
        let mut rel_indices = Vec::with_capacity(time * time);
        let mut rel_valid = Vec::with_capacity(time * time);
        for i in 0..time {
            for j in 0..time {
                let delta = j as i64 - i as i64;
                let inside = delta.unsigned_abs() <= window as u64;
                rel_indices.push(if inside {window as i64 + delta} else {0});
                rel_valid.push(if inside {1.0f32} else {0.0});
            }
        }
        let indices = Tensor::<2, Int>::from_data(TensorData::new(rel_indices,[time,time]),&device);
        let valid = Tensor::<5>::from_data(TensorData::new(rel_valid,[1,1,time,time,1]),&device);
        let relative_keys = module::embedding(rkeys.clone(),indices.clone())
            .reshape([1,1,time,time,depth]) * valid.clone();
        let relative_values = module::embedding(rvalues.clone(),indices)
            .reshape([1,1,time,time,depth]) * valid;
        let scores = q.clone().matmul(k.clone().transpose()) +
            (q.clone().reshape([batch,heads,time,1,depth]) * relative_keys)
            .sum_dim(4).reshape([batch,heads,time,time]);
        let scores = scores.mask_fill((mask.clone().reshape([batch,1,time,1])
            * mask.clone().reshape([batch,1,1,time]))
            .expand([batch,heads,time,time]).equal_elem(0.0),-1e4);
        let p = burn::tensor::activation::softmax(scores,3);
        let dense = (p.clone().matmul(v.clone()) +
            (p.reshape([batch,heads,time,time,1]) * relative_values)
            .sum_dim(3).reshape([batch,heads,time,depth]))
            .into_data().try_into_vec::<f32>().unwrap();
        let actual = forward(q,k,v,Some(mask.clone()),Some(mask),
            Some(RelativeAttention {keys:rkeys,values:rvalues,window}), 16,11)
            .into_data().try_into_vec::<f32>().unwrap();
        let error = dense.iter().zip(&actual).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
        assert!(error < 7e-5, "relative online vs dense max abs error {error}");
    }

    #[test]
    fn online_softmax_matches_dense_with_nondivisible_tiles_and_key_mask() {
        let device = Device::cpu();
        let (batch, heads, time, depth) = (2, 3, 13, 8);
        let data = |salt: usize| {
            (0..batch*heads*time*depth).map(|i| ((i*31 + salt*47) as f32 * 0.083).sin() * 2.0).collect::<Vec<_>>()
        };
        let q = Tensor::from_data(TensorData::new(data(0), [batch, heads, time, depth]), &device);
        let k = Tensor::from_data(TensorData::new(data(1), [batch, heads, time, depth]), &device);
        let v = Tensor::from_data(TensorData::new(data(2), [batch, heads, time, depth]), &device);
        let mask = Tensor::from_data(TensorData::new(
            (0..batch).flat_map(|b| (0..time).map(move |i| if i < (if b == 0 { 11 } else { 7 }) { 1.0 } else { 0.0 }))
                .collect::<Vec<f32>>(), [batch, time]), &device);
        let scores = q.clone().matmul(k.clone().transpose()).mask_fill(
            mask.clone().reshape([batch,1,1,time]).expand([batch,heads,time,time]).equal_elem(0.0), -1e4);
        let dense = burn::tensor::activation::softmax(scores,3).matmul(v.clone())
            .into_data().try_into_vec::<f32>().unwrap();
        let result = forward(q,k,v,Some(mask),None,None,4,5)
            .into_data().try_into_vec::<f32>().unwrap();
        let error = dense.iter().zip(result).map(|(a,b)|(a-b).abs()).fold(0.0f32, f32::max);
        assert!(error < 5e-5,"online vs dense max abs error {error}");
    }
}
