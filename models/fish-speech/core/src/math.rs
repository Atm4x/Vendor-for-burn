use crate::error::{FishError, Result};

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.is_empty() || x.len() != weight.len() || eps <= 0.0 || !eps.is_finite() {
        return Err(FishError::InvalidShape(
            "RMSNorm dimensions/epsilon invalid",
        ));
    }
    let scale = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64
        + eps as f64)
        .sqrt()
        .recip();
    Ok(x.iter()
        .zip(weight)
        .map(|(&v, &w)| (v as f64 * scale * w as f64) as f32)
        .collect())
}

pub fn swiglu(w1: &[f32], w3: &[f32]) -> Result<Vec<f32>> {
    if w1.len() != w3.len() {
        return Err(FishError::InvalidShape("SwiGLU projections incompatible"));
    }
    Ok(w1
        .iter()
        .zip(w3)
        .map(|(&a, &b)| a / (1.0 + (-a).exp()) * b)
        .collect())
}

pub fn rotary_emb(x: &[f32], position: usize, base: f32) -> Result<Vec<f32>> {
    if x.is_empty() || x.len() % 2 != 0 || !base.is_finite() || base <= 0.0 {
        return Err(FishError::InvalidShape(
            "RoPE needs positive even head dimension and positive base",
        ));
    }
    let mut result = Vec::with_capacity(x.len());
    for i in (0..x.len()).step_by(2) {
        let freq = base.powf(-(i as f32) / (x.len() as f32));
        let (sin, cos) = (position as f32 * freq).sin_cos();
        result.push(x[i] * cos - x[i + 1] * sin);
        result.push(x[i] * sin + x[i + 1] * cos);
    }
    Ok(result)
}

pub fn gqa_attention_single_query(
    query: &[Vec<f32>],
    keys: &[Vec<Vec<f32>>],
    values: &[Vec<Vec<f32>>],
    query_pos: usize,
    active_len: usize,
) -> Result<Vec<Vec<f32>>> {
    if query.is_empty()
        || keys.is_empty()
        || keys.len() != values.len()
        || query.len() % keys.len() != 0
        || active_len == 0
        || query_pos >= active_len
    {
        return Err(FishError::InvalidShape(
            "GQA head layout, position or prefix invalid",
        ));
    }
    let d = query[0].len();
    if d == 0
        || query.iter().any(|v| v.len() != d)
        || keys.iter().zip(values).any(|(ks, vs)| {
            ks.len() < active_len
                || vs.len() < active_len
                || ks.iter().take(active_len).any(|k| k.len() != d)
                || vs.iter().take(active_len).any(|v| v.len() != d)
        })
    {
        return Err(FishError::InvalidShape("GQA tensor dimensions mismatch"));
    }
    let group = query.len() / keys.len();
    let mut outputs = Vec::with_capacity(query.len());
    for (head, q) in query.iter().enumerate() {
        let kvh = head / group;
        let num_keys = active_len.min(query_pos + 1);
        let scores: Vec<f64> = keys[kvh][..num_keys]
            .iter()
            .map(|k| {
                q.iter()
                    .zip(k)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>()
                    / (d as f64).sqrt()
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let probs: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
        let total: f64 = probs.iter().sum();
        let mut out = vec![0.0_f32; d];
        for (j, scalar) in out.iter_mut().enumerate() {
            *scalar = probs
                .iter()
                .zip(&values[kvh])
                .take(num_keys)
                .map(|(&p, v)| p * v[j] as f64)
                .sum::<f64>() as f32
                / total as f32;
        }
        outputs.push(out);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rmsnorm_unity() {
        let x = rms_norm(&[3.0, 4.0], &[1.0, 1.0], 1e-5).unwrap();
        let scale = (12.5_f32 + 1e-5).sqrt();
        assert!((x[0] - 3.0 / scale).abs() < 1e-6);
    }
    #[test]
    fn swiglu_baseline() {
        let x = swiglu(&[0.0, 1.0], &[3.0, 2.0]).unwrap();
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 2.0 / (1.0 + (-1.0_f32).exp())).abs() < 1e-6);
    }
    #[test]
    fn rope_zero_position_identity() {
        assert_eq!(
            rotary_emb(&[1.0, 2.0, 3.0, 4.0], 0, 10000.0).unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );
    }
    #[test]
    fn rope_rotation_preserves_norm() {
        let x = rotary_emb(&[1.0, 2.0, 3.0, 4.0], 19, 10000.0).unwrap();
        assert!((x.iter().map(|v| v * v).sum::<f32>() - 30.0).abs() < 2e-5);
    }
    #[test]
    fn gqa_query_heads_share_kv_heads() {
        let q = vec![vec![1.0, 0.0]; 4];
        let k = vec![vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]]];
        let v = vec![vec![vec![2.0, 0.0]], vec![vec![0.0, 3.0]]];
        let out = gqa_attention_single_query(&q, &k, &v, 0, 1).unwrap();
        assert_eq!(
            out,
            [
                vec![2.0, 0.0],
                vec![2.0, 0.0],
                vec![0.0, 3.0],
                vec![0.0, 3.0]
            ]
        );
    }
    #[test]
    fn gqa_poisoned_tail_unread() {
        let q = vec![vec![1.0, 1.0]];
        let k = vec![vec![vec![1.0, 1.0], vec![f32::NAN, f32::NAN]]];
        let v = vec![vec![vec![9.0, 4.0], vec![f32::NAN, f32::NAN]]];
        let out = gqa_attention_single_query(&q, &k, &v, 0, 1).unwrap();
        assert_eq!(out, [vec![9.0, 4.0]]);
    }
    #[test]
    fn gqa_causal_query_does_not_see_future() {
        let q = vec![vec![1.0, 0.0]];
        let k = vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]];
        let v = vec![vec![vec![10.0, 0.0], vec![99.0, 0.0]]];
        let out = gqa_attention_single_query(&q, &k, &v, 0, 2).unwrap();
        assert_eq!(out[0][0], 10.0);
    }
}
