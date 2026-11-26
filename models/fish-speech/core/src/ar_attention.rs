use crate::error::{FishError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaShape {
    pub batch: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    pub first_position: usize,
}

impl GqaShape {
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || self.query_heads % self.kv_heads != 0
            || self.query_len == 0
            || self.head_dim == 0
            || self.kv_len == 0
            || self.first_position.checked_add(self.query_len) != Some(self.kv_len)
        {
            return Err(FishError::InvalidShape(
                "GQA requires right-aligned active-prefix causal positions",
            ));
        }
        Ok(())
    }
}

/// Additive right-aligned causal mask for a fixed [query_len, kv_len] attention tile.
/// TensorRT must softmax along the key axis; negative entries are never visible.
pub fn right_aligned_causal_bias(shape: GqaShape) -> Result<Vec<f32>> {
    shape.validate()?;
    let count = shape
        .query_len
        .checked_mul(shape.kv_len)
        .ok_or(FishError::InvalidShape("GQA causal mask size overflow"))?;
    let mut mask = vec![0.0f32; count];
    for q in 0..shape.query_len {
        let last_visible = shape.first_position + q;
        for k in (last_visible + 1)..shape.kv_len {
            mask[q * shape.kv_len + k] = -1.0e9;
        }
    }
    Ok(mask)
}

pub fn causal_gqa_cpu(q: &[f32], k: &[f32], v: &[f32], s: GqaShape) -> Result<Vec<f32>> {
    s.validate()?;
    let total = |a: usize, b: usize, c: usize, d: usize| {
        a.checked_mul(b)
            .and_then(|n| n.checked_mul(c))
            .and_then(|n| n.checked_mul(d))
            .ok_or(FishError::InvalidShape("GQA size overflow"))
    };
    if q.len() != total(s.batch, s.query_heads, s.query_len, s.head_dim)?
        || k.len() != total(s.batch, s.kv_heads, s.kv_len, s.head_dim)?
        || v.len() != k.len()
    {
        return Err(FishError::InvalidShape("GQA Q/K/V tensor sizes mismatch"));
    }
    let width = s.query_heads * s.head_dim;
    let mut out = vec![0f32; total(s.batch, s.query_len, s.query_heads, s.head_dim)?];
    for b in 0..s.batch {
        for h in 0..s.query_heads {
            let kvh = h / (s.query_heads / s.kv_heads);
            for t in 0..s.query_len {
                let visible = s.first_position + t + 1;
                let qstart = ((b * s.query_heads + h) * s.query_len + t) * s.head_dim;
                let qs = &q[qstart..qstart + s.head_dim];
                let mut scores = Vec::with_capacity(visible);
                for kt in 0..visible {
                    let kstart = ((b * s.kv_heads + kvh) * s.kv_len + kt) * s.head_dim;
                    let sum = qs
                        .iter()
                        .zip(&k[kstart..kstart + s.head_dim])
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                    scores.push(sum / (s.head_dim as f32).sqrt());
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denominator = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    denominator += *score;
                }
                let dst = (b * s.query_len + t) * width + h * s.head_dim;
                for (kt, score) in scores.iter().enumerate() {
                    let vstart = ((b * s.kv_heads + kvh) * s.kv_len + kt) * s.head_dim;
                    let weight = *score / denominator;
                    for d in 0..s.head_dim {
                        out[dst + d] += weight * v[vstart + d];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefill_and_decode_masks_are_right_aligned() {
        let base = GqaShape {
            batch: 2,
            query_heads: 4,
            kv_heads: 2,
            query_len: 3,
            kv_len: 3,
            head_dim: 8,
            first_position: 0,
        };
        let mask = right_aligned_causal_bias(base).unwrap();
        assert_eq!(mask, [0., -1.0e9, -1.0e9, 0., 0., -1.0e9, 0., 0., 0.]);
        let decode = GqaShape {
            query_len: 1,
            kv_len: 100,
            first_position: 99,
            ..base
        };
        assert!(right_aligned_causal_bias(decode)
            .unwrap()
            .iter()
            .all(|v| *v == 0.0));
        let prefix = GqaShape {
            query_len: 2,
            kv_len: 5,
            first_position: 3,
            ..base
        };
        assert_eq!(
            right_aligned_causal_bias(prefix).unwrap(),
            [0., 0., 0., 0., -1.0e9, 0., 0., 0., 0., 0.]
        );
    }

    #[test]
    fn refuses_left_aligned_cached_decode() {
        let s = GqaShape {
            batch: 1,
            query_heads: 4,
            kv_heads: 2,
            query_len: 1,
            kv_len: 5,
            head_dim: 2,
            first_position: 0,
        };
        assert!(s.validate().is_err());
        assert!(GqaShape {
            first_position: 4,
            ..s
        }
        .validate()
        .is_ok());
    }
    #[test]
    fn causal_mask_excludes_future_in_prefill() {
        let s = GqaShape {
            batch: 1,
            query_heads: 2,
            kv_heads: 1,
            query_len: 3,
            kv_len: 3,
            head_dim: 1,
            first_position: 0,
        };
        let y = causal_gqa_cpu(&[0.; 6], &[0.; 3], &[1., 2., 9.], s).unwrap();
        assert_eq!(y, [1., 1., 1.5, 1.5, 4., 4.]);
    }
    #[test]
    fn poison_tail_not_part_of_attention() {
        let s = GqaShape {
            batch: 1,
            query_heads: 2,
            kv_heads: 1,
            query_len: 1,
            kv_len: 2,
            head_dim: 1,
            first_position: 1,
        };
        let y = causal_gqa_cpu(&[0., 0.], &[0., 0.], &[1., 3.], s).unwrap();
        assert_eq!(y, [2., 2.]);
    }
}
