use crate::error::{FishError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    DualAr,
    Naive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerDims {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub intermediate_size: usize,
    pub qkv_bias: bool,
}

impl TransformerDims {
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.n_heads == 0 || self.n_kv_heads == 0 || self.layers == 0 {
            return Err(FishError::InvalidConfig(
                "transformer dimensions must be positive",
            ));
        }
        if self.dim % self.n_heads != 0 || self.n_heads % self.n_kv_heads != 0 {
            return Err(FishError::InvalidConfig("invalid GQA head divisibility"));
        }
        if self.intermediate_size == 0 || self.head_dim() % 2 != 0 {
            return Err(FishError::InvalidConfig("invalid FFN or RoPE head size"));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
    pub fn qkv_width(&self) -> usize {
        (self.n_heads + 2 * self.n_kv_heads) * self.head_dim()
    }
}

#[derive(Debug, Clone)]
pub struct FishModelConfig {
    pub kind: ModelKind,
    pub slow: TransformerDims,
    pub fast: Option<TransformerDims>,
    pub vocab_size: usize,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub share_codebook_embeddings: bool,
}

impl FishModelConfig {
    pub fn validate(&self) -> Result<()> {
        self.slow.validate()?;
        if self.vocab_size == 0
            || self.num_codebooks == 0
            || self.codebook_size == 0
            || self.max_seq_len == 0
        {
            return Err(FishError::InvalidConfig(
                "vocabulary/codebook/context dimensions must be positive",
            ));
        }
        if !self.rope_base.is_finite()
            || self.rope_base <= 0.0
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
        {
            return Err(FishError::InvalidConfig(
                "invalid rotary base or RMSNorm epsilon",
            ));
        }
        if self.num_codebooks.checked_mul(self.codebook_size).is_none() {
            return Err(FishError::InvalidConfig(
                "codebook embedding dimensions overflow",
            ));
        }
        match (self.kind, &self.fast) {
            (ModelKind::DualAr, Some(fast)) => {
                fast.validate()?;
                if self.num_codebooks < 1 {
                    return Err(FishError::InvalidConfig("empty Fast AR"));
                }
            }
            (ModelKind::Naive, None) => {}
            _ => {
                return Err(FishError::InvalidConfig(
                    "DualAR requires Fast transformer; Naive must not have one",
                ))
            }
        }
        Ok(())
    }

    pub fn synthetic_dual_ar_fixture() -> Self {
        Self {
            kind: ModelKind::DualAr,
            slow: TransformerDims {
                dim: 64,
                n_heads: 8,
                n_kv_heads: 2,
                layers: 2,
                intermediate_size: 128,
                qkv_bias: false,
            },
            fast: Some(TransformerDims {
                dim: 64,
                n_heads: 8,
                n_kv_heads: 2,
                layers: 2,
                intermediate_size: 128,
                qkv_bias: false,
            }),
            vocab_size: 1024,
            num_codebooks: 8,
            codebook_size: 1024,
            max_seq_len: 256,
            rope_base: 10_000.0,
            norm_eps: 1e-5,
            tie_word_embeddings: true,
            share_codebook_embeddings: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_is_valid() {
        FishModelConfig::synthetic_dual_ar_fixture()
            .validate()
            .unwrap();
    }
    #[test]
    fn qkv_width_matches_gqa() {
        let d = FishModelConfig::synthetic_dual_ar_fixture().slow;
        assert_eq!(d.qkv_width(), 96);
        assert_eq!(d.head_dim(), 8);
    }
    #[test]
    fn reject_bad_gqa() {
        let mut c = FishModelConfig::synthetic_dual_ar_fixture();
        c.slow.n_kv_heads = 3;
        assert!(c.validate().is_err());
    }
    #[test]
    fn reject_naive_with_fast() {
        let mut c = FishModelConfig::synthetic_dual_ar_fixture();
        c.kind = ModelKind::Naive;
        assert!(c.validate().is_err());
    }
}
