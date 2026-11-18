use std::fs;
use std::path::Path;

use fish_speech_core::config::{FishModelConfig, ModelKind, TransformerDims};
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelConfigError {
    #[error("checkpoint I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("checkpoint JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("checkpoint config is invalid: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, ModelConfigError>;

#[derive(Debug, Deserialize)]
struct RawModelArgs {
    model_type: String,
    #[serde(default = "default_vocab")]
    vocab_size: usize,
    #[serde(default = "default_layers")]
    n_layer: usize,
    #[serde(default = "default_heads")]
    n_head: usize,
    #[serde(default = "default_dim")]
    dim: usize,
    intermediate_size: Option<usize>,
    #[serde(default = "negative_one")]
    n_local_heads: isize,
    head_dim: Option<usize>,
    #[serde(default = "default_rope")]
    rope_base: f32,
    #[serde(default = "default_eps")]
    norm_eps: f32,
    #[serde(default = "default_context")]
    max_seq_len: usize,
    #[serde(default = "default_codebook_size")]
    codebook_size: usize,
    #[serde(default = "default_codebooks")]
    num_codebooks: usize,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default = "default_true")]
    share_codebook_embeddings: bool,
    #[serde(default)]
    attention_qkv_bias: bool,
    #[serde(default = "default_fast_layers")]
    n_fast_layer: usize,
    fast_dim: Option<usize>,
    fast_n_head: Option<usize>,
    fast_n_local_heads: Option<usize>,
    fast_head_dim: Option<usize>,
    fast_intermediate_size: Option<usize>,
    fast_attention_qkv_bias: Option<bool>,
}

fn default_vocab() -> usize {
    32000
}
fn default_layers() -> usize {
    32
}
fn default_heads() -> usize {
    32
}
fn default_dim() -> usize {
    4096
}
fn negative_one() -> isize {
    -1
}
fn default_rope() -> f32 {
    10000.0
}
fn default_eps() -> f32 {
    1e-5
}
fn default_context() -> usize {
    2048
}
fn default_codebook_size() -> usize {
    160
}
fn default_codebooks() -> usize {
    4
}
fn default_true() -> bool {
    true
}
fn default_fast_layers() -> usize {
    4
}

fn intermediate_size(dim: usize) -> Result<usize> {
    let hidden = dim
        .checked_mul(4)
        .ok_or_else(|| ModelConfigError::Invalid("FFN dimension overflow".into()))?;
    let n_hidden = hidden
        .checked_mul(2)
        .map(|v| v / 3)
        .ok_or_else(|| ModelConfigError::Invalid("FFN dimension overflow".into()))?;
    n_hidden
        .checked_add(255)
        .map(|v| v / 256 * 256)
        .ok_or_else(|| ModelConfigError::Invalid("FFN dimension overflow".into()))
}

pub fn parse_checkpoint_config(json: &[u8]) -> Result<FishModelConfig> {
    let raw: RawModelArgs = serde_json::from_slice(json)?;
    let kind = match raw.model_type.as_str() {
        "dual_ar" => ModelKind::DualAr,
        "naive" => ModelKind::Naive,
        other => {
            return Err(ModelConfigError::Invalid(format!(
                "unknown Fish Speech 1.5 model_type: {other}"
            )));
        }
    };
    let local_heads = if raw.n_local_heads == -1 {
        raw.n_head
    } else {
        usize::try_from(raw.n_local_heads)
            .map_err(|_| ModelConfigError::Invalid("invalid n_local_heads".into()))?
    };
    if raw.n_head == 0 || raw.dim == 0 || raw.dim % raw.n_head != 0 {
        return Err(ModelConfigError::Invalid(
            "invalid slow attention head dimensions".into(),
        ));
    }
    let head_dim = raw.dim / raw.n_head;
    if raw.head_dim.is_some_and(|head| head != head_dim) {
        // The Python __post_init__ overwrites head_dim with dim // n_head.
        // Reject inconsistent checkpoints instead of constructing an incompatible graph.
        return Err(ModelConfigError::Invalid(
            "checkpoint head_dim disagrees with dim/n_head".into(),
        ));
    }
    let slow = TransformerDims {
        dim: raw.dim,
        n_heads: raw.n_head,
        n_kv_heads: local_heads,
        layers: raw.n_layer,
        intermediate_size: match raw.intermediate_size {
            Some(value) => value,
            None => intermediate_size(raw.dim)?,
        },
        qkv_bias: raw.attention_qkv_bias,
    };
    let fast = if kind == ModelKind::DualAr {
        let dim = raw.fast_dim.filter(|v| *v > 0).unwrap_or(raw.dim);
        let n_head = raw.fast_n_head.filter(|v| *v > 0).unwrap_or(raw.n_head);
        let n_local_heads = raw
            .fast_n_local_heads
            .filter(|v| *v > 0)
            .unwrap_or(local_heads);
        if n_head == 0 || dim % n_head != 0 {
            return Err(ModelConfigError::Invalid(
                "invalid Fast AR head dimensions".into(),
            ));
        }
        let effective_head_dim = raw.fast_head_dim.filter(|v| *v > 0).unwrap_or(head_dim);
        if effective_head_dim != dim / n_head {
            return Err(ModelConfigError::Invalid(
                "Fast AR head_dim differs from dim/n_head; unsupported checkpoint geometry".into(),
            ));
        }
        Some(TransformerDims {
            dim,
            n_heads: n_head,
            n_kv_heads: n_local_heads,
            layers: raw.n_fast_layer,
            intermediate_size: raw
                .fast_intermediate_size
                .filter(|v| *v > 0)
                .unwrap_or(slow.intermediate_size),
            qkv_bias: raw
                .fast_attention_qkv_bias
                .unwrap_or(raw.attention_qkv_bias),
        })
    } else {
        None
    };
    let parsed = FishModelConfig {
        kind,
        slow,
        fast,
        vocab_size: raw.vocab_size,
        num_codebooks: raw.num_codebooks,
        codebook_size: raw.codebook_size,
        max_seq_len: raw.max_seq_len,
        rope_base: raw.rope_base,
        norm_eps: raw.norm_eps,
        tie_word_embeddings: raw.tie_word_embeddings,
        share_codebook_embeddings: raw.share_codebook_embeddings,
    };
    parsed
        .validate()
        .map_err(|e| ModelConfigError::Invalid(e.to_string()))?;
    Ok(parsed)
}

pub fn load_checkpoint_config(path: impl AsRef<Path>) -> Result<FishModelConfig> {
    let p = path.as_ref();
    let p = if p.is_dir() {
        p.join("config.json")
    } else {
        p.to_owned()
    };
    if fs::metadata(&p)?.len() > 1024 * 1024 {
        return Err(ModelConfigError::Invalid(
            "config.json exceeds 1 MiB".into(),
        ));
    }
    parse_checkpoint_config(&fs::read(p)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_dual_ar_defaults_and_overrides() {
        let parsed = parse_checkpoint_config(br#"{"model_type":"dual_ar","n_layer":24,"n_fast_layer":4,"dim":1024,"n_head":16,"n_local_heads":2,"num_codebooks":8,"vocab_size":32000,"max_seq_len":4096,"fast_dim":768,"fast_n_head":12,"fast_head_dim":64,"fast_n_local_heads":3,"fast_attention_qkv_bias":true}"#).unwrap();
        assert_eq!(parsed.kind, ModelKind::DualAr);
        assert_eq!(parsed.slow.intermediate_size, 2816);
        assert_eq!(parsed.slow.qkv_width(), 1280);
        assert_eq!(parsed.fast.as_ref().unwrap().qkv_width(), 1152);
        assert_eq!(parsed.fast.as_ref().unwrap().intermediate_size, 2816);
        assert_eq!(parsed.fast.as_ref().unwrap().layers, 4);
        assert!(parsed.fast.as_ref().unwrap().qkv_bias);
        assert_eq!(parsed.num_codebooks, 8);
    }

    #[test]
    fn naive_skips_fast_even_when_defaults_exist() {
        let parsed = parse_checkpoint_config(br#"{"model_type":"naive","dim":64,"n_head":8,"n_layer":2,"n_local_heads":-1,"attention_qkv_bias":true}"#).unwrap();
        assert_eq!(parsed.kind, ModelKind::Naive);
        assert!(parsed.fast.is_none());
        assert_eq!(parsed.slow.n_kv_heads, 8);
        assert_eq!(parsed.slow.intermediate_size, 256);
        assert!(parsed.slow.qkv_bias);
    }

    #[test]
    fn invalid_head_and_unsupported_kind_fail() {
        for json in [
            r#"{"model_type":"dual_ar","dim":1000,"n_head":16}"#,
            r#"{"model_type":"naive","dim":64,"n_head":8,"head_dim":16}"#,
            r#"{"model_type":"llama3"}"#,
            r#"{"model_type":"dual_ar","dim":64,"n_head":8,"fast_dim":128}"#,
        ] {
            assert!(parse_checkpoint_config(json.as_bytes()).is_err());
        }
    }
}
