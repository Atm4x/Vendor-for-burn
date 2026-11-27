use super::language::LanguageConfig;
use crate::audio::MelConfig;
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Silu,
    Swish,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StackConfig {
    pub factor: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub query_dim: usize,
    pub value_dim: usize,
    pub ff_multiple: usize,
    pub conv_kernel: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextMode {
    pub chunk_size: i64,
    pub window: [i64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ZapformerConfig {
    pub in_channels: usize,
    pub frontend_channels: [usize; 3],
    pub embed_dim: usize,
    pub stacks: Vec<StackConfig>,
    pub causal_frontend: bool,
    pub causal: bool,
    pub dual_mode: bool,
    pub contexts: Vec<ContextMode>,
    pub chunk_window: [i64; 2],
    pub max_seq_len: usize,
    pub activation: Activation,
}

fn integer(root: &Value, key: &str, default: usize) -> Result<usize> {
    root.get(key).map_or(Ok(default), |v| {
        v.as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .with_context(|| format!("invalid integer {key}"))
    })
}

fn boolean(root: &Value, key: &str, default: bool) -> Result<bool> {
    root.get(key).map_or(Ok(default), |v| {
        v.as_bool()
            .with_context(|| format!("invalid boolean {key}"))
    })
}

fn signed_values(value: &Value, name: &str) -> Result<Vec<i64>> {
    let bad = || format!("invalid integer tuple {name}");
    let values = match value {
        Value::String(s) => s
            .split(',')
            .map(|x| x.trim().parse::<i64>().with_context(bad))
            .collect::<Result<Vec<_>>>()?,
        Value::Array(a) => a
            .iter()
            .map(|x| x.as_i64().with_context(bad))
            .collect::<Result<Vec<_>>>()?,
        Value::Number(n) => vec![n.as_i64().with_context(bad)?],
        _ => bail!("{}", bad()),
    };
    ensure!(!values.is_empty(), "empty tuple {name}");
    Ok(values)
}

fn dimensions(value: &Value, name: &str, count: Option<usize>) -> Result<Vec<usize>> {
    let mut values = signed_values(value, name)?
        .into_iter()
        .map(|n| usize::try_from(n).with_context(|| format!("negative {name}")))
        .collect::<Result<Vec<_>>>()?;
    if let Some(count) = count {
        if values.len() == 1 {
            values.resize(count, values[0]);
        }
        ensure!(
            values.len() == count,
            "{name} must have one or {count} entries"
        );
    }
    Ok(values)
}

fn window(value: &Value, name: &str) -> Result<[i64; 2]> {
    let values = signed_values(value, name)?;
    ensure!(
        values.len() == 2 && values.iter().all(|n| *n >= -1),
        "invalid {name} window"
    );
    Ok([values[0], values[1]])
}

impl ZapformerConfig {
    pub fn from_upstream(encoder: &Value) -> Result<Self> {
        ensure!(encoder.as_object().is_some(), "encoder must be an object");
        ensure!(
            encoder.get("_target_").and_then(Value::as_str)
                == Some("gigaam.encoder.ZapformerEncoder"),
            "expected ZapformerEncoder target"
        );
        ensure!(
            !boolean(encoder, "flash_attn", false)?,
            "Zapformer does not support flash_attn"
        );
        let field = |name: &str, default: Value, n: Option<usize>| {
            dimensions(encoder.get(name).unwrap_or(&default), name, n)
        };
        let factors = field("downsampling_factor", json!("1,2,4,2"), None)?;
        ensure!((1..=64).contains(&factors.len()), "invalid stack count");
        let n = Some(factors.len());
        let dims = field("encoder_dim", json!("320,512,768,512"), n)?;
        let layers = field("num_encoder_layers", json!("6,8,14,8"), n)?;
        let heads = field("num_heads", json!("4,4,4,4"), n)?;
        ensure!(heads.iter().all(|&h| h > 0), "num_heads must be positive");
        let qdims = match encoder.get("query_head_dim").filter(|v| !v.is_null()) {
            Some(v) => dimensions(v, "query_head_dim", n)?,
            None => dims.iter().zip(&heads).map(|(d, h)| d / h).collect(),
        };
        let vdims = field("value_head_dim", json!("96,96,96,96"), n)?;
        let multiples = field("feedforward_multiple", json!("4,4,3,4"), n)?;
        let kernels = field("conv_params", json!("31,31,15,31"), n)?;
        let channels = field("frontend_channels", json!("8,32,128"), Some(3))?;
        let embed_dim = match encoder.get("embed_dim").filter(|v| !v.is_null()) {
            Some(_) => integer(encoder, "embed_dim", dims[0])?,
            None => dims[0],
        };
        let causal = boolean(encoder, "causal", false)?;
        let dual_mode = boolean(encoder, "dual_mode", false)?;
        let chunks = signed_values(
            encoder.get("chunk_size").unwrap_or(&json!(-1)),
            "chunk_size",
        )?;
        let window_default = json!([-1, -1]);
        let windows = encoder.get("window_size").unwrap_or(&window_default);
        let contexts = if dual_mode {
            let windows = if let Some(a) = windows
                .as_array()
                .filter(|a| a.first().is_some_and(Value::is_array))
            {
                ensure!(
                    a.len() == chunks.len(),
                    "one window per chunk mode required"
                );
                a.iter()
                    .map(|v| window(v, "window_size"))
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![window(windows, "window_size")?; chunks.len()]
            };
            chunks
                .into_iter()
                .zip(windows)
                .map(|(chunk_size, window)| ContextMode { chunk_size, window })
                .collect()
        } else {
            ensure!(
                chunks.len() == 1,
                "non-dual encoder requires scalar chunk_size"
            );
            vec![ContextMode {
                chunk_size: if causal { chunks[0] } else { -1 },
                window: window(windows, "window_size")?,
            }]
        };
        let activation = match encoder
            .get("activation")
            .map_or(Some("swish"), Value::as_str)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("swish") => Activation::Swish,
            Some("silu") => Activation::Silu,
            _ => bail!("unsupported Zapformer activation"),
        };
        let stacks = (0..factors.len())
            .map(|i| StackConfig {
                factor: factors[i],
                dim: dims[i],
                layers: layers[i],
                heads: heads[i],
                query_dim: qdims[i],
                value_dim: vdims[i],
                ff_multiple: multiples[i],
                conv_kernel: kernels[i],
            })
            .collect();
        let config = Self {
            in_channels: integer(encoder, "in_channels", 64)?,
            frontend_channels: [channels[0], channels[1], channels[2]],
            embed_dim,
            stacks,
            causal_frontend: boolean(encoder, "causal_frontend", false)?,
            causal: causal && !dual_mode,
            dual_mode,
            contexts,
            chunk_window: window(
                encoder.get("chunk_window").unwrap_or(&window_default),
                "chunk_window",
            )?,
            max_seq_len: integer(encoder, "max_seq_len", 10000)?,
            activation,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            (7..=256).contains(&self.in_channels),
            "invalid frontend mel width"
        );
        ensure!(
            self.embed_dim > 0 && self.frontend_channels.iter().all(|&x| x > 0),
            "invalid frontend dimensions"
        );
        ensure!(
            !self.stacks.is_empty() && self.max_seq_len > 0,
            "invalid encoder geometry"
        );
        let max_factor = self.stacks.iter().map(|s| s.factor).max().unwrap();
        for s in &self.stacks {
            ensure!(
                s.factor > 0 && max_factor % s.factor == 0,
                "stack factors must divide maximum factor"
            );
            ensure!(
                s.dim > 0
                    && s.layers > 0
                    && s.heads > 0
                    && s.query_dim > 0
                    && s.query_dim % 2 == 0
                    && s.value_dim > 0
                    && s.ff_multiple > 0,
                "invalid stack geometry: {s:?}"
            );
            ensure!(
                s.conv_kernel == 0 || s.conv_kernel % 2 == 1,
                "convolution kernel must be zero or odd"
            );
            for (a, b) in [
                (self.embed_dim, s.factor),
                (s.heads, s.query_dim),
                (s.heads, s.value_dim),
                (s.dim, s.ff_multiple),
            ] {
                a.checked_mul(b)
                    .and_then(|v| v.checked_mul(3))
                    .context("stack dimension overflow")?;
            }
        }
        let mut seen = BTreeSet::new();
        ensure!(!self.contexts.is_empty(), "missing context modes");
        for mode in &self.contexts {
            ensure!(
                mode.chunk_size == -1 || mode.chunk_size > 0,
                "invalid chunk size"
            );
            ensure!(seen.insert(mode.chunk_size), "duplicate chunk mode");
            if mode.chunk_size > 0 {
                ensure!(
                    usize::try_from(mode.chunk_size)? % max_factor == 0,
                    "chunk must be divisible by maximum downsampling factor"
                );
            }
        }
        Ok(())
    }

    pub fn projection_width(&self) -> Result<usize> {
        self.frontend_channels[2]
            .checked_mul(((self.in_channels - 1) / 2 - 1) / 2)
            .context("frontend projection overflow")
    }

    pub fn supports_full_context(&self) -> bool {
        self.contexts.iter().any(|m| m.chunk_size == -1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TokenizerConfig {
    Vocabulary { tokens: Vec<String> },
    SentencePiece { checkpoint_path: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HeadConfig {
    Ctc {
        num_classes: usize,
        tokenizer: TokenizerConfig,
    },
    Rnnt {
        num_classes: usize,
        pred_hidden: usize,
        pred_rnn_layers: usize,
        joint_hidden: usize,
        max_symbols_per_step: usize,
        tokenizer: TokenizerConfig,
    },
}

fn parse_head(head: &Value, decoding: &Value, width: usize) -> Result<HeadConfig> {
    let target = head
        .get("_target_")
        .and_then(Value::as_str)
        .context("missing head target")?;
    let decoding_target = decoding
        .get("_target_")
        .and_then(Value::as_str)
        .context("missing decoding target")?;
    let tokenizer = if let Some(path) = decoding.get("model_path").filter(|x| !x.is_null()) {
        TokenizerConfig::SentencePiece {
            checkpoint_path: path
                .as_str()
                .filter(|p| !p.is_empty())
                .context("invalid tokenizer model_path")?
                .to_owned(),
        }
    } else {
        let tokens: Vec<String> = serde_json::from_value(
            decoding
                .get("vocabulary")
                .context("missing vocabulary")?
                .clone(),
        )?;
        TokenizerConfig::Vocabulary { tokens }
    };
    let validate_classes = |classes: usize| -> Result<()> {
        ensure!(classes > 1, "invalid head num_classes");
        if let TokenizerConfig::Vocabulary { tokens } = &tokenizer {
            ensure!(
                tokens.len() == classes - 1,
                "vocabulary does not match classes"
            );
        }
        Ok(())
    };
    match (target, decoding_target) {
        ("gigaam.decoder.CTCHead", "gigaam.decoding.CTCGreedyDecoding") => {
            let num_classes = integer(head, "num_classes", 0)?;
            validate_classes(num_classes)?;
            ensure!(
                integer(head, "feat_in", 0)? == width,
                "CTC encoder width mismatch"
            );
            Ok(HeadConfig::Ctc {
                num_classes,
                tokenizer,
            })
        }
        ("gigaam.decoder.RNNTHead", "gigaam.decoding.RNNTGreedyDecoding") => {
            let decoder = head.get("decoder").context("missing predictor")?;
            let joint = head.get("joint").context("missing joint")?;
            let num_classes = integer(decoder, "num_classes", 0)?;
            validate_classes(num_classes)?;
            let pred_hidden = integer(decoder, "pred_hidden", 0)?;
            let pred_rnn_layers = integer(decoder, "pred_rnn_layers", 0)?;
            let joint_hidden = integer(joint, "joint_hidden", 0)?;
            let max_symbols_per_step = integer(decoding, "max_symbols_per_step", 10)?;
            ensure!(
                pred_hidden > 0
                    && pred_rnn_layers > 0
                    && joint_hidden > 0
                    && max_symbols_per_step > 0,
                "invalid RNNT dimensions"
            );
            ensure!(
                integer(joint, "num_classes", 0)? == num_classes
                    && integer(joint, "enc_hidden", 0)? == width
                    && integer(joint, "pred_hidden", 0)? == pred_hidden,
                "RNNT joint geometry mismatch"
            );
            Ok(HeadConfig::Rnnt {
                num_classes,
                pred_hidden,
                pred_rnn_layers,
                joint_hidden,
                max_symbols_per_step,
                tokenizer,
            })
        }
        _ => bail!("unsupported head/decoding targets {target}/{decoding_target}"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HeadsConfig {
    EncoderOnly,
    Single { head: HeadConfig },
    Dual { raw: HeadConfig, e2e: HeadConfig },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CheckpointConfig {
    pub encoder: ZapformerConfig,
    pub mel: MelConfig,
    pub heads: HeadsConfig,
    pub language_conditioning: Option<LanguageConfig>,
    pub streaming: Option<Value>,
}

impl CheckpointConfig {
    pub fn from_upstream(cfg: &Value) -> Result<Self> {
        let encoder =
            ZapformerConfig::from_upstream(cfg.get("encoder").context("missing encoder")?)?;
        let pre = cfg.get("preprocessor").context("missing preprocessor")?;
        let mel = MelConfig::from_upstream(pre)?;
        ensure!(
            integer(pre, "features", 0)? == encoder.in_channels,
            "preprocessor.features and encoder.in_channels differ"
        );
        let heads = match (
            cfg.get("heads").filter(|v| !v.is_null()),
            cfg.get("head").filter(|v| !v.is_null()),
        ) {
            (Some(heads), _) => {
                let decodings = cfg.get("decodings").context("missing dual decodings")?;
                for section in [heads, decodings] {
                    ensure!(
                        section.as_object().is_some_and(|x| x.len() == 2
                            && x.contains_key("raw")
                            && x.contains_key("e2e")),
                        "expected raw/e2e head pair"
                    );
                }
                HeadsConfig::Dual {
                    raw: parse_head(&heads["raw"], &decodings["raw"], encoder.embed_dim)?,
                    e2e: parse_head(&heads["e2e"], &decodings["e2e"], encoder.embed_dim)?,
                }
            }
            (None, Some(head)) => HeadsConfig::Single {
                head: parse_head(
                    head,
                    cfg.get("decoding").context("missing decoding")?,
                    encoder.embed_dim,
                )?,
            },
            (None, None) => HeadsConfig::EncoderOnly,
        };
        let language_conditioning = cfg
            .get("language_conditioning")
            .filter(|v| !v.is_null())
            .map(|v| LanguageConfig::from_upstream(v, &encoder))
            .transpose()?
            .flatten();
        ensure!(
            language_conditioning.is_none() || matches!(heads, HeadsConfig::Dual { .. }),
            "language conditioning requires dual ASR heads"
        );
        Ok(Self {
            encoder,
            mel,
            heads,
            language_conditioning,
            streaming: cfg.get("streaming").filter(|v| !v.is_null()).cloned(),
        })
    }
}
