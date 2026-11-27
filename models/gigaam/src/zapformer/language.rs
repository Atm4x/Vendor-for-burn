use super::config::ZapformerConfig;
use crate::{model::Linear, weights::Weights};
use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Device, Tensor, TensorData, activation};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::{BTreeMap, BTreeSet}, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LanguageConfig {
    pub languages: Vec<String>,
    pub num_slots: usize,
    pub embedding_dim: usize,
    pub default_tag: String,
    pub film_layers: Vec<usize>,
    pub late_fusion_hidden: Option<usize>,
}

impl LanguageConfig {
    pub fn from_upstream(cfg: &Value, encoder: &ZapformerConfig) -> Result<Option<Self>> {
        ensure!(cfg.is_object(), "language_conditioning must be an object");
        let enabled = |section: &str| -> Result<bool> {
            cfg.get(section)
                .and_then(|v| v.get("enabled"))
                .map_or(Ok(false), |v| {
                    v.as_bool().context("invalid language enabled flag")
                })
        };
        let film = enabled("film")?;
        let late = enabled("late_fusion")?;
        if !film && !late {
            return Ok(None);
        }
        let positive = |v: &Value, key: &str, default: usize| -> Result<usize> {
            let n = v.get(key).map_or(Ok(default), |x| {
                x.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .with_context(|| format!("invalid {key}"))
            })?;
            ensure!(n > 0, "{key} must be positive");
            Ok(n)
        };
        let languages: Vec<String> =
            serde_json::from_value(cfg.get("languages").context("missing languages")?.clone())?;
        let mut seen = BTreeSet::new();
        ensure!(
            !languages.is_empty()
                && languages.iter().all(|s| !s.is_empty()
                    && s == &s.trim().to_ascii_lowercase()
                    && s != "auto"
                    && s != "kz"
                    && !s.contains('-')
                    && seen.insert(s.clone())),
            "invalid language slot codes"
        );
        let num_slots = positive(cfg, "num_slots", 0)?;
        ensure!(
            num_slots > languages.len(),
            "language slots do not include auto plus all languages"
        );
        let embedding_dim = positive(cfg, "embedding_dim", 0)?;
        let default_tag = cfg
            .get("default_tag")
            .filter(|v| !v.is_null())
            .map_or(Ok("auto"), |v| {
                v.as_str().context("invalid default language")
            })?;
        let default_tag = if default_tag.is_empty() {
            "auto"
        } else {
            default_tag
        }
        .to_owned();
        let layer_count = encoder.stacks.iter().try_fold(0usize, |a, s| {
            a.checked_add(s.layers).context("layer count overflow")
        })?;
        let film_layers = if film {
            let f = &cfg["film"];
            match f.get("placement").map_or(Some("every_n"), Value::as_str) {
                Some("all") => (0..layer_count).collect(),
                Some("every_n") => {
                    let n = positive(f, "every_n", 4)?;
                    ((n - 1)..layer_count).step_by(n).collect()
                }
                Some("custom") => serde_json::from_value::<Vec<usize>>(
                    f.get("layers")
                        .context("missing FiLM layer indices")?
                        .clone(),
                )?,
                _ => bail!("invalid FiLM placement"),
            }
        } else {
            Vec::new()
        };
        let mut seen = BTreeSet::new();
        ensure!(
            film_layers
                .iter()
                .all(|&i| i < layer_count && seen.insert(i)),
            "invalid or duplicate FiLM layer index"
        );
        let late_fusion_hidden = if late {
            Some(positive(&cfg["late_fusion"], "hidden", 2048)?)
        } else {
            None
        };
        let config = Self {
            languages,
            num_slots,
            embedding_dim,
            default_tag,
            film_layers,
            late_fusion_hidden,
        };
        config.vector(&LanguageTag::Tag(config.default_tag.clone()))?;
        Ok(Some(config))
    }

    pub fn vector(&self, tag: &LanguageTag) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; self.num_slots];
        let mut assign = |code: &str, weight: f32| -> Result<()> {
            let code = code.trim().to_ascii_lowercase();
            let code = if code == "kz" { "kk" } else { code.as_str() };
            let slot = if code == "auto" {
                0
            } else {
                self.languages
                    .iter()
                    .position(|s| s == code)
                    .with_context(|| format!("unknown language {code:?}"))?
                    + 1
            };
            ensure!(
                weight.is_finite() && weight >= 0.0,
                "language weight must be finite and nonnegative"
            );
            vector[slot] += weight;
            Ok(())
        };
        match tag {
            LanguageTag::Weighted(weights) => {
                for (code, &weight) in weights {
                    assign(code, weight)?;
                }
            }
            _ => {
                let codes = match tag {
                    LanguageTag::Tag(value) if value.starts_with('[') => {
                        serde_json::from_str::<Vec<String>>(value)
                            .context("invalid JSON language list")?
                    }
                    LanguageTag::Tag(value) => vec![value.clone()],
                    LanguageTag::Codes(values) => values.clone(),
                    LanguageTag::Weighted(_) => unreachable!(),
                };
                // Upstream deduplicates normalized literal tags before resolving the kz alias.
                let unique = codes
                    .iter()
                    .flat_map(|value| {
                        value
                            .trim()
                            .to_ascii_lowercase()
                            .split('-')
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .collect::<BTreeSet<_>>();
                for code in unique {
                    assign(&code, 1.0)?;
                }
            }
        }
        let total: f32 = vector.iter().sum();
        ensure!(
            total.is_finite() && total > 0.0,
            "language mixture must have positive total weight"
        );
        for value in &mut vector {
            *value /= total;
        }
        Ok(vector)
    }
}

/// One utterance's language selection, not mutable model-wide state.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum LanguageTag {
    Tag(String),
    Codes(Vec<String>),
    Weighted(BTreeMap<String, f32>),
}

struct Fusion {
    input: Linear,
    output: Linear,
}

pub struct LanguageConditioner {
    config: LanguageConfig,
    embedding: Tensor<2>,
    film: BTreeMap<usize, Linear>,
    fusion: Option<Fusion>,
}

pub struct LanguageRequest {
    owner: Arc<LanguageConditioner>,
    embedding: Tensor<2>,
}

impl LanguageConditioner {
    pub fn load(
        w: &mut Weights,
        config: LanguageConfig,
        encoder: &ZapformerConfig,
        d: &Device,
    ) -> Result<Arc<Self>> {
        let embedding = w.checked_tensor(
            "language_embedding.weight",
            [config.num_slots, config.embedding_dim],
            d,
        )?;
        let widths = encoder
            .stacks
            .iter()
            .flat_map(|s| std::iter::repeat_n(s.dim, s.layers))
            .collect::<Vec<_>>();
        let mut film = BTreeMap::new();
        for &i in &config.film_layers {
            let width = *widths.get(i).context("FiLM layer outside encoder")?;
            film.insert(
                i,
                Linear::load_checked(
                    w,
                    &format!("encoder.language_film.projections.{i}"),
                    d,
                    config.embedding_dim,
                    2 * width,
                )?,
            );
        }
        let fusion = config
            .late_fusion_hidden
            .map(|hidden| -> Result<Fusion> {
                Ok(Fusion {
                    input: Linear::load_checked(
                        w,
                        "language_fusion.kernel.0",
                        d,
                        encoder.embed_dim + config.embedding_dim,
                        hidden,
                    )?,
                    output: Linear::load_checked(
                        w,
                        "language_fusion.kernel.3",
                        d,
                        hidden,
                        encoder.embed_dim,
                    )?,
                })
            })
            .transpose()?;
        Ok(Arc::new(Self {
            config,
            embedding,
            film,
            fusion,
        }))
    }

    pub fn request(
        self: &Arc<Self>,
        tags: Option<&[LanguageTag]>,
        batch: usize,
    ) -> Result<LanguageRequest> {
        ensure!(batch > 0, "empty language batch");
        let vectors = if let Some(tags) = tags {
            ensure!(tags.len() == batch, "language batch size mismatch");
            tags.iter()
                .map(|tag| self.config.vector(tag))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect()
        } else {
            self.config
                .vector(&LanguageTag::Tag(self.config.default_tag.clone()))?
                .repeat(batch)
        };
        let vectors = Tensor::<2>::from_data(
            TensorData::new(vectors, [batch, self.config.num_slots]),
            &self.embedding.device(),
        );
        Ok(LanguageRequest {
            owner: Arc::clone(self),
            embedding: vectors.matmul(self.embedding.clone()),
        })
    }
}
impl LanguageRequest {
    pub fn batch_size(&self) -> usize {
        self.embedding.dims()[0]
    }
    pub fn embedding(&self) -> Tensor<2> {
        self.embedding.clone()
    }
    pub(super) fn apply_layer(&self, index: usize, x: Tensor<3>) -> Tensor<3> {
        let Some(proj) = self.owner.film.get(&index) else {
            return x;
        };
        let [b, _, width] = x.dims();
        let params = proj.forward(self.embedding.clone().unsqueeze_dim::<3>(1));
        let gamma = params.clone().slice([0..b, 0..1, 0..width]);
        let beta = params.slice([0..b, 0..1, width..2 * width]);
        x * (gamma + 1.0) + beta
    }
    pub fn fuse(&self, x: Tensor<3>) -> Tensor<3> {
        let Some(fusion) = &self.owner.fusion else {
            return x;
        };
        let time = x.dims()[1];
        let language = self
            .embedding
            .clone()
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, time);
        x.clone()
            + fusion.output.forward(activation::relu(
                fusion.input.forward(Tensor::cat(vec![x, language], 2)),
            ))
    }
}
