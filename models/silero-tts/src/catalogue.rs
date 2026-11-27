//! Offline copy of the upstream TTS catalogue. Download/cache ownership stays with the supervisor.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclaredFormat { Package, Jit }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogueSpeaker {
    pub lang: String,
    pub example: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelEntry {
    pub language: String,
    pub model_id: String,
    pub declared_format: DeclaredFormat,
    pub url: String,
    pub sample_rates: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speakers: Option<BTreeMap<String,CatalogueSpeaker>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Catalogue {
    pub source: String,
    pub source_sha256: String,
    pub models: Vec<ModelEntry>,
}

impl Catalogue {
    pub fn embedded() -> Result<Self> {
        let catalogue: Self = serde_json::from_str(include_str!("../official_catalogue.json"))?;
        let mut ids = HashSet::new();
        for m in &catalogue.models {
            ensure!(!m.language.is_empty() && !m.model_id.is_empty() && ids.insert((&m.language, &m.model_id)), "invalid or duplicate catalogue model");
            ensure!(m.url.starts_with("https://") && !m.sample_rates.is_empty() && m.sample_rates.iter().all(|&r| r > 0), "invalid catalogue artifact: {}", m.model_id);
            ensure!(m.declared_format != DeclaredFormat::Jit || m.tokenset.as_ref().is_some_and(|s| !s.is_empty()), "legacy model missing tokenset: {}", m.model_id);
        }
        Ok(catalogue)
    }

    pub fn resolve(&self, language: &str, model_id: &str) -> Result<&ModelEntry> {
        self.models.iter().find(|m| m.language == language && m.model_id == model_id)
            .with_context(|| format!("unknown Silero catalogue entry {language}/{model_id}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_catalogue_resolves_both_loading_routes() {
        let c = Catalogue::embedded().unwrap();
        assert_eq!(c.models.len(), 66);
        assert_eq!(c.models.iter().filter(|m| m.declared_format == DeclaredFormat::Package).count(), 46);
        let ru = c.resolve("ru", "v5_ru").unwrap();
        assert_eq!(ru.declared_format, DeclaredFormat::Package);
        assert_eq!(ru.sample_rates, [8000,24000,48000]);
        let jit = c.resolve("ru", "aidar_8khz").unwrap();
        assert_eq!(jit.declared_format, DeclaredFormat::Jit);
        assert!(jit.tokenset.as_ref().is_some_and(|s| s.starts_with("_~")));
        assert!(c.resolve("en", "v5_ru").is_err());
        assert_eq!(c.resolve("multi", "multi_v2").unwrap().speakers.as_ref().unwrap()["aidar"].lang, "ru");
        assert_eq!(c.resolve("ru", "v5_cis_base_nostress_jit").unwrap().declared_format, DeclaredFormat::Package);
    }
}
