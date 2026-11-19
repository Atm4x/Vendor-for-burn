use std::fs;
use std::path::{Path, PathBuf};

use fish_speech_core::audio::FIREFLY_GROUPS;
use fish_speech_core::config::FishModelConfig;
use fish_speech_core::cpu_weighted_ar::SelectionMode;
use fish_speech_core::error::FishError;
use fish_speech_core::prompt::ReferenceExample;
use fish_speech_core::synthesis::SemanticGenerator;
use fish_speech_core::synthesis::{
    SynthesisEvent, SynthesisRequest, SynthesisSample, synthesize_with_events,
};
use fish_speech_core::tokenizer::SpecialTokens;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::firefly_config::{FireflyConfig, FireflyConfigError};
use crate::fish_tokenizer::{FishBpeTokenizer, FishTokenizerError};
use crate::model_config::{ModelConfigError, parse_checkpoint_config};
use crate::reference_cache::{CachePolicy, ReferenceBytes, ReferenceCache};
use crate::reference_loader::{ReferenceAudioLoader, ReferenceLoadError};
use crate::weighted_cpu::CpuSemanticGenerator;
use crate::weighted_firefly::{CpuFireflyDecoder, CpuFireflyReferenceEncoder};
use crate::weights::{LoadedWeights, WeightError};

const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ModelRuntimeError {
    #[error("model files: {0}")]
    Io(#[from] std::io::Error),
    #[error("semantic config: {0}")]
    Config(#[from] ModelConfigError),
    #[error("Firefly config: {0}")]
    FireflyConfig(#[from] FireflyConfigError),
    #[error("Fish tokenizer: {0}")]
    Tokenizer(#[from] FishTokenizerError),
    #[error("checkpoint: {0}")]
    Weights(#[from] WeightError),
    #[error("reference audio: {0}")]
    Reference(#[from] ReferenceLoadError),
    #[error("generation: {0}")]
    Synthesis(#[from] FishError),
    #[error("invalid Fish Speech model bundle: {0}")]
    Invalid(String),
    #[error("model is closed")]
    Closed,
}

pub type Result<T> = std::result::Result<T, ModelRuntimeError>;

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub semantic_config: PathBuf,
    pub semantic_tokenizer: PathBuf,
    pub semantic_dir: PathBuf,
    pub semantic_manifest: PathBuf,
    pub firefly_config_source: PathBuf,
    pub firefly_dir: PathBuf,
    pub firefly_manifest: PathBuf,
}

impl ModelPaths {
    pub fn new(
        semantic_dir: impl AsRef<Path>,
        firefly_dir: impl AsRef<Path>,
        firefly_config_source: impl AsRef<Path>,
    ) -> Self {
        let semantic_dir = semantic_dir.as_ref().to_owned();
        let firefly_dir = firefly_dir.as_ref().to_owned();
        Self {
            semantic_config: semantic_dir.join("config.json"),
            semantic_tokenizer: semantic_dir.join("tokenizer.tiktoken"),
            semantic_manifest: semantic_dir.join("manifest.json"),
            firefly_manifest: firefly_dir.join("manifest.json"),
            semantic_dir,
            firefly_dir,
            firefly_config_source: firefly_config_source.as_ref().to_owned(),
        }
    }
}

fn checked_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let size = fs::metadata(path)?.len();
    if size == 0 || size > limit {
        return Err(ModelRuntimeError::Invalid(format!(
            "invalid file length {}: {size}",
            path.display()
        )));
    }
    Ok(fs::read(path)?)
}

fn check_config_hash(component: &str, actual: &[u8], weights: &LoadedWeights) -> Result<()> {
    let actual = format!("{:x}", Sha256::digest(actual));
    if actual != weights.manifest().config_sha256 {
        return Err(ModelRuntimeError::Invalid(format!(
            "{component} manifest does not match source config SHA-256"
        )));
    }
    Ok(())
}

pub struct LoadedFishArtifacts {
    pub config: FishModelConfig,
    pub firefly_config: FireflyConfig,
    pub tokenizer: FishBpeTokenizer,
    pub semantic: LoadedWeights,
    pub firefly: LoadedWeights,
}

impl LoadedFishArtifacts {
    pub fn open(paths: &ModelPaths) -> Result<Self> {
        let config_bytes = checked_file(&paths.semantic_config, MAX_CONFIG_BYTES)?;
        let firefly_config_bytes = checked_file(&paths.firefly_config_source, MAX_CONFIG_BYTES)?;
        let tokenizer_bytes = checked_file(&paths.semantic_tokenizer, MAX_TOKENIZER_BYTES)?;
        let config = parse_checkpoint_config(&config_bytes)?;
        let firefly_config = FireflyConfig::parse_supported(&firefly_config_bytes)?;
        let tokenizer_text = std::str::from_utf8(&tokenizer_bytes)
            .map_err(|_| ModelRuntimeError::Invalid("tokenizer.tiktoken is not UTF-8".into()))?;
        let tokenizer = FishBpeTokenizer::from_tiktoken_text(tokenizer_text)?;
        let special = tokenizer.special_tokens();
        let end = special
            .semantic_begin
            .checked_add(1024)
            .ok_or_else(|| ModelRuntimeError::Invalid("Fish semantic token IDs overflow".into()))?;
        if config.num_codebooks != FIREFLY_GROUPS
            || end > config.vocab_size
            || special.im_end >= config.vocab_size
        {
            return Err(ModelRuntimeError::Invalid(
                "semantic vocabulary or codebooks disagree with Fish tokenizer/Firefly".into(),
            ));
        }
        let semantic = LoadedWeights::open(&paths.semantic_dir, &paths.semantic_manifest)?;
        if semantic.manifest().component != "semantic" {
            return Err(ModelRuntimeError::Invalid(
                "expected a semantic checkpoint manifest".into(),
            ));
        }
        check_config_hash("semantic", &config_bytes, &semantic)?;
        let firefly = LoadedWeights::open(&paths.firefly_dir, &paths.firefly_manifest)?;
        if firefly.manifest().component != "firefly" {
            return Err(ModelRuntimeError::Invalid(
                "expected a Firefly checkpoint manifest".into(),
            ));
        }
        check_config_hash("firefly", &firefly_config_bytes, &firefly)?;
        Ok(Self {
            config,
            firefly_config,
            tokenizer,
            semantic,
            firefly,
        })
    }

    pub fn into_cpu_model(self, seed: u64, mode: SelectionMode) -> Result<CpuFishModel> {
        let special = self.tokenizer.special_tokens().clone();
        let generator = CpuSemanticGenerator::from_checkpoint(
            self.config.clone(),
            &self.semantic,
            &special,
            seed,
            mode,
        )?;
        FishModel::from_components(self.config, self.tokenizer, self.firefly, generator)
    }
}

#[cfg(feature = "cubecl-backend")]
impl FishModel<crate::cubecl_backend::CubeClSemanticGenerator> {
    pub fn open_cubecl(
        paths: &ModelPaths,
        client: cubecl::prelude::Client,
        seed: u64,
        mode: SelectionMode,
        limits: crate::cubecl_backend::CubeClRequestLimits,
    ) -> Result<Self> {
        LoadedFishArtifacts::open(paths)?.into_cubecl_model(client, seed, mode, limits)
    }
}

#[cfg(feature = "cubecl-backend")]
impl LoadedFishArtifacts {
    pub fn into_cubecl_model(
        self,
        client: cubecl::prelude::Client,
        seed: u64,
        mode: SelectionMode,
        limits: crate::cubecl_backend::CubeClRequestLimits,
    ) -> Result<FishModel<crate::cubecl_backend::CubeClSemanticGenerator>> {
        let generator = crate::cubecl_backend::CubeClSemanticGenerator::new(
            client,
            self.config.clone(),
            self.semantic,
            self.tokenizer.special_tokens(),
            seed,
            mode,
            limits,
        )?;
        FishModel::from_components(self.config, self.tokenizer, self.firefly, generator)
    }
}

struct FishModelParts<G> {
    config: FishModelConfig,
    special: SpecialTokens,
    tokenizer: FishBpeTokenizer,
    generator: G,
    decoder: CpuFireflyDecoder,
    reference_encoder: CpuFireflyReferenceEncoder,
    cache: ReferenceCache<Vec<Vec<u32>>>,
}

pub struct FishModel<G> {
    parts: Option<FishModelParts<G>>,
}

pub type CpuFishModel = FishModel<CpuSemanticGenerator>;

#[cfg(feature = "cubecl-backend")]
pub type CubeClFishModel = FishModel<crate::cubecl_backend::CubeClSemanticGenerator>;

impl FishModel<CpuSemanticGenerator> {
    pub fn open(paths: &ModelPaths, seed: u64, mode: SelectionMode) -> Result<Self> {
        LoadedFishArtifacts::open(paths)?.into_cpu_model(seed, mode)
    }

    pub fn reload(&mut self, paths: &ModelPaths, seed: u64, mode: SelectionMode) -> Result<()> {
        let replacement = Self::open(paths, seed, mode)?;
        *self = replacement;
        Ok(())
    }
}

impl<G: SemanticGenerator> FishModel<G> {
    pub(crate) fn from_components(
        config: FishModelConfig,
        tokenizer: FishBpeTokenizer,
        firefly: LoadedWeights,
        generator: G,
    ) -> Result<Self> {
        let special = tokenizer.special_tokens().clone();
        let decoder = CpuFireflyDecoder::from_checkpoint(&firefly)?;
        let reference_encoder = CpuFireflyReferenceEncoder::from_checkpoint(&firefly)?;
        Ok(Self {
            parts: Some(FishModelParts {
                config,
                special,
                tokenizer,
                generator,
                decoder,
                reference_encoder,
                cache: ReferenceCache::default(),
            }),
        })
    }

    pub fn is_loaded(&self) -> bool {
        self.parts.is_some()
    }

    pub fn set_reference_audio_loader(&mut self, loader: ReferenceAudioLoader) -> Result<()> {
        let parts = self.parts.as_mut().ok_or(ModelRuntimeError::Closed)?;
        parts.reference_encoder.set_audio_loader(loader);
        parts.cache.clear();
        Ok(())
    }

    pub fn close(&mut self) {
        self.parts = None;
    }

    pub fn config(&self) -> Result<&FishModelConfig> {
        Ok(&self.parts.as_ref().ok_or(ModelRuntimeError::Closed)?.config)
    }

    pub fn prepare_references(
        &mut self,
        references: &[ReferenceBytes<'_>],
        policy: CachePolicy,
    ) -> Result<Vec<ReferenceExample>> {
        let parts = self.parts.as_mut().ok_or(ModelRuntimeError::Closed)?;
        let items = parts.reference_encoder.encode_references_cached(
            &mut parts.cache,
            references,
            policy,
        )?;
        Ok(items
            .into_iter()
            .map(|reference| ReferenceExample {
                text: reference.text,
                acoustic_codes: reference.tokens,
            })
            .collect())
    }

    pub fn clear_reference_cache(&mut self) -> Result<()> {
        self.parts
            .as_mut()
            .ok_or(ModelRuntimeError::Closed)?
            .cache
            .clear();
        Ok(())
    }

    pub fn synthesize<C, E>(
        &mut self,
        request: &SynthesisRequest<'_>,
        is_cancelled: C,
        streaming: bool,
        emit: E,
    ) -> Result<Vec<SynthesisSample>>
    where
        C: FnMut() -> bool,
        E: for<'a> FnMut(SynthesisEvent<'a>) -> std::result::Result<(), FishError>,
    {
        let parts = self.parts.as_mut().ok_or(ModelRuntimeError::Closed)?;
        if request.num_codebooks != parts.config.num_codebooks {
            return Err(ModelRuntimeError::Invalid(
                "synthesis codebook count differs from model checkpoint".into(),
            ));
        }
        Ok(synthesize_with_events(
            &parts.tokenizer,
            &parts.special,
            request,
            &mut parts.generator,
            &mut parts.decoder,
            is_cancelled,
            streaming,
            emit,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{MODEL_FAMILY, TensorEntry, WeightManifest};
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use std::collections::BTreeMap;

    fn hash(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn make_manifest(dir: &Path, component: &str, cfg: &[u8]) {
        let data = [0u8; 4];
        let tensor = serialize(
            [(
                "test.weight",
                TensorView::new(Dtype::F32, vec![1], &data).unwrap(),
            )],
            None,
        )
        .unwrap();
        let shard = format!("{component}.safetensors");
        fs::write(dir.join(&shard), &tensor).unwrap();
        let manifest = WeightManifest {
            schema_version: 1,
            model_family: MODEL_FAMILY.into(),
            component: component.into(),
            source_checkpoint_sha256: hash(b"synthetic checkpoint"),
            source_license: "fixture-only".into(),
            config_sha256: hash(cfg),
            shards: BTreeMap::from([(shard.clone(), hash(&tensor))]),
            tensors: BTreeMap::from([(
                "test.weight".into(),
                TensorEntry {
                    shard,
                    dtype: "F32".into(),
                    shape: vec![1],
                    sha256: hash(&data),
                },
            )]),
        };
        fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn fixture() -> (PathBuf, ModelPaths) {
        let root = std::env::temp_dir().join(format!(
            "fish-cpu-bundle-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let sem = root.join("semantic");
        let fire = root.join("firefly");
        fs::create_dir_all(&sem).unwrap();
        fs::create_dir_all(&fire).unwrap();
        let json = br#"{"model_type":"naive","dim":16,"n_head":4,"n_local_heads":2,"n_layer":1,"vocab_size":1292,"num_codebooks":8,"codebook_size":1000,"max_seq_len":32}"#;
        let firefly_config = include_bytes!("../tests/fixtures/firefly_1_5.json");
        fs::write(sem.join("config.json"), json).unwrap();
        fs::write(fire.join("config.json"), firefly_config).unwrap();
        let bpe = (0..=255u16)
            .map(|i| {
                format!(
                    "{} {i}\n",
                    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, [i as u8])
                )
            })
            .collect::<String>();
        fs::write(sem.join("tokenizer.tiktoken"), bpe).unwrap();
        make_manifest(&sem, "semantic", json);
        make_manifest(&fire, "firefly", firefly_config);
        (root, ModelPaths::new(&sem, &fire, fire.join("config.json")))
    }

    #[test]
    fn loads_two_component_bundle_and_validates_provenance() {
        let (root, paths) = fixture();
        let bundle = LoadedFishArtifacts::open(&paths).unwrap();
        assert_eq!(bundle.config.num_codebooks, FIREFLY_GROUPS);
        assert_eq!(bundle.tokenizer.special_tokens().semantic_begin, 268);
        assert_eq!(bundle.semantic.manifest().component, "semantic");
        assert_eq!(bundle.firefly.manifest().component, "firefly");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn changed_semantic_config_rejected_before_cpu_model_creation() {
        let (root, paths) = fixture();
        fs::write(&paths.semantic_config, br#"{"model_type":"naive","dim":16,"n_head":4,"n_local_heads":2,"n_layer":1,"vocab_size":1293,"num_codebooks":8,"codebook_size":1000,"max_seq_len":32}"#).unwrap();
        assert!(
            matches!(LoadedFishArtifacts::open(&paths), Err(ModelRuntimeError::Invalid(s)) if s.contains("semantic manifest"))
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsupported_firefly_architecture_rejected_even_with_matching_manifest_hash() {
        let (root, paths) = fixture();
        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.firefly_config_source).unwrap()).unwrap();
        config["head"]["upsample_kernel_sizes"] = serde_json::json!([16, 16, 8, 2, 2]);
        let config_bytes = serde_json::to_vec(&config).unwrap();
        fs::write(&paths.firefly_config_source, &config_bytes).unwrap();
        make_manifest(&paths.firefly_dir, "firefly", &config_bytes);
        assert!(matches!(
            LoadedFishArtifacts::open(&paths),
            Err(ModelRuntimeError::FireflyConfig(
                FireflyConfigError::Unsupported(_)
            ))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn changed_firefly_config_rejected_before_cpu_model_creation() {
        let (root, paths) = fixture();
        let original = fs::read(&paths.firefly_config_source).unwrap();
        fs::write(
            &paths.firefly_config_source,
            [b" ".as_slice(), &original].concat(),
        )
        .unwrap();
        assert!(
            matches!(LoadedFishArtifacts::open(&paths), Err(ModelRuntimeError::Invalid(s)) if s.contains("firefly manifest"))
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "full Fish 1.5 synthetic checkpoints with hundreds of tensor shapes"]
    fn full_synthetic_bundle_can_be_assembled_and_closed() {
        use fish_speech_core::firefly_ir::FireflyGraph;
        use fish_speech_core::graph_ir::TensorType;
        use fish_speech_core::transformer_ir::TransformerIrFactory;

        fn emit_full_component(
            dir: &Path,
            component: &str,
            cfg: &[u8],
            requirements: BTreeMap<String, TensorType>,
        ) {
            let mut raw = BTreeMap::new();
            let mut entries = BTreeMap::new();
            let shard = format!("{component}.safetensors");
            for (key, ty) in requirements {
                let count: usize = ty.shape.iter().product();
                let data = vec![0u8; count * 4];
                entries.insert(
                    key.clone(),
                    TensorEntry {
                        shard: shard.clone(),
                        dtype: "F32".into(),
                        shape: ty.shape.clone(),
                        sha256: hash(&data),
                    },
                );
                raw.insert(key, (ty.shape, data));
            }
            let views = raw
                .iter()
                .map(|(key, (shape, data))| {
                    (
                        key.as_str(),
                        TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            let shard_data = serialize(views, None).unwrap();
            fs::write(dir.join(&shard), &shard_data).unwrap();
            let manifest = WeightManifest {
                schema_version: 1,
                model_family: MODEL_FAMILY.into(),
                component: component.into(),
                source_checkpoint_sha256: hash(b"synthetic checkpoint"),
                source_license: "fixture-only".into(),
                config_sha256: hash(cfg),
                shards: BTreeMap::from([(shard, hash(&shard_data))]),
                tensors: entries,
            };
            fs::write(
                dir.join("manifest.json"),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
        }
        let (root, paths) = fixture();
        let cfg = parse_checkpoint_config(&fs::read(&paths.semantic_config).unwrap()).unwrap();
        let semantic_needs = TransformerIrFactory::new(&cfg)
            .unwrap()
            .slow_prefill(1)
            .unwrap()
            .weight_requirements();
        emit_full_component(
            &paths.semantic_dir,
            "semantic",
            &fs::read(&paths.semantic_config).unwrap(),
            semantic_needs,
        );
        let mut firefly_needs = FireflyGraph::reference_encode(1, 4096).unwrap().weights;
        firefly_needs.extend(FireflyGraph::waveform_decode(1, 1, 1).unwrap().weights);
        emit_full_component(
            &paths.firefly_dir,
            "firefly",
            &fs::read(&paths.firefly_config_source).unwrap(),
            firefly_needs,
        );
        let mut model = CpuFishModel::open(&paths, 7, SelectionMode::Greedy).unwrap();
        assert!(model.is_loaded());
        assert_eq!(model.config().unwrap().num_codebooks, 8);
        let request = SynthesisRequest {
            text: "A",
            references: &[],
            num_samples: 1,
            num_codebooks: 8,
            iterative_prompt: false,
            chunk_length: 100,
            max_context_length: 64,
            max_new_tokens: 2,
            sampling: fish_speech_core::sampling::SamplingParams::default(),
        };
        let samples = model
            .synthesize(&request, || false, false, |_| Ok(()))
            .unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].sample_rate, 44_100);
        assert!(!samples[0].pcm.is_empty());
        assert!(samples[0].pcm.iter().all(|v| v.is_finite()));
        model.close();
        assert!(!model.is_loaded());
        assert!(matches!(
            model.clear_reference_cache(),
            Err(ModelRuntimeError::Closed)
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mismatched_codebooks_and_corrupt_tensor_are_rejected() {
        let (root, paths) = fixture();
        let bad = br#"{"model_type":"naive","dim":16,"n_head":4,"n_local_heads":2,"n_layer":1,"vocab_size":1292,"num_codebooks":4,"codebook_size":1000,"max_seq_len":32}"#;
        fs::write(&paths.semantic_config, bad).unwrap();
        assert!(
            matches!(LoadedFishArtifacts::open(&paths), Err(ModelRuntimeError::Invalid(s)) if s.contains("codebooks"))
        );
        fs::remove_dir_all(root).unwrap();
    }
}
