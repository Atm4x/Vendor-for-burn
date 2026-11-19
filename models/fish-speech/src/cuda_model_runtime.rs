use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::Arc,
};

use fish_speech_core::{
    config::{FishModelConfig, ModelKind},
    conversation::EncodedConversation,
    cpu_weighted_ar::SelectionMode,
    error::{FishError, Result as FishResult},
    sampling::SamplingParams,
    synthesis::SemanticGenerator,
    tokenizer::{SEMANTIC_TOKEN_COUNT, SpecialTokens},
};
use trtx::Logger;

use crate::{
    ar_weight_bindings::required_ar_weights,
    cuda_generation::{PreparedDualArPlans, generate_prepared_cuda_dual_ar},
    cuda_trtx::CachedArDispatchConfig,
    device_memory::DeviceMemoryBudget,
    model_runtime::{FishModel, LoadedFishArtifacts, ModelPaths, ModelRuntimeError, Result},
    trtx_cache::EngineRegistry,
    weights::{LoadedWeights, TensorEntry},
};

#[derive(Debug, Clone)]
pub struct CudaRuntimeOptions {
    pub device_ordinal: usize,
    pub engine_cache_dir: PathBuf,
    pub engine_cache_per_engine_bytes: usize,
    pub engine_cache_resident_bytes: usize,
    pub workspace_bytes: usize,
    pub device_memory_limit_bytes: usize,
    pub max_profiled_new_tokens: usize,
    pub gpu_identity: String,
    pub tensorrt_version: String,
    pub driver_version: String,
    pub precision: String,
    pub build_flags: String,
}

impl CudaRuntimeOptions {
    pub fn validate(&self) -> Result<()> {
        if self.engine_cache_dir.as_os_str().is_empty()
            || self.workspace_bytes == 0
            || self.device_memory_limit_bytes == 0
            || !(1..=32768).contains(&self.max_profiled_new_tokens)
            || self.engine_cache_per_engine_bytes == 0
            || self.engine_cache_resident_bytes < self.engine_cache_per_engine_bytes
            || [
                &self.gpu_identity,
                &self.tensorrt_version,
                &self.driver_version,
                &self.precision,
                &self.build_flags,
            ]
            .iter()
            .any(|field| field.is_empty() || field.len() > 256)
        {
            return Err(ModelRuntimeError::Invalid(
                "invalid CUDA runtime options or engine identity".into(),
            ));
        }
        Ok(())
    }
}

const PREPARED_PLAN_CACHE_CAPACITY: usize = 8;

#[derive(Default)]
struct PreparedPlanCache {
    entries: VecDeque<((usize, usize), Arc<PreparedDualArPlans>)>,
}

impl PreparedPlanCache {
    fn get_or_prepare(
        &mut self,
        prompt_len: usize,
        max_new_tokens: usize,
        build: impl FnOnce() -> FishResult<PreparedDualArPlans>,
    ) -> FishResult<Arc<PreparedDualArPlans>> {
        let key = (prompt_len, max_new_tokens);
        if let Some(index) = self.entries.iter().position(|(stored, _)| *stored == key) {
            let (_, cached) = self.entries.remove(index).expect("existing cache entry");
            let shared = Arc::clone(&cached);
            self.entries.push_back((key, cached));
            return Ok(shared);
        }
        let built = Arc::new(build()?);
        if self.entries.len() >= PREPARED_PLAN_CACHE_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back((key, Arc::clone(&built)));
        Ok(built)
    }
}

pub struct CudaSemanticGenerator {
    config: FishModelConfig,
    checkpoint: LoadedWeights,
    options: CudaRuntimeOptions,
    registry: EngineRegistry,
    memory: DeviceMemoryBudget,
    semantic_lut: Vec<i32>,
    semantic_begin_id: u32,
    eos_id: u32,
    seed: u64,
    chunk_index: u64,
    prepared: PreparedPlanCache,
}

impl CudaSemanticGenerator {
    pub fn new(
        config: FishModelConfig,
        checkpoint: LoadedWeights,
        special: &SpecialTokens,
        options: CudaRuntimeOptions,
        seed: u64,
        mode: SelectionMode,
    ) -> Result<Self> {
        options.validate()?;
        config.validate()?;
        if config.kind != ModelKind::DualAr {
            return Err(ModelRuntimeError::Invalid(
                "native GPU model supports DualAR; Naive TensorRT is F14".into(),
            ));
        }
        if mode != SelectionMode::Sample {
            return Err(ModelRuntimeError::Invalid(
                "native GPU model currently supports sampled selection only".into(),
            ));
        }
        let end = special
            .semantic_begin
            .checked_add(SEMANTIC_TOKEN_COUNT)
            .ok_or_else(|| ModelRuntimeError::Invalid("semantic token range overflow".into()))?;
        if end > config.vocab_size || special.im_end >= config.vocab_size {
            return Err(ModelRuntimeError::Invalid(
                "semantic tokens exceed GPU model vocabulary".into(),
            ));
        }
        if checkpoint.manifest().component != "semantic" {
            return Err(ModelRuntimeError::Invalid(
                "GPU generator requires semantic checkpoint".into(),
            ));
        }
        let mut semantic_lut = vec![0; config.vocab_size];
        semantic_lut[special.semantic_begin..end].fill(1);
        let registry = EngineRegistry::new_with_resident_budget(
            &options.engine_cache_dir,
            options.engine_cache_per_engine_bytes,
            options.engine_cache_resident_bytes,
        )
        .map_err(|error| {
            ModelRuntimeError::Invalid(format!("TensorRT engine registry: {error:?}"))
        })?;
        let memory = DeviceMemoryBudget::new(options.device_memory_limit_bytes)
            .map_err(|error| ModelRuntimeError::Invalid(format!("GPU budget: {error:?}")))?;
        Ok(Self {
            config,
            checkpoint,
            options,
            registry,
            memory,
            semantic_lut,
            semantic_begin_id: special.semantic_begin as u32,
            eos_id: special.im_end as u32,
            seed,
            chunk_index: 0,
            prepared: PreparedPlanCache::default(),
        })
    }

    pub fn prepare_request(
        &self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
    ) -> FishResult<PreparedDualArPlans> {
        if max_new_tokens > self.options.max_profiled_new_tokens {
            return Err(FishError::InvalidConfig(
                "CUDA request exceeds configured decode token limit",
            ));
        }
        let plans = PreparedDualArPlans::new(&self.config, prompt.length(), max_new_tokens)?;
        self.validate_plan_weights(&plans)?;
        Ok(plans)
    }

    fn validate_plan_weights(&self, plans: &PreparedDualArPlans) -> FishResult<()> {
        validate_plan_weight_manifest(plans, &self.checkpoint.manifest().tensors)
    }

    fn cached_request(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
    ) -> FishResult<Arc<PreparedDualArPlans>> {
        if max_new_tokens > self.options.max_profiled_new_tokens {
            return Err(FishError::InvalidConfig(
                "CUDA request exceeds configured decode token limit",
            ));
        }
        let config = &self.config;
        let weights = &self.checkpoint.manifest().tensors;
        self.prepared
            .get_or_prepare(prompt.length(), max_new_tokens, || {
                let plans = PreparedDualArPlans::new(config, prompt.length(), max_new_tokens)?;
                validate_plan_weight_manifest(&plans, weights)?;
                Ok(plans)
            })
    }
}

fn validate_plan_weight_manifest(
    plans: &PreparedDualArPlans,
    entries: &BTreeMap<String, TensorEntry>,
) -> FishResult<()> {
    for plan in plans.ar_plans() {
        let needed = required_ar_weights(plan)
            .map_err(|_| FishError::InvalidShape("invalid CUDA AR weight requirements"))?;
        for (key, ty) in needed {
            let entry = entries
                .get(&key)
                .ok_or(FishError::InvalidShape("CUDA checkpoint weight missing"))?;
            if entry.shape != ty.shape || !matches!(entry.dtype.as_str(), "F32" | "F16" | "BF16") {
                return Err(FishError::InvalidShape(
                    "CUDA checkpoint weight dimensions or dtype mismatch",
                ));
            }
        }
    }
    Ok(())
}

impl SemanticGenerator for CudaSemanticGenerator {
    fn begin_sample(&mut self) -> FishResult<()> {
        self.chunk_index = 0;
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
    ) -> FishResult<Vec<Vec<u32>>> {
        self.generate_with_cancel(prompt, max_new_tokens, sampling, &mut || false)
    }

    fn generate_with_cancel(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> FishResult<Vec<Vec<u32>>> {
        if is_cancelled() {
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        sampling.validate()?;
        let plans = self.cached_request(prompt, max_new_tokens)?;
        let logger = Logger::stderr()
            .map_err(|_| FishError::InvalidTransition("TensorRT logger initialization failed"))?;
        let config = CachedArDispatchConfig {
            model_sha256: &self.checkpoint.manifest().source_checkpoint_sha256,
            gpu_identity: &self.options.gpu_identity,
            tensorrt_version: &self.options.tensorrt_version,
            driver_version: &self.options.driver_version,
            precision: &self.options.precision,
            shape_profile: "source-1.5-bucketed-slow-v1",
            build_flags: &self.options.build_flags,
            workspace_bytes: self.options.workspace_bytes,
        };
        let result = generate_prepared_cuda_dual_ar(
            &self.registry,
            &logger,
            config,
            &self.config,
            prompt,
            plans.as_ref(),
            &self.checkpoint,
            self.options.device_ordinal,
            self.memory.clone(),
            &self.semantic_lut,
            self.semantic_begin_id,
            self.eos_id,
            sampling,
            self.seed.wrapping_add(self.chunk_index),
            is_cancelled,
        );
        if result.is_ok() {
            self.chunk_index = self.chunk_index.wrapping_add(1);
        }
        result
    }
}

pub type CudaFishModel = FishModel<CudaSemanticGenerator>;

impl LoadedFishArtifacts {
    pub fn into_cuda_model(
        self,
        options: CudaRuntimeOptions,
        seed: u64,
        mode: SelectionMode,
    ) -> Result<CudaFishModel> {
        let special = self.tokenizer.special_tokens().clone();
        let generator = CudaSemanticGenerator::new(
            self.config.clone(),
            self.semantic,
            &special,
            options,
            seed,
            mode,
        )?;
        FishModel::from_components(self.config, self.tokenizer, self.firefly, generator)
    }
}

impl FishModel<CudaSemanticGenerator> {
    pub fn open_cuda(
        paths: &ModelPaths,
        options: CudaRuntimeOptions,
        seed: u64,
        mode: SelectionMode,
    ) -> Result<Self> {
        LoadedFishArtifacts::open(paths)?.into_cuda_model(options, seed, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> CudaRuntimeOptions {
        CudaRuntimeOptions {
            device_ordinal: 0,
            engine_cache_dir: std::env::temp_dir().join("fish-test-trtx-cache"),
            engine_cache_per_engine_bytes: 1024,
            engine_cache_resident_bytes: 4096,
            workspace_bytes: 1024,
            device_memory_limit_bytes: 16384,
            max_profiled_new_tokens: 4,
            gpu_identity: "test-sm89".into(),
            tensorrt_version: "1.7-test".into(),
            driver_version: "test-driver".into(),
            precision: "fp32".into(),
            build_flags: "test".into(),
        }
    }

    #[test]
    fn prepared_plan_cache_reuses_same_request_and_bounds_distinct_geometries() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let mut cache = PreparedPlanCache::default();
        let first = cache
            .get_or_prepare(4, 2, || PreparedDualArPlans::new(&config, 4, 2))
            .unwrap();
        let same = cache
            .get_or_prepare(4, 2, || panic!("prepared geometry rebuilt"))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        for prompt_len in 5..5 + PREPARED_PLAN_CACHE_CAPACITY {
            cache
                .get_or_prepare(prompt_len, 2, || {
                    PreparedDualArPlans::new(&config, prompt_len, 2)
                })
                .unwrap();
        }
        assert_eq!(cache.entries.len(), PREPARED_PLAN_CACHE_CAPACITY);
        assert!(!cache.entries.iter().any(|(key, _)| *key == (4, 2)));
        assert!(
            cache
                .get_or_prepare(config.max_seq_len, 2, || PreparedDualArPlans::new(
                    &config,
                    config.max_seq_len,
                    2
                ))
                .is_err()
        );
        assert_eq!(cache.entries.len(), PREPARED_PLAN_CACHE_CAPACITY);
    }

    #[test]
    fn runtime_settings_validate_without_initializing_a_gpu() {
        assert!(settings().validate().is_ok());
        let mut bad = settings();
        bad.max_profiled_new_tokens = 0;
        assert!(bad.validate().is_err());
        let mut bad = settings();
        bad.engine_cache_resident_bytes = 1;
        assert!(bad.validate().is_err());
        let mut bad = settings();
        bad.gpu_identity.clear();
        assert!(bad.validate().is_err());
        let mut bad = settings();
        bad.max_profiled_new_tokens = 32769;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn source_derived_gpu_weight_manifest_is_verified_before_any_enqueue() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let plans = PreparedDualArPlans::new(&config, 4, 3).unwrap();
        let mut entries = BTreeMap::new();
        for plan in plans.ar_plans() {
            for (key, ty) in required_ar_weights(plan).unwrap() {
                entries.insert(
                    key,
                    TensorEntry {
                        shard: "synthetic.safetensors".into(),
                        dtype: "F16".into(),
                        shape: ty.shape,
                        sha256: "0".repeat(64),
                    },
                );
            }
        }
        assert!(!entries.is_empty());
        assert!(validate_plan_weight_manifest(&plans, &entries).is_ok());
        let key = entries.keys().next().unwrap().clone();
        let saved = entries.remove(&key).unwrap();
        assert!(validate_plan_weight_manifest(&plans, &entries).is_err());
        entries.insert(key.clone(), saved);
        entries.get_mut(&key).unwrap().shape[0] += 1;
        assert!(validate_plan_weight_manifest(&plans, &entries).is_err());
    }

    #[test]
    fn automatically_bounded_plans_reject_an_oversized_decode_before_gpu_use() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        assert!(PreparedDualArPlans::new(&config, 4, 4).is_ok());
        assert!(PreparedDualArPlans::new(&config, 4, config.max_seq_len).is_err());
    }
}
