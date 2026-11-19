use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::Arc,
};

use cudarc::driver::CudaContext;
use fish_speech_core::{
    ar_stream_plan::{ArStreamPhase, ArStreamPlan},
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
    cuda_fast_sampling::CudaFastSampler,
    cuda_generation::{PreparedDualArPlans, generate_prepared_cuda_dual_ar},
    cuda_naive_generation::{PreparedNaiveArPlans, generate_prepared_cuda_naive},
    cuda_semantic_sampling::CudaSemanticSampler,
    cuda_stream::{CudaScratchPool, CudaWorkspace},
    cuda_trtx::CachedArDispatchConfig,
    device_memory::DeviceMemoryBudget,
    model_runtime::{FishModel, LoadedFishArtifacts, ModelPaths, ModelRuntimeError, Result},
    trtx_budget_allocator::BudgetedTrtAllocator,
    trtx_cache::{EngineIdentity, EngineRegistry},
    trtx_engine::{DeviceRegion, EngineError},
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

struct PreparedPlanCache<P> {
    entries: VecDeque<((usize, usize), Arc<P>)>,
}

impl<P> Default for PreparedPlanCache<P> {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }
}

impl<P> PreparedPlanCache<P> {
    fn clear(&mut self) {
        self.entries.clear();
    }

    fn get_or_prepare(
        &mut self,
        prompt_len: usize,
        max_new_tokens: usize,
        build: impl FnOnce() -> FishResult<P>,
    ) -> FishResult<Arc<P>> {
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

enum PreparedCudaArPlans {
    Dual(PreparedDualArPlans),
    Naive(PreparedNaiveArPlans),
}

impl PreparedCudaArPlans {
    fn new(config: &FishModelConfig, prompt_len: usize, max_new_tokens: usize) -> FishResult<Self> {
        match config.kind {
            ModelKind::DualAr => Ok(Self::Dual(PreparedDualArPlans::new(
                config,
                prompt_len,
                max_new_tokens,
            )?)),
            ModelKind::Naive => Ok(Self::Naive(PreparedNaiveArPlans::new(
                config,
                prompt_len,
                max_new_tokens,
            )?)),
        }
    }

    fn ar_plans(&self) -> Vec<&ArStreamPlan> {
        match self {
            Self::Dual(plans) => plans.ar_plans(),
            Self::Naive(plans) => plans.ar_plans(),
        }
    }
}

fn normalized_naive_tokens(
    config: &FishModelConfig,
    prompt_len: usize,
    requested: usize,
) -> FishResult<usize> {
    if config.kind != ModelKind::Naive || prompt_len == 0 || prompt_len >= config.max_seq_len {
        return Err(FishError::InvalidShape("invalid Naive prompt length"));
    }
    let remaining = config.max_seq_len - prompt_len;
    Ok(if requested == 0 {
        remaining
    } else {
        requested.min(remaining)
    })
}

pub struct CudaSemanticGenerator {
    config: FishModelConfig,
    checkpoint: LoadedWeights,
    options: CudaRuntimeOptions,
    registry: EngineRegistry,
    memory: DeviceMemoryBudget,
    scratch_pool: CudaScratchPool,
    semantic_lut: Vec<i32>,
    semantic_begin_id: u32,
    eos_id: u32,
    seed: u64,
    chunk_index: u64,
    prepared: PreparedPlanCache<PreparedCudaArPlans>,
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
        let scratch_pool = CudaScratchPool::new(
            options.device_ordinal,
            options
                .device_memory_limit_bytes
                .saturating_div(8)
                .min(64 * 1024 * 1024),
        );
        Ok(Self {
            config,
            checkpoint,
            options,
            registry,
            memory,
            scratch_pool,
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

    pub fn prepare_naive_request(
        &self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
    ) -> FishResult<PreparedNaiveArPlans> {
        let max_new_tokens =
            normalized_naive_tokens(&self.config, prompt.length(), max_new_tokens)?;
        if max_new_tokens > self.options.max_profiled_new_tokens {
            return Err(FishError::InvalidConfig(
                "CUDA request exceeds configured decode token limit",
            ));
        }
        let plans = PreparedNaiveArPlans::new(&self.config, prompt.length(), max_new_tokens)?;
        validate_ar_plan_weights(&plans.ar_plans(), &self.checkpoint.manifest().tensors)?;
        Ok(plans)
    }

    fn validate_plan_weights(&self, plans: &PreparedDualArPlans) -> FishResult<()> {
        validate_plan_weight_manifest(plans, &self.checkpoint.manifest().tensors)
    }

    /// Compiles and deserializes all request engine profiles, then loads the CUDA samplers.
    /// Execution contexts are request-scoped and are not retained across warmup and inference.
    pub fn warmup_request(
        &mut self,
        prompt: &EncodedConversation,
        requested_new_tokens: usize,
    ) -> FishResult<usize> {
        let tokens = if self.config.kind == ModelKind::Naive {
            normalized_naive_tokens(&self.config, prompt.length(), requested_new_tokens)?
        } else {
            requested_new_tokens
        };
        let plans = self.cached_request(prompt, tokens)?;
        let context = CudaContext::new(self.options.device_ordinal)
            .map_err(|_| FishError::InvalidTransition("CUDA warmup context failed"))?;
        context
            .bind_to_thread()
            .map_err(|_| FishError::InvalidTransition("CUDA warmup context bind failed"))?;
        let _fast = CudaFastSampler::new(&context)?;
        let _semantic = CudaSemanticSampler::new(&context)?;
        let logger = Logger::stderr()
            .map_err(|_| FishError::InvalidTransition("TensorRT warmup logger failed"))?;
        let mut entries = BTreeMap::new();
        for plan in plans.ar_plans() {
            for phase in &plan.phases {
                let ArStreamPhase::Network(blueprint) = phase else {
                    continue;
                };
                let identity = EngineIdentity::for_blueprint(
                    blueprint,
                    &self.checkpoint.manifest().source_checkpoint_sha256,
                    &self.options.gpu_identity,
                    &self.options.tensorrt_version,
                    &self.options.driver_version,
                    &self.options.precision,
                    match self.config.kind {
                        ModelKind::DualAr => "source-1.5-bucketed-slow-v1",
                        ModelKind::Naive => "source-1.5-naive-bucketed-v1",
                    },
                    &self.options.build_flags,
                )
                .map_err(|_| FishError::InvalidShape("invalid warmup TensorRT identity"))?;
                let key = identity
                    .key()
                    .map_err(|_| FishError::InvalidShape("invalid warmup engine key"))?;
                entries.entry(key).or_insert((blueprint, identity));
            }
        }
        let prepared = entries.into_values().collect::<Vec<_>>();
        let stream = context
            .new_stream()
            .map_err(|_| FishError::InvalidTransition("CUDA warmup stream creation failed"))?;
        let memory = &self.memory;
        self.registry
            .with_resident_sessions_managed(
                &logger,
                &prepared,
                self.options.workspace_bytes,
                || {
                    trtx::interfaces::GpuAllocator::new(Box::new(BudgetedTrtAllocator::new(
                        Arc::clone(&context),
                        memory.clone(),
                        self.scratch_pool.clone(),
                    )))
                    .map_err(|e| EngineError::TensorRt(e.to_string()))
                },
                |bytes| {
                    let allocation = CudaWorkspace::allocate(&stream, memory, bytes)
                        .or_else(|_| {
                            self.scratch_pool.clear()?;
                            CudaWorkspace::allocate(&stream, memory, bytes)
                        })
                        .map_err(|e| EngineError::TensorRt(e.to_string()))?;
                    let (address, size) = allocation.region(&stream);
                    Ok((DeviceRegion::new(address, size)?, allocation))
                },
                |contexts| Ok(contexts.len()),
                || {
                    stream
                        .synchronize()
                        .map_err(|e| EngineError::TensorRt(e.to_string()))
                },
            )
            .map_err(|_| FishError::InvalidTransition("TensorRT engine warmup failed"))
    }

    /// Releases cached CPU plans, serialized TensorRT engines and idle GPU scratch allocations.
    /// Active generation is excluded by `&mut self`; per-request contexts have already retired.
    pub fn unload(&mut self) -> Result<()> {
        self.scratch_pool
            .clear()
            .map_err(|e| ModelRuntimeError::Invalid(format!("GPU scratch unload: {e}")))?;
        self.registry
            .evict_memory()
            .map_err(|e| ModelRuntimeError::Invalid(format!("TensorRT engine unload: {e:?}")))?;
        self.prepared.clear();
        self.chunk_index = 0;
        if self.memory.usage().allocated_bytes != 0 {
            return Err(ModelRuntimeError::Invalid(
                "CUDA unload found outstanding device allocations".into(),
            ));
        }
        Ok(())
    }

    pub fn cached_scratch_bytes(&self) -> usize {
        self.scratch_pool.cached_bytes()
    }

    pub fn memory_usage(&self) -> crate::device_memory::DeviceMemoryUsage {
        self.memory.usage()
    }

    fn cached_request(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
    ) -> FishResult<Arc<PreparedCudaArPlans>> {
        if max_new_tokens > self.options.max_profiled_new_tokens {
            return Err(FishError::InvalidConfig(
                "CUDA request exceeds configured decode token limit",
            ));
        }
        let config = &self.config;
        let weights = &self.checkpoint.manifest().tensors;
        self.prepared
            .get_or_prepare(prompt.length(), max_new_tokens, || {
                let plans = PreparedCudaArPlans::new(config, prompt.length(), max_new_tokens)?;
                validate_ar_plan_weights(&plans.ar_plans(), weights)?;
                Ok(plans)
            })
    }
}

fn validate_plan_weight_manifest(
    plans: &PreparedDualArPlans,
    entries: &BTreeMap<String, TensorEntry>,
) -> FishResult<()> {
    validate_ar_plan_weights(&plans.ar_plans(), entries)
}

fn validate_ar_plan_weights(
    plans: &[&ArStreamPlan],
    entries: &BTreeMap<String, TensorEntry>,
) -> FishResult<()> {
    for &plan in plans {
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
        let max_new_tokens = if self.config.kind == ModelKind::Naive {
            normalized_naive_tokens(&self.config, prompt.length(), max_new_tokens)?
        } else {
            max_new_tokens
        };
        let plans = self.cached_request(prompt, max_new_tokens)?;
        let logger = Logger::stderr()
            .map_err(|_| FishError::InvalidTransition("TensorRT logger initialization failed"))?;
        let config = CachedArDispatchConfig {
            model_sha256: &self.checkpoint.manifest().source_checkpoint_sha256,
            gpu_identity: &self.options.gpu_identity,
            tensorrt_version: &self.options.tensorrt_version,
            driver_version: &self.options.driver_version,
            precision: &self.options.precision,
            shape_profile: match self.config.kind {
                ModelKind::DualAr => "source-1.5-bucketed-slow-v1",
                ModelKind::Naive => "source-1.5-naive-bucketed-v1",
            },
            build_flags: &self.options.build_flags,
            workspace_bytes: self.options.workspace_bytes,
        };
        let seed = self.seed.wrapping_add(self.chunk_index);
        let result = match plans.as_ref() {
            PreparedCudaArPlans::Dual(plans) => generate_prepared_cuda_dual_ar(
                &self.registry,
                &logger,
                config,
                &self.config,
                prompt,
                plans,
                &self.checkpoint,
                self.options.device_ordinal,
                self.memory.clone(),
                self.scratch_pool.clone(),
                &self.semantic_lut,
                self.semantic_begin_id,
                self.eos_id,
                sampling,
                seed,
                is_cancelled,
            ),
            PreparedCudaArPlans::Naive(plans) => generate_prepared_cuda_naive(
                &self.registry,
                &logger,
                config,
                &self.config,
                prompt,
                plans,
                &self.checkpoint,
                self.options.device_ordinal,
                self.memory.clone(),
                self.scratch_pool.clone(),
                &self.semantic_lut,
                self.eos_id,
                sampling,
                seed,
                is_cancelled,
            ),
        };
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
    fn lifecycle_clears_prepared_plan_cache_without_leaking_external_owners() {
        let mut cache: PreparedPlanCache<usize> = PreparedPlanCache::default();
        let first = cache.get_or_prepare(4, 2, || Ok(5)).unwrap();
        cache.clear();
        assert_eq!(*first, 5);
        assert!(cache.entries.is_empty());
        let replacement = cache.get_or_prepare(4, 2, || Ok(6)).unwrap();
        assert_eq!(*replacement, 6);
        assert!(!Arc::ptr_eq(&first, &replacement));
    }

    #[test]
    fn naive_unbounded_and_overlong_requests_follow_python_max_sequence_rule() {
        let mut naive = FishModelConfig::synthetic_dual_ar_fixture();
        naive.kind = ModelKind::Naive;
        naive.fast = None;
        assert_eq!(
            normalized_naive_tokens(&naive, 4, 0).unwrap(),
            naive.max_seq_len - 4
        );
        assert_eq!(
            normalized_naive_tokens(&naive, 4, usize::MAX).unwrap(),
            naive.max_seq_len - 4
        );
        assert_eq!(normalized_naive_tokens(&naive, 4, 2).unwrap(), 2);
        assert!(normalized_naive_tokens(&naive, 0, 2).is_err());
        assert!(normalized_naive_tokens(&naive, naive.max_seq_len, 2).is_err());
    }

    #[test]
    fn model_runtime_routes_dual_and_naive_to_separate_engine_contracts() {
        let dual = FishModelConfig::synthetic_dual_ar_fixture();
        let plans = PreparedCudaArPlans::new(&dual, 4, 3).unwrap();
        assert!(matches!(plans, PreparedCudaArPlans::Dual(_)));
        assert!(!plans.ar_plans().is_empty());
        let mut naive = dual;
        naive.kind = ModelKind::Naive;
        naive.fast = None;
        let plans = PreparedCudaArPlans::new(&naive, 4, 3).unwrap();
        assert!(matches!(plans, PreparedCudaArPlans::Naive(_)));
        assert_eq!(plans.ar_plans().len(), 2);
        for plan in plans.ar_plans() {
            let weights = required_ar_weights(plan).unwrap();
            assert!(weights.contains_key("codebook_output.weight"));
            assert!(!weights.contains_key("fast_output.weight"));
        }
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
