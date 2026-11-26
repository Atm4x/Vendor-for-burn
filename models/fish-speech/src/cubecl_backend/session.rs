//! Model-scoped CubeCL weights and request-scoped stream-ordered KV ownership.
//! This layer does not sample tokens or depend on TensorRT.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

use cubecl::{prelude::*, server::Handle};
use fish_speech_core::{
    config::{FishModelConfig, ModelKind},
    cpu_ir::NamedTensors,
    error::{FishError, Result},
    graph::Stage,
    graph_ir::{Graph, Layout, Op, Origin, ScalarType, TensorType},
    sampling::SamplingParams,
    transformer_ir::TransformerIrFactory,
};

use super::{
    CubeClGraph, CubeClGraphExecutor,
    runtime::{is_cache_name, validate_binding},
    sampling::{
        decode_sample, enqueue_greedy, enqueue_top_p, greedy_device_token_with_handle,
        pack_selected_token_kernel, top_p_device_token_with_handle,
    },
};

mod fast_replay;
mod slow_replay;
use fast_replay::CapturedFastFrame;

// Full prefill keeps the temporary tensors for a voice reference and text in flight.
const DEFAULT_MAX_PENDING_BYTES: usize = 2 * 1024 * 1024 * 1024;
const DEFAULT_MAX_KV_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_PENDING_STEPS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeClRequestLimits {
    pub max_pending_bytes: usize,
    pub max_kv_bytes: usize,
    pub max_pending_steps: usize,
}

impl Default for CubeClRequestLimits {
    fn default() -> Self {
        Self {
            max_pending_bytes: DEFAULT_MAX_PENDING_BYTES,
            max_kv_bytes: DEFAULT_MAX_KV_BYTES,
            max_pending_steps: DEFAULT_MAX_PENDING_STEPS,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeClRequestCursor {
    pub slow_active_len: usize,
    pub fast_active_len: usize,
}

impl CubeClRequestCursor {
    pub fn validate_step(&self, graph: &Graph) -> Result<()> {
        graph.validate()?;
        let active = graph.active_kv_len.ok_or(FishError::InvalidShape(
            "CubeCL autoregression requires an active KV prefix",
        ))?;
        let capacity = graph.kv_capacity.ok_or(FishError::InvalidShape(
            "CubeCL autoregression requires a KV capacity",
        ))?;
        let expected_position = match graph.stage {
            Stage::SlowPrefill if self.slow_active_len == 0 => 0,
            Stage::SlowDecode if self.slow_active_len != 0 => self.slow_active_len,
            Stage::FastDecode if self.slow_active_len > 0 => self.fast_active_len,
            _ => {
                return Err(FishError::InvalidTransition(
                    "invalid CubeCL AR stage transition",
                ));
            }
        };
        let expected_active = match graph.stage {
            Stage::SlowPrefill => active,
            Stage::SlowDecode | Stage::FastDecode => expected_position
                .checked_add(1)
                .ok_or(FishError::InvalidShape("CubeCL prefix length overflow"))?,
            _ => unreachable!(),
        };
        if active != expected_active || active > capacity {
            return Err(FishError::InvalidTransition(
                "CubeCL AR prefix not contiguous",
            ));
        }
        let mut cache_updates = BTreeSet::new();
        for node in &graph.nodes {
            if let Op::KvUpdate { position } = node.op {
                let cache = &graph.values[node.inputs[0].0];
                if position != expected_position
                    || !matches!(cache.origin, Origin::Input)
                    || !is_cache_name(&cache.name)
                    || !cache_updates.insert(cache.name.as_str())
                    || !graph.outputs.contains(&node.result)
                {
                    return Err(FishError::InvalidShape(
                        "CubeCL KV update must be a unique exported cache write at expected position",
                    ));
                }
            }
        }
        if cache_updates.is_empty() {
            return Err(FishError::InvalidShape("CubeCL AR graph lacks KV updates"));
        }
        Ok(())
    }

    fn advance(&mut self, graph: &Graph) {
        let active = graph.active_kv_len.expect("validated active KV length");
        match graph.stage {
            Stage::SlowPrefill | Stage::SlowDecode => self.slow_active_len = active,
            Stage::FastDecode => self.fast_active_len = active,
            _ => unreachable!(),
        }
    }

    pub fn reset_fast(&mut self) {
        self.fast_active_len = 0;
    }
}

#[derive(Clone)]
pub struct CubeClDeviceTensor {
    ty: TensorType,
    handle: Handle,
    request_id: u64,
    verified_tokens: Option<Vec<i32>>,
}

impl CubeClDeviceTensor {
    pub fn shape_elements(&self) -> usize {
        self.ty.shape.iter().product()
    }

    pub(super) fn checked_token_shadow(&self, expected: &[i32]) -> Result<()> {
        if self.verified_tokens.as_deref() != Some(expected) {
            return Err(FishError::InvalidTransition(
                "CubeCL device tokens differ from host trace",
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct CubeClSampledToken {
    pub(super) token: u32,
    handle: Handle,
    request_id: u64,
}

pub(super) struct PendingFastSample {
    handle: Handle,
    request_id: u64,
}

pub struct CubeClRunResult {
    pub host: NamedTensors,
    pub device: BTreeMap<String, CubeClDeviceTensor>,
}

struct SlowDecodeProfiles {
    config: FishModelConfig,
    first: usize,
    slots: Vec<OnceLock<CubeClGraph>>,
}

/// Model plan collection. Preparing or compiling a plan does not allocate GPU memory.
pub struct CubeClPreparedModel {
    graphs: BTreeMap<String, CubeClGraph>,
    slow_decode: Option<SlowDecodeProfiles>,
    weights: BTreeMap<String, TensorType>,
    caches: BTreeMap<String, TensorType>,
}

impl CubeClPreparedModel {
    pub fn compile(graphs: impl IntoIterator<Item = (String, Graph)>) -> Result<Self> {
        let mut result = Self {
            graphs: BTreeMap::new(),
            slow_decode: None,
            weights: BTreeMap::new(),
            caches: BTreeMap::new(),
        };
        for (name, graph) in graphs {
            if name.is_empty() || result.graphs.contains_key(&name) {
                return Err(FishError::InvalidConfig(
                    "duplicate or empty CubeCL model plan key",
                ));
            }
            let compiled = CubeClGraph::compile(&graph)?;
            let mut has_cache_updates = false;
            for node in &graph.nodes {
                if matches!(node.op, Op::KvUpdate { .. }) {
                    has_cache_updates = true;
                    let cache = &graph.values[node.inputs[0].0];
                    if !matches!(cache.origin, Origin::Input)
                        || !is_cache_name(&cache.name)
                        || !graph.outputs.contains(&node.result)
                        || graph.values[node.result.0].ty != cache.ty
                    {
                        return Err(FishError::InvalidShape(
                            "CubeCL KV writes must export a tensor matching the source cache",
                        ));
                    }
                }
            }
            if !has_cache_updates {
                return Err(FishError::InvalidShape("CubeCL AR plan has no KV updates"));
            }
            for value in &graph.values {
                let requirements = match value.origin {
                    Origin::Weight { .. } => Some(&mut result.weights),
                    Origin::Input if is_cache_name(&value.name) => Some(&mut result.caches),
                    _ => None,
                };
                if let Some(requirements) = requirements {
                    if let Some(previous) =
                        requirements.insert(value.name.clone(), value.ty.clone())
                    {
                        if previous != value.ty {
                            return Err(FishError::InvalidShape(
                                "CubeCL incompatible weight or cache tensor across plans",
                            ));
                        }
                    }
                }
            }
            result.graphs.insert(name, compiled);
        }
        if result.graphs.is_empty() {
            return Err(FishError::InvalidConfig(
                "CubeCL model has no compiled plans",
            ));
        }
        Ok(result)
    }

    /// Prepare prefill, Fast AR and the first Slow decode position. Later
    /// position-specific graphs are materialized only when generation reaches them.
    pub fn from_transformer(
        config: &FishModelConfig,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> Result<Self> {
        if prompt_len == 0
            || max_new_tokens == 0
            || prompt_len >= config.max_seq_len
            || max_new_tokens > config.max_seq_len - prompt_len
        {
            return Err(FishError::InvalidShape(
                "invalid CubeCL Transformer profile bounds",
            ));
        }
        let factory = TransformerIrFactory::new(config)?;
        let mut graphs = vec![("slow_prefill".into(), factory.slow_prefill(prompt_len)?)];
        let first = prompt_len + 1;
        if max_new_tokens > 1 {
            graphs.push((format!("slow_decode_{first}"), factory.slow_decode(first)?));
        }
        if config.kind == ModelKind::DualAr {
            for step in 0..config.num_codebooks {
                graphs.push((format!("fast_{step}"), factory.fast_decode(step)?));
            }
        }
        let mut prepared = Self::compile(graphs)?;
        if max_new_tokens > 2 {
            prepared.slow_decode = Some(SlowDecodeProfiles {
                config: config.clone(),
                first: first + 1,
                slots: (0..max_new_tokens - 2).map(|_| OnceLock::new()).collect(),
            });
        }
        Ok(prepared)
    }

    pub fn graph(&self, key: &str) -> Result<&CubeClGraph> {
        if let Some(graph) = self.graphs.get(key) {
            return Ok(graph);
        }
        let prepared = self
            .slow_decode
            .as_ref()
            .and_then(|profile| {
                let active = key.strip_prefix("slow_decode_")?.parse::<usize>().ok()?;
                let slot = profile.slots.get(active.checked_sub(profile.first)?)?;
                Some((profile, active, slot))
            })
            .ok_or(FishError::InvalidConfig(
                "CubeCL model profile is not prepared",
            ))?;
        let (profile, active, slot) = prepared;
        if let Some(graph) = slot.get() {
            return Ok(graph);
        }
        let graph = TransformerIrFactory::new(&profile.config)?.slow_decode(active)?;
        let compiled = CubeClGraph::compile(&graph)?;
        for value in &graph.values {
            let known = match value.origin {
                Origin::Weight { .. } => self.weights.get(&value.name),
                Origin::Input if is_cache_name(&value.name) => self.caches.get(&value.name),
                _ => continue,
            };
            if known != Some(&value.ty) {
                return Err(FishError::InvalidShape(
                    "CubeCL lazy profile changed weight or cache layout",
                ));
            }
        }
        let _ = slot.set(compiled);
        Ok(slot.get().expect("Slow decode profile was initialized"))
    }

    pub fn plan_count(&self) -> usize {
        self.graphs.len()
            + self
                .slow_decode
                .as_ref()
                .map_or(0, |profile| profile.slots.len())
    }

    pub fn weight_specs(&self) -> &BTreeMap<String, TensorType> {
        &self.weights
    }

    pub fn cache_specs(&self) -> &BTreeMap<String, TensorType> {
        &self.caches
    }

    pub fn validate_weights(&self, weights: &NamedTensors) -> Result<()> {
        for (name, ty) in &self.weights {
            let weight = weights.get(name).ok_or(FishError::InvalidShape(
                "missing CubeCL model checkpoint weight",
            ))?;
            validate_binding(ty, weight)?;
        }
        if weights.len() != self.weights.len() {
            return Err(FishError::InvalidShape(
                "unknown CubeCL model checkpoint weight",
            ));
        }
        Ok(())
    }
}

/// Immutable device weights shared between requests; no other backend resources are created.
pub struct CubeClModel {
    prepared: CubeClPreparedModel,
    executor: CubeClGraphExecutor,
    weights: BTreeMap<String, Handle>,
}

impl CubeClModel {
    pub fn new(
        client: Client,
        prepared: CubeClPreparedModel,
        weights: &NamedTensors,
    ) -> Result<Self> {
        prepared.validate_weights(weights)?;
        let executor = CubeClGraphExecutor::new(client);
        let mut device_weights = BTreeMap::new();
        for (name, ty) in &prepared.weights {
            device_weights.insert(name.clone(), executor.upload(ty, &weights[name]));
        }
        Ok(Self {
            prepared,
            executor,
            weights: device_weights,
        })
    }

    /// Upload semantic safetensors directly, one tensor at a time, avoiding a
    /// complete intermediate F32 host checkpoint. Eligible BF16 projection
    /// weights retain their exact bits in packed U32 device buffers.
    pub fn from_checkpoint(
        client: Client,
        prepared: CubeClPreparedModel,
        checkpoint: &crate::weights::LoadedWeights,
    ) -> crate::weights::Result<Self> {
        use crate::weights::WeightError;
        use safetensors::Dtype;

        if checkpoint.manifest().component != "semantic" {
            return Err(WeightError::Invalid(
                "CubeCL requires semantic model weights".into(),
            ));
        }
        for (key, ty) in prepared.weight_specs() {
            let view = checkpoint.tensor(key)?;
            if view.shape() != ty.shape
                || ty.dtype != ScalarType::F32
                || !matches!(view.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16)
            {
                return Err(WeightError::Invalid(format!(
                    "CubeCL checkpoint weight dtype or shape mismatch: {key}"
                )));
            }
        }
        let executor = CubeClGraphExecutor::new(client);
        // The tied word table supports packed BF16 in both lookup and output projection.
        let mut unpacked = BTreeSet::new();
        for compiled in prepared.graphs.values() {
            let graph = compiled.graph();
            for node in &graph.nodes {
                for (index, id) in node.inputs.iter().enumerate() {
                    let value = &graph.values[id.0];
                    if matches!(value.origin, Origin::Weight { .. })
                        && (index != 1
                            || !matches!(
                                node.op,
                                Op::Linear | Op::LinearBias | Op::FishEmbedding { .. }
                            ))
                    {
                        unpacked.insert(value.name.as_str());
                    }
                }
            }
            unpacked.extend(
                graph
                    .outputs
                    .iter()
                    .map(|id| graph.values[id.0].name.as_str()),
            );
        }
        let mut weights = BTreeMap::new();
        for (key, ty) in prepared.weight_specs() {
            let view = checkpoint.tensor(key)?;
            let handle = if view.dtype() == Dtype::BF16
                && ty.shape.len() == 2
                && ty.shape[1] % 2 == 0
                && !unpacked.contains(key.as_str())
            {
                let data = view.data();
                if data
                    .chunks_exact(2)
                    .any(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]) & 0x7f80 == 0x7f80)
                {
                    return Err(WeightError::Invalid(format!(
                        "non-finite CubeCL weight: {key}"
                    )));
                }
                executor.client().create_from_slice(data)
            } else {
                let bytes = checkpoint.semantic_f32_upload_bytes(key, ty)?;
                executor.client().create_from_slice(&bytes)
            };
            weights.insert(key.clone(), handle);
        }
        Ok(Self {
            prepared,
            executor,
            weights,
        })
    }

    pub fn start_request(&self, limits: CubeClRequestLimits) -> Result<CubeClRequest<'_>> {
        if limits.max_pending_bytes == 0
            || limits.max_kv_bytes == 0
            || limits.max_pending_steps == 0
        {
            return Err(FishError::InvalidConfig(
                "CubeCL request limits must be positive",
            ));
        }
        Ok(CubeClRequest {
            model: self,
            request_id: NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed),
            limits,
            cursor: CubeClRequestCursor::default(),
            caches: BTreeMap::new(),
            retained: Vec::new(),
            pending_bytes: 0,
            pending_steps: 0,
            kv_bytes: 0,
            poisoned: false,
            last_sample: None,
            replay_context: None,
            slow_replay: None,
            fast_replay: None,
        })
    }

    pub fn prepared(&self) -> &CubeClPreparedModel {
        &self.prepared
    }
}

pub struct CubeClRequest<'a> {
    model: &'a CubeClModel,
    request_id: u64,
    limits: CubeClRequestLimits,
    cursor: CubeClRequestCursor,
    caches: BTreeMap<String, Handle>,
    retained: Vec<Handle>,
    pending_bytes: usize,
    pending_steps: usize,
    kv_bytes: usize,
    poisoned: bool,
    last_sample: Option<CubeClSampledToken>,
    replay_context: Option<CubeClDeviceTensor>,
    slow_replay: Option<slow_replay::SlowReplay>,
    fast_replay: Option<CapturedFastFrame>,
}

impl CubeClRequest<'_> {
    pub fn cursor(&self) -> CubeClRequestCursor {
        self.cursor
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn run(&mut self, key: &str, runtime: &NamedTensors) -> Result<NamedTensors> {
        self.run_with_cancel(key, runtime, &mut || false)
    }

    /// Cancellation is checked before dispatch and after returning from device
    /// readback. An in-flight cancellation poisons the request until an
    /// explicit stream-synchronized reset; it never reuses speculative KV.
    pub fn run_with_cancel(
        &mut self,
        key: &str,
        runtime: &NamedTensors,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<NamedTensors> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "CubeCL stream poisoned; reset required",
            ));
        }
        if is_cancelled() {
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        let result = self.run_inner(key, runtime, &BTreeMap::new(), None, None)?;
        if is_cancelled() {
            self.poisoned = true;
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        Ok(result.host)
    }

    /// Execute a graph with explicitly selected host readbacks. Other exported
    /// tensors remain device-owned and can be forwarded to a later graph.
    /// Only validated F32 fast_context may enter a graph from a device output;
    /// token IDs still require host-side bounds checks.
    pub fn sample_greedy(
        &mut self,
        logits: &CubeClDeviceTensor,
        offset: usize,
        vocabulary: usize,
        history: &[usize],
        params: SamplingParams,
        eos_id: u32,
    ) -> Result<(u32, bool)> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "CubeCL stream poisoned; reset required",
            ));
        }
        if logits.request_id != self.request_id {
            return Err(FishError::InvalidTransition(
                "CubeCL sampled tensor belongs to another request",
            ));
        }
        if logits.ty.dtype != ScalarType::F32 {
            return Err(FishError::InvalidShape(
                "CubeCL GPU sampler requires F32 logits",
            ));
        }
        let total = logits.ty.shape.iter().product();
        let extra = history
            .len()
            .max(1)
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or(FishError::InvalidShape(
                "CubeCL sampler scratch byte overflow",
            ))?;
        let projected = self
            .pending_bytes
            .checked_add(extra)
            .ok_or(FishError::InvalidShape(
                "CubeCL sampler scratch byte overflow",
            ))?;
        if projected > self.limits.max_pending_bytes
            || self.pending_steps >= self.limits.max_pending_steps
        {
            return Err(FishError::InvalidTransition(
                "CubeCL GPU sampler pending budget exceeded",
            ));
        }
        self.last_sample = None;
        let result = greedy_device_token_with_handle(
            self.model.executor.client(),
            &logits.handle,
            total,
            offset,
            vocabulary,
            history,
            params,
            eos_id,
            &mut self.retained,
        );
        self.pending_steps += 1;
        self.pending_bytes = projected;
        match result {
            Ok((token, eos, handle)) => {
                self.retire_after_readback();
                self.last_sample = Some(CubeClSampledToken {
                    token,
                    handle,
                    request_id: self.request_id,
                });
                Ok((token, eos))
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    pub fn sample_top_p(
        &mut self,
        logits: &CubeClDeviceTensor,
        offset: usize,
        vocabulary: usize,
        history: &[usize],
        params: SamplingParams,
        eos_id: u32,
        rng: &mut fish_speech_core::sampling::SessionRng,
    ) -> Result<(u32, bool)> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "CubeCL stream poisoned; reset required",
            ));
        }
        if logits.request_id != self.request_id {
            return Err(FishError::InvalidTransition(
                "CubeCL sampled tensor belongs to another request",
            ));
        }
        if logits.ty.dtype != ScalarType::F32 {
            return Err(FishError::InvalidShape(
                "CubeCL GPU sampler requires F32 logits",
            ));
        }
        let total = logits.ty.shape.iter().product();
        let extra = history
            .len()
            .max(1)
            .checked_mul(4)
            .and_then(|bytes| {
                vocabulary
                    .checked_mul(12)
                    .and_then(|scratch| bytes.checked_add(scratch))
            })
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or(FishError::InvalidShape(
                "CubeCL sampler scratch byte overflow",
            ))?;
        let projected = self
            .pending_bytes
            .checked_add(extra)
            .ok_or(FishError::InvalidShape(
                "CubeCL sampler scratch byte overflow",
            ))?;
        if projected > self.limits.max_pending_bytes
            || self.pending_steps >= self.limits.max_pending_steps
        {
            return Err(FishError::InvalidTransition(
                "CubeCL GPU sampler pending budget exceeded",
            ));
        }
        self.last_sample = None;
        let sampled = top_p_device_token_with_handle(
            self.model.executor.client(),
            &logits.handle,
            total,
            offset,
            vocabulary,
            history,
            params,
            eos_id,
            rng,
            &mut self.retained,
        );
        self.pending_steps += 1;
        self.pending_bytes = projected;
        match sampled {
            Ok((token, eos, handle)) => {
                self.retire_after_readback();
                self.last_sample = Some(CubeClSampledToken {
                    token,
                    handle,
                    request_id: self.request_id,
                });
                Ok((token, eos))
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    pub(super) fn run_with_pending_fast(
        &mut self,
        key: &str,
        token: &PendingFastSample,
    ) -> Result<CubeClRunResult> {
        self.run_inner(
            key,
            &NamedTensors::new(),
            &BTreeMap::new(),
            Some(&[]),
            Some(token),
        )
    }

    pub(super) fn enqueue_fast_sample(
        &mut self,
        logits: &CubeClDeviceTensor,
        vocabulary: usize,
        history: &[usize],
        params: SamplingParams,
        rng: Option<&mut fish_speech_core::sampling::SessionRng>,
    ) -> Result<PendingFastSample> {
        if self.poisoned
            || logits.request_id != self.request_id
            || logits.ty.dtype != ScalarType::F32
        {
            return Err(FishError::InvalidTransition(
                "invalid CubeCL Fast sampler input",
            ));
        }
        let extra = vocabulary
            .checked_mul(if rng.is_some() { 12 } else { 0 })
            .and_then(|bytes| bytes.checked_add(history.len().max(1).checked_mul(4)?))
            .and_then(|bytes| bytes.checked_add(8))
            .and_then(|bytes| bytes.checked_add(self.pending_bytes))
            .ok_or(FishError::InvalidShape(
                "CubeCL Fast sampler scratch byte overflow",
            ))?;
        if extra > self.limits.max_pending_bytes
            || self.pending_steps >= self.limits.max_pending_steps
        {
            return Err(FishError::InvalidTransition(
                "CubeCL GPU sampler pending budget exceeded",
            ));
        }
        let client = self.model.executor.client();
        let total = logits.shape_elements();
        let eos = i32::MAX as u32;
        let result = match rng {
            Some(rng) => enqueue_top_p(
                client,
                &logits.handle,
                total,
                0,
                vocabulary,
                history,
                params,
                eos,
                rng,
                &mut self.retained,
            ),
            None => enqueue_greedy(
                client,
                &logits.handle,
                total,
                0,
                vocabulary,
                history,
                params,
                eos,
                &mut self.retained,
            ),
        };
        self.pending_steps += 1;
        self.pending_bytes = extra;
        match result {
            Ok(handle) => Ok(PendingFastSample {
                handle,
                request_id: self.request_id,
            }),
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    pub(super) fn read_fast_samples(
        &mut self,
        pending: Vec<PendingFastSample>,
        vocabulary: usize,
    ) -> Result<Vec<CubeClSampledToken>> {
        if pending
            .iter()
            .any(|sample| sample.request_id != self.request_id)
        {
            return Err(FishError::InvalidTransition(
                "invalid pending Fast AR token ownership",
            ));
        }
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        let result = (|| {
            let handles = pending.iter().map(|sample| sample.handle.clone()).collect();
            let values =
                self.model.executor.client().read(handles).map_err(|_| {
                    FishError::InvalidTransition("CubeCL Fast samples readback failed")
                })?;
            if values.len() != pending.len() {
                return Err(FishError::InvalidShape("missing CubeCL Fast sample"));
            }
            pending
                .into_iter()
                .zip(values)
                .map(|(sample, bytes)| {
                    let (token, _) = decode_sample(&bytes, vocabulary, i32::MAX as u32)?;
                    Ok(CubeClSampledToken {
                        token,
                        handle: sample.handle,
                        request_id: self.request_id,
                    })
                })
                .collect()
        })();
        if result.is_err() {
            self.poisoned = true;
        } else {
            self.retire_after_readback();
        }
        result
    }

    pub(super) fn take_sampled(&mut self, expected: u32) -> Result<CubeClSampledToken> {
        let selected = self.last_sample.take().ok_or(FishError::InvalidTransition(
            "CubeCL selected token has no device handle",
        ))?;
        if selected.request_id != self.request_id || selected.token != expected {
            return Err(FishError::InvalidTransition(
                "CubeCL selected token identity mismatch",
            ));
        }
        Ok(selected)
    }

    /// Pack already validated sampled IDs into a complete Fish token column.
    /// A derived initial acoustic code is computed from the semantic device token.
    pub(super) fn pack_token_column(
        &mut self,
        sources: &[(&CubeClSampledToken, bool)],
        host_tokens: &[u32],
        semantic_begin: u32,
        codebook_size: usize,
        ty: TensorType,
    ) -> Result<CubeClDeviceTensor> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "CubeCL stream poisoned; reset required",
            ));
        }
        if sources.is_empty() || sources.len() != host_tokens.len() || codebook_size == 0 {
            return Err(FishError::InvalidShape(
                "CubeCL token column has invalid length",
            ));
        }
        let begin = i32::try_from(semantic_begin)
            .map_err(|_| FishError::InvalidShape("CubeCL semantic start exceeds i32"))?;
        let mut verified = Vec::with_capacity(sources.len());
        for ((selected, derived), &host) in sources.iter().zip(host_tokens) {
            if selected.request_id != self.request_id {
                return Err(FishError::InvalidTransition(
                    "CubeCL token belongs to another request",
                ));
            }
            let value = if *derived {
                selected.token.saturating_sub(semantic_begin)
            } else {
                selected.token
            };
            if value != host || value > i32::MAX as u32 {
                return Err(FishError::InvalidTransition(
                    "CubeCL host/device token column mismatch",
                ));
            }
            if *derived && (value as usize >= codebook_size || selected.token != sources[0].0.token)
            {
                return Err(FishError::InvalidShape(
                    "CubeCL semantic-derived acoustic code invalid",
                ));
            }
            verified.push(value as i32);
        }
        if ty.dtype != ScalarType::I32 || ty.shape.iter().product::<usize>() != verified.len() {
            return Err(FishError::InvalidShape(
                "CubeCL packed token type or shape mismatch",
            ));
        }
        let scratch = verified
            .len()
            .checked_mul(4)
            .ok_or(FishError::InvalidShape("CubeCL token scratch overflow"))?;
        let projected = self
            .pending_bytes
            .checked_add(scratch)
            .ok_or(FishError::InvalidShape("CubeCL token scratch overflow"))?;
        if projected > self.limits.max_pending_bytes
            || self.pending_steps >= self.limits.max_pending_steps
        {
            return Err(FishError::InvalidTransition(
                "CubeCL token column pending budget exceeded",
            ));
        }
        let output = self.model.executor.client().empty(scratch);
        for (index, (selected, derived)) in sources.iter().enumerate() {
            pack_selected_token_kernel::launch(
                self.model.executor.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { BufferArg::from_raw_parts(selected.handle.clone(), 2) },
                unsafe { BufferArg::from_raw_parts(output.clone(), verified.len()) },
                index,
                *derived,
                begin,
            );
        }
        self.retained.push(output.clone());
        self.pending_bytes = projected;
        self.pending_steps += 1;
        Ok(CubeClDeviceTensor {
            ty,
            handle: output,
            request_id: self.request_id,
            verified_tokens: Some(verified),
        })
    }

    pub fn run_selected(
        &mut self,
        key: &str,
        runtime: &NamedTensors,
        device_inputs: &BTreeMap<String, CubeClDeviceTensor>,
        host_readbacks: &[&str],
    ) -> Result<CubeClRunResult> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "CubeCL stream poisoned; reset required",
            ));
        }
        self.run_inner(key, runtime, device_inputs, Some(host_readbacks), None)
    }

    fn run_inner(
        &mut self,
        key: &str,
        runtime: &NamedTensors,
        device_inputs: &BTreeMap<String, CubeClDeviceTensor>,
        host_readbacks: Option<&[&str]>,
        pending_fast: Option<&PendingFastSample>,
    ) -> Result<CubeClRunResult> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "CubeCL stream poisoned; reset required",
            ));
        }
        let plan = self.model.prepared.graph(key)?;
        if device_inputs
            .values()
            .any(|tensor| tensor.request_id != self.request_id)
        {
            return Err(FishError::InvalidTransition(
                "CubeCL graph input belongs to another request",
            ));
        }
        let mut device_types: BTreeMap<_, _> = device_inputs
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.ty.clone()))
            .collect();
        let mut validation_inputs = runtime.clone();
        for name in ["tokens", "previous_acoustic_code"] {
            if let Some(tokens) = device_inputs.get(name) {
                if runtime.contains_key(name) {
                    return Err(FishError::InvalidShape(
                        "CubeCL token has duplicate host and device bindings",
                    ));
                }
                if tokens.ty.dtype != ScalarType::I32 {
                    return Err(FishError::InvalidShape(
                        "CubeCL forwarded token dtype must be I32",
                    ));
                }
                let ids = tokens
                    .verified_tokens
                    .as_ref()
                    .ok_or(FishError::InvalidTransition(
                        "CubeCL device tokens lack validated sampled origin",
                    ))?;
                validation_inputs.insert(
                    name.into(),
                    fish_speech_core::cpu_ir::TensorData::I32(ids.clone()),
                );
            }
        }
        if let Some(sample) = pending_fast {
            if sample.request_id != self.request_id
                || plan.graph().stage != Stage::FastDecode
                || runtime.contains_key("previous_acoustic_code")
                || device_types.contains_key("previous_acoustic_code")
            {
                return Err(FishError::InvalidTransition(
                    "invalid pending Fast AR token",
                ));
            }
            device_types.insert(
                "previous_acoustic_code".into(),
                TensorType::new([1, 1], ScalarType::I32, Layout::Plain),
            );
        }
        plan.validate_runtime_bindings(&validation_inputs, &device_types, pending_fast.is_some())?;
        self.cursor.validate_step(plan.graph())?;
        let graph = plan.graph();
        if let Some(selected) = host_readbacks {
            for &name in selected {
                if !graph.outputs.iter().any(|id| {
                    let value = &graph.values[id.0];
                    value.name == name && !is_cache_name(&value.name)
                }) {
                    return Err(FishError::InvalidShape(
                        "CubeCL requested readback is not a public graph output",
                    ));
                }
            }
        }
        let mut missing_kv = Vec::new();
        for value in &graph.values {
            if matches!(value.origin, Origin::Input) && is_cache_name(&value.name) {
                if self.caches.contains_key(&value.name) {
                    continue;
                }
                let expected_first = match graph.stage {
                    Stage::SlowPrefill => true,
                    Stage::FastDecode => self.cursor.fast_active_len == 0,
                    _ => false,
                };
                if !expected_first {
                    return Err(FishError::InvalidTransition(
                        "CubeCL decode attempted without previous KV state",
                    ));
                }
                missing_kv.push((value.name.clone(), value.ty.clone()));
            }
        }
        let new_kv = missing_kv.iter().try_fold(0usize, |bytes, (_, ty)| {
            bytes
                .checked_add(self.model.executor.tensor_bytes(ty)?)
                .ok_or(FishError::InvalidShape("CubeCL KV byte count overflow"))
        })?;
        let projected_kv = self
            .kv_bytes
            .checked_add(new_kv)
            .ok_or(FishError::InvalidShape("CubeCL KV byte count overflow"))?;
        let scratch = self.model.executor.scratch_bytes(plan)?;
        let projected_pending =
            self.pending_bytes
                .checked_add(scratch)
                .ok_or(FishError::InvalidShape(
                    "CubeCL scratch byte count overflow",
                ))?;
        if projected_kv > self.limits.max_kv_bytes
            || projected_pending > self.limits.max_pending_bytes
            || self.pending_steps >= self.limits.max_pending_steps
        {
            return Err(FishError::InvalidTransition(
                "CubeCL GPU memory or pending-submission budget exceeded",
            ));
        }
        let mut inputs = self.model.weights.clone();
        for value in &graph.values {
            if matches!(value.origin, Origin::Input) && !is_cache_name(&value.name) {
                inputs.insert(
                    value.name.clone(),
                    if let Some(sample) =
                        pending_fast.filter(|_| value.name == "previous_acoustic_code")
                    {
                        sample.handle.clone()
                    } else if let Some(device) = device_inputs.get(&value.name) {
                        device.handle.clone()
                    } else {
                        self.model.executor.upload(&value.ty, &runtime[&value.name])
                    },
                );
            }
        }
        for (name, ty) in missing_kv {
            self.caches.insert(name, self.model.executor.zeros(&ty)?);
        }
        self.kv_bytes = projected_kv;
        for value in &graph.values {
            if matches!(value.origin, Origin::Input) && is_cache_name(&value.name) {
                inputs.insert(value.name.clone(), self.caches[&value.name].clone());
            }
        }
        let result = self
            .model
            .executor
            .execute_handles(plan, &inputs, &mut self.retained, true);
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(err) => {
                self.poisoned = true;
                self.pending_steps += 1;
                self.pending_bytes = projected_pending;
                return Err(err);
            }
        };
        let mut next_caches = BTreeMap::new();
        for node in &graph.nodes {
            if let Op::KvUpdate { .. } = node.op {
                let name = &graph.values[node.inputs[0].0].name;
                next_caches.insert(
                    name.clone(),
                    outputs[&graph.values[node.result.0].name].clone(),
                );
            }
        }
        let cache_output_ids = graph
            .nodes
            .iter()
            .filter_map(|node| matches!(node.op, Op::KvUpdate { .. }).then_some(node.result))
            .collect::<BTreeSet<_>>();
        let mut host = NamedTensors::new();
        let mut device = BTreeMap::new();
        for output in &graph.outputs {
            if cache_output_ids.contains(output) {
                continue;
            }
            let value = &graph.values[output.0];
            let handle = &outputs[&value.name];
            device.insert(
                value.name.clone(),
                CubeClDeviceTensor {
                    ty: value.ty.clone(),
                    handle: handle.clone(),
                    request_id: self.request_id,
                    verified_tokens: None,
                },
            );
            if host_readbacks.is_none_or(|selected| selected.contains(&value.name.as_str())) {
                match self.model.executor.read(&value.ty, handle) {
                    Ok(data) => {
                        host.insert(value.name.clone(), data);
                    }
                    Err(err) => {
                        self.poisoned = true;
                        self.pending_steps += 1;
                        self.pending_bytes = projected_pending;
                        return Err(err);
                    }
                }
            }
        }
        self.caches.extend(next_caches);
        self.pending_steps += 1;
        self.pending_bytes = projected_pending;
        self.cursor.advance(graph);
        Ok(CubeClRunResult { host, device })
    }

    /// A new semantic token starts a fresh Fast AR history, without touching Slow KV.
    /// Old handles remain retained until `retire_completed` synchronizes the stream.
    pub fn reset_fast(&mut self) {
        self.caches
            .retain(|name, _| !name.starts_with("fast_layers."));
        self.kv_bytes = self
            .caches
            .keys()
            .filter_map(|key| self.model.prepared.caches.get(key))
            .filter_map(|ty| self.model.executor.tensor_bytes(ty).ok())
            .sum::<usize>()
            + self
                .fast_replay
                .as_ref()
                .map_or(0, |replay| replay.kv_bytes);
        self.cursor.reset_fast();
    }

    /// Explicitly wait for all outstanding submissions before retiring scratch handles.
    pub async fn retire_completed(&mut self) -> Result<()> {
        if self.model.executor.client().sync().await.is_err() {
            self.poisoned = true;
            return Err(FishError::InvalidTransition(
                "CubeCL stream synchronization failed",
            ));
        }
        self.retire_after_readback();
        Ok(())
    }

    // A successful blocking read orders all earlier work on this client's stream.
    fn retire_after_readback(&mut self) {
        self.retained.clear();
        self.pending_steps = 0;
        self.pending_bytes = self
            .fast_replay
            .as_ref()
            .map_or(0, |replay| replay.scratch_bytes)
            + self.slow_replay.as_ref().map_or(0, |s| s.bytes);
    }

    pub(crate) fn retire_pending(&mut self) -> Result<()> {
        if self.pending_steps > 0 {
            self.retire_blocking()?;
        }
        Ok(())
    }

    /// Synchronous retirement for the synchronous `SemanticGenerator` boundary.
    /// A wake-capable parker is used rather than spinning on a pending device future.
    pub fn retire_blocking(&mut self) -> Result<()> {
        use std::future::Future;
        use std::sync::Arc;
        use std::task::{Context, Poll, Wake, Waker};

        struct ThreadWake(std::thread::Thread);
        impl Wake for ThreadWake {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }
        let waker: Waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut pending = std::pin::pin!(self.retire_completed());
        loop {
            match pending.as_mut().poll(&mut context) {
                Poll::Ready(result) => return result,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    pub async fn reset_request(&mut self) -> Result<()> {
        self.replay_context = None;
        self.retire_completed().await?;
        self.caches.clear();
        self.slow_replay = None;
        self.fast_replay = None;
        self.pending_bytes = 0;
        self.kv_bytes = 0;
        self.cursor = CubeClRequestCursor::default();
        self.last_sample = None;
        self.poisoned = false;
        Ok(())
    }
}

impl Drop for CubeClRequest<'_> {
    fn drop(&mut self) {
        if self.pending_steps != 0 {
            let _ = self.retire_blocking();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fish_speech_core::{
        config::FishModelConfig,
        graph_ir::{Layout, ScalarType, TensorType},
        transformer_ir::TransformerIrFactory,
    };

    #[test]
    fn compiled_profiles_share_weight_specs_and_separate_slow_fast_kv() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        let plans = CubeClPreparedModel::compile([
            ("prefill".into(), factory.slow_prefill(3).unwrap()),
            ("decode4".into(), factory.slow_decode(4).unwrap()),
            ("fast0".into(), factory.fast_decode(0).unwrap()),
            ("fast1".into(), factory.fast_decode(1).unwrap()),
        ])
        .unwrap();
        assert_eq!(plans.plan_count(), 4);
        assert!(plans.weight_specs().len() > 30);
        assert!(plans.cache_specs().contains_key("layers.0.k_cache"));
        assert!(plans.cache_specs().contains_key("fast_layers.0.k_cache"));
        assert_eq!(
            plans.cache_specs()["layers.0.k_cache"].dtype,
            ScalarType::BF16
        );
        assert!(plans.graph("missing").is_err());
        let mut weights = NamedTensors::new();
        for (name, ty) in plans.weight_specs() {
            weights.insert(
                name.clone(),
                fish_speech_core::cpu_ir::TensorData::F32(vec![0.0; ty.shape.iter().product()]),
            );
        }
        plans.validate_weights(&weights).unwrap();
        weights.remove("norm.weight");
        assert!(plans.validate_weights(&weights).is_err());
        weights.insert(
            "norm.weight".into(),
            fish_speech_core::cpu_ir::TensorData::F32(vec![0.]),
        );
        assert!(plans.validate_weights(&weights).is_err());
    }

    #[test]
    fn shape_conflicts_and_duplicate_profiles_fail_before_device_use() {
        fn weighted_graph(width: usize) -> Graph {
            let mut g = Graph::new(Stage::FastDecode, Some(1), Some(2));
            let x = g
                .input(
                    "x",
                    TensorType::new([1, 1, width], ScalarType::F32, Layout::SequenceBtd),
                )
                .unwrap();
            let w = g
                .weight(
                    "weight",
                    TensorType::new([width, width], ScalarType::F32, Layout::Plain),
                )
                .unwrap();
            let y = g.emit("y", Op::Linear, &[x, w]).unwrap();
            g.output(y).unwrap();
            g
        }
        let a = weighted_graph(4);
        assert!(
            CubeClPreparedModel::compile([("same".into(), a.clone()), ("same".into(), a.clone())])
                .is_err()
        );
        assert!(
            CubeClPreparedModel::compile([("four".into(), a), ("eight".into(), weighted_graph(8))])
                .is_err()
        );
    }

    #[test]
    fn request_cursor_enforces_prefill_decode_and_fast_reset() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        let prefill = factory.slow_prefill(3).unwrap();
        let decode4 = factory.slow_decode(4).unwrap();
        let decode5 = factory.slow_decode(5).unwrap();
        let fast0 = factory.fast_decode(0).unwrap();
        let fast1 = factory.fast_decode(1).unwrap();
        let mut cursor = CubeClRequestCursor::default();
        assert!(cursor.validate_step(&fast0).is_err());
        assert!(cursor.validate_step(&decode4).is_err());
        cursor.validate_step(&prefill).unwrap();
        cursor.advance(&prefill);
        assert_eq!(cursor.slow_active_len, 3);
        assert!(cursor.validate_step(&prefill).is_err());
        assert!(cursor.validate_step(&decode5).is_err());
        cursor.validate_step(&decode4).unwrap();
        cursor.advance(&decode4);
        cursor.validate_step(&decode5).unwrap();
        cursor.validate_step(&fast0).unwrap();
        cursor.advance(&fast0);
        assert_eq!(cursor.fast_active_len, 1);
        assert!(cursor.validate_step(&fast0).is_err());
        cursor.validate_step(&fast1).unwrap();
        cursor.advance(&fast1);
        cursor.reset_fast();
        assert_eq!(cursor.fast_active_len, 0);
        assert_eq!(cursor.slow_active_len, 4);
        cursor.validate_step(&fast0).unwrap();
    }

    #[test]
    fn automatic_dual_ar_and_naive_profiles_have_exact_position_keys() {
        let dual = FishModelConfig::synthetic_dual_ar_fixture();
        let profiles = CubeClPreparedModel::from_transformer(&dual, 3, 4).unwrap();
        assert_eq!(profiles.plan_count(), dual.num_codebooks + 4);
        assert!(profiles.graph("slow_prefill").is_ok());
        for active in [4, 5, 6] {
            assert_eq!(
                profiles
                    .graph(&format!("slow_decode_{active}"))
                    .unwrap()
                    .graph()
                    .active_kv_len,
                Some(active)
            );
        }
        for i in 0..dual.num_codebooks {
            assert!(profiles.graph(&format!("fast_{i}")).is_ok());
        }
        assert!(CubeClPreparedModel::from_transformer(&dual, 0, 2).is_err());
        let long = CubeClPreparedModel::from_transformer(&dual, 3, 65).unwrap();
        assert_eq!(long.plan_count(), dual.num_codebooks + 65);
        assert!(
            long.slow_decode
                .as_ref()
                .unwrap()
                .slots
                .iter()
                .all(|slot| slot.get().is_none())
        );
        assert!(long.graph("slow_decode_67").is_ok());
        assert!(long.graph("slow_decode_68").is_err());
        assert!(CubeClPreparedModel::from_transformer(&dual, 3, 0).is_err());

        let mut naive = dual.clone();
        naive.kind = ModelKind::Naive;
        naive.fast = None;
        naive.validate().unwrap();
        let plans = CubeClPreparedModel::from_transformer(&naive, 3, 3).unwrap();
        assert_eq!(plans.plan_count(), 3);
        assert!(plans.graph("fast_0").is_err());
        assert!(plans.graph("slow_decode_4").is_ok());
        assert!(
            plans
                .graph("slow_prefill")
                .unwrap()
                .graph()
                .values
                .iter()
                .any(|value| value.name == "codebook_logits")
        );
    }

    #[test]
    fn request_inputs_validate_without_loading_weights_every_step() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let graph = TransformerIrFactory::new(&config)
            .unwrap()
            .slow_decode(4)
            .unwrap();
        let plan = CubeClGraph::compile(&graph).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert(
            "tokens".into(),
            fish_speech_core::cpu_ir::TensorData::I32(vec![0; config.num_codebooks + 1]),
        );
        inputs.insert(
            "semantic_membership_lut".into(),
            fish_speech_core::cpu_ir::TensorData::I32(vec![0; config.vocab_size]),
        );
        plan.validate_runtime_inputs(&inputs).unwrap();
        inputs.insert(
            "layers.0.k_cache".into(),
            fish_speech_core::cpu_ir::TensorData::F32(vec![0.0]),
        );
        assert!(plan.validate_runtime_inputs(&inputs).is_err());
        inputs.remove("layers.0.k_cache");
        inputs.insert(
            "tokens".into(),
            fish_speech_core::cpu_ir::TensorData::I32(vec![-1; config.num_codebooks + 1]),
        );
        assert!(plan.validate_runtime_inputs(&inputs).is_err());
    }
}
