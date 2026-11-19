use super::kernels::*;
use cubecl::prelude::*;
use cubecl::server::Handle;
use fish_speech_core::{
    cpu_ir::{NamedTensors, TensorData},
    error::{FishError, Result},
    graph_ir::{Graph, Op, Origin, QkvPart, ScalarType, TensorType},
    graph_stream_plan::GraphExecutionPlan,
};
use half::bf16;
use std::collections::BTreeMap;

fn count(ty: &TensorType) -> Result<usize> {
    ty.shape
        .iter()
        .try_fold(1usize, |n, &x| n.checked_mul(x))
        .ok_or(FishError::InvalidShape(
            "CubeCL tensor element count overflow",
        ))
}

fn validated_scalar(ty: &TensorType) -> Result<usize> {
    count(ty)
}

fn verify_binding(ty: &TensorType, data: &TensorData) -> Result<()> {
    let n = validated_scalar(ty)?;
    let valid = match (ty.dtype, data) {
        (ScalarType::F32 | ScalarType::BF16, TensorData::F32(v)) => v.len() == n,
        (ScalarType::I32, TensorData::I32(v)) => v.len() == n,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(FishError::InvalidShape(
            "CubeCL input type or shape mismatch",
        ))
    }
}

fn supported(op: Op) -> bool {
    matches!(
        op,
        Op::Identity
            | Op::Linear
            | Op::LinearBias
            | Op::RmsNorm { .. }
            | Op::Add
            | Op::Mul
            | Op::SiLu
            | Op::LastToken
            | Op::TokenEmbedding
            | Op::FishEmbedding { .. }
            | Op::SplitQkv { .. }
            | Op::RoPe { .. }
            | Op::KvUpdate { .. }
            | Op::KvPrefix { .. }
            | Op::GqaAttention { .. }
    )
}

pub(super) fn is_cache_name(name: &str) -> bool {
    name.ends_with(".k_cache") || name.ends_with(".v_cache")
}

pub(super) fn tensor_bytes(ty: &TensorType) -> Result<usize> {
    count(ty)?
        .checked_mul(match ty.dtype {
            ScalarType::BF16 => 2,
            _ => 4,
        })
        .ok_or(FishError::InvalidShape("CubeCL tensor byte count overflow"))
}

pub(super) fn validate_binding(ty: &TensorType, data: &TensorData) -> Result<()> {
    verify_binding(ty, data)
}

pub struct CubeClGraph {
    graph: Graph,
    execution: GraphExecutionPlan,
}

impl CubeClGraph {
    pub fn compile(graph: &Graph) -> Result<Self> {
        graph.validate()?;
        for node in &graph.nodes {
            if !supported(node.op) {
                return Err(FishError::Unsupported(
                    "CubeCL graph operation not implemented yet",
                ));
            }
        }
        for value in &graph.values {
            validated_scalar(&value.ty)?;
        }
        for node in &graph.nodes {
            if matches!(node.op, Op::TokenEmbedding | Op::FishEmbedding { .. })
                && !matches!(graph.values[node.inputs[0].0].origin, Origin::Input)
            {
                return Err(FishError::Unsupported(
                    "CubeCL token IDs produced by device graph require device-side error reporting",
                ));
            }
            if matches!(node.op, Op::FishEmbedding { .. })
                && !matches!(graph.values[node.inputs[3].0].origin, Origin::Input)
            {
                return Err(FishError::Unsupported(
                    "CubeCL semantic LUT produced by device graph is not yet supported",
                ));
            }
        }
        Ok(Self {
            graph: graph.clone(),
            execution: GraphExecutionPlan::from_graph(graph)?,
        })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn execution(&self) -> &GraphExecutionPlan {
        &self.execution
    }

    pub fn validate_inputs(&self, inputs: &NamedTensors) -> Result<()> {
        for value in &self.graph.values {
            if matches!(value.origin, Origin::Input | Origin::Weight { .. }) {
                let data = inputs
                    .get(&value.name)
                    .ok_or(FishError::InvalidShape("missing CubeCL graph input"))?;
                verify_binding(&value.ty, data)?;
            }
        }
        for supplied in inputs.keys() {
            if !self.graph.values.iter().any(|v| {
                v.name == *supplied && matches!(v.origin, Origin::Input | Origin::Weight { .. })
            }) {
                return Err(FishError::InvalidShape("unknown CubeCL graph input"));
            }
        }
        self.validate_tokens(inputs)?;
        Ok(())
    }

    /// Validate only request-owned inputs. KV buffers and immutable weights are owned by
    /// the request and model respectively, never accepted from a caller at every step.
    pub fn validate_runtime_inputs(&self, inputs: &NamedTensors) -> Result<()> {
        self.validate_runtime_bindings(inputs, &BTreeMap::new())
    }

    pub(crate) fn validate_runtime_bindings(
        &self,
        inputs: &NamedTensors,
        device_types: &BTreeMap<String, TensorType>,
    ) -> Result<()> {
        for value in &self.graph.values {
            if matches!(value.origin, Origin::Input) && !is_cache_name(&value.name) {
                if let Some(ty) = device_types.get(&value.name) {
                    let forwarded_tokens =
                        matches!(value.name.as_str(), "tokens" | "previous_acoustic_code")
                            && ty.dtype == ScalarType::I32;
                    if (!forwarded_tokens && value.name != "fast_context")
                        || *ty != value.ty
                        || (inputs.contains_key(&value.name) && !forwarded_tokens)
                    {
                        return Err(FishError::InvalidShape(
                            "invalid CubeCL device input binding",
                        ));
                    }
                    if forwarded_tokens {
                        let shadow = inputs.get(&value.name).ok_or(FishError::InvalidShape(
                            "CubeCL forwarded tokens require validated host shadow",
                        ))?;
                        verify_binding(&value.ty, shadow)?;
                    }
                } else {
                    let data = inputs
                        .get(&value.name)
                        .ok_or(FishError::InvalidShape("missing CubeCL request input"))?;
                    verify_binding(&value.ty, data)?;
                }
            }
        }
        for name in inputs.keys().chain(device_types.keys()) {
            if !self.graph.values.iter().any(|value| {
                value.name == *name && matches!(value.origin, Origin::Input) && !is_cache_name(name)
            }) {
                return Err(FishError::InvalidShape("unknown CubeCL request input"));
            }
        }
        self.validate_tokens(inputs)
    }

    fn validate_tokens(&self, inputs: &NamedTensors) -> Result<()> {
        for node in &self.graph.nodes {
            match node.op {
                Op::TokenEmbedding => {
                    let tokens = match &inputs[&self.graph.values[node.inputs[0].0].name] {
                        TensorData::I32(tokens) => tokens,
                        _ => unreachable!("validated graph token dtype"),
                    };
                    let vocab = self.graph.values[node.inputs[1].0].ty.shape[0];
                    if vocab > i32::MAX as usize
                        || tokens.iter().any(|&id| id < 0 || id as usize >= vocab)
                    {
                        return Err(FishError::InvalidShape(
                            "CubeCL embedding token outside vocabulary",
                        ));
                    }
                }
                Op::FishEmbedding {
                    num_codebooks,
                    codebook_size,
                    ..
                } => {
                    let tokens = match &inputs[&self.graph.values[node.inputs[0].0].name] {
                        TensorData::I32(tokens) => tokens,
                        _ => unreachable!("validated Fish token dtype"),
                    };
                    let lut = match &inputs[&self.graph.values[node.inputs[3].0].name] {
                        TensorData::I32(lut) => lut,
                        _ => unreachable!("validated semantic LUT dtype"),
                    };
                    if lut.len() > i32::MAX as usize
                        || codebook_size > i32::MAX as usize
                        || lut.iter().any(|&flag| flag != 0 && flag != 1)
                    {
                        return Err(FishError::InvalidShape(
                            "CubeCL invalid semantic LUT or embedding range",
                        ));
                    }
                    let frames = self.graph.values[node.inputs[0].0].ty.shape[2];
                    let batch = self.graph.values[node.inputs[0].0].ty.shape[0];
                    for b in 0..batch {
                        for t in 0..frames {
                            let word = tokens[b * (num_codebooks + 1) * frames + t];
                            if word < 0 || word as usize >= lut.len() {
                                return Err(FishError::InvalidShape(
                                    "CubeCL semantic token outside vocabulary",
                                ));
                            }
                            if lut[word as usize] == 1 {
                                for c in 0..num_codebooks {
                                    let code =
                                        tokens[(b * (num_codebooks + 1) + c + 1) * frames + t];
                                    if code < 0 || code as usize >= codebook_size {
                                        return Err(FishError::InvalidShape(
                                            "CubeCL acoustic token outside codebook",
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

pub struct CubeClGraphExecutor {
    client: Client,
}

impl CubeClGraphExecutor {
    pub(super) fn client(&self) -> &Client {
        &self.client
    }

    pub fn new(client: Client) -> Self {
        Self { client }
    }

    pub fn run(&self, compiled: &CubeClGraph, inputs: &NamedTensors) -> Result<NamedTensors> {
        compiled.validate_inputs(inputs)?;
        self.ensure_bf16_supported(compiled)?;
        let graph = compiled.graph();
        let mut external = BTreeMap::new();
        for value in &graph.values {
            if matches!(value.origin, Origin::Input | Origin::Weight { .. }) {
                external.insert(
                    value.name.clone(),
                    self.upload(&value.ty, &inputs[&value.name]),
                );
            }
        }
        let mut retained = Vec::new();
        let outputs = self.execute_handles(compiled, &external, &mut retained)?;
        let mut host = BTreeMap::new();
        for id in &graph.outputs {
            let value = &graph.values[id.0];
            host.insert(
                value.name.clone(),
                self.read(&value.ty, &outputs[&value.name])?,
            );
        }
        Ok(host)
    }

    pub(super) fn ensure_bf16_supported(&self, graph: &CubeClGraph) -> Result<()> {
        if graph
            .graph
            .values
            .iter()
            .any(|v| v.ty.dtype == ScalarType::BF16)
        {
            use cubecl::features::TypeUsage;
            let supported = bf16::supported_uses(&self.client);
            if !supported.contains(TypeUsage::Buffer) || !supported.contains(TypeUsage::Conversion)
            {
                return Err(FishError::Unsupported(
                    "CubeCL runtime lacks native BF16 buffer or conversion support",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn upload(&self, ty: &TensorType, data: &TensorData) -> Handle {
        match (ty.dtype, data) {
            (ScalarType::BF16, TensorData::F32(v)) => {
                let halfs = v
                    .iter()
                    .copied()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>();
                self.client.create_from_slice(bytemuck::cast_slice(&halfs))
            }
            (_, TensorData::F32(v)) => self.client.create_from_slice(bytemuck::cast_slice(v)),
            (_, TensorData::I32(v)) => self.client.create_from_slice(bytemuck::cast_slice(v)),
        }
    }

    pub(super) fn zeros(&self, ty: &TensorType) -> Result<Handle> {
        let size = tensor_bytes(ty)?;
        Ok(self.client.create_from_slice(&vec![0u8; size]))
    }

    pub(super) fn read(&self, ty: &TensorType, buffer: &Handle) -> Result<TensorData> {
        let bytes = self
            .client
            .read_one(buffer.clone())
            .map_err(|_| FishError::InvalidTransition("CubeCL device read failed"))?;
        let data = match ty.dtype {
            ScalarType::F32 => TensorData::F32(
                bytes
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().expect("chunk size")))
                    .collect(),
            ),
            ScalarType::I32 => TensorData::I32(
                bytes
                    .chunks_exact(4)
                    .map(|x| i32::from_le_bytes(x.try_into().expect("chunk size")))
                    .collect(),
            ),
            ScalarType::BF16 => TensorData::F32(
                bytes
                    .chunks_exact(2)
                    .map(|x| {
                        half::bf16::from_bits(u16::from_le_bytes(x.try_into().expect("chunk size")))
                            .to_f32()
                    })
                    .collect(),
            ),
        };
        verify_binding(ty, &data)?;
        Ok(data)
    }

    /// Device-resident execution. The caller owns all handles in `retained` until
    /// a stream synchronization; output handles can feed a later compiled graph.
    pub(super) fn execute_handles(
        &self,
        compiled: &CubeClGraph,
        external: &BTreeMap<String, Handle>,
        retained: &mut Vec<Handle>,
    ) -> Result<BTreeMap<String, Handle>> {
        let graph = compiled.graph();
        let mut buffers = vec![None; graph.values.len()];
        for (index, value) in graph.values.iter().enumerate() {
            if matches!(value.origin, Origin::Input | Origin::Weight { .. }) {
                let handle = external
                    .get(&value.name)
                    .ok_or(FishError::InvalidShape("missing CubeCL device binding"))?
                    .clone();
                retained.push(handle.clone());
                buffers[index] = Some(handle);
            }
        }
        for phase in compiled.execution().phases() {
            for &node_index in phase.node_indices() {
                let node = &graph.nodes[node_index];
                let args = node
                    .inputs
                    .iter()
                    .map(|id| {
                        buffers[id.0]
                            .as_ref()
                            .ok_or(FishError::InvalidShape("CubeCL missing graph dependency"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let ty = &graph.values[node.result.0].ty;
                let n = count(ty)?;
                let output = if matches!(node.op, Op::Identity) {
                    args[0].clone()
                } else {
                    let bytes = tensor_bytes(ty)?;
                    let output = self.client.empty(bytes);
                    retained.push(output.clone());
                    self.dispatch(node.op, &args, &output, n, graph, node)?;
                    output
                };
                buffers[node.result.0] = Some(output);
            }
        }
        let mut results = BTreeMap::new();
        for id in &graph.outputs {
            let value = &graph.values[id.0];
            let handle = buffers[id.0]
                .as_ref()
                .ok_or(FishError::InvalidShape("CubeCL missing graph output"))?;
            results.insert(value.name.clone(), handle.clone());
        }
        Ok(results)
    }

    fn dispatch(
        &self,
        op: Op,
        args: &[&Handle],
        output: &Handle,
        n: usize,
        graph: &Graph,
        node: &fish_speech_core::graph_ir::Node,
    ) -> Result<()> {
        let dim = CubeDim::new_1d(128);
        let cubes = u32::try_from(n.div_ceil(128))
            .map_err(|_| FishError::InvalidShape("CubeCL launch grid exceeds u32"))?;
        let grid = CubeCount::Static(cubes, 1, 1);
        let count_arg = |i: usize| -> Result<usize> { count(&graph.values[node.inputs[i].0].ty) };
        let input_arg = |i: usize| -> Result<BufferArg> {
            let len = count_arg(i)?;
            Ok(unsafe { BufferArg::from_raw_parts((*args[i]).clone(), len) })
        };
        let out_arg = || unsafe { BufferArg::from_raw_parts(output.clone(), n) };
        match op {
            Op::Add => add_kernel::launch(
                &self.client,
                grid,
                dim,
                input_arg(0)?,
                input_arg(1)?,
                out_arg(),
            ),
            Op::Mul => mul_kernel::launch(
                &self.client,
                grid,
                dim,
                input_arg(0)?,
                input_arg(1)?,
                out_arg(),
            ),
            Op::SiLu => silu_kernel::launch(&self.client, grid, dim, input_arg(0)?, out_arg()),
            Op::Linear | Op::LinearBias => {
                let w_shape = &graph.values[node.inputs[1].0].ty.shape;
                let input_features = w_shape[1];
                let output_features = w_shape[0];
                linear_kernel::launch(
                    &self.client,
                    grid.clone(),
                    dim.clone(),
                    input_arg(0)?,
                    input_arg(1)?,
                    out_arg(),
                    input_features,
                    output_features,
                );
                if matches!(op, Op::LinearBias) {
                    add_bias_kernel::launch(
                        &self.client,
                        grid,
                        dim,
                        input_arg(2)?,
                        out_arg(),
                        output_features,
                    );
                }
            }
            Op::RmsNorm { epsilon_bits } => {
                let width = graph.values[node.inputs[1].0].ty.shape[0];
                let rows = n / width;
                let row_cubes = u32::try_from(rows.div_ceil(128))
                    .map_err(|_| FishError::InvalidShape("CubeCL RMS grid exceeds u32"))?;
                rms_norm_kernel::launch(
                    &self.client,
                    CubeCount::Static(row_cubes, 1, 1),
                    dim,
                    input_arg(0)?,
                    input_arg(1)?,
                    out_arg(),
                    width,
                    epsilon_bits,
                );
            }
            Op::TokenEmbedding => {
                let w = &graph.values[node.inputs[1].0].ty.shape;
                token_embedding_kernel::launch(
                    &self.client,
                    grid,
                    dim,
                    input_arg(0)?,
                    input_arg(1)?,
                    out_arg(),
                    w[1],
                    w[0],
                );
            }
            Op::FishEmbedding {
                num_codebooks,
                codebook_size,
                share_codebook_embeddings,
            } => {
                let tokens = &graph.values[node.inputs[0].0].ty.shape;
                let dimension = graph.values[node.inputs[1].0].ty.shape[1];
                let vocabulary = graph.values[node.inputs[1].0].ty.shape[0];
                fish_embedding_kernel::launch(
                    &self.client,
                    grid,
                    dim,
                    input_arg(0)?,
                    input_arg(1)?,
                    input_arg(2)?,
                    input_arg(3)?,
                    out_arg(),
                    tokens[2],
                    dimension,
                    num_codebooks,
                    codebook_size,
                    vocabulary,
                    share_codebook_embeddings,
                );
            }
            Op::SplitQkv {
                part,
                q_heads,
                kv_heads,
                head_dim,
            } => {
                let input = &graph.values[node.inputs[0].0].ty.shape;
                let (heads, offset) = match part {
                    QkvPart::Query => (q_heads, 0),
                    QkvPart::Key => (kv_heads, q_heads * head_dim),
                    QkvPart::Value => (kv_heads, (q_heads + kv_heads) * head_dim),
                };
                split_qkv_kernel::launch(
                    &self.client,
                    grid,
                    dim,
                    input_arg(0)?,
                    out_arg(),
                    input[1],
                    heads,
                    head_dim,
                    input[2],
                    offset,
                );
            }
            Op::RoPe {
                position,
                base_bits,
            } => {
                let source = &graph.values[node.inputs[0].0].ty.shape;
                let frames = source[2];
                let head_dim = source[3];
                let coefficients = fish_speech_core::ar_rotary::fish_rope_coefficients(
                    position,
                    frames,
                    head_dim,
                    f32::from_bits(base_bits),
                )?;
                let cosine = self
                    .client
                    .create_from_slice(bytemuck::cast_slice(&coefficients.cosine));
                let signed_sine = self
                    .client
                    .create_from_slice(bytemuck::cast_slice(&coefficients.signed_sine));
                rope_kernel::launch(
                    &self.client,
                    grid,
                    dim,
                    input_arg(0)?,
                    unsafe { BufferArg::from_raw_parts(cosine, coefficients.cosine.len()) },
                    unsafe {
                        BufferArg::from_raw_parts(signed_sine, coefficients.signed_sine.len())
                    },
                    out_arg(),
                    frames,
                    head_dim,
                );
            }
            Op::KvUpdate { position } => {
                let cache = &graph.values[node.inputs[0].0].ty.shape;
                let fresh = &graph.values[node.inputs[1].0].ty.shape;
                kv_update_kernel::launch(
                    &self.client,
                    grid,
                    dim,
                    input_arg(0)?,
                    input_arg(1)?,
                    out_arg(),
                    cache[1],
                    cache[2],
                    fresh[2],
                    cache[3],
                    position,
                );
            }
            Op::KvPrefix { active_len } => {
                let cache = &graph.values[node.inputs[0].0].ty.shape;
                kv_prefix_kernel::launch(
                    &self.client,
                    grid,
                    dim,
                    input_arg(0)?,
                    out_arg(),
                    cache[2],
                    active_len,
                    cache[3],
                );
            }
            Op::GqaAttention {
                position,
                q_heads,
                kv_heads,
            } => {
                let query = &graph.values[node.inputs[0].0].ty.shape;
                let key = &graph.values[node.inputs[1].0].ty.shape;
                causal_gqa_kernel::launch(
                    &self.client,
                    grid,
                    dim,
                    input_arg(0)?,
                    input_arg(1)?,
                    input_arg(2)?,
                    out_arg(),
                    q_heads,
                    kv_heads,
                    query[2],
                    key[2],
                    query[3],
                    position,
                );
            }
            Op::LastToken => {
                let source = &graph.values[node.inputs[0].0].ty.shape;
                let sequence = source[1];
                let width = source[2];
                last_token_kernel::launch(
                    &self.client,
                    grid,
                    dim,
                    input_arg(0)?,
                    out_arg(),
                    sequence,
                    width,
                );
            }
            _ => {
                return Err(FishError::Unsupported(
                    "CubeCL graph operation not implemented",
                ));
            }
        }
        Ok(())
    }
}
