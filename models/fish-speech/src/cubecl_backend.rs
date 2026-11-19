//! First independent CubeCL graph-execution slice for Fish Speech.
//! This is a synchronous operator-parity path, not yet a streaming AR backend.

use std::collections::BTreeMap;

use cubecl::prelude::*;
use cubecl::server::Handle;
use fish_speech_core::{
    cpu_ir::{NamedTensors, TensorData},
    error::{FishError, Result},
    graph_ir::{Graph, Op, Origin, ScalarType, TensorType},
};

#[cube(launch)]
fn add_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

#[cube(launch)]
fn mul_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS];
    }
}

#[cube(launch)]
fn silu_kernel(a: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        let x = a[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = x / (1.0 + (-x).exp());
    }
}

#[cube(launch)]
fn linear_kernel(
    a: &[f32],
    w: &[f32],
    out: &mut [f32],
    #[comptime] input_features: usize,
    #[comptime] output_features: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let row = ABSOLUTE_POS / output_features;
        let col = ABSOLUTE_POS % output_features;
        let mut sum = 0.0f32;
        for k in 0..input_features {
            sum += a[row * input_features + k] * w[col * input_features + k];
        }
        out[ABSOLUTE_POS] = sum;
    }
}

#[cube(launch)]
fn add_bias_kernel(bias: &[f32], out: &mut [f32], #[comptime] width: usize) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] += bias[ABSOLUTE_POS % width];
    }
}

#[cube(launch)]
fn rms_norm_kernel(
    a: &[f32],
    gamma: &[f32],
    out: &mut [f32],
    #[comptime] width: usize,
    #[comptime] epsilon_bits: u32,
) {
    let rows = out.len() / width;
    if ABSOLUTE_POS < rows {
        let start = ABSOLUTE_POS * width;
        let mut squared = 0.0f32;
        for k in 0..width {
            let x = a[start + k];
            squared += x * x;
        }
        let scale = 1.0 / (squared / (width as f32) + f32::from_bits(epsilon_bits)).sqrt();
        for k in 0..width {
            out[start + k] = a[start + k] * scale * gamma[k];
        }
    }
}

#[cube(launch)]
fn last_token_kernel(
    a: &[f32],
    out: &mut [f32],
    #[comptime] sequence: usize,
    #[comptime] width: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let batch = ABSOLUTE_POS / width;
        let feature = ABSOLUTE_POS % width;
        out[ABSOLUTE_POS] = a[(batch * sequence + sequence - 1) * width + feature];
    }
}

fn count(ty: &TensorType) -> Result<usize> {
    ty.shape
        .iter()
        .try_fold(1usize, |n, &x| n.checked_mul(x))
        .ok_or(FishError::InvalidShape(
            "CubeCL tensor element count overflow",
        ))
}

fn validated_scalar(ty: &TensorType) -> Result<usize> {
    if !matches!(ty.dtype, ScalarType::F32 | ScalarType::I32) {
        return Err(FishError::Unsupported(
            "CubeCL BF16 kernel not yet implemented",
        ));
    }
    count(ty)
}

fn verify_binding(ty: &TensorType, data: &TensorData) -> Result<()> {
    let n = validated_scalar(ty)?;
    let valid = match (ty.dtype, data) {
        (ScalarType::F32, TensorData::F32(v)) => v.len() == n,
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
    )
}

pub struct CubeClGraph {
    graph: Graph,
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
        Ok(Self {
            graph: graph.clone(),
        })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
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
        Ok(())
    }
}

pub struct CubeClGraphExecutor {
    client: Client,
}

impl CubeClGraphExecutor {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    pub fn run(&self, compiled: &CubeClGraph, inputs: &NamedTensors) -> Result<NamedTensors> {
        let graph = compiled.graph();
        compiled.validate_inputs(inputs)?;
        let mut buffers = vec![None; graph.values.len()];
        for (i, value) in graph.values.iter().enumerate() {
            if matches!(value.origin, Origin::Input | Origin::Weight { .. }) {
                let data = &inputs[&value.name];
                let bytes: &[u8] = match data {
                    TensorData::F32(v) => bytemuck::cast_slice(v),
                    TensorData::I32(v) => bytemuck::cast_slice(v),
                };
                buffers[i] = Some(self.client.create_from_slice(bytes));
            }
        }
        for node in &graph.nodes {
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
            let output =
                if matches!(node.op, Op::Identity) {
                    args[0].clone()
                } else {
                    let bytes = n.checked_mul(std::mem::size_of::<f32>()).ok_or(
                        FishError::InvalidShape("CubeCL allocation byte count overflow"),
                    )?;
                    let output = self.client.empty(bytes);
                    self.dispatch(node.op, &args, &output, n, graph, node)?;
                    output
                };
            buffers[node.result.0] = Some(output);
        }
        let mut results = BTreeMap::new();
        for id in &graph.outputs {
            let value = &graph.values[id.0];
            let buffer = buffers[id.0]
                .as_ref()
                .ok_or(FishError::InvalidShape("CubeCL missing graph output"))?;
            let bytes = self
                .client
                .read_one(buffer.clone())
                .map_err(|_| FishError::InvalidTransition("CubeCL device read failed"))?;
            let data = match value.ty.dtype {
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
                ScalarType::BF16 => {
                    return Err(FishError::Unsupported("CubeCL BF16 read not implemented"));
                }
            };
            verify_binding(&value.ty, &data)?;
            results.insert(value.name.clone(), data);
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

#[cfg(test)]
mod preflight_tests {
    use super::*;
    use fish_speech_core::{
        graph::Stage,
        graph_ir::{Layout, TensorType},
    };

    #[test]
    fn supported_graph_compiles_without_runtime_or_tensorrt() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let a = g
            .input(
                "a",
                TensorType::new([1, 2, 3], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let b = g
            .input(
                "b",
                TensorType::new([1, 2, 3], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let out = g.emit("sum", Op::Add, &[a, b]).unwrap();
        g.output(out).unwrap();
        let compiled = CubeClGraph::compile(&g).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("a".into(), TensorData::F32(vec![1.0; 6]));
        inputs.insert("b".into(), TensorData::F32(vec![2.0; 6]));
        assert!(compiled.validate_inputs(&inputs).is_ok());
        inputs.remove("b");
        assert!(compiled.validate_inputs(&inputs).is_err());
        inputs.insert("b".into(), TensorData::I32(vec![2; 6]));
        assert!(compiled.validate_inputs(&inputs).is_err());
        inputs.insert("b".into(), TensorData::F32(vec![2.0; 6]));
        inputs.insert("unknown".into(), TensorData::F32(vec![1.0]));
        assert!(compiled.validate_inputs(&inputs).is_err());
    }

    #[test]
    fn unsupported_embedding_fails_during_compile() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let tokens = g
            .input(
                "tokens",
                TensorType::new([1, 2], ScalarType::I32, Layout::Plain),
            )
            .unwrap();
        let weight = g
            .weight(
                "weight",
                TensorType::new([12, 4], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let out = g
            .emit("embedding", Op::TokenEmbedding, &[tokens, weight])
            .unwrap();
        g.output(out).unwrap();
        assert!(matches!(
            CubeClGraph::compile(&g),
            Err(FishError::Unsupported(_))
        ));
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use cubecl::Device;
    use fish_speech_core::{
        cpu_ir::CpuIrInterpreter,
        graph::Stage,
        graph_ir::{Layout, TensorType},
    };

    fn ty(shape: &[usize]) -> TensorType {
        TensorType::new(shape.to_vec(), ScalarType::F32, Layout::SequenceBtd)
    }

    fn run_parity(graph: Graph, inputs: NamedTensors) {
        let expected = CpuIrInterpreter::run(&graph, &inputs).unwrap();
        let executor = CubeClGraphExecutor::new(Device::Cpu(Default::default()).client());
        let actual = executor
            .run(&CubeClGraph::compile(&graph).unwrap(), &inputs)
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (name, target) in expected {
            let got = &actual[&name];
            match (got, target) {
                (TensorData::F32(a), TensorData::F32(b)) => {
                    assert_eq!(a.len(), b.len());
                    for (a, b) in a.iter().zip(b.iter()) {
                        assert!(
                            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                            "{name}: {a} != {b}"
                        );
                    }
                }
                (a, b) => assert_eq!(*a, b),
            }
        }
    }

    #[test]
    fn cubecl_f32_linear_bias_silu_add_rms_last_token() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let x = g.input("x", ty(&[1, 3, 2])).unwrap();
        let w = g
            .weight("w", TensorType::new([4, 2], ScalarType::F32, Layout::Plain))
            .unwrap();
        let bias = g
            .weight("bias", TensorType::new([4], ScalarType::F32, Layout::Plain))
            .unwrap();
        let gamma = g
            .weight(
                "gamma",
                TensorType::new([4], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let h = g.emit("h", Op::LinearBias, &[x, w, bias]).unwrap();
        let z = g.emit("z", Op::SiLu, &[h]).unwrap();
        let r = g.emit("r", Op::Add, &[h, z]).unwrap();
        let m = g.emit("m", Op::Mul, &[r, z]).unwrap();
        let norm = g
            .emit(
                "norm",
                Op::RmsNorm {
                    epsilon_bits: 1e-5f32.to_bits(),
                },
                &[m, gamma],
            )
            .unwrap();
        let last = g.emit("last", Op::LastToken, &[norm]).unwrap();
        g.output(norm).unwrap();
        g.output(last).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert(
            "x".into(),
            TensorData::F32(vec![-1.0, 0.25, 0.0, 1.0, 2.0, -1.0]),
        );
        inputs.insert(
            "w".into(),
            TensorData::F32(vec![1.0, 2.0, 0.5, -1.0, -1.0, 0.25, 2.0, 0.1]),
        );
        inputs.insert("bias".into(), TensorData::F32(vec![0.5, -0.5, 1.0, 0.25]));
        inputs.insert("gamma".into(), TensorData::F32(vec![1.0, 0.75, 2.0, -0.5]));
        run_parity(g, inputs);
    }

    #[test]
    fn cubecl_compile_rejects_unsupported_before_device_use() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let ids = g
            .input(
                "tokens",
                TensorType::new([1, 2], ScalarType::I32, Layout::Plain),
            )
            .unwrap();
        let weights = g
            .weight(
                "embed",
                TensorType::new([8, 3], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let out = g
            .emit("embed_out", Op::TokenEmbedding, &[ids, weights])
            .unwrap();
        g.output(out).unwrap();
        assert!(matches!(
            CubeClGraph::compile(&g),
            Err(FishError::Unsupported(_))
        ));
    }
}
