//! A pure Rust ONNX parser that produces a typed, framework-independent
//! intermediate representation.
//!
//! `onnx-ir` reads an ONNX model and returns an [`OnnxGraph`]: a list of
//! [`Node`]s, one enum variant per operator, each carrying its inputs, outputs,
//! and a strongly typed config struct holding the operator's ONNX attributes.
//! Along the way it infers element types, ranks, and static shapes, lifts
//! constants, and optionally simplifies the graph. It powers
//! [`burn-onnx`](https://docs.rs/burn-onnx), but knows nothing about Burn and
//! can back any code generator or analysis tool.
//!
//! # Example
//!
//! ```no_run
//! use onnx_ir::{Node, OnnxGraphBuilder};
//!
//! # fn main() -> Result<(), onnx_ir::Error> {
//! let graph = OnnxGraphBuilder::new().parse_file("model.onnx")?;
//!
//! for input in &graph.inputs {
//!     println!("input {}: {:?}", input.name, input.ty);
//! }
//!
//! for node in &graph.nodes {
//!     match node {
//!         Node::Softmax(softmax) => println!("{}: softmax on axis {}", node.name(), softmax.config.axis),
//!         Node::Conv2d(conv) => println!("{}: conv kernel {:?}", node.name(), conv.config.kernel_size),
//!         // Operators outside the standard domains keep their ONNX identity.
//!         Node::Custom(custom) => println!("{}: {}::{}", node.name(), custom.domain, custom.op_type),
//!         _ => println!("{}", node.name()),
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # What you get
//!
//! - **Every opset.** Each supported operator handles ONNX opsets 1 through 24,
//!   including attributes that later became inputs and defaults that changed.
//! - **Typed configs.** Each operator's attributes are parsed into a config
//!   struct of typed, normalized settings, independent of any one framework:
//!   opset-dependent defaults are applied and negative axes are resolved.
//!   Attribute values the parser cannot represent are skipped with a warning.
//! - **Type and shape inference.** Every [`Argument`] has an [`ArgType`]:
//!   a tensor with dtype, rank, and static shape when known, a scalar, or a
//!   shape value.
//! - **Simplification.** Enabled by default with
//!   [`OnnxGraphBuilder::simplify`]: constant folding, shape propagation,
//!   common subexpression elimination, dead node removal, and attention fusion.
//! - **Tolerant of unknown operators.** Ops from vendor or custom domains parse
//!   as [`Node::Custom`] instead of failing. Implement [`CustomOpInference`]
//!   and pass it to [`OnnxGraphBuilder::with_custom_op_inference`] to give them
//!   real type inference.
//! - **Memory mapping.** With the default `mmap` feature, large weight tensors
//!   are read from the file on demand instead of being copied up front.
//!
//! The [development guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md)
//! describes the parsing pipeline phase by phase.

#[macro_use]
extern crate derive_new;

mod external_data;
mod graph_state;
pub mod ir;
pub mod node;
mod phases;
mod pipeline;
mod processor;
mod proto_conversion;
mod protos;
mod registry;
mod simplify;
mod tensor_store;

// Public API - only expose essentials
pub use ir::*;
pub use node::custom::{CustomNode, CustomOpInference, HookCoverage, MissingReason, OpsetRange};
pub use node::*;
pub use pipeline::{Error, MissingHook, OnnxGraphBuilder, normalize_domain};
pub use processor::ProcessError;

/// The trait `ModelProto::parse_from_bytes` and `write_to_bytes` come from, so a
/// caller reading or writing a model needs no `protobuf` dependency of its own.
pub use protobuf::Message;
/// Generated protobuf bindings for the ONNX wire format messages that
/// sibling crates need to decode on-disk artifacts.
///
/// Re-exported so callers (e.g. `onnx-official-tests`) can read `.pb`
/// reference tensors and inspect `model.onnx` input/output signatures
/// without rebuilding the bindings themselves.
///
/// * `TensorProto` decodes individual tensor `.pb` reference files.
/// * `ModelProto` / `GraphProto` / `ValueInfoProto` / `TypeProto` /
///   `TensorShapeProto` are used by `onnx-official-tests`'s build script
///   to walk a `model.onnx` header and extract per-test input/output
///   shape and dtype metadata so it can emit per-test harness glue with
///   the correct rank and element type.
///
/// * `NodeProto` / `AttributeProto` / `AttributeType` let a caller rewrite
///   a graph before importing it (set an attribute on every node of a
///   kind, add a node); an attribute's type is an enum the caller has to
///   name to set it, which is why that one nested type is re-exported.
///
/// The other inner namespaces (`type_proto::Tensor`, `tensor_shape_proto::
/// Dimension`, etc.) stay private. Callers can still reach them via
/// method calls on values of the re-exported outer types — Rust allows
/// public method calls on references to private types as long as the
/// private type is never named in user code.
pub use protos::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto, TypeProto,
    ValueInfoProto, attribute_proto::AttributeType,
};
