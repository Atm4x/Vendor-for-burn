//! Export captured Burn operation graphs to ONNX.
//!
//! [`OnnxExporter`] runs a module's forward pass once on a private capture
//! device, records every tensor operation, and lowers the recorded graph to an
//! ONNX model with the weights embedded. Any [`Module`](burn::module::Module)
//! works; there is nothing to derive or implement beyond the module itself.
//!
//! This module is **experimental**. It covers the operations common in
//! convolutional and fully connected networks (ResNet-18 exports and passes
//! the ONNX checker), and fails with [`ExportError::UnsupportedOperation`]
//! when the forward pass uses an operation it cannot lower yet.
//!
//! # Static shapes
//!
//! [`OnnxExporter::export`] bakes the sample input's shape into the model:
//!
//! ```
//! use burn::module::Module;
//! use burn::nn::{Linear, LinearConfig};
//! use burn::tensor::{Device, Tensor, activation::relu};
//! use burn_onnx::export::OnnxExporter;
//!
//! #[derive(Module, Debug)]
//! struct Mlp {
//!     fc1: Linear,
//!     fc2: Linear,
//! }
//!
//! impl Mlp {
//!     fn forward(&self, x: Tensor<2>) -> Tensor<2> {
//!         self.fc2.forward(relu(self.fc1.forward(x)))
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::default();
//! let model = Mlp {
//!     fc1: LinearConfig::new(4, 8).init(&device),
//!     fc2: LinearConfig::new(8, 2).init(&device),
//! };
//! let sample = Tensor::<2>::zeros([1, 4], &device);
//!
//! let onnx = OnnxExporter::new().export(&model, sample, Mlp::forward)?;
//! onnx.save(std::env::temp_dir().join("mlp.onnx"))?;
//! # Ok(())
//! # }
//! ```
//!
//! # Dynamic axes
//!
//! A trace only sees concrete numbers, so a single capture cannot tell a batch
//! dimension apart from a constant. [`OnnxExporter::export_dynamic`] captures
//! the forward pass twice, once per input set, and marks the axes you annotate
//! with [`AxisSpec::dynamic`] as symbolic. Annotated axes must differ between
//! the two inputs and every other axis must match:
//!
//! ```
//! # use burn::module::Module;
//! # use burn::tensor::{Device, Tensor};
//! use burn_onnx::export::{AxisSpec, InputSpec, OnnxExporter};
//!
//! #[derive(Module, Debug)]
//! struct Flatten;
//!
//! impl Flatten {
//!     fn forward(&self, x: Tensor<3>) -> Tensor<2> {
//!         let [batch, channels, width] = x.dims();
//!         x.reshape([batch, channels * width])
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::default();
//! let sample = Tensor::<3>::zeros([2, 3, 4], &device);
//! let validation = Tensor::<3>::zeros([5, 3, 4], &device);
//! let specs = [InputSpec::new([
//!     AxisSpec::dynamic("batch"),
//!     AxisSpec::Static,
//!     AxisSpec::Static,
//! ])];
//!
//! let onnx = OnnxExporter::new().export_dynamic(
//!     &Flatten,
//!     sample,
//!     validation,
//!     &specs,
//!     Flatten::forward,
//! )?;
//! assert!(!onnx.as_bytes().is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! The two traces must record the same sequence of operations. Export fails
//! with [`ExportError::DynamicGraphMismatch`] when they do not, and with
//! [`ExportError::DynamicShapeLost`] when an operation consumed a shape that
//! capture already reduced to a constant (slicing a dynamic axis, for one).
//!
//! # Design
//!
//! Shape validation and resolution deliberately precede ONNX lowering. This
//! keeps trace-based inference replaceable by a future symbolic capture pass.

mod error;
mod exporter;
mod lower;
mod model;
mod resolved;
mod shape;
mod validate;

pub use error::ExportError;
#[doc(hidden)]
pub use exporter::{ExportInput, ExportOutput};
pub use exporter::{OnnxExporter, Opset};
pub use model::OnnxModel;
pub(crate) use resolved::{DynamicAxis, ResolvedExportGraph, ResolvedShape, ShapeExpr};
pub use shape::{AxisSpec, InputSpec};
pub(crate) use validate::GraphStructureValidator;
