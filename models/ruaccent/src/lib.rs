//! Native Rust port of RUAccent. Neural inference is available with the `onnx` feature.

pub mod text;
pub mod tokenization;

#[cfg(feature = "onnx")]
pub mod dict;
#[cfg(feature = "onnx")]
pub mod models;
#[cfg(feature = "onnx")]
pub mod pipeline;

#[cfg(feature = "onnx")]
pub use pipeline::RuAccent;
