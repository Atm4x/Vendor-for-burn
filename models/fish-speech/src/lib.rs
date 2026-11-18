//! Fish Speech 1.5 native Rust model prototype.
//! CPU reference contracts are separate from TensorRT-RTX bindings.

pub use fish_speech_core::*;

#[cfg(any(feature = "tensorrt-mock", feature = "tensorrt-api"))]
pub mod trtx_lowering;

pub mod trtx_cache;
pub mod trtx_engine;
