//! Native Silero v5 Russian TTS components based on Burn/CubeCL.
//!
//! Weight conversion is a build-time operation. Runtime tensor execution never
//! depends on Python or PyTorch, and CubeCL CPU is the default test backend.

pub mod frontend;
pub mod vocoder;
pub mod spectral;
pub mod synthesis;
pub mod store;
pub mod tokenizer;

pub use burn::tensor::{Device, Tensor};

pub mod transformer;
pub mod predictors;
pub mod acoustic;
pub mod runtime;

pub mod accentor;
pub mod homosolver;
