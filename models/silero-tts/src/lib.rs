//! Native Silero v5 Russian and v3 English TTS components based on Burn.
//!
//! Runtime loading accepts official TorchPackage checkpoints directly.
//! Prepared safetensors remain supported; neither path requires Python or PyTorch.

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
mod english;
mod english_vocoder;

pub mod accentor;
pub mod homosolver;

pub mod ssml;
pub mod audio;

pub mod worker;

#[cfg(feature = "reference-flex")]
mod flex_pointwise;
#[cfg(feature = "reference-flex")]
mod flex_spectral;

#[cfg(feature = "reference-flex")]
mod flex_depthwise;

#[cfg(feature = "reference-flex")]
mod flex_convnext;

mod longform;
