//! Native Silero V5 Russian and multilingual V3 HiFi-GAN TTS components based on Burn.
//!
//! Runtime loading accepts official TorchPackage checkpoints directly.
//! Prepared safetensors remain supported; neither path requires Python or PyTorch.

pub mod frontend;
pub mod catalogue;
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
mod v2;
mod tacotron;
mod v3;
mod v4;
mod v5_plain;
mod utterance;
mod english_vocoder;
#[cfg(feature = "reference-flex")]
mod flex_english_conv;
#[cfg(feature = "reference-flex")]
mod flex_english_pqmf;
#[cfg(feature = "reference-flex")]
mod flex_transformer_ffn;

pub mod accentor;
pub mod homosolver;

pub mod ssml;
pub mod audio;

pub mod worker;
pub mod ipc;

#[cfg(feature = "reference-flex")]
mod flex_pointwise;
#[cfg(feature = "reference-flex")]
mod flex_spectral;

#[cfg(feature = "reference-flex")]
mod flex_depthwise;

#[cfg(feature = "reference-flex")]
mod flex_convnext;

mod longform;
