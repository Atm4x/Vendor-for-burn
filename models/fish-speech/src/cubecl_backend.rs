//! Independent CubeCL Graph IR executor for Fish Speech.
//! Streaming autoregression and model-scoped device ownership are not implemented yet.

mod kernels;
mod runtime;

pub use runtime::{CubeClGraph, CubeClGraphExecutor};

#[cfg(test)]
mod tests;
