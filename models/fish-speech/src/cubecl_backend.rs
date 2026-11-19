//! Independent CubeCL Graph IR executor for Fish Speech.
//! Streaming autoregression and model-scoped device ownership are not implemented yet.

mod kernels;
mod runtime;
mod session;

pub use runtime::{CubeClGraph, CubeClGraphExecutor};
pub use session::{
    CubeClModel, CubeClPreparedModel, CubeClRequest, CubeClRequestCursor, CubeClRequestLimits,
};

#[cfg(test)]
mod tests;
