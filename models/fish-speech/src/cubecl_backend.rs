//! Independent CubeCL Graph IR executor for Fish Speech.
//! Graph execution, model-scoped weights and DualAR/Naive generation are independent of TensorRT.

mod generation;
mod kernels;
mod runtime;
mod session;

pub use generation::CubeClSemanticGenerator;
pub use runtime::{CubeClGraph, CubeClGraphExecutor};
pub use session::{
    CubeClModel, CubeClPreparedModel, CubeClRequest, CubeClRequestCursor, CubeClRequestLimits,
};

#[cfg(test)]
mod tests;
