//! Independent CubeCL Graph IR executor for Fish Speech.
//! Graph execution, model-scoped weights and DualAR/Naive generation are independent of TensorRT.

mod convnext;
mod fsq;
mod logmel;
mod generation;
mod kernels;
mod runtime;
mod sampling;
mod session;

pub use convnext::{CubeClConvNextExecutor, CubeClPreparedFireflyEncoder};
pub use fsq::CubeClFsqExecutor;
pub use logmel::CubeClLogMel;
pub use generation::CubeClSemanticGenerator;
pub use runtime::{CubeClGraph, CubeClGraphExecutor};
pub use session::{
    CubeClDeviceTensor, CubeClModel, CubeClPreparedModel, CubeClRequest, CubeClRequestCursor,
    CubeClRequestLimits, CubeClRunResult,
};

#[cfg(test)]
mod tests;
