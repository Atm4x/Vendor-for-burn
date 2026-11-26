//! Independent CubeCL Graph IR executor for Fish Speech.
//! Graph execution, model-scoped weights and DualAR/Naive generation are independent of TensorRT.

mod convnext;
mod fsq;
mod generation;
mod kernels;
mod logmel;
mod runtime;
mod sampling;
mod session;

pub use convnext::{
    CubeClConvNextExecutor, CubeClFireflyAcousticDecoder, CubeClFireflyCodec, CubeClFireflyDecoder,
    CubeClPreparedFireflyEncoder,
};
pub use fsq::CubeClFsqExecutor;
pub use generation::CubeClSemanticGenerator;
pub use logmel::CubeClLogMel;
pub use runtime::{CubeClGraph, CubeClGraphExecutor};
pub use session::{
    CubeClDeviceTensor, CubeClModel, CubeClPreparedModel, CubeClRequest, CubeClRequestCursor,
    CubeClRequestLimits, CubeClRunResult,
};

#[cfg(test)]
mod tests;

pub(crate) use convnext::CubeClFireflyReferenceEncoder;
