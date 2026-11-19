//! Fish Speech 1.5 native Rust model prototype.
//! CPU reference contracts are separate from TensorRT-RTX bindings.

pub use fish_speech_core::*;

#[cfg(feature = "tensorrt-api")]
mod trtx_ar_ops;
#[cfg(all(feature = "cuda-driver", feature = "tensorrt-api"))]
pub mod trtx_budget_allocator;
pub mod trtx_cache;
pub mod trtx_engine;
#[cfg(feature = "tensorrt-api")]
pub mod trtx_firefly_fsq;
#[cfg(feature = "tensorrt-api")]
pub mod trtx_firefly_mask;
#[cfg(feature = "tensorrt-api")]
pub mod trtx_firefly_network;
#[cfg(feature = "tensorrt-api")]
pub mod trtx_firefly_pointwise;
#[cfg(feature = "tensorrt-api")]
pub mod trtx_firefly_segment;
#[cfg(any(feature = "tensorrt-mock", feature = "tensorrt-api"))]
pub mod trtx_lowering;

pub mod ar_input_contract;
pub mod ar_weight_bindings;
pub mod device_memory;
pub mod firefly_config;
pub mod fish_tokenizer;
pub mod model_config;
pub mod model_runtime;
pub mod reference_cache;
pub mod reference_loader;
pub mod request_queue;
pub mod scratch_pool;
pub mod weighted_cpu;
pub mod weighted_firefly;
pub mod weights;

#[cfg(feature = "cuda-driver")]
pub mod cuda_stream;

#[cfg(all(feature = "cuda-driver", feature = "tensorrt-api"))]
pub mod cuda_trtx;

#[cfg(all(feature = "cuda-driver", feature = "tensorrt-api"))]
pub mod cuda_generation;

#[cfg(all(feature = "cuda-driver", feature = "tensorrt-api"))]
mod cuda_fast_sampling;

#[cfg(all(feature = "cuda-driver", feature = "tensorrt-api"))]
mod cuda_semantic_sampling;

#[cfg(all(feature = "cuda-driver", feature = "tensorrt-api"))]
pub mod cuda_model_runtime;

#[cfg(all(feature = "cuda-driver", feature = "tensorrt-api"))]
mod dynamic_decode;

#[cfg(all(feature = "cuda-driver", feature = "tensorrt-api"))]
pub mod cuda_naive_generation;

#[cfg(feature = "cubecl-backend")]
pub mod cubecl_backend;
