use std::sync::{Arc, Mutex, OnceLock};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{Ptx, compile_ptx},
};
use fish_speech_core::{
    ar_stream_executor::ArStreamDevice,
    error::{FishError, Result},
    graph_ir::{Layout, ScalarType, TensorType},
    sampling::SamplingParams,
    trtx_blueprint::{Binding, TrtBlueprint},
};

use crate::{cuda_stream::CudaBuffer, cuda_trtx::WeightedCudaArSession};

const THREADS: u32 = 256;
const MAX_SEMANTIC_VOCAB: usize = 1_048_576;

pub struct CudaSemanticSampler {
    kernel: CudaFunction,
    tiles: CudaFunction,
}

impl CudaSemanticSampler {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self> {
        static PTX: OnceLock<Mutex<Option<String>>> = OnceLock::new();
        let cache = PTX.get_or_init(|| Mutex::new(None));
        let ptx = {
            let mut cached = cache.lock().map_err(|_| {
                FishError::InvalidTransition("CUDA semantic sampler PTX cache poisoned")
            })?;
            if let Some(ptx) = cached.as_ref() {
                ptx.clone()
            } else {
                let ptx = compile_ptx(include_str!("cuda_semantic_sampling.cu"))
                    .map_err(|error| {
                        FishError::Backend(format!("CUDA semantic sampler NVRTC: {error:?}"))
                    })?
                    .to_src();
                *cached = Some(ptx.clone());
                ptx
            }
        };
        let module = context
            .load_module(Ptx::from_src(ptx))
            .map_err(|_| FishError::InvalidTransition("CUDA semantic sampler PTX load failed"))?;
        let kernel = module
            .load_function("fish_semantic_sample")
            .map_err(|_| FishError::InvalidTransition("CUDA semantic sampler kernel absent"))?;
        let tiles = module
            .load_function("fish_semantic_tiles")
            .map_err(|_| FishError::InvalidTransition("CUDA semantic tile kernel absent"))?;
        Ok(Self { kernel, tiles })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaBuffer,
        history_mask: &CudaBuffer,
        vocabulary: usize,
        params: SamplingParams,
        seed: u64,
        semantic_step: u64,
        eos_id: u32,
        result: &CudaBuffer,
    ) -> Result<()> {
        self.enqueue_with_column(
            stream,
            logits,
            history_mask,
            vocabulary,
            params,
            seed,
            semantic_step,
            eos_id,
            result,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_with_column(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaBuffer,
        history_mask: &CudaBuffer,
        vocabulary: usize,
        params: SamplingParams,
        seed: u64,
        semantic_step: u64,
        eos_id: u32,
        result: &CudaBuffer,
        output_column: Option<&CudaBuffer>,
    ) -> Result<()> {
        params.validate()?;
        validate_vocabulary(vocabulary)?;
        if output_column.is_some_and(|buffer| buffer.bytes() < 4) {
            return Err(FishError::InvalidShape(
                "CUDA semantic output column too short",
            ));
        }
        let packed_history = i32::from(history_mask.bytes() != vocabulary * 4);
        if logits.bytes() != vocabulary * 4
            || (packed_history != 0 && history_mask.bytes() != vocabulary.div_ceil(32) * 4)
            || result.bytes() != 8
        {
            return Err(FishError::InvalidShape(
                "CUDA semantic sampling buffer size mismatch",
            ));
        }
        let eos_id = i32::try_from(eos_id)
            .map_err(|_| FishError::InvalidShape("CUDA semantic EOS exceeds i32"))?;
        let vocab = i32::try_from(vocabulary)
            .map_err(|_| FishError::InvalidShape("CUDA semantic vocabulary exceeds i32"))?;
        let logits_ptr = logits.address(stream);
        let history_ptr = history_mask.address(stream);
        let result_ptr = result.address(stream);
        let column_ptr = output_column.unwrap_or(result).address(stream);
        let write_column = i32::from(output_column.is_some());
        let groups = (vocab as u32).div_ceil(THREADS);
        let mut summaries = stream
            .alloc_zeros::<f64>(2 * groups as usize)
            .map_err(|_| FishError::InvalidTransition("CUDA semantic scratch allocation failed"))?;
        let mut prepare = stream.launch_builder(&self.tiles);
        prepare
            .arg(&logits_ptr)
            .arg(&history_ptr)
            .arg(&packed_history)
            .arg(&vocab)
            .arg(&params.repetition_penalty)
            .arg(&mut summaries);
        unsafe {
            prepare.launch(LaunchConfig {
                grid_dim: (groups, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|_| FishError::InvalidTransition("CUDA semantic tile launch failed"))?;
        let mut launch = stream.launch_builder(&self.kernel);
        launch
            .arg(&logits_ptr)
            .arg(&history_ptr)
            .arg(&packed_history)
            .arg(&vocab)
            .arg(&params.top_p)
            .arg(&params.temperature)
            .arg(&params.repetition_penalty)
            .arg(&seed)
            .arg(&semantic_step)
            .arg(&eos_id)
            .arg(&result_ptr)
            .arg(&column_ptr)
            .arg(&write_column)
            .arg(&summaries);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|_| FishError::InvalidTransition("CUDA semantic sampler launch failed"))?;
        Ok(())
    }
}

fn validate_vocabulary(vocabulary: usize) -> Result<()> {
    if vocabulary == 0 || vocabulary > MAX_SEMANTIC_VOCAB {
        return Err(FishError::Unsupported(
            "CUDA semantic sampler vocabulary outside supported range",
        ));
    }
    Ok(())
}

pub(crate) fn upload_semantic_bitset<F>(
    session: &mut WeightedCudaArSession<F>,
    vocabulary: usize,
    history: &[usize],
) -> Result<CudaBuffer>
where
    F: FnMut(
        &TrtBlueprint,
        &std::collections::BTreeMap<Binding, CudaBuffer>,
        &Arc<CudaStream>,
    ) -> Result<()>,
{
    validate_vocabulary(vocabulary)?;
    let mut words = vec![0u32; vocabulary.div_ceil(32)];
    for &id in history {
        if id >= vocabulary {
            return Err(FishError::InvalidSampling(
                "semantic repetition history exceeds vocabulary",
            ));
        }
        words[id / 32] |= 1 << (id % 32);
    }
    session.upload_runtime_input(
        &TensorType::new(vec![words.len()], ScalarType::I32, Layout::Plain),
        bytemuck::cast_slice(&words),
    )
}

pub fn allocate_semantic_result<F>(session: &mut WeightedCudaArSession<F>) -> Result<CudaBuffer>
where
    F: FnMut(
        &TrtBlueprint,
        &std::collections::BTreeMap<Binding, CudaBuffer>,
        &Arc<CudaStream>,
    ) -> Result<()>,
{
    session
        .device_mut()
        .allocate_tensor(&TensorType::new(vec![2], ScalarType::I32, Layout::Plain))
}

pub fn parse_semantic_result(bytes: &[u8], vocabulary: usize, eos_id: u32) -> Result<(u32, bool)> {
    validate_vocabulary(vocabulary)?;
    if bytes.len() != 8 {
        return Err(FishError::InvalidShape(
            "CUDA semantic result is not two I32 values",
        ));
    }
    let token = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let eos = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if token < 0
        || token as usize >= vocabulary
        || (eos != 0 && eos != 1)
        || (eos == 1) != (token as u32 == eos_id)
    {
        return Err(FishError::InvalidSampling(
            "CUDA semantic token/EOS result invalid",
        ));
    }
    Ok((token as u32, eos == 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_vocabulary_is_not_restricted_to_fast_codebook_size() {
        assert!(validate_vocabulary(4097).is_ok());
        assert!(validate_vocabulary(155_000).is_ok());
        assert!(validate_vocabulary(0).is_err());
        assert!(validate_vocabulary(MAX_SEMANTIC_VOCAB + 1).is_err());
    }

    #[test]
    fn semantic_result_validates_eos_and_error_sentinel() {
        let bytes = [123_i32, 1_i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            parse_semantic_result(&bytes, 4096, 123).unwrap(),
            (123, true)
        );
        assert!(parse_semantic_result(&bytes, 4096, 124).is_err());
        assert!(parse_semantic_result(&bytes, 100, 123).is_err());
        let invalid = [-1_i32, -1_i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        assert!(parse_semantic_result(&invalid, 4096, 123).is_err());
    }
}
