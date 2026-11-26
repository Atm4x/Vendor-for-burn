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

const MAX_FAST_VOCAB: usize = 4096;
const THREADS: u32 = 256;

fn token_type(length: usize) -> TensorType {
    TensorType::new(vec![length], ScalarType::I32, Layout::Plain)
}

fn sampler_geometry(vocab: usize) -> Result<(usize, u32)> {
    let padded = vocab
        .checked_next_power_of_two()
        .ok_or(FishError::InvalidShape(
            "CUDA Fast sampler vocabulary overflow",
        ))?;
    if vocab == 0 || vocab > MAX_FAST_VOCAB {
        return Err(FishError::Unsupported(
            "CUDA Fast sampler supports at most 4096 acoustic codes",
        ));
    }
    let shared_bytes = padded
        .checked_mul(8)
        .and_then(|size| u32::try_from(size).ok())
        .ok_or(FishError::InvalidShape(
            "CUDA sampler shared memory overflow",
        ))?;
    Ok((padded, shared_bytes))
}

fn check_logits_slice(total_bytes: usize, vocabulary: usize, offset: usize) -> Result<()> {
    sampler_geometry(vocabulary)?;
    let end = offset
        .checked_add(vocabulary)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or(FishError::InvalidShape("CUDA Fast logits offset overflow"))?;
    if end > total_bytes {
        return Err(FishError::InvalidShape(
            "CUDA Fast logits slice exceeds output",
        ));
    }
    Ok(())
}

pub struct CudaFastSampler {
    kernel: CudaFunction,
}

impl CudaFastSampler {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self> {
        static PTX: OnceLock<Mutex<Option<String>>> = OnceLock::new();
        let source = PTX.get_or_init(|| Mutex::new(None));
        let ptx = {
            let mut cached = source.lock().map_err(|_| {
                FishError::InvalidTransition("CUDA Fast sampler PTX cache poisoned")
            })?;
            if let Some(compiled) = cached.as_ref() {
                compiled.clone()
            } else {
                let compiled = compile_ptx(include_str!("cuda_fast_sampling.cu"))
                    .map_err(|error| FishError::Backend(format!("CUDA Fast sampler NVRTC: {error:?}")))?
                    .to_src();
                *cached = Some(compiled.clone());
                compiled
            }
        };
        let module = context
            .load_module(Ptx::from_src(ptx))
            .map_err(|_| FishError::InvalidTransition("CUDA Fast sampler PTX loading failed"))?;
        let kernel = module
            .load_function("fish_fast_sample")
            .map_err(|_| FishError::InvalidTransition("CUDA Fast sampler kernel absent"))?;
        Ok(Self { kernel })
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
        codebook_index: usize,
        result_token: &CudaBuffer,
        result_column: &CudaBuffer,
    ) -> Result<()> {
        self.enqueue_slice(
            stream,
            logits,
            history_mask,
            vocabulary,
            0,
            params,
            seed,
            codebook_index,
            result_token,
            result_column,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_slice(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaBuffer,
        history_mask: &CudaBuffer,
        vocabulary: usize,
        logits_offset: usize,
        params: SamplingParams,
        seed: u64,
        codebook_index: usize,
        result_token: &CudaBuffer,
        result_column: &CudaBuffer,
    ) -> Result<()> {
        params.validate()?;
        let (padded, shared_bytes) = sampler_geometry(vocabulary)?;
        check_logits_slice(logits.bytes(), vocabulary, logits_offset)?;
        if history_mask.bytes() != vocabulary * 4
            || result_token.bytes() != 4
            || result_column.bytes() < (codebook_index + 1) * 4
        {
            return Err(FishError::InvalidShape(
                "CUDA Fast sampler buffer size mismatch",
            ));
        }
        let logits_ptr = logits.address(stream);
        let history_ptr = history_mask.address(stream);
        let token_ptr = result_token.address(stream);
        let column_ptr = result_column.address(stream);
        let vocabulary = i32::try_from(vocabulary)
            .map_err(|_| FishError::InvalidShape("CUDA Fast sampler vocabulary overflow"))?;
        let padded = i32::try_from(padded)
            .map_err(|_| FishError::InvalidShape("CUDA Fast sampler padded size overflow"))?;
        let logits_offset = i32::try_from(logits_offset)
            .map_err(|_| FishError::InvalidShape("CUDA Fast logits offset exceeds i32"))?;
        let codebook = i32::try_from(codebook_index)
            .map_err(|_| FishError::InvalidShape("CUDA Fast sampler codebook overflow"))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut launch = stream.launch_builder(&self.kernel);
        launch
            .arg(&logits_ptr)
            .arg(&history_ptr)
            .arg(&vocabulary)
            .arg(&logits_offset)
            .arg(&padded)
            .arg(&params.top_p)
            .arg(&params.temperature)
            .arg(&params.repetition_penalty)
            .arg(&seed)
            .arg(&codebook)
            .arg(&token_ptr)
            .arg(&column_ptr);
        unsafe { launch.launch(cfg) }
            .map_err(|_| FishError::InvalidTransition("CUDA Fast sampler launch failed"))?;
        Ok(())
    }
}

pub fn upload_history_mask<F>(
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
    sampler_geometry(vocabulary)?;
    let mut mask = vec![0i32; vocabulary];
    for &token in history {
        let cell = mask.get_mut(token).ok_or(FishError::InvalidShape(
            "CUDA Fast sampler history index exceeds codebook",
        ))?;
        *cell = 1;
    }
    let mut bytes = Vec::with_capacity(vocabulary * 4);
    for entry in mask {
        bytes.extend_from_slice(&entry.to_le_bytes());
    }
    session.upload_runtime_input(&token_type(vocabulary), &bytes)
}

pub fn allocate_token<F>(session: &mut WeightedCudaArSession<F>) -> Result<CudaBuffer>
where
    F: FnMut(
        &TrtBlueprint,
        &std::collections::BTreeMap<Binding, CudaBuffer>,
        &Arc<CudaStream>,
    ) -> Result<()>,
{
    session.device_mut().allocate_tensor(&token_type(1))
}

pub fn upload_column<F>(
    session: &mut WeightedCudaArSession<F>,
    initial: u32,
    count: usize,
) -> Result<CudaBuffer>
where
    F: FnMut(
        &TrtBlueprint,
        &std::collections::BTreeMap<Binding, CudaBuffer>,
        &Arc<CudaStream>,
    ) -> Result<()>,
{
    let initial = i32::try_from(initial)
        .map_err(|_| FishError::InvalidShape("CUDA acoustic code exceeds i32"))?;
    if count == 0 {
        return Err(FishError::InvalidShape("CUDA acoustic column is empty"));
    }
    let mut bytes = Vec::with_capacity(count * 4);
    bytes.extend_from_slice(&initial.to_le_bytes());
    for _ in 1..count {
        bytes.extend_from_slice(&(-1i32).to_le_bytes());
    }
    session.upload_runtime_input(&token_type(count), &bytes)
}

pub fn parse_column(bytes: &[u8], count: usize, vocabulary: usize) -> Result<Vec<u32>> {
    if bytes.len() != count * 4 {
        return Err(FishError::InvalidShape(
            "CUDA acoustic column byte length mismatch",
        ));
    }
    bytes
        .chunks_exact(4)
        .map(|bytes| {
            let value = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            usize::try_from(value)
                .ok()
                .filter(|&value| value < vocabulary)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(FishError::InvalidShape(
                    "CUDA acoustic sampler produced an invalid code",
                ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_shared_memory_is_bounded() {
        assert_eq!(sampler_geometry(1024).unwrap(), (1024, 8192));
        assert_eq!(sampler_geometry(1025).unwrap(), (2048, 16384));
        assert!(sampler_geometry(0).is_err());
        assert!(sampler_geometry(4097).is_err());
    }

    #[test]
    fn naive_codebook_slices_cannot_cross_logit_boundaries() {
        assert!(check_logits_slice(3 * 16 * 4, 16, 0).is_ok());
        assert!(check_logits_slice(3 * 16 * 4, 16, 16).is_ok());
        assert!(check_logits_slice(3 * 16 * 4, 16, 32).is_ok());
        assert!(check_logits_slice(3 * 16 * 4, 16, 33).is_err());
        assert!(check_logits_slice(3 * 16 * 4, 16, usize::MAX).is_err());
        assert!(check_logits_slice(3 * 16 * 4, 4097, 0).is_err());
    }

    #[test]
    fn device_column_detects_bad_codes_without_any_gpu() {
        let valid = [5i32, 1, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(parse_column(&valid, 3, 10).unwrap(), [5, 1, 8]);
        assert!(parse_column(&valid, 2, 10).is_err());
        let invalid = [5i32, -1, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        assert!(parse_column(&invalid, 3, 10).is_err());
    }
}
