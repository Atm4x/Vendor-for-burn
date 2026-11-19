use std::sync::Arc;

use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use fish_speech_core::{
    audio::FIREFLY_GROUPS,
    error::{FishError, Result},
    firefly_ir::{FireflyGraph, FireflyStage},
    synthesis::AcousticDecoder,
};
use trtx::{Builder, MemoryPoolType};

use crate::{
    cuda_stream::CudaScratchPool,
    cuda_trtx::ResidentTrtContexts,
    device_memory::DeviceMemoryBudget,
    reference_loader::{ReferenceAudioLoader, ReferenceLoadError},
    trtx_firefly_network::define_firefly_network,
    weights::LoadedWeights,
};

pub struct CudaFireflyDecoder {
    runtime: FireflyRuntime,
    max_frames: usize,
}

pub(crate) struct CudaFireflyReferenceEncoder {
    runtime: FireflyRuntime,
    loader: ReferenceAudioLoader,
}

#[derive(Clone)]
pub(crate) struct FireflyRuntimeOptions {
    pub device: usize,
    pub workspace_bytes: usize,
    pub memory: DeviceMemoryBudget,
    pub scratch: CudaScratchPool,
}

struct FireflyRuntime {
    cached: Option<FireflyEngine>,
    checkpoint: Arc<LoadedWeights>,
    options: FireflyRuntimeOptions,
}

struct FireflyEngine {
    profile: usize,
    input_name: String,
    output_elements: usize,
    resident: ResidentTrtContexts,
}

fn backend(error: impl std::fmt::Debug) -> FishError {
    FishError::Backend(format!("Firefly TensorRT: {error:?}"))
}

impl FireflyRuntime {
    fn new(checkpoint: Arc<LoadedWeights>, options: FireflyRuntimeOptions) -> Self {
        Self {
            cached: None,
            checkpoint,
            options,
        }
    }

    fn prepare(&self, profile: usize, graph: FireflyGraph) -> Result<FireflyEngine> {
        let options = &self.options;
        let mut resident =
            ResidentTrtContexts::empty(options.device, &options.memory, &options.scratch)
                .map_err(backend)?;
        let (image, spec) = {
            let mut builder = Builder::new(&resident.logger).map_err(backend)?;
            builder.set_gpu_allocator(
                resident
                    .allocator(&options.memory, &options.scratch)
                    .map_err(backend)?,
            );
            let mut network = builder.create_network(0).map_err(backend)?;
            let spec =
                define_firefly_network(&mut network, &graph, &self.checkpoint).map_err(backend)?;
            let mut config = builder.create_builder_config().map_err(backend)?;
            config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, options.workspace_bytes);
            if graph.stage == FireflyStage::ReferenceEncode {
                config.clear_flag(trtx::BuilderFlag::kTF32);
            }
            let image = builder
                .build_serialized_network(&mut network, &mut config)
                .map_err(backend)?;
            (image.as_ref().to_vec(), spec)
        };
        resident
            .insert("firefly".into(), &image, &options.memory, &options.scratch)
            .map_err(backend)?;
        Ok(FireflyEngine {
            profile,
            input_name: spec.external_name,
            output_elements: spec.output_shape.iter().product(),
            resident,
        })
    }

    fn run<I: DeviceRepr, O: DeviceRepr>(
        &mut self,
        profile: usize,
        graph: impl FnOnce() -> Result<FireflyGraph>,
        input: &[I],
    ) -> Result<Vec<O>> {
        if self
            .cached
            .as_ref()
            .is_none_or(|engine| engine.profile != profile)
        {
            self.cached = None;
            self.cached = Some(self.prepare(profile, graph()?)?);
        }
        let engine = self.cached.as_mut().expect("Firefly profile was prepared");
        let result = engine.run(input, &self.options);
        if result.is_err() {
            self.cached = None;
        }
        result
    }
}

impl CudaFireflyDecoder {
    pub(crate) fn new(
        checkpoint: Arc<LoadedWeights>,
        options: FireflyRuntimeOptions,
        max_frames: usize,
    ) -> Self {
        Self {
            runtime: FireflyRuntime::new(checkpoint, options),
            max_frames,
        }
    }
}

impl AcousticDecoder for CudaFireflyDecoder {
    fn decode(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let frames = codes.first().map_or(0, Vec::len);
        if codes.len() != FIREFLY_GROUPS
            || frames == 0
            || frames > self.max_frames
            || codes
                .iter()
                .any(|row| row.len() != frames || row.iter().any(|&code| code >= 1000))
        {
            return Err(FishError::InvalidShape(
                "invalid Firefly codebook indices or frame count",
            ));
        }
        let indices: Vec<i32> = codes.iter().flatten().map(|&code| code as i32).collect();
        let pcm: Vec<f32> = self.runtime.run(
            frames,
            || FireflyGraph::waveform_decode(1, frames, frames),
            &indices,
        )?;
        if pcm.iter().any(|sample| !sample.is_finite()) {
            self.runtime.cached = None;
            return Err(FishError::InvalidShape("Firefly returned non-finite PCM"));
        }
        Ok(pcm)
    }
}

impl CudaFireflyReferenceEncoder {
    pub(crate) fn new(checkpoint: Arc<LoadedWeights>, options: FireflyRuntimeOptions) -> Self {
        Self {
            runtime: FireflyRuntime::new(checkpoint, options),
            loader: ReferenceAudioLoader::default(),
        }
    }

    pub(crate) fn set_audio_loader(&mut self, loader: ReferenceAudioLoader) {
        self.loader = loader;
    }

    pub(crate) fn encode_reference_bytes(
        &mut self,
        audio: &[u8],
    ) -> std::result::Result<Vec<Vec<u32>>, ReferenceLoadError> {
        let mel = self.loader.load_bytes(audio)?.prepare_mel()?;
        let frames = mel.lengths.allocated_code_frames;
        let valid_frames = mel.lengths.valid_code_frames;
        let codes: Vec<i32> = self.runtime.run(
            mel.lengths.input_audio_samples,
            || mel.reference_graph(),
            &mel.mels.data,
        )?;
        if codes.len() != FIREFLY_GROUPS * frames || codes.iter().any(|&i| !(0..1000).contains(&i))
        {
            self.runtime.cached = None;
            return Err(FishError::InvalidShape("Firefly returned invalid reference codes").into());
        }
        Ok(codes
            .chunks_exact(frames)
            .map(|row| {
                row[..valid_frames]
                    .iter()
                    .map(|&code| code as u32)
                    .collect()
            })
            .collect())
    }
}

impl FireflyEngine {
    fn run<I: DeviceRepr, O: DeviceRepr>(
        &mut self,
        input: &[I],
        options: &FireflyRuntimeOptions,
    ) -> Result<Vec<O>> {
        self.resident.device.bind_to_thread().map_err(backend)?;
        let stream = Arc::clone(&self.resident.stream);
        let io_bytes = std::mem::size_of_val(input) + self.output_elements * size_of::<O>();
        let _lease = options
            .memory
            .reserve(io_bytes)
            .or_else(|_| {
                let _ = options.scratch.clear();
                options.memory.reserve(io_bytes)
            })
            .map_err(backend)?;
        let input = stream.clone_htod(input).map_err(backend)?;
        let mut output = unsafe { stream.alloc::<O>(self.output_elements) }.map_err(backend)?;
        unsafe {
            let (src, _src_access) = input.device_ptr(&stream);
            let (dst, _dst_access) = output.device_ptr_mut(&stream);
            let enqueued =
                self.resident
                    .enqueue_firefly(&self.input_name, src as *mut _, dst as *mut _);
            stream.synchronize().map_err(backend)?;
            enqueued.map_err(backend)?;
        }
        stream.clone_dtoh(&output).map_err(backend)
    }
}
