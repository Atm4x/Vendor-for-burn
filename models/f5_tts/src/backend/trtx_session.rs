//! Model-specific TensorRT-RTX inference session. The engine executor and the CUDA
//! transport are independent of the host-side F5 sampling and reference contracts.

use std::ffi::c_void;

use crate::{
    F5Error, Result, cfm::{CfmSampleMode, OdeMethod, SamplingConfig, duplicate_test_initial_state}, mel::VocosMel, noise::StandardNormal,
    reference::ReferenceAudio, speech_edit::SpeechEditPlan,
};

use super::{
    trtx::{DeviceBuffer, timestep_frequencies},
    trtx_cfm::{CfmStepBindings, enqueue_cfm_step},
    trtx_input::{TrtCfgInputs, TrtInputShape},
    trtx_speed::{
        TrtSpeedBindings, TrtSpeedInputs, TrtSpeedShape, decode_predicted_speeds, enqueue_speed,
    },
    trtx_vocos::{TrtVocosBindings, TrtVocosShape, decode_vocos_spectrum, enqueue_vocos},
};

pub trait CudaTransport {
    fn allocate(&mut self, bytes: usize) -> Result<DeviceBuffer>;
    fn free(&mut self, buffer: DeviceBuffer) -> Result<()>;
    fn upload(&mut self, buffer: DeviceBuffer, contents: &[u8]) -> Result<()>;
    fn download(&mut self, buffer: DeviceBuffer, contents: &mut [u8]) -> Result<()>;
    fn synchronize(&mut self) -> Result<()>;
    fn stream(&self) -> *mut c_void;
}

pub trait TrtExecutor {
    fn cfm_step(&mut self, bindings: CfmStepBindings, stream: *mut c_void) -> Result<()>;
    fn vocos(&mut self, bindings: TrtVocosBindings, stream: *mut c_void) -> Result<()>;
    fn speed(&mut self, bindings: TrtSpeedBindings, stream: *mut c_void) -> Result<()>;
}

pub struct TrtGraphExecutor<'a, 'engine> {
    pub cfm: &'a mut trtx::ExecutionContext<'engine>,
    pub vocos: &'a mut trtx::ExecutionContext<'engine>,
    pub speed: Option<&'a mut trtx::ExecutionContext<'engine>>,
    pub cfm_shape: TrtInputShape,
    pub vocos_shape: TrtVocosShape,
    pub speed_shape: Option<TrtSpeedShape>,
}

impl TrtExecutor for TrtGraphExecutor<'_, '_> {
    fn cfm_step(&mut self, bindings: CfmStepBindings, stream: *mut c_void) -> Result<()> {
        unsafe { enqueue_cfm_step(self.cfm, self.cfm_shape, bindings, stream) }
    }

    fn vocos(&mut self, bindings: TrtVocosBindings, stream: *mut c_void) -> Result<()> {
        unsafe { enqueue_vocos(self.vocos, self.vocos_shape, bindings, stream) }
    }

    fn speed(&mut self, bindings: TrtSpeedBindings, stream: *mut c_void) -> Result<()> {
        let (Some(context), Some(shape)) = (self.speed.as_deref_mut(), self.speed_shape) else {
            return Err(F5Error::InvalidConfiguration(
                "Cross-Lingual engine is not loaded",
            ));
        };
        unsafe { enqueue_speed(context, shape, bindings, stream) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SessionShape {
    pub cfm: TrtInputShape,
    pub vocos: TrtVocosShape,
    pub reference_frames: usize,
    pub speed: Option<TrtSpeedShape>,
}

impl SessionShape {
    pub fn validate(self) -> Result<()> {
        self.cfm.validate()?;
        self.vocos.validate()?;
        let cfg = self.cfm.backbone;
        if cfg.batch != 2
            || cfg.mel_channels != 100
            || self.reference_frames >= cfg.frames
            || self.vocos.frames != cfg.frames - self.reference_frames
            || self.vocos.mel_channels != cfg.mel_channels
        {
            return Err(F5Error::InvalidShape(
                "Incompatible fixed-shape TRT F5/CFM/Vocos models",
            ));
        }
        if let Some(speed) = self.speed {
            if self.reference_frames == 0 {
                return Err(F5Error::InvalidConfiguration(
                    "Speech edit TRT engine has no reference speed model",
                ));
            }
            speed.validate()?;
            if speed.batch != 1 {
                return Err(F5Error::InvalidConfiguration(
                    "Cross-Lingual session requires one reference",
                ));
            }
        }
        Ok(())
    }

    fn mel_bytes(self) -> usize {
        self.cfm.backbone.frames * 100 * 4
    }
}

const A: usize = 0;
const B: usize = 1;
const CONDITION: usize = 2;
const TOKENS: usize = 3;
const TEXT_MASK: usize = 4;
const SINCOS: usize = 5;
const DT: usize = 6;
const CFG: usize = 7;
const RESTORE: usize = 8;
const REFERENCE_MASK: usize = 9;
const SPECTRAL: usize = 10;
const SPEED_MEL: usize = 11;
const SPEED_MASK: usize = 12;
const SPEED_LOGITS: usize = 13;

pub struct TrtF5Session<I: CudaTransport, E: TrtExecutor> {
    io: I,
    executor: E,
    shape: SessionShape,
    buffers: Vec<DeviceBuffer>,
    midpoint_buffer: usize,
}

pub struct SynthesisInputs<'a> {
    pub condition: &'a [f32],
    pub initial_noise: &'a [f32],
    pub tokens: &'a [i32],
    pub restoration_gain: f32,
    pub sampling: SamplingConfig,
}

impl<I: CudaTransport, E: TrtExecutor> TrtF5Session<I, E> {
    pub fn open(mut io: I, executor: E, shape: SessionShape) -> Result<Self> {
        shape.validate()?;
        if io.stream().is_null() {
            return Err(F5Error::InvalidConfiguration(
                "TensorRT session requires a CUDA stream",
            ));
        }
        let frames = shape.cfm.backbone.frames;
        let s = shape.speed;
        let mut bytes = vec![
            shape.mel_bytes(),
            shape.mel_bytes(),
            shape.mel_bytes(),
            frames * 2 * 4,
            frames * 2 * 4,
            2 * 256 * 4,
            4,
            4,
            4,
            frames * 4,
            shape.vocos.head_bytes()?,
        ];
        if let Some(s) = s {
            bytes.extend_from_slice(&[
                s.batch * s.frames * 100 * 4,
                s.batch * s.frames * 4,
                s.batch * 32 * 4,
            ]);
        }
        let midpoint_buffer = bytes.len();
        bytes.push(shape.mel_bytes());
        let mut buffers = Vec::with_capacity(bytes.len());
        for size in bytes {
            match io.allocate(size) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    let _ = io.synchronize();
                    for buffer in buffers.into_iter().rev() {
                        let _ = io.free(buffer);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self {
            io,
            executor,
            shape,
            buffers,
            midpoint_buffer,
        })
    }

    pub fn shape(&self) -> SessionShape {
        self.shape
    }

    pub fn synthesize(&mut self, request: SynthesisInputs<'_>) -> Result<Vec<f32>> {
        self.synthesize_with_method(request, OdeMethod::Euler)
    }

    pub fn synthesize_with_method(&mut self, request: SynthesisInputs<'_>, method: OdeMethod) -> Result<Vec<f32>> {
        self.synthesize_with_sample_mode(request, method, CfmSampleMode::default())
    }

    pub fn synthesize_with_sample_mode(
        &mut self, request: SynthesisInputs<'_>, method: OdeMethod, mode: CfmSampleMode,
    ) -> Result<Vec<f32>> {
        let result = self.synthesize_inner(request, method, mode);
        let synchronized = self.io.synchronize();
        match (result, synchronized) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(waveform), Ok(())) => Ok(waveform),
        }
    }

    /// Speech editing runs CFM with an arbitrary preserved-frame mask and
    /// decodes the complete waveform, instead of trimming off a reference prefix.
    /// Requires an edit-shaped engine (reference_frames=0, vocos.frames=total).
    pub fn edit_speech(
        &mut self, plan: &SpeechEditPlan, tokens: &[i32], noise: &[f32],
        restoration_gain: f32, sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        self.edit_speech_with_method(plan, tokens, noise, restoration_gain, sampling, OdeMethod::Euler)
    }

    pub fn edit_speech_with_method(
        &mut self, plan: &SpeechEditPlan, tokens: &[i32], noise: &[f32],
        restoration_gain: f32, sampling: SamplingConfig, method: OdeMethod,
    ) -> Result<Vec<f32>> {
        let frames = self.shape.cfm.backbone.frames;
        if self.shape.reference_frames != 0
            || self.shape.vocos.frames != frames
            || plan.total_frames != frames
            || plan.keep_frames.len() != frames
        {
            return Err(F5Error::InvalidShape(
                "Speech editing needs a full-length TRT edit engine",
            ));
        }
        let mask: Vec<f32> = plan
            .keep_frames
            .iter()
            .map(|&keep| if keep { 1.0 } else { 0.0 })
            .collect();
        let result = self.synthesize_inner_masked(
            SynthesisInputs {
                condition: &plan.conditioning_mel,
                initial_noise: noise,
                tokens,
                restoration_gain,
                sampling,
            },
            &mask,
            0,
            method,
            CfmSampleMode::default(),
        );
        let synchronized = self.io.synchronize();
        match (result, synchronized) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(waveform), Ok(())) => Ok(waveform),
        }
    }

    pub fn synthesize_seeded(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let prepared = reference.prepare(
            &VocosMel::new(),
            self.shape.cfm.backbone.frames,
            tokens.len(),
        )?;
        if prepared.frames.reference != self.shape.reference_frames
            || prepared.frames.target != self.shape.cfm.backbone.frames
        {
            return Err(F5Error::InvalidShape(
                "Reference duration differs from the warmed TensorRT engine",
            ));
        }
        let mut rng = StandardNormal::new(seed);
        let noise = rng.sample_vec(self.shape.mel_bytes() / 4)?;
        self.synthesize(SynthesisInputs {
            condition: &prepared.conditioning_mel,
            initial_noise: &noise,
            tokens,
            restoration_gain: prepared.restoration_gain(),
            sampling,
        })
    }

    fn synthesize_inner(
        &mut self, request: SynthesisInputs<'_>, method: OdeMethod, mode: CfmSampleMode,
    ) -> Result<Vec<f32>> {
        let frames = self.shape.cfm.backbone.frames;
        let r = self.shape.reference_frames;
        if r == 0 {
            return Err(F5Error::InvalidConfiguration(
                "Standard synthesis requires a reference prefix",
            ));
        }
        if request.condition.len() != frames * 100
            || request.condition[r * 100..].iter().any(|v| *v != 0.0)
        {
            return Err(F5Error::InvalidShape(
                "Reference conditioning has nonzero generated suffix",
            ));
        }
        let mut mask = vec![0.0; frames];
        mask[..r].fill(1.0);
        self.synthesize_inner_masked(request, &mask, r, method, mode)
    }

    fn synthesize_inner_masked(
        &mut self,
        request: SynthesisInputs<'_>,
        mask: &[f32],
        decoded_from: usize,
        method: OdeMethod,
        mode: CfmSampleMode,
    ) -> Result<Vec<f32>> {
        let cfg = self.shape.cfm;
        mode.validate()?;
        let frames = cfg.backbone.frames;
        if request.condition.len() != frames * 100
            || request.initial_noise.len() != frames * 100
            || request.tokens.is_empty()
            || request.tokens.len() > frames
            || !request.restoration_gain.is_finite()
            || request.restoration_gain < 0.0
            || request
                .condition
                .iter()
                .chain(request.initial_noise.iter())
                .any(|value| !value.is_finite())
            || mask.len() != frames
            || decoded_from != self.shape.reference_frames
            || mask.iter().any(|value| *value != 0.0 && *value != 1.0)
            || request
                .condition
                .iter()
                .enumerate()
                .any(|(i, &value)| mask[i / 100] == 0.0 && value != 0.0)
        {
            return Err(F5Error::InvalidShape(
                "Invalid TensorRT synthesis mel, tokens or gain",
            ));
        }
        let cfg_inputs = TrtCfgInputs::from_original_tokens(
            cfg,
            request.tokens.len(),
            request.initial_noise,
            request.condition,
            request.tokens,
        )?;
        let mut initial_state = request.initial_noise.to_vec();
        if mode.duplicate_test {
            duplicate_test_initial_state(
                &mut initial_state, request.condition, decoded_from.max(1), 100, mode.t_inter,
            )?;
        }
        self.upload_f32(A, &initial_state)?;
        if mode.no_ref_audio {
            self.upload_f32(CONDITION, &vec![0.0; frames * 100])?;
        } else {
            self.upload_f32(CONDITION, request.condition)?;
        }
        self.upload_i32(TOKENS, &cfg_inputs.tokens)?;
        self.upload_i32(TEXT_MASK, &cfg_inputs.text_mask)?;
        self.upload_f32(CFG, &[request.sampling.cfg_strength])?;
        self.upload_f32(REFERENCE_MASK, mask)?;

        let timeline = mode.timeline(request.sampling)?;
        let mut current = A;
        let mut next = B;
        for (index, times) in timeline.windows(2).enumerate() {
            let dt = times[1] - times[0];
            let last = f32::from(index + 2 == timeline.len());
            let base_state = self.buffers[current];
            let stream = self.io.stream();
            if method == OdeMethod::Midpoint {
                let freq = timestep_frequencies(&[times[0], times[0]])?;
                self.upload_f32(SINCOS, &freq)?;
                self.upload_f32(DT, &[dt * 0.5])?;
                self.upload_f32(RESTORE, &[0.0])?;
                let mid = CfmStepBindings {
                    state: base_state,
                    base_state,
                    condition: self.buffers[CONDITION],
                    tokens: self.buffers[TOKENS],
                    text_mask: self.buffers[TEXT_MASK],
                    sincos: self.buffers[SINCOS],
                    dt: self.buffers[DT],
                    cfg_strength: self.buffers[CFG],
                    restore: self.buffers[RESTORE],
                    reference_mask: self.buffers[REFERENCE_MASK],
                    next_state: self.buffers[self.midpoint_buffer],
                };
                mid.validate(cfg)?;
                self.executor.cfm_step(mid, stream)?;
            }
            let eval_time = if method == OdeMethod::Midpoint { times[0] + 0.5 * dt } else { times[0] };
            let freq = timestep_frequencies(&[eval_time, eval_time])?;
            self.upload_f32(SINCOS, &freq)?;
            self.upload_f32(DT, &[dt])?;
            self.upload_f32(RESTORE, &[last])?;
            let bindings = CfmStepBindings {
                state: if method == OdeMethod::Midpoint {
                    self.buffers[self.midpoint_buffer]
                } else {
                    base_state
                },
                base_state,
                condition: self.buffers[CONDITION],
                tokens: self.buffers[TOKENS],
                text_mask: self.buffers[TEXT_MASK],
                sincos: self.buffers[SINCOS],
                dt: self.buffers[DT],
                cfg_strength: self.buffers[CFG],
                restore: self.buffers[RESTORE],
                reference_mask: self.buffers[REFERENCE_MASK],
                next_state: self.buffers[next],
            };
            bindings.validate(cfg)?;
            self.executor.cfm_step(bindings, stream)?;
            std::mem::swap(&mut current, &mut next);
        }
        let generated_bytes = self.shape.vocos.mel_bytes()?;
        let mel = DeviceBuffer {
            address: self.buffers[current].address + decoded_from * 100 * 4,
            bytes: generated_bytes,
        };
        let bindings = TrtVocosBindings {
            mel,
            spectral_head: self.buffers[SPECTRAL],
        };
        bindings.validate(self.shape.vocos)?;
        self.executor.vocos(bindings, self.io.stream())?;
        self.io.synchronize()?;
        let mut spectral = vec![0.0; self.shape.vocos.head_bytes()? / 4];
        self.download_f32(SPECTRAL, &mut spectral)?;
        decode_vocos_spectrum(&spectral, request.restoration_gain)
    }

    pub fn predict_speed(&mut self, reference_mel: &[f32]) -> Result<f32> {
        let result = self.predict_speed_inner(reference_mel);
        let synchronized = self.io.synchronize();
        match (result, synchronized) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        }
    }

    pub fn predict_reference_speed(&mut self, reference: &ReferenceAudio) -> Result<f32> {
        let mel = VocosMel::new().extract(&reference.samples)?;
        self.predict_speed(mel.as_flat_frames())
    }

    fn predict_speed_inner(&mut self, reference_mel: &[f32]) -> Result<f32> {
        let shape = self.shape.speed.ok_or(F5Error::InvalidConfiguration(
            "Cross-Lingual speed engine is not loaded",
        ))?;
        if reference_mel.is_empty()
            || reference_mel.len() % 100 != 0
            || reference_mel.len() > shape.frames * 100
            || reference_mel.iter().any(|x| !x.is_finite())
        {
            return Err(F5Error::InvalidShape(
                "Invalid reference mel for Cross-Lingual speed",
            ));
        }
        let count = reference_mel.len() / 100;
        let mut padded = vec![0.0; shape.frames * 100];
        padded[..reference_mel.len()].copy_from_slice(reference_mel);
        let mask = TrtSpeedInputs::from_lengths(shape, &[count])?;
        self.upload_f32(SPEED_MEL, &padded)?;
        self.upload_i32(SPEED_MASK, &mask.valid_frames)?;
        let bindings = TrtSpeedBindings {
            mel: self.buffers[SPEED_MEL],
            valid_frames: self.buffers[SPEED_MASK],
            logits: self.buffers[SPEED_LOGITS],
        };
        bindings.validate(shape)?;
        self.executor.speed(bindings, self.io.stream())?;
        self.io.synchronize()?;
        let mut logits = vec![0.0; 32];
        self.download_f32(SPEED_LOGITS, &mut logits)?;
        Ok(decode_predicted_speeds(&logits, 1)?[0])
    }

    fn upload_f32(&mut self, index: usize, data: &[f32]) -> Result<()> {
        if data.len() * 4 > self.buffers[index].bytes {
            return Err(F5Error::InvalidShape(
                "F5 GPU upload exceeds allocated region",
            ));
        }
        self.io.upload(self.buffers[index], unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4)
        })
    }

    fn upload_i32(&mut self, index: usize, data: &[i32]) -> Result<()> {
        if data.len() * 4 > self.buffers[index].bytes {
            return Err(F5Error::InvalidShape(
                "F5 GPU token upload exceeds allocated region",
            ));
        }
        self.io.upload(self.buffers[index], unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4)
        })
    }

    fn download_f32(&mut self, index: usize, values: &mut [f32]) -> Result<()> {
        if values.len() * 4 > self.buffers[index].bytes {
            return Err(F5Error::InvalidShape(
                "F5 GPU download exceeds allocated region",
            ));
        }
        self.io.download(self.buffers[index], unsafe {
            std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), values.len() * 4)
        })
    }
}

impl<I: CudaTransport, E: TrtExecutor> Drop for TrtF5Session<I, E> {
    fn drop(&mut self) {
        let _ = self.io.synchronize();
        for buffer in self.buffers.drain(..).rev() {
            let _ = self.io.free(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

    use super::*;
    use crate::backend::trtx::{BackboneShape, DeviceBuffer};

    #[derive(Default)]
    struct TestGpu {
        regions: BTreeMap<usize, Vec<u8>>,
        allocation_count: usize,
        release_count: usize,
        sync_count: usize,
        downloads: Vec<usize>,
        observed_mel: Vec<f32>,
        stages: usize,
        fail_step: bool,
        affine_velocity: bool,
    }

    impl TestGpu {
        fn locate(&self, buffer: DeviceBuffer, count: usize) -> Result<(usize, usize)> {
            for (address, contents) in &self.regions {
                if buffer.address >= *address && buffer.address - address <= contents.len() {
                    let offset = buffer.address - address;
                    if count <= contents.len() - offset {
                        return Ok((*address, offset));
                    }
                }
            }
            Err(F5Error::InvalidShape("Test GPU access outside allocation"))
        }

        fn read(&self, buffer: DeviceBuffer, count: usize) -> Vec<u8> {
            let (address, offset) = self.locate(buffer, count).unwrap();
            self.regions[&address][offset..offset + count].to_vec()
        }

        fn write(&mut self, buffer: DeviceBuffer, bytes: &[u8]) {
            let (address, offset) = self.locate(buffer, bytes.len()).unwrap();
            self.regions.get_mut(&address).unwrap()[offset..offset + bytes.len()]
                .copy_from_slice(bytes);
        }

        fn floats(&self, buffer: DeviceBuffer) -> Vec<f32> {
            self.read(buffer, buffer.bytes)
                .chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                .collect()
        }

        fn write_floats(&mut self, buffer: DeviceBuffer, values: &[f32]) {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            self.write(buffer, &data);
        }
    }

    struct TestIo(Rc<RefCell<TestGpu>>);

    impl CudaTransport for TestIo {
        fn allocate(&mut self, bytes: usize) -> Result<DeviceBuffer> {
            let mut data = self.0.borrow_mut();
            let address = 0x100_000 + data.allocation_count * 0x1_000_000;
            data.allocation_count += 1;
            data.regions.insert(address, vec![0; bytes]);
            Ok(DeviceBuffer { address, bytes })
        }

        fn free(&mut self, buffer: DeviceBuffer) -> Result<()> {
            let mut data = self.0.borrow_mut();
            if data.regions.remove(&buffer.address).is_none() {
                return Err(F5Error::InvalidShape("Unknown test CUDA allocation"));
            }
            data.release_count += 1;
            Ok(())
        }

        fn upload(&mut self, buffer: DeviceBuffer, contents: &[u8]) -> Result<()> {
            self.0.borrow_mut().write(buffer, contents);
            Ok(())
        }

        fn download(&mut self, buffer: DeviceBuffer, contents: &mut [u8]) -> Result<()> {
            let mut data = self.0.borrow_mut();
            data.downloads.push(buffer.address);
            contents.copy_from_slice(&data.read(buffer, contents.len()));
            Ok(())
        }

        fn synchronize(&mut self) -> Result<()> {
            self.0.borrow_mut().sync_count += 1;
            Ok(())
        }

        fn stream(&self) -> *mut c_void {
            0x1000 as *mut c_void
        }
    }

    struct TestExecutor(Rc<RefCell<TestGpu>>);

    impl TrtExecutor for TestExecutor {
        fn cfm_step(&mut self, bindings: CfmStepBindings, _stream: *mut c_void) -> Result<()> {
            let mut data = self.0.borrow_mut();
            if data.fail_step {
                return Err(F5Error::ModelExecution("Injected CFM failure".into()));
            }
            let state = data.floats(bindings.state);
            let base = data.floats(bindings.base_state);
            let condition = data.floats(bindings.condition);
            let reference = data.floats(bindings.reference_mask);
            let cfg = data.floats(bindings.cfg_strength)[0];
            let dt = data.floats(bindings.dt)[0];
            let restore = data.floats(bindings.restore)[0];
            let guided_velocity = 0.1 + cfg * (0.1 - (-0.05));
            let affine = data.affine_velocity;
            let output: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    let velocity = if affine { 0.5 * state[i] + 0.1 } else { guided_velocity };
                    let sample = *value + velocity * dt;
                    if restore > 0.5 && reference[i / 100] > 0.5 {
                        condition[i]
                    } else {
                        sample
                    }
                })
                .collect();
            data.write_floats(bindings.next_state, &output);
            data.stages += 1;
            Ok(())
        }

        fn vocos(&mut self, bindings: TrtVocosBindings, _stream: *mut c_void) -> Result<()> {
            let mut data = self.0.borrow_mut();
            data.observed_mel = data.floats(bindings.mel);
            let mut head = vec![0.0; bindings.spectral_head.bytes / 4];
            for frame in head.chunks_exact_mut(1026) {
                frame[..513].fill(-8.0);
            }
            data.write_floats(bindings.spectral_head, &head);
            Ok(())
        }

        fn speed(&mut self, bindings: TrtSpeedBindings, _stream: *mut c_void) -> Result<()> {
            let mut data = self.0.borrow_mut();
            let mut logits = vec![0.0; bindings.logits.bytes / 4];
            logits[6] = 1.0;
            data.write_floats(bindings.logits, &logits);
            Ok(())
        }
    }

    fn test_shape() -> SessionShape {
        SessionShape {
            cfm: TrtInputShape {
                backbone: BackboneShape {
                    batch: 2,
                    frames: 6,
                    dim: 16,
                    heads: 4,
                    depth: 1,
                    ff_mult: 2,
                    mel_channels: 100,
                },
                text_dim: 4,
                vocab_size: 20,
                text_blocks: 1,
            },
            vocos: TrtVocosShape {
                frames: 4,
                mel_channels: 100,
                dim: 8,
                intermediate_dim: 16,
                layers: 1,
            },
            reference_frames: 2,
            speed: Some(TrtSpeedShape {
                batch: 1,
                frames: 5,
                dim: 8,
                depth: 1,
                heads: 2,
                ff_mult: 2,
                position_groups: 2,
            }),
        }
    }

    #[test]
    fn speech_edit_reuses_gpu_cfm_and_decodes_full_mel() {
        let data = Rc::new(RefCell::new(TestGpu::default()));
        let mut shape = test_shape();
        shape.reference_frames = 0;
        shape.vocos.frames = shape.cfm.backbone.frames;
        shape.speed = None;
        let mut session =
            TrtF5Session::open(TestIo(data.clone()), TestExecutor(data.clone()), shape).unwrap();
        let mel = vec![0.75; 5 * 100];
        let second = crate::mel::HOP_SIZE as f64 / crate::mel::SAMPLE_RATE as f64;
        let plan = SpeechEditPlan::new(
            &mel,
            5,
            &[crate::speech_edit::EditSpan {
                start_seconds: second,
                end_seconds: 3.0 * second,
                replacement_seconds: Some(2.0 * second),
            }],
            3,
        )
        .unwrap();
        assert_eq!(plan.total_frames, 6);
        assert_eq!(
            plan.keep_frames,
            vec![true, false, false, true, true, false]
        );
        let audio = session
            .edit_speech(
                &plan,
                &[1, 2, 3],
                &vec![0.25; 600],
                1.0,
                SamplingConfig {
                    steps: 5,
                    cfg_strength: 2.0,
                    sway_sampling_coef: None,
                    empirically_pruned_steps: true,
                },
            )
            .unwrap();
        assert!(!audio.is_empty());
        assert!(audio.iter().all(|v| v.is_finite()));
        let observed = data.borrow().observed_mel.clone();
        assert_eq!(observed.len(), 600);
        for (frame, &keep) in plan.keep_frames.iter().enumerate() {
            let expected = if keep { 0.75 } else { 0.65 };
            let actual = &observed[frame * 100..(frame + 1) * 100];
            assert!(
                actual.iter().all(|v| (v - expected).abs() < 2e-5),
                "incorrect edit restoration at frame {frame}"
            );
        }
        assert_eq!(data.borrow().stages, 5);
        assert!(
            session
                .synthesize(SynthesisInputs {
                    condition: &plan.conditioning_mel,
                    initial_noise: &vec![0.25; 600],
                    tokens: &[1, 2, 3],
                    restoration_gain: 1.0,
                    sampling: SamplingConfig::default(),
                })
                .is_err()
        );
    }

    #[test]
    fn full_session_keeps_cfm_states_on_gpu_and_reuses_memory() {
        let data = Rc::new(RefCell::new(TestGpu::default()));
        let shape = test_shape();
        let mut session =
            TrtF5Session::open(TestIo(data.clone()), TestExecutor(data.clone()), shape).unwrap();
        let mut conditioning = vec![0.0; 600];
        conditioning[..200].fill(0.75);
        let noise = vec![0.25; 600];
        let sampling = SamplingConfig {
            steps: 5,
            cfg_strength: 2.0,
            sway_sampling_coef: None,
            empirically_pruned_steps: true,
        };
        let make_request = || SynthesisInputs {
            condition: &conditioning,
            initial_noise: &noise,
            tokens: &[1, 2, 3],
            restoration_gain: 0.5,
            sampling,
        };
        let waveform1 = session.synthesize(make_request()).unwrap();
        assert!(waveform1.iter().all(|v| v.is_finite()));
        let waveform2 = session.synthesize(make_request()).unwrap();
        assert_eq!(waveform1, waveform2);
        let speed = session.predict_speed(&vec![0.2; 300]).unwrap();
        assert_eq!(speed, 1.75);
        {
            let observed = data.borrow();
            assert_eq!(observed.stages, 10);
            assert_eq!(observed.observed_mel.len(), 400);
            let oracle: serde_json::Value = serde_json::from_str(include_str!(
                "../../tests/fixtures/trtx_session_pytorch_expected.json"
            ))
            .unwrap();
            for (actual, expected) in observed
                .observed_mel
                .iter()
                .zip(oracle["generated_mel"].as_array().unwrap())
            {
                assert!((*actual - expected.as_f64().unwrap() as f32).abs() < 5e-6);
            }
            assert_eq!(observed.downloads.len(), 3);
            assert_ne!(observed.downloads[0], observed.downloads[2]);
            assert_eq!(observed.allocation_count, 15);
            assert_eq!(observed.release_count, 0);
        }
        drop(session);
        let observed = data.borrow();
        assert_eq!(observed.allocation_count, observed.release_count);
        assert!(observed.sync_count >= 4);
    }

    #[test]
    fn midpoint_retains_original_state_and_keeps_mel_on_mock_gpu() {
        let data = Rc::new(RefCell::new(TestGpu::default()));
        let mut session = TrtF5Session::open(
            TestIo(data.clone()), TestExecutor(data.clone()), test_shape()
        ).unwrap();
        let mut condition = vec![0.0; 600];
        condition[..200].fill(0.5);
        let sampling = SamplingConfig {
            steps: 5, cfg_strength: 1.0, sway_sampling_coef: None, empirically_pruned_steps: true,
        };
        let audio = session.synthesize_with_method(SynthesisInputs {
            condition: &condition, initial_noise: &vec![0.0; 600], tokens: &[1, 2],
            restoration_gain: 1.0, sampling,
        }, OdeMethod::Midpoint).unwrap();
        assert!(!audio.is_empty());
        let state = data.borrow();
        assert_eq!(state.stages, 10);
        assert!(state.observed_mel.iter().all(|x| (*x - 0.25).abs() < 1e-5));
    }

    #[test]
    fn midpoint_state_dependent_velocity_agrees_with_torch_midpoint_not_euler() {
        let data = Rc::new(RefCell::new(TestGpu { affine_velocity: true, ..Default::default() }));
        let mut session = TrtF5Session::open(
            TestIo(data.clone()), TestExecutor(data.clone()), test_shape()
        ).unwrap();
        let condition = vec![0.0; 600];
        let noise = vec![0.25; 600];
        let sampling = SamplingConfig {
            steps: 2, cfg_strength: 0.0,
            sway_sampling_coef: None, empirically_pruned_steps: false,
        };
        session.synthesize_with_method(SynthesisInputs {
            condition: &condition, initial_noise: &noise, tokens: &[1, 2],
            restoration_gain: 1.0, sampling,
        }, OdeMethod::Midpoint).unwrap();
        let observed = data.borrow();
        assert_eq!(observed.stages, 4);
        for value in &observed.observed_mel {
            // Independent PyTorch torch.float32 explicit midpoint, f(x)=0.5*x+0.1.
            assert!((*value - 0.5387207).abs() < 2e-7, "midpoint state {value}");
            assert!((*value - 0.503125).abs() > 0.03, "incorrectly used Euler");
        }
    }

    #[test]
    fn duplicate_test_no_ref_and_t_inter_reuse_warmed_cfm_engine() {
        let data = Rc::new(RefCell::new(TestGpu::default()));
        let mut session = TrtF5Session::open(
            TestIo(data.clone()), TestExecutor(data.clone()), test_shape()
        ).unwrap();
        let mut condition = vec![0.0; 600];
        condition[..200].fill(0.8);
        let noise = vec![0.25; 600];
        let sampling = SamplingConfig {
            steps: 4, cfg_strength: 1.0,
            sway_sampling_coef: None, empirically_pruned_steps: true,
        };
        let mode = CfmSampleMode {
            no_ref_audio: true, duplicate_test: true, t_inter: 0.5,
        };
        session.synthesize_with_sample_mode(SynthesisInputs {
            condition: &condition, initial_noise: &noise, tokens: &[1, 2],
            restoration_gain: 1.0, sampling,
        }, OdeMethod::Euler, mode).unwrap();
        let state = data.borrow();
        assert_eq!(state.stages, 2);
        assert_eq!(state.observed_mel.len(), 400);
        for (i, value) in state.observed_mel.iter().enumerate() {
            let expected = if i < 200 { 0.65 } else { 0.25 };
            assert!((*value - expected).abs() < 2e-7, "sample {i}: {value}");
        }
    }

    #[test]
    fn f5_host_timestep_embeddings_match_independent_pytorch() {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/trtx_session_pytorch_expected.json"
        ))
        .unwrap();
        for (index, time) in oracle["timeline"]
            .as_array()
            .unwrap()
            .iter()
            .take(5)
            .enumerate()
        {
            let frequencies = timestep_frequencies(&[time.as_f64().unwrap() as f32]).unwrap();
            let expected = oracle["frequency_embeddings"][index].as_array().unwrap();
            assert_eq!(frequencies.len(), expected.len());
            for (actual, expected) in frequencies.iter().zip(expected) {
                assert!((*actual - expected.as_f64().unwrap() as f32).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn session_rejects_invalid_reference_and_syncs_on_failure() {
        let data = Rc::new(RefCell::new(TestGpu::default()));
        let mut session = TrtF5Session::open(
            TestIo(data.clone()),
            TestExecutor(data.clone()),
            test_shape(),
        )
        .unwrap();
        let condition = vec![0.5; 600];
        let noise = vec![0.25; 600];
        let invalid = session.synthesize(SynthesisInputs {
            condition: &condition,
            initial_noise: &noise,
            tokens: &[0],
            restoration_gain: 1.0,
            sampling: SamplingConfig::default(),
        });
        assert!(invalid.is_err());
        let mut condition = condition;
        condition[200..].fill(0.0);
        data.borrow_mut().fail_step = true;
        let broken = session.synthesize(SynthesisInputs {
            condition: &condition,
            initial_noise: &noise,
            tokens: &[0],
            restoration_gain: 1.0,
            sampling: SamplingConfig::default(),
        });
        assert!(matches!(broken, Err(F5Error::ModelExecution(_))));
        assert!(data.borrow().sync_count >= 2);
        drop(session);
        assert!(data.borrow().regions.is_empty());
    }
}

pub struct TrtEngineImages<'a> {
    pub cfm: &'a [u8],
    pub vocos: &'a [u8],
    pub speed: Option<&'a [u8]>,
}

pub struct TrtLibraryPaths<'a> {
    pub cuda_runtime: &'a std::path::Path,
    pub tensorrt_rtx: &'a std::path::Path,
}

pub trait TrtSynthesis {
    fn synthesize(&mut self, request: SynthesisInputs<'_>) -> Result<Vec<f32>>;
    fn synthesize_with_sample_mode(
        &mut self, request: SynthesisInputs<'_>, method: OdeMethod, mode: CfmSampleMode,
    ) -> Result<Vec<f32>>;
    fn edit_speech(
        &mut self,
        plan: &SpeechEditPlan,
        tokens: &[i32],
        noise: &[f32],
        restoration_gain: f32,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>>;
    fn synthesize_seeded(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>>;
    fn predict_speed(&mut self, reference_mel: &[f32]) -> Result<f32>;
}

impl<I: CudaTransport, E: TrtExecutor> TrtSynthesis for TrtF5Session<I, E> {
    fn edit_speech(
        &mut self,
        plan: &SpeechEditPlan,
        tokens: &[i32],
        noise: &[f32],
        restoration_gain: f32,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        TrtF5Session::edit_speech(self, plan, tokens, noise, restoration_gain, sampling)
    }

    fn synthesize(&mut self, request: SynthesisInputs<'_>) -> Result<Vec<f32>> {
        TrtF5Session::synthesize(self, request)
    }

    fn synthesize_with_sample_mode(
        &mut self, request: SynthesisInputs<'_>, method: OdeMethod, mode: CfmSampleMode,
    ) -> Result<Vec<f32>> {
        TrtF5Session::synthesize_with_sample_mode(self, request, method, mode)
    }

    fn synthesize_seeded(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        TrtF5Session::synthesize_seeded(self, reference, tokens, seed, sampling)
    }

    fn predict_speed(&mut self, reference_mel: &[f32]) -> Result<f32> {
        TrtF5Session::predict_speed(self, reference_mel)
    }
}

/// Creates all TensorRT engines and contexts on the stack, keeps them warmed for
/// every request inside `serve`, then synchronizes and releases device memory before
/// the contexts, engines, runtime and logger are destroyed. No self-referential owner.
pub fn with_warm_session<R>(
    logger: &trtx::Logger,
    paths: TrtLibraryPaths<'_>,
    images: TrtEngineImages<'_>,
    shape: SessionShape,
    serve: impl FnOnce(&mut dyn TrtSynthesis) -> Result<R>,
) -> Result<R> {
    shape.validate()?;
    if images.cfm.is_empty()
        || images.vocos.is_empty()
        || images.speed.is_some_and(|data| data.is_empty())
        || images.speed.is_some() != shape.speed.is_some()
    {
        return Err(F5Error::InvalidConfiguration(
            "Missing TensorRT-RTX engine image",
        ));
    }
    let io = super::trtx_cuda::CudaRuntimeIo::open(paths.cuda_runtime)?;
    super::trtx_cuda::CudaRuntimeIo::load_tensorrt_rtx(paths.tensorrt_rtx)?;
    let mut runtime = trtx::Runtime::new(logger)
        .map_err(|e| F5Error::ModelExecution(format!("TensorRT-RTX runtime: {e}")))?;
    let mut cfm_engine = runtime
        .deserialize_cuda_engine(images.cfm)
        .map_err(|e| F5Error::ModelExecution(format!("TensorRT-RTX CFM deserialize: {e}")))?;
    let mut vocos_engine = runtime
        .deserialize_cuda_engine(images.vocos)
        .map_err(|e| F5Error::ModelExecution(format!("TensorRT-RTX Vocos deserialize: {e}")))?;
    let mut speed_engine = images
        .speed
        .map(|data| {
            runtime.deserialize_cuda_engine(data).map_err(|e| {
                F5Error::ModelExecution(format!("TensorRT-RTX Speed deserialize: {e}"))
            })
        })
        .transpose()?;
    let mut cfm = cfm_engine
        .create_execution_context()
        .map_err(|e| F5Error::ModelExecution(format!("TensorRT-RTX CFM context: {e}")))?;
    let mut vocos = vocos_engine
        .create_execution_context()
        .map_err(|e| F5Error::ModelExecution(format!("TensorRT-RTX Vocos context: {e}")))?;
    let mut speed = speed_engine
        .as_mut()
        .map(|engine| {
            engine
                .create_execution_context()
                .map_err(|e| F5Error::ModelExecution(format!("TensorRT-RTX Speed context: {e}")))
        })
        .transpose()?;
    let executor = TrtGraphExecutor {
        cfm: &mut cfm,
        vocos: &mut vocos,
        speed: speed.as_mut(),
        cfm_shape: shape.cfm,
        vocos_shape: shape.vocos,
        speed_shape: shape.speed,
    };
    let mut session = TrtF5Session::open(io, executor, shape)?;
    let result = serve(&mut session);
    drop(session);
    result
}
