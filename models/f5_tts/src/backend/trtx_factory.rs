//! Direct TensorRT-RTX image compiler for arbitrary F5 reference/target frame shapes.
//! The image cache is persistent. Per-request execution uses a scope-bound engine owner.

use std::{path::PathBuf, sync::mpsc, thread};

use crate::{
    F5Error, Result, cfm::{CfmSampleMode, OdeMethod, SamplingConfig}, checkpoint::Checkpoint, mel::VocosMel,
    reference::ReferenceAudio, speech_edit::SpeechEditPlan,
};

use super::{
    trtx::{BackboneShape, TrtBackboneWeights, TrtTimeWeights},
    trtx_cfm::build_serialized_cfm_step_with_attention,
    trtx_cuda::CudaRuntimeIo,
    trtx_input::{TrtInputShape, TrtInputWeights},
    trtx_session::{
        SessionShape, SynthesisInputs, TrtEngineImages, TrtLibraryPaths, with_warm_session,
    },
    trtx_speed::{TrtSpeedShape, TrtSpeedWeights, build_serialized_speed},
    trtx_text::{TrtChunkKernel, TrtKernelFactory},
    trtx_vocos::{TrtVocosShape, TrtVocosWeights, build_serialized_vocos},
};

pub struct TrtModelFiles {
    pub dit_checkpoint: PathBuf,
    pub vocos_checkpoint: PathBuf,
    pub speed_checkpoint: Option<PathBuf>,
    pub cuda_runtime: PathBuf,
    pub tensorrt_rtx: PathBuf,
    pub vocabulary_size: usize,
    pub workspace_bytes: usize,
}

impl TrtModelFiles {
    fn validate(&self) -> Result<()> {
        if self.vocabulary_size == 0 || self.vocabulary_size > 100_000 || self.workspace_bytes == 0
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid TensorRT F5 model factory configuration",
            ));
        }
        for path in [
            Some(&self.dit_checkpoint),
            Some(&self.vocos_checkpoint),
            self.speed_checkpoint.as_ref(),
            Some(&self.cuda_runtime),
            Some(&self.tensorrt_rtx),
        ]
        .into_iter()
        .flatten()
        {
            if path.as_os_str().is_empty() {
                return Err(F5Error::InvalidConfiguration(
                    "Missing explicit F5 model or runtime path",
                ));
            }
        }
        Ok(())
    }
}

pub use super::trtx::TrtAttentionPrecision;

pub struct TrtDirectFactory {
    files: TrtModelFiles,
    attention_precision: TrtAttentionPrecision,
}

impl TrtDirectFactory {
    pub fn new(files: TrtModelFiles) -> Result<Self> {
        files.validate()?;
        Ok(Self {
            files,
            attention_precision: TrtAttentionPrecision::default(),
        })
    }

    /// Select the DiT attention precision for engines built by this factory.
    pub fn with_attention_precision(mut self, precision: TrtAttentionPrecision) -> Self {
        self.attention_precision = precision;
        self
    }

    fn shape(&self, reference_frames: usize, target_frames: usize) -> Result<SessionShape> {
        if target_frames <= reference_frames || target_frames > 65_536 {
            return Err(F5Error::InvalidShape(
                "Invalid F5 reference or target frames",
            ));
        }
        let backbone = BackboneShape {
            frames: target_frames,
            ..BackboneShape::V1
        };
        let cfm = TrtInputShape {
            backbone,
            ..TrtInputShape::v1(self.files.vocabulary_size)
        };
        let vocos = TrtVocosShape {
            frames: target_frames - reference_frames,
            ..TrtVocosShape::V1
        };
        let speed = self
            .files
            .speed_checkpoint
            .as_ref()
            .filter(|_| reference_frames > 0)
            .map(|_| TrtSpeedShape {
                frames: reference_frames,
                ..TrtSpeedShape::CROSS_LINGUAL
            });
        let shape = SessionShape {
            cfm,
            vocos,
            speed,
            reference_frames,
        };
        shape.validate()?;
        Ok(shape)
    }

    fn compile(&self, shape: SessionShape) -> Result<TrtCompiledImages> {
        CudaRuntimeIo::load_tensorrt_rtx(&self.files.tensorrt_rtx)?;
        let logger = trtx::Logger::stderr()
            .map_err(|e| F5Error::ModelExecution(format!("TensorRT-RTX logger: {e}")))?;
        let dit = Checkpoint::open(&self.files.dit_checkpoint)?;
        let vocos = Checkpoint::open(&self.files.vocos_checkpoint)?;
        let input = TrtInputWeights::from_checkpoint(&dit, shape.cfm)?;
        let time = TrtTimeWeights::from_checkpoint(
            &dit,
            shape.cfm.backbone.batch,
            shape.cfm.backbone.dim,
        )?;
        let backbone = TrtBackboneWeights::from_checkpoint(&dit, shape.cfm.backbone)?;
        let vocos_weights = TrtVocosWeights::from_checkpoint(&vocos, shape.vocos)?;
        let cfm = build_serialized_cfm_step_with_attention(
            &logger,
            &input,
            &time,
            &backbone,
            self.files.workspace_bytes,
            self.attention_precision,
        )?;
        let vocos = build_serialized_vocos(&logger, &vocos_weights, self.files.workspace_bytes)?;
        let speed = self
            .files
            .speed_checkpoint
            .as_ref()
            .map(|path| {
                let checkpoint = Checkpoint::open(path)?;
                let speed_shape = shape.speed.ok_or(F5Error::InvalidConfiguration(
                    "Missing Cross-Lingual speed shape",
                ))?;
                let weights = TrtSpeedWeights::from_checkpoint(&checkpoint, speed_shape)?;
                build_serialized_speed(&logger, &weights, self.files.workspace_bytes)
            })
            .transpose()?;
        Ok(TrtCompiledImages {
            shape,
            cfm,
            vocos,
            speed,
        })
    }
}

impl TrtKernelFactory for TrtDirectFactory {
    fn validate_model(&mut self) -> Result<()> {
        let dit = Checkpoint::open(&self.files.dit_checkpoint)?;
        let vocos = Checkpoint::open(&self.files.vocos_checkpoint)?;
        let shape = self.shape(1, 256)?;
        TrtInputWeights::from_checkpoint(&dit, shape.cfm)?;
        TrtTimeWeights::from_checkpoint(&dit, shape.cfm.backbone.batch, shape.cfm.backbone.dim)?;
        TrtBackboneWeights::from_checkpoint(&dit, shape.cfm.backbone)?;
        TrtVocosWeights::from_checkpoint(&vocos, shape.vocos)?;
        if let Some(path) = &self.files.speed_checkpoint {
            let speed = Checkpoint::open(path)?;
            let speed_shape = shape.speed.ok_or(F5Error::InvalidConfiguration(
                "Missing Cross-Lingual speed shape",
            ))?;
            TrtSpeedWeights::from_checkpoint(&speed, speed_shape)?;
        }
        Ok(())
    }

    fn build(
        &mut self,
        reference_frames: usize,
        target_frames: usize,
    ) -> Result<Box<dyn TrtChunkKernel>> {
        let shape = self.shape(reference_frames, target_frames)?;
        let images = self.compile(shape)?;
        Ok(Box::new(TrtCompiledKernel::open(
            images,
            self.files.cuda_runtime.clone(),
            self.files.tensorrt_rtx.clone(),
        )?))
    }
}

struct TrtCompiledImages {
    shape: SessionShape,
    cfm: Vec<u8>,
    vocos: Vec<u8>,
    speed: Option<Vec<u8>>,
}

enum Request {
    Synthesize {
        condition: Vec<f32>,
        noise: Vec<f32>,
        tokens: Vec<i32>,
        restoration_gain: f32,
        sampling: SamplingConfig,
        method: OdeMethod,
        mode: CfmSampleMode,
        reply: mpsc::SyncSender<Result<Vec<f32>>>,
    },
    Edit {
        plan: SpeechEditPlan,
        noise: Vec<f32>,
        tokens: Vec<i32>,
        restoration_gain: f32,
        sampling: SamplingConfig,
        method: OdeMethod,
        reply: mpsc::SyncSender<Result<Vec<f32>>>,
    },
    Speed {
        mel: Vec<f32>,
        reply: mpsc::SyncSender<Result<f32>>,
    },
    Shutdown,
}

/// Each cached shape owns one thread with scope-bound TensorRT engine/context lifetimes.
/// A model replacement or LRU eviction sends Shutdown and joins that thread before release.
struct TrtCompiledKernel {
    shape: SessionShape,
    sender: mpsc::Sender<Request>,
    worker: Option<thread::JoinHandle<()>>,
}

impl TrtCompiledKernel {
    fn open(
        images: TrtCompiledImages,
        cuda_runtime: PathBuf,
        tensorrt_rtx: PathBuf,
    ) -> Result<Self> {
        let shape = images.shape;
        let (sender, receiver) = mpsc::channel();
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name(format!(
                "f5-trtx-{}/{}",
                shape.reference_frames, shape.cfm.backbone.frames
            ))
            .spawn(move || {
                let started = (|| -> Result<()> {
                    let logger = trtx::Logger::stderr().map_err(|error| {
                        F5Error::ModelExecution(format!("TensorRT-RTX logger: {error}"))
                    })?;
                    with_warm_session(
                        &logger,
                        TrtLibraryPaths {
                            cuda_runtime: &cuda_runtime,
                            tensorrt_rtx: &tensorrt_rtx,
                        },
                        TrtEngineImages {
                            cfm: &images.cfm,
                            vocos: &images.vocos,
                            speed: images.speed.as_deref(),
                        },
                        shape,
                        |session| {
                            let _ = ready_sender.send(Ok(()));
                            while let Ok(message) = receiver.recv() {
                                match message {
                                    Request::Synthesize {
                                        condition,
                                        noise,
                                        tokens,
                                        restoration_gain,
                                        sampling,
                                        method,
                                        mode,
                                        reply,
                                    } => {
                                        let _ = reply.send(session.synthesize_with_sample_mode(SynthesisInputs {
                                            condition: &condition,
                                            initial_noise: &noise,
                                            tokens: &tokens,
                                            restoration_gain,
                                            sampling,
                                        }, method, mode));
                                    }
                                    Request::Edit {
                                        plan,
                                        noise,
                                        tokens,
                                        restoration_gain,
                                        sampling,
                                        method,
                                        reply,
                                    } => {
                                        let _ = reply.send(session.edit_speech_with_method(
                                            &plan,
                                            &tokens,
                                            &noise,
                                            restoration_gain,
                                            sampling,
                                            method,
                                        ));
                                    }
                                    Request::Speed { mel, reply } => {
                                        let _ = reply.send(session.predict_speed(&mel));
                                    }
                                    Request::Shutdown => break,
                                }
                            }
                            Ok(())
                        },
                    )
                })();
                if let Err(error) = started {
                    let _ = ready_sender.send(Err(error));
                }
            })
            .map_err(|error| {
                F5Error::ModelExecution(format!("Cannot start F5 TensorRT engine thread: {error}"))
            })?;
        match ready_receiver.recv() {
            Ok(Ok(())) => Ok(Self {
                shape,
                sender,
                worker: Some(worker),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker.join();
                Err(F5Error::ModelExecution(
                    "F5 TensorRT engine thread stopped before ready".into(),
                ))
            }
        }
    }
}

impl Drop for TrtCompiledKernel {
    fn drop(&mut self) {
        let _ = self.sender.send(Request::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl TrtChunkKernel for TrtCompiledKernel {
    fn edit_speech(
        &mut self,
        plan: &SpeechEditPlan,
        tokens: &[i32],
        noise: &[f32],
        restoration_gain: f32,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        if self.shape.reference_frames != 0 || plan.total_frames != self.shape.cfm.backbone.frames {
            return Err(F5Error::InvalidShape("Wrong warmed TRT speech edit shape"));
        }
        let (reply, receive) = mpsc::sync_channel(1);
        self.sender
            .send(Request::Edit {
                plan: plan.clone(),
                noise: noise.to_vec(),
                tokens: tokens.to_vec(),
                restoration_gain,
                sampling,
                method: OdeMethod::Euler,
                reply,
            })
            .map_err(|_| F5Error::ModelExecution("TRT speech edit worker stopped".into()))?;
        receive
            .recv()
            .map_err(|_| F5Error::ModelExecution("TRT speech edit interrupted".into()))?
    }

    fn edit_speech_with_method(
        &mut self, plan: &SpeechEditPlan, tokens: &[i32], noise: &[f32],
        restoration_gain: f32, sampling: SamplingConfig, method: OdeMethod,
    ) -> Result<Vec<f32>> {
        if self.shape.reference_frames != 0 || plan.total_frames != self.shape.cfm.backbone.frames {
            return Err(F5Error::InvalidShape("Wrong warmed speech-edit shape"));
        }
        let (reply, receive) = mpsc::sync_channel(1);
        self.sender.send(Request::Edit {
            plan: plan.clone(), noise: noise.to_vec(), tokens: tokens.to_vec(),
            restoration_gain, sampling, method, reply,
        }).map_err(|_| F5Error::ModelExecution("TRT edit worker stopped".into()))?;
        receive.recv().map_err(|_| F5Error::ModelExecution("TRT edit interrupted".into()))?
    }

    fn synthesize_chunk(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let shape = self.shape;
        if target_frames != shape.cfm.backbone.frames {
            return Err(F5Error::InvalidShape("Incorrect cached F5 total duration"));
        }
        let condition = reference.prepare(&VocosMel::new(), target_frames, tokens.len())?;
        if condition.frames.reference != shape.reference_frames {
            return Err(F5Error::InvalidShape(
                "Incorrect cached F5 reference duration",
            ));
        }
        let (reply, result) = mpsc::sync_channel(1);
        let restoration_gain = condition.restoration_gain();
        self.sender
            .send(Request::Synthesize {
                condition: condition.conditioning_mel,
                noise: noise.to_vec(),
                tokens: tokens.to_vec(),
                restoration_gain,
                sampling,
                method: OdeMethod::Euler,
                mode: CfmSampleMode::default(),
                reply,
            })
            .map_err(|_| F5Error::ModelExecution("F5 TensorRT engine thread is stopped".into()))?;
        result
            .recv()
            .map_err(|_| F5Error::ModelExecution("F5 TensorRT synthesis interrupted".into()))?
    }

    fn synthesize_chunk_with_sample_mode(
        &mut self, reference: &ReferenceAudio, tokens: &[i32],
        target_frames: usize, noise: &[f32], sampling: SamplingConfig,
        method: OdeMethod, mode: CfmSampleMode,
    ) -> Result<Vec<f32>> {
        let shape = self.shape;
        if target_frames != shape.cfm.backbone.frames {
            return Err(F5Error::InvalidShape("Incorrect cached F5 duration"));
        }
        let condition = reference.prepare(&VocosMel::new(), target_frames, tokens.len())?;
        if condition.frames.reference != shape.reference_frames {
            return Err(F5Error::InvalidShape("Incorrect cached F5 reference duration"));
        }
        let (reply, result) = mpsc::sync_channel(1);
        let restoration_gain = condition.restoration_gain();
        self.sender.send(Request::Synthesize {
            condition: condition.conditioning_mel,
            noise: noise.to_vec(), tokens: tokens.to_vec(),
            restoration_gain,
            sampling, method, mode, reply,
        }).map_err(|_| F5Error::ModelExecution("F5 TensorRT engine thread stopped".into()))?;
        result.recv().map_err(|_| F5Error::ModelExecution("F5 TensorRT synthesis interrupted".into()))?
    }

    fn predict_reference_speed(&mut self, reference: &ReferenceAudio) -> Result<f32> {
        if self.shape.speed.is_none() {
            return Err(F5Error::InvalidConfiguration(
                "No Cross-Lingual speed model in this shape",
            ));
        }
        let mel = VocosMel::new().extract(&reference.samples)?;
        if mel.frames != self.shape.reference_frames {
            return Err(F5Error::InvalidShape(
                "Incorrect cached speed reference shape",
            ));
        }
        let (reply, result) = mpsc::sync_channel(1);
        self.sender
            .send(Request::Speed {
                mel: mel.as_flat_frames().to_vec(),
                reply,
            })
            .map_err(|_| {
                F5Error::ModelExecution("F5 TensorRT speed engine thread is stopped".into())
            })?;
        result.recv().map_err(|_| {
            F5Error::ModelExecution("F5 TensorRT speed prediction interrupted".into())
        })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arbitrary_frames_share_production_weight_schema_and_reject_invalid_shapes() {
        let factory = TrtDirectFactory::new(TrtModelFiles {
            dit_checkpoint: "f5.safetensors".into(),
            vocos_checkpoint: "vocos.safetensors".into(),
            speed_checkpoint: Some("speed.safetensors".into()),
            cuda_runtime: "cudart64_12.dll".into(),
            tensorrt_rtx: "tensorrt_rtx.dll".into(),
            vocabulary_size: 2545,
            workspace_bytes: 1 << 30,
        })
        .unwrap();
        let s = factory.shape(92, 230).unwrap();
        assert_eq!(s.cfm.backbone.depth, 22);
        assert_eq!(s.vocos.frames, 138);
        assert_eq!(s.speed.unwrap().frames, 92);
        assert_eq!(s.cfm.backbone.frames, 230);
        assert_eq!(s.cfm.vocab_size, 2545);
        assert!(factory.shape(100, 100).is_err());
        assert!(factory.shape(100, 65_537).is_err());
        assert!(factory.shape(8193, 8200).is_err());
    }
}
