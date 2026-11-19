use std::collections::BTreeMap;
use std::sync::Arc;

use fish_speech_core::audio::FIREFLY_GROUPS;
use fish_speech_core::error::{FishError, Result as FishResult};
use fish_speech_core::firefly_cpu::AudioTensor;
use fish_speech_core::firefly_ir::FireflyGraph;
use fish_speech_core::firefly_weighted::WeightedFireflyExecutor;
use fish_speech_core::synthesis::AcousticDecoder;

use crate::reference_cache::{CachePolicy, ReferenceBytes, ReferenceCache, ReferenceEntry};
use crate::reference_loader::{
    ReferenceAudioLoader, ReferenceLoadError, Result as ReferenceResult,
};
use crate::weights::{LoadedWeights, Result as WeightResult};

pub struct CpuFireflyDecoder {
    weights: Arc<BTreeMap<String, Vec<f32>>>,
    cached: Option<(usize, WeightedFireflyExecutor)>,
}

impl CpuFireflyDecoder {
    pub fn from_checkpoint(checkpoint: &LoadedWeights) -> WeightResult<Self> {
        let graph = FireflyGraph::waveform_decode(1, 1, 1)
            .map_err(|e| crate::weights::WeightError::Invalid(format!("Firefly decoder: {e}")))?;
        let weights = checkpoint.firefly_cpu_weights(&graph)?;
        Ok(Self {
            weights: Arc::new(weights),
            cached: None,
        })
    }

    pub fn decode_indices(&mut self, indices: &[i32], frames: usize) -> FishResult<Vec<f32>> {
        if frames == 0 || indices.len() != FIREFLY_GROUPS * frames {
            return Err(FishError::InvalidShape(
                "Firefly acoustic codes must be [8, frames]",
            ));
        }
        if self
            .cached
            .as_ref()
            .is_none_or(|(known, _)| *known != frames)
        {
            let graph = FireflyGraph::waveform_decode(1, frames, frames)?;
            let executor =
                WeightedFireflyExecutor::with_shared_weights(graph, Arc::clone(&self.weights))?;
            self.cached = Some((frames, executor));
        }
        self.cached.as_ref().unwrap().1.decode(indices)
    }
}

impl AcousticDecoder for CpuFireflyDecoder {
    fn decode(&mut self, acoustic_codes: &[Vec<u32>]) -> FishResult<Vec<f32>> {
        if acoustic_codes.len() != FIREFLY_GROUPS {
            return Err(FishError::InvalidShape(
                "Fish Firefly requires 8 acoustic codebooks",
            ));
        }
        let frames = acoustic_codes[0].len();
        if frames == 0 || acoustic_codes.iter().any(|codes| codes.len() != frames) {
            return Err(FishError::InvalidShape(
                "inconsistent acoustic codebook frames",
            ));
        }
        let mut indices = Vec::with_capacity(FIREFLY_GROUPS * frames);
        for codes in acoustic_codes {
            for &index in codes {
                if index >= 1000 {
                    return Err(FishError::InvalidShape("invalid Fish FSQ index"));
                }
                indices.push(index as i32);
            }
        }
        self.decode_indices(&indices, frames)
    }
}

pub struct CpuFireflyReferenceEncoder {
    weights: Arc<BTreeMap<String, Vec<f32>>>,
    audio_loader: ReferenceAudioLoader,
}

impl CpuFireflyReferenceEncoder {
    pub fn from_checkpoint(checkpoint: &LoadedWeights) -> WeightResult<Self> {
        let graph = FireflyGraph::reference_encode(1, 4096)
            .map_err(|e| crate::weights::WeightError::Invalid(format!("Firefly encoder: {e}")))?;
        let weights = checkpoint.firefly_cpu_weights(&graph)?;
        Ok(Self {
            weights: Arc::new(weights),
            audio_loader: ReferenceAudioLoader::default(),
        })
    }

    pub fn set_audio_loader(&mut self, loader: ReferenceAudioLoader) {
        self.audio_loader = loader;
    }

    pub fn encode_reference_bytes(&self, audio: &[u8]) -> ReferenceResult<Vec<Vec<u32>>> {
        let decoded = self.audio_loader.load_bytes(audio)?;
        let (codes, _) = self
            .encode_mono(&decoded.pcm)
            .map_err(ReferenceLoadError::Mel)?;
        Ok(codes)
    }

    pub fn encode_references_cached(
        &self,
        cache: &mut ReferenceCache<Vec<Vec<u32>>>,
        references: &[ReferenceBytes<'_>],
        policy: CachePolicy,
    ) -> ReferenceResult<Vec<ReferenceEntry<Vec<Vec<u32>>>>> {
        cache.load_by_hash(references, policy, |audio| {
            self.encode_reference_bytes(audio)
        })
    }

    pub fn encode_mono(&self, audio: &[f32]) -> FishResult<(Vec<Vec<u32>>, usize)> {
        let graph = FireflyGraph::reference_encode(1, audio.len())?;
        let executor =
            WeightedFireflyExecutor::with_shared_weights(graph, Arc::clone(&self.weights))?;
        let input = AudioTensor::new(1, 1, audio.len(), audio.to_vec())?;
        let (indices, valid_frames) = executor.encode(input)?;
        let mut channels = Vec::with_capacity(FIREFLY_GROUPS);
        for group in 0..FIREFLY_GROUPS {
            channels.push(
                indices[group * valid_frames..(group + 1) * valid_frames]
                    .iter()
                    .map(|&index| index as u32)
                    .collect(),
            );
        }
        Ok((channels, valid_frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{MODEL_FAMILY, TensorEntry, WeightManifest};
    use fish_speech_core::firefly_ir::{FireflyNode, FireflyOp, FireflyStage, FireflyTensor};
    use fish_speech_core::graph_ir::{Layout, ScalarType, TensorType};
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::process::{Command, Stdio};

    #[test]
    fn configured_codec_reaches_reference_encoder_without_an_ffmpeg_global_default() {
        if Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return;
        }
        let output = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=0.2",
                "-f",
                "mp3",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        let mut encoder = CpuFireflyReferenceEncoder {
            weights: Arc::new(BTreeMap::new()),
            audio_loader: ReferenceAudioLoader::default(),
        };
        assert!(matches!(
            encoder.encode_reference_bytes(&output.stdout),
            Err(ReferenceLoadError::Decode(_))
        ));
        encoder.set_audio_loader(ReferenceAudioLoader::default().with_ffmpeg("ffmpeg"));
        assert!(matches!(
            encoder.encode_reference_bytes(&output.stdout),
            Err(ReferenceLoadError::Mel(_))
        ));
    }

    fn digest(data: &[u8]) -> String {
        format!("{:x}", Sha256::digest(data))
    }

    #[test]
    fn firefly_checkpoint_bridge_reads_fp16_and_rejects_semantic_manifest() {
        let key = "some.weight".to_string();
        let data = [0x00u8, 0x3c, 0x00, 0xc0];
        let shape = vec![2];
        let shard = serialize(
            [(
                key.as_str(),
                TensorView::new(Dtype::F16, shape.clone(), &data).unwrap(),
            )],
            None,
        )
        .unwrap();
        let manifest = WeightManifest {
            schema_version: 1,
            model_family: MODEL_FAMILY.into(),
            component: "firefly".into(),
            source_checkpoint_sha256: digest(b"fixture"),
            source_license: "test-only".into(),
            config_sha256: digest(b"test-config"),
            shards: BTreeMap::from([("firefly.safetensors".into(), digest(&shard))]),
            tensors: BTreeMap::from([(
                key.clone(),
                TensorEntry {
                    shard: "firefly.safetensors".into(),
                    dtype: "F16".into(),
                    shape: shape.clone(),
                    sha256: digest(&data),
                },
            )]),
        };
        let path =
            std::env::temp_dir().join(format!("fish-firefly-weight-bridge-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("firefly.safetensors"), shard).unwrap();
        let loaded = LoadedWeights::load(&path, manifest.clone()).unwrap();
        let graph = FireflyGraph {
            stage: FireflyStage::WaveformDecode,
            nodes: vec![
                FireflyNode {
                    name: "codes".into(),
                    operation: FireflyOp::CodeInput,
                    inputs: vec![],
                    output: FireflyTensor {
                        batch: 1,
                        channels: 8,
                        frames: 1,
                        dtype: ScalarType::I32,
                    },
                    checkpoint_keys: vec![],
                },
                FireflyNode {
                    name: "tanh".into(),
                    operation: FireflyOp::Tanh,
                    inputs: vec![0],
                    output: FireflyTensor {
                        batch: 1,
                        channels: 8,
                        frames: 1,
                        dtype: ScalarType::I32,
                    },
                    checkpoint_keys: vec![key.clone()],
                },
            ],
            weights: BTreeMap::from([(
                key,
                TensorType::new(vec![2], ScalarType::F32, Layout::Plain),
            )]),
            input: 0,
            output: 1,
            valid_frames: 1,
        };
        assert_eq!(
            loaded.firefly_cpu_weights(&graph).unwrap()["some.weight"],
            vec![1.0, -2.0]
        );
        let mut wrong_manifest = manifest;
        wrong_manifest.component = "semantic".into();
        let wrong = LoadedWeights::load(&path, wrong_manifest).unwrap();
        assert!(wrong.firefly_cpu_weights(&graph).is_err());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn acoustic_decoder_checks_all_eight_codebooks_before_execution() {
        let mut decoder = CpuFireflyDecoder {
            weights: Arc::new(BTreeMap::new()),
            cached: None,
        };
        assert!(decoder.decode(&vec![vec![1]; 7]).is_err());
        assert!(
            decoder
                .decode(
                    &vec![vec![1]; 8]
                        .into_iter()
                        .enumerate()
                        .map(|(index, mut codes)| {
                            if index == 3 {
                                codes.clear();
                            }
                            codes
                        })
                        .collect::<Vec<_>>()
                )
                .is_err()
        );
        assert!(decoder.decode(&vec![vec![1000]; 8]).is_err());
        assert!(decoder.cached.is_none());
    }
}

#[cfg(test)]
mod weighted_synthesis_smoke {
    use super::*;
    use fish_speech_core::conversation::{EncodedConversation, TokenEncoder};
    use fish_speech_core::sampling::SamplingParams;
    use fish_speech_core::synthesis::{SemanticGenerator, SynthesisRequest, synthesize};
    use fish_speech_core::tokenizer::SpecialTokens;

    struct FixtureTokenizer;
    impl TokenEncoder for FixtureTokenizer {
        fn encode(&self, value: &str) -> FishResult<Vec<u32>> {
            Ok(value.bytes().map(u32::from).collect())
        }
    }

    struct FixtureGenerator;
    impl SemanticGenerator for FixtureGenerator {
        fn begin_sample(&mut self) -> FishResult<()> {
            Ok(())
        }
        fn generate(
            &mut self,
            _: &EncodedConversation,
            _: usize,
            _: SamplingParams,
        ) -> FishResult<Vec<Vec<u32>>> {
            let mut channels = vec![vec![100, 101]];
            channels.extend((0..8).map(|_| vec![0, 0]));
            Ok(channels)
        }
    }

    #[test]
    #[ignore = "full-size synthetic Firefly HiFiGAN; execute with --release --ignored"]
    fn synthesizer_calls_weighted_firefly_decoder_and_collects_pcm() {
        let graph = FireflyGraph::waveform_decode(1, 1, 1).unwrap();
        let weights = graph
            .weights
            .iter()
            .map(|(key, ty)| (key.clone(), vec![0.0; ty.shape.iter().product()]))
            .collect();
        let mut decoder = CpuFireflyDecoder {
            weights: Arc::new(weights),
            cached: None,
        };
        let request = SynthesisRequest {
            text: "Hello.",
            references: &[],
            num_samples: 1,
            num_codebooks: 8,
            iterative_prompt: false,
            chunk_length: 64,
            max_context_length: 4096,
            max_new_tokens: 4,
            sampling: SamplingParams::default(),
        };
        let samples = synthesize(
            &FixtureTokenizer,
            &SpecialTokens::new(128).unwrap(),
            &request,
            &mut FixtureGenerator,
            &mut decoder,
            || false,
        )
        .unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].sample_rate, 44_100);
        assert_eq!(samples[0].segments.len(), 1);
        assert_eq!(samples[0].pcm, vec![0.0; 2048]);
    }
}
