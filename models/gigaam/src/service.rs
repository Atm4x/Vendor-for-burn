use crate::{AsrOptions, GigaAm, ModelKind, audio, decode::Word};
use anyhow::{Context, Result, ensure};
use burn::tensor::Device;
use serde::Serialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Debug, Clone, Copy)]
pub enum Segmentation<'a> {
    FullRecording,
    Energy,
    SpeechRegions(&'a [audio::SpeechRegion]),
    GroupedSpeechRegions {
        regions: &'a [audio::SpeechRegion],
        grouping: audio::VadGrouping,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct TranscribeOptions<'a> {
    pub segmentation: Segmentation<'a>,
    pub word_timestamps: bool,
}

impl Default for TranscribeOptions<'_> {
    fn default() -> Self {
        Self {
            segmentation: Segmentation::FullRecording,
            word_timestamps: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WordResult {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

impl From<Word> for WordResult {
    fn from(value: Word) -> Self {
        Self {
            text: value.text,
            start: value.start,
            end: value.end,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TranscriptSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub words: Option<Vec<WordResult>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TranscriptResult {
    pub sample_rate: u32,
    pub duration_seconds: f64,
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ShortTranscriptionResult {
    pub text: String,
    pub words: Option<Vec<WordResult>>,
}

impl TranscriptResult {
    pub fn words(&self) -> Vec<&WordResult> {
        self.segments
            .iter()
            .flat_map(|segment| segment.words.iter().flatten())
            .collect()
    }

    pub fn has_word_timestamps(&self) -> bool {
        self.segments
            .first()
            .is_some_and(|segment| segment.words.is_some())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbeddingResult {
    pub frames: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ServiceStatus {
    pub model_kind: ModelKind,
    pub sample_rate: u32,
}

/// Reuses one loaded model across repeated calls; the host owns synchronization.
pub struct GigaAmService {
    model: GigaAm,
    batch_size: usize,
}

impl GigaAmService {
    pub fn from_loaded(model: GigaAm) -> Self {
        Self {
            model,
            batch_size: 16,
        }
    }

    /// Set Python longform's fr_batch_size (default 16) to cap GPU allocations.
    pub fn with_batch_size(mut self, batch_size: usize) -> Result<Self> {
        ensure!(
            (1..=256).contains(&batch_size),
            "batch_size must be 1..=256"
        );
        self.batch_size = batch_size;
        Ok(self)
    }

    pub fn load(path: &Path, device: Device) -> Result<Self> {
        Ok(Self {
            model: GigaAm::load(path, device)
                .with_context(|| format!("loading GigaAM service model {}", path.display()))?,
            batch_size: 16,
        })
    }

    /// Resolve Python model aliases from an explicit cache; optional HTTPS downloads
    /// are controlled by the caller. Local fine-tuned checkpoints remain supported.
    pub fn load_named(
        name_or_path: &str,
        device: Device,
        options: &crate::model_registry::CacheOptions,
    ) -> Result<Self> {
        let path = crate::model_registry::resolve_model_path(name_or_path, options)?;
        Ok(Self::from_loaded(GigaAm::load_with_cache(&path, device, options)?))
    }

    /// Start a native utterance stream. The caller decides utterance boundaries (external VAD).
    pub fn start_stream(&self, future_context_ms: Option<usize>, language: Option<&crate::zapformer::language::LanguageTag>, word_timestamps: bool) -> Result<crate::zapformer::ZapformerStream> {
        self.model.start_stream(future_context_ms, language, word_timestamps)
    }

    pub fn is_zapformer(&self) -> bool {
        self.model.is_zapformer()
    }

    pub fn status(&self) -> ServiceStatus {
        ServiceStatus {
            model_kind: self.model.model_kind(),
            sample_rate: audio::SAMPLE_RATE,
        }
    }

    pub fn transcribe_file(
        &self,
        path: &Path,
        options: TranscribeOptions<'_>,
    ) -> Result<TranscriptResult> {
        let pcm = audio::load_audio_16k(path)
            .with_context(|| format!("loading audio {}", path.display()))?;
        self.transcribe_16k(&pcm, options)
    }

    pub fn transcribe_16k(
        &self,
        pcm: &[f32],
        options: TranscribeOptions<'_>,
    ) -> Result<TranscriptResult> {
        self.transcribe_16k_with_asr(pcm, options, AsrOptions::default())
    }

    pub fn transcribe_16k_with_asr(&self, pcm: &[f32], options: TranscribeOptions<'_>, asr: AsrOptions<'_>) -> Result<TranscriptResult> {
        self.model.validate_asr_options(asr)?;
        ensure!(
            pcm.iter().all(|x| x.is_finite()),
            "PCM contains nonfinite samples"
        );

        let chunks = match options.segmentation {
            Segmentation::FullRecording => {
                ensure!(
                    !pcm.is_empty() && pcm.len() <= 25 * audio::SAMPLE_RATE as usize,
                    "Python GigaAM.transcribe requires 0 < audio duration <= 25s; provide external VAD segments for longform"
                );
                vec![audio::AudioChunk {
                    start_sample: 0,
                    samples: pcm,
                }]
            }
            Segmentation::Energy => {
                audio::speech_chunks_with_offsets(pcm, 24 * audio::SAMPLE_RATE as usize)
            }
            Segmentation::SpeechRegions(regions) => {
                audio::speech_chunks_in_regions(pcm, regions, 24 * audio::SAMPLE_RATE as usize)?
            }
            Segmentation::GroupedSpeechRegions { regions, grouping } => {
                audio::speech_chunks_grouped_vad(pcm, regions, grouping)?
            }
        };

        if chunks.len() > 1 {
            let mut segments = Vec::with_capacity(chunks.len());
            for group in chunks.chunks(self.batch_size) {
                let recordings: Vec<&[f32]> = group.iter().map(|chunk| chunk.samples).collect();
                if options.word_timestamps {
                    let results = self
                        .model
                        .transcribe_segments_padded_with_timestamps_16k(&recordings, asr)?;
                    for (chunk, result) in group.iter().zip(results) {
                        let offset = chunk.start_sample as f64 / audio::SAMPLE_RATE as f64;
                        segments.push(TranscriptSegment {
                            text: result.transcription.text,
                            start: offset,
                            end: (chunk.start_sample + chunk.samples.len()) as f64
                                / audio::SAMPLE_RATE as f64,
                            words: Some(
                                result
                                    .words
                                    .into_iter()
                                    .map(|word| WordResult {
                                        text: word.text,
                                        start: ((word.start + offset) * 1000.0).round_ties_even()
                                            / 1000.0,
                                        end: ((word.end + offset) * 1000.0).round_ties_even()
                                            / 1000.0,
                                    })
                                    .collect(),
                            ),
                        });
                    }
                } else {
                    let results = self.model.transcribe_segments_padded_16k(&recordings, asr)?;
                    for (chunk, result) in group.iter().zip(results) {
                        segments.push(TranscriptSegment {
                            text: result.text,
                            start: chunk.start_sample as f64 / audio::SAMPLE_RATE as f64,
                            end: (chunk.start_sample + chunk.samples.len()) as f64
                                / audio::SAMPLE_RATE as f64,
                            words: None,
                        });
                    }
                }
            }
            return Ok(TranscriptResult {
                sample_rate: audio::SAMPLE_RATE,
                duration_seconds: pcm.len() as f64 / audio::SAMPLE_RATE as f64,
                text: segments
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                segments,
            });
        }

        let mut segments = Vec::with_capacity(chunks.len());
        for (index, chunk) in chunks.into_iter().enumerate() {
            let offset = chunk.start_sample as f64 / audio::SAMPLE_RATE as f64;
            let end = (chunk.start_sample + chunk.samples.len()) as f64 / audio::SAMPLE_RATE as f64;
            let segment = if options.word_timestamps {
                let result = self
                    .model
                    .transcribe_segment_with_timestamps_16k(chunk.samples, asr)
                    .with_context(|| format!("transcribing segment {index}"))?;
                let words = result
                    .words
                    .into_iter()
                    .map(|word| WordResult {
                        text: word.text,
                        start: ((word.start + offset) * 1000.0).round_ties_even() / 1000.0,
                        end: ((word.end + offset) * 1000.0).round_ties_even() / 1000.0,
                    })
                    .collect();
                TranscriptSegment {
                    text: result.transcription.text,
                    start: offset,
                    end,
                    words: Some(words),
                }
            } else {
                let result = self
                    .model
                    .transcribe_segment_16k(chunk.samples, asr)
                    .with_context(|| format!("transcribing segment {index}"))?;
                TranscriptSegment {
                    text: result.text,
                    start: offset,
                    end,
                    words: None,
                }
            };
            segments.push(segment);
        }
        let text = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(TranscriptResult {
            sample_rate: audio::SAMPLE_RATE,
            duration_seconds: pcm.len() as f64 / audio::SAMPLE_RATE as f64,
            text,
            segments,
        })
    }

    /// Equivalent to Python GigaAMASR.transcribe: one clip, at most 25 seconds.
    pub fn transcribe_short_16k(
        &self,
        pcm: &[f32],
        word_timestamps: bool,
    ) -> Result<ShortTranscriptionResult> {
        let result = self.transcribe_16k(
            pcm,
            TranscribeOptions {
                segmentation: Segmentation::FullRecording,
                word_timestamps,
            },
        )?;
        let segment = result
            .segments
            .into_iter()
            .next()
            .context("missing short transcription")?;
        Ok(ShortTranscriptionResult {
            text: segment.text,
            words: segment.words,
        })
    }

    pub fn transcribe_short_file(
        &self,
        path: &Path,
        word_timestamps: bool,
    ) -> Result<ShortTranscriptionResult> {
        let pcm = audio::load_audio_16k(path)
            .with_context(|| format!("loading short audio {}", path.display()))?;
        self.transcribe_short_16k(&pcm, word_timestamps)
    }

    /// Equivalent to Python transcribe_longform after segmentation by a caller-owned VAD.
    pub fn transcribe_longform_regions_16k(
        &self,
        pcm: &[f32],
        regions: &[audio::SpeechRegion],
        word_timestamps: bool,
        grouping: Option<audio::VadGrouping>,
    ) -> Result<TranscriptResult> {
        let segmentation = Segmentation::GroupedSpeechRegions {
            regions,
            grouping: grouping.unwrap_or_default(),
        };
        self.transcribe_16k(
            pcm,
            TranscribeOptions {
                segmentation,
                word_timestamps,
            },
        )
    }

    /// Runs Python-style short-form transcription for each recording, in input order.
    /// GPU encoder batches are capped at the configured batch_size.
    pub fn transcribe_batch_16k<T: AsRef<[f32]>>(
        &self,
        recordings: &[T],
        word_timestamps: bool,
    ) -> Result<Vec<TranscriptResult>> {
        if word_timestamps {
            let mut results = Vec::with_capacity(recordings.len());
            for group in recordings.chunks(self.batch_size) {
                results.extend(self.model.transcribe_batch_with_timestamps_16k(group)?);
            }
            return Ok(results
                .into_iter()
                .map(|result| {
                    let duration = result.end;
                    TranscriptResult {
                        sample_rate: audio::SAMPLE_RATE,
                        duration_seconds: duration,
                        text: result.transcription.text.clone(),
                        segments: vec![TranscriptSegment {
                            text: result.transcription.text,
                            start: 0.0,
                            end: duration,
                            words: Some(
                                result
                                    .words
                                    .into_iter()
                                    .map(|word| WordResult {
                                        text: word.text,
                                        start: (word.start * 1000.0).round_ties_even() / 1000.0,
                                        end: (word.end * 1000.0).round_ties_even() / 1000.0,
                                    })
                                    .collect(),
                            ),
                        }],
                    }
                })
                .collect());
        }
        let mut results = Vec::with_capacity(recordings.len());
        for group in recordings.chunks(self.batch_size) {
            results.extend(self.model.transcribe_batch_16k(group)?);
        }
        Ok(recordings
            .iter()
            .zip(results)
            .map(|(pcm, transcription)| {
                let duration = pcm.as_ref().len() as f64 / audio::SAMPLE_RATE as f64;
                TranscriptResult {
                    sample_rate: audio::SAMPLE_RATE,
                    duration_seconds: duration,
                    text: transcription.text.clone(),
                    segments: vec![TranscriptSegment {
                        text: transcription.text,
                        start: 0.0,
                        end: duration,
                        words: None,
                    }],
                }
            })
            .collect())
    }

    pub fn transcribe_batch_files<P: AsRef<Path>>(
        &self,
        paths: &[P],
        word_timestamps: bool,
    ) -> Result<Vec<TranscriptResult>> {
        let recordings = paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                audio::load_audio_16k(path.as_ref()).with_context(|| {
                    format!("loading batch file {index}: {}", path.as_ref().display())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.transcribe_batch_16k(&recordings, word_timestamps)
    }

    pub fn embed_file(&self, path: &Path) -> Result<EmbeddingResult> {
        let pcm = audio::load_audio_16k(path)?;
        self.embed_16k(&pcm)
    }

    pub fn embed_16k(&self, pcm: &[f32]) -> Result<EmbeddingResult> {
        ensure!(
            self.model.model_kind().is_ssl(),
            "embeddings require an SSL model"
        );
        let (data, frames, channels) = self.model.infer_embeddings_16k(pcm)?;
        Ok(EmbeddingResult {
            frames,
            channels,
            data,
        })
    }

    /// Match Python GigaAMEmo.get_probs: emotion name -> probability.
    pub fn emotions_dict_16k(&self, pcm: &[f32]) -> Result<BTreeMap<String, f32>> {
        Ok(self.emotions_16k(pcm)?.into_iter().collect())
    }

    pub fn emotions_dict_file(&self, path: &Path) -> Result<BTreeMap<String, f32>> {
        let pcm = audio::load_audio_16k(path)?;
        self.emotions_dict_16k(&pcm)
    }

    pub fn emotions_file(&self, path: &Path) -> Result<Vec<(String, f32)>> {
        let pcm = audio::load_audio_16k(path)?;
        self.emotions_16k(&pcm)
    }

    pub fn emotions_16k(&self, pcm: &[f32]) -> Result<Vec<(String, f32)>> {
        self.model.emotion_probs_16k(pcm)
    }
}
