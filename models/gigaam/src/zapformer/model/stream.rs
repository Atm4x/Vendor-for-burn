use super::{AsrHead, Decoder, Heads, ZapformerModel};
use crate::{audio::{SAMPLE_RATE, StreamingMel}, decode::{Word, ctc_argmax, words_at_frame_shift}, rnnt::RnntDecodeState, zapformer::{encoder::ZapformerEncoderStream, language::{LanguageRequest, LanguageTag}}};
use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use std::sync::Arc;

/// Text and words are publication deltas; token IDs and frames are newly decoded tokens.
/// Frame and word coordinates are relative to the start of this utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamTranscription {
    pub text: String,
    pub words: Option<Vec<Word>>,
    pub token_ids: Vec<usize>,
    pub frames: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StreamPacket {
    Single(StreamTranscription),
    Dual { raw: StreamTranscription, e2e: StreamTranscription },
}

enum DecodeState {
    Ctc { previous: Option<usize>, offset: usize },
    Rnnt(RnntDecodeState),
}

struct HeadState {
    head: Arc<AsrHead>,
    state: DecodeState,
    ids: Vec<usize>,
    frames: Vec<usize>,
    published: String,
    published_words: usize,
}

impl HeadState {
    fn new(head: &Arc<AsrHead>) -> Self {
        let state = match head.decoder {
            Decoder::Ctc(_) => DecodeState::Ctc { previous: None, offset: 0 },
            Decoder::Rnnt(_) => DecodeState::Rnnt(RnntDecodeState::default()),
        };
        Self { head: Arc::clone(head), state, ids: Vec::new(), frames: Vec::new(), published: String::new(), published_words: 0 }
    }

    fn consume(&mut self, encoded: Tensor<3>) -> Result<()> {
        let count = encoded.dims()[1];
        match (&self.head.decoder, &mut self.state) {
            (Decoder::Ctc(head), DecodeState::Ctc { previous, offset }) => {
                let classes = head.num_classes();
                let labels = ctc_argmax(&head.logits(encoded)?.into_data().try_to_vec::<f32>()?, count, classes)?;
                let end = offset.checked_add(count).context("CTC frame offset overflow")?;
                for (frame, id) in labels.into_iter().enumerate() {
                    if id != classes - 1 && Some(id) != *previous {
                        self.ids.push(id);
                        self.frames.push(*offset + frame);
                    }
                    *previous = Some(id);
                }
                *offset = end;
            }
            (Decoder::Rnnt(head), DecodeState::Rnnt(state)) => {
                let device = encoded.device();
                let (ids, frames, _) = head.decode_chunk(encoded, &device, state)?;
                self.ids.extend(ids);
                self.frames.extend(frames);
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn packet(&mut self, encoded: Option<Tensor<3>>, final_packet: bool, timestamps: bool, shift: f64, total_frames: usize, samples: usize) -> Result<StreamTranscription> {
        let start = self.ids.len();
        if let Some(encoded) = encoded { self.consume(encoded)?; }
        let mut count = self.ids.len();
        if timestamps && !final_packet {
            count = 0;
            for i in (0..self.ids.len()).rev() {
                let piece = self.head.tokenizer.id_to_str(self.ids[i])?;
                if piece == " " || piece.starts_with('▁') {
                    count = i + usize::from(piece == " ");
                    break;
                }
            }
        }
        let mut text = self.head.tokenizer.decode(&self.ids[..count])?;
        let words = if timestamps {
            let all = words_at_frame_shift(&self.head.tokenizer, &self.ids[..count], &self.frames[..count], shift, total_frames)?;
            ensure!(all.len() >= self.published_words, "stream published word prefix shrank");
            let mut words = all[self.published_words..].to_vec();
            self.published_words = all.len();
            if final_packet {
                for w in &mut words { w.end = w.end.min(samples as f64 / SAMPLE_RATE as f64); }
            }
            Some(words)
        } else {
            if !final_packet { text.truncate(text.trim_end_matches('\u{fffd}').len()); }
            None
        };
        let delta = text.chars().skip(self.published.chars().count()).collect();
        self.published = text;
        Ok(StreamTranscription { text: delta, words, token_ids: self.ids[start..].to_vec(), frames: self.frames[start..].to_vec() })
    }
}

enum HeadStates {
    Single(HeadState),
    Dual { raw: HeadState, e2e: HeadState },
}

/// One utterance owns encoder, mel and independent decoder states; weights remain shared.
pub struct ZapformerStream {
    device: Device,
    encoder: ZapformerEncoderStream,
    mel: StreamingMel,
    language: Option<LanguageRequest>,
    heads: HeadStates,
    timestamps: bool,
    shift: f64,
    finished: bool,
    failed: bool,
}

impl ZapformerModel {
    pub fn start_stream(&self, future_context_ms: Option<usize>, tag: Option<&LanguageTag>, word_timestamps: bool) -> Result<ZapformerStream> {
        let contexts = self.config.streaming.as_ref().and_then(|v|v["contexts"].as_array()).context("checkpoint declares no streaming contexts")?;
        let mut modes = Vec::with_capacity(contexts.len());
        for mode in contexts {
            let ms = usize::try_from(mode["future_context_ms"].as_u64().context("invalid future context")?)?;
            let left = usize::try_from(mode["left_context"].as_u64().context("invalid left context")?)?;
            ensure!(!modes.iter().any(|(m,_)| *m==ms), "duplicate streaming context");
            modes.push((ms,left));
        }
        let ms = future_context_ms.or_else(||modes.iter().map(|(m,_)|*m).max()).context("empty streaming contexts")?;
        let (_,left) = modes.into_iter().find(|(m,_)|*m==ms).context("unsupported future context")?;
        let frame_samples = self.config.mel.hop_length.checked_mul(4).context("frame shift overflow")?;
        let numerator = ms.checked_mul(SAMPLE_RATE as usize).context("context overflow")?;
        let denominator = frame_samples.checked_mul(1000).context("frame shift overflow")?;
        ensure!(numerator % denominator == 0, "stream context is not frame aligned");
        let chunk = numerator / denominator;
        self.validate_language(tag)?;
        let language = self.language.as_ref().map(|l|l.request(tag.map(std::slice::from_ref),1)).transpose()?;
        let heads = match &self.heads {
            Heads::EncoderOnly => bail!("encoder-only checkpoint does not support streaming ASR"),
            Heads::Single(head) => HeadStates::Single(HeadState::new(head)),
            Heads::Dual { raw,e2e } => HeadStates::Dual { raw: HeadState::new(raw), e2e: HeadState::new(e2e) },
        };
        Ok(ZapformerStream { device: self.device.clone(), encoder: self.encoder.start_stream(chunk,left)?, mel: StreamingMel::new(self.config.mel,self.config.encoder.in_channels)?, language, heads, timestamps: word_timestamps, shift: frame_samples as f64 / SAMPLE_RATE as f64, finished: false, failed: false })
    }
}

impl ZapformerStream {
    pub fn finished(&self) -> bool { self.finished }
    pub fn encoded_frames(&self) -> usize { self.encoder.position() }
    pub fn flush(&mut self) -> Result<StreamPacket> { self.push(&[],true) }
    pub fn push(&mut self, pcm: &[f32], final_packet: bool) -> Result<StreamPacket> {
        ensure!(!self.failed, "failed stream must be discarded");
        if self.finished {
            let empty = || StreamTranscription { text: String::new(), words: self.timestamps.then(Vec::new), token_ids: Vec::new(), frames: Vec::new() };
            return Ok(match self.heads { HeadStates::Single(_) => StreamPacket::Single(empty()), HeadStates::Dual { .. } => StreamPacket::Dual { raw: empty(), e2e: empty() } });
        }
        let result = self.push_inner(pcm,final_packet);
        self.failed = result.is_err();
        result
    }

    fn push_inner(&mut self, pcm: &[f32], final_packet: bool) -> Result<StreamPacket> {
        let features = self.mel.push(pcm)?.map(|f| Tensor::<3>::from_data(TensorData::new(f.values,[1,f.n_mels,f.frames]),&self.device).swap_dims(1,2));
        let encoded = self.encoder.push_conditioned(features, final_packet, self.language.as_ref())?.map(|x|match &self.language { Some(l) => l.fuse(x), None => x });
        let frames = self.encoder.position();
        let result = match &mut self.heads {
            HeadStates::Single(head) => StreamPacket::Single(head.packet(encoded,final_packet,self.timestamps,self.shift,frames,self.mel.samples())?),
            HeadStates::Dual { raw,e2e } => StreamPacket::Dual {
                raw: raw.packet(encoded.clone(),final_packet,self.timestamps,self.shift,frames,self.mel.samples())?,
                e2e: e2e.packet(encoded,final_packet,self.timestamps,self.shift,frames,self.mel.samples())?,
            },
        };
        self.finished = final_packet;
        Ok(result)
    }
}
