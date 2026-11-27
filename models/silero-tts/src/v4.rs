use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Tensor};
use silero_package::V4Checkpoint;
use std::collections::{BTreeMap, HashSet};
use crate::{accentor::Accentor, runtime::TextOptions};
use crate::{acoustic::TextToMel, english_vocoder::EnglishVocoder, longform, store::TensorStore};

pub(crate) struct V4Tts {
    symbols: String,
    symbol_ids: BTreeMap<String, usize>,
    ascii_symbols: BTreeMap<char, String>,
    pub(crate) speakers: BTreeMap<String, usize>,
    accent: Option<(Accentor, String, HashSet<String>)>,
    acoustic: TextToMel,
    vocoder: EnglishVocoder,
}

impl V4Tts {
    pub(crate) fn load(checkpoint: V4Checkpoint, device: &Device) -> Result<Self> {
        let accent = checkpoint.accent.map(|a| -> Result<_> { Ok((Accentor::from_parts(&TensorStore::native(a.weights), a.ngrams, a.exceptions, device)?, a.vowels, a.stop_words)) }).transpose()?;
        let weights = TensorStore::native(checkpoint.weights);
        Ok(Self { symbols: checkpoint.symbols, symbol_ids: checkpoint.symbol_ids, ascii_symbols: checkpoint.ascii_symbols,
            speakers: checkpoint.speakers, accent, acoustic: TextToMel::load_variant(&weights, device, false)?,
            vocoder: EnglishVocoder::load_variant(&weights, true, device)? })
    }

    fn encode(&self, text: &str, options: &TextOptions) -> Result<Vec<i64>> {
        let mut sentence = silero_package::normalize_v3(&self.symbols, &self.ascii_symbols, text)?;
        if let Some((accent,vowels,stop)) = &self.accent {
            if options.put_accent { sentence = accent.accentuate_v4_ascii(&sentence, vowels, stop, options.put_yo)?; }
        }
        std::iter::once('|').chain(sentence.chars()).chain(std::iter::once('~')).map(|ch| {
            self.symbol_ids.get(&ch.to_string()).map(|&id| id as i64).with_context(|| format!("unknown V4 symbol {ch:?}"))
        }).collect()
    }

    pub(crate) fn warmup_input(&self) -> Result<(String, String)> {
        let speaker = self.speakers.keys().find(|s| s.as_str() != "random").context("no V4 speaker")?;
        let letter = self.symbols.chars().skip(3).filter(|ch| ch.is_alphabetic())
            .find(|ch| self.encode(&ch.to_string(), &TextOptions::default()).is_ok()).context("no V4 warmup symbol")?;
        Ok((letter.to_string().repeat(3), speaker.clone()))
    }

    pub(crate) fn infer_text(&self, text: &str, speaker: &str, rate: u32, options: &TextOptions) -> Result<Vec<f32>> {
        ensure!(speaker != "random", "random voice generation is not implemented");
        let id = *self.speakers.get(speaker).context("unknown V4 speaker")? as i64;
        ensure!(matches!(rate, 8000 | 24000 | 48000), "unsupported V4 sample rate");
        longform::infer_text_chunks(text, rate, |fragment| {
            let tokens = self.encode(fragment, options)?;
            self.infer_token_ids(&tokens, id, rate)?.into_data().try_to_vec::<f32>().map_err(Into::into)
        })
    }

    pub(crate) fn infer_token_ids(&self, tokens: &[i64], speaker: i64, rate: u32) -> Result<Tensor<2>> {
        ensure!(self.speakers.values().any(|&id| id as i64 == speaker), "unknown V4 speaker ID");
        ensure!(tokens.iter().all(|id| self.symbol_ids.values().any(|&v| v as i64 == *id)), "unknown V4 token ID");
        let (mel, _) = self.acoustic.infer_v3(tokens, speaker)?;
        self.vocoder.forward(mel, rate)
    }
}
