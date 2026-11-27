use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Tensor};
use silero_package::V5PlainCheckpoint;
use std::collections::{BTreeMap, HashSet};
use crate::{acoustic::TextToMel, longform, store::TensorStore, synthesis::Synthesizer};

pub(crate) struct V5PlainTts {
    pub(crate) standalone: bool,
    symbols: HashSet<char>,
    symbol_ids: BTreeMap<String, usize>,
    alphabet: HashSet<char>,
    sos: i64,
    eos: i64,
    pub(crate) speakers: BTreeMap<String, usize>,
    transliteration: BTreeMap<String, BTreeMap<char, String>>,
    acoustic: TextToMel,
    synthesizer: Synthesizer,
}

impl V5PlainTts {
    pub(crate) fn load(checkpoint: V5PlainCheckpoint, device: &Device) -> Result<Self> {
        let weights = TensorStore::native(checkpoint.acoustic);
        let standalone = checkpoint.pqmf.is_none();
        let synthesizer = match checkpoint.pqmf {
            Some(filters) => Synthesizer::load(&weights, &TensorStore::native(filters), device)?,
            None => Synthesizer::load_fullband(&weights, device)?,
        };
        let frontend = checkpoint.frontend;
        Ok(Self {
            standalone,
            symbols: frontend.input_symbols.chars().skip(3).collect(),
            sos: frontend.symbol_ids[&frontend.sos] as i64,
            eos: frontend.symbol_ids[&frontend.eos] as i64,
            symbol_ids: frontend.symbol_ids,
            alphabet: frontend.alphabet,
            speakers: frontend.speakers,
            transliteration: frontend.transliteration,
            acoustic: TextToMel::load_variant(&weights, device, !standalone)?,
            synthesizer,
        })
    }

    pub(crate) fn sample_rates(&self) -> &[u32] { if self.standalone { &[48000] } else { &[8000,24000,48000] } }

    fn encode(&self, text: &str, speaker: &str) -> Result<Vec<i64>> {
        ensure!(!text.contains('^'), "V5 plain text does not yet support caret controls");
        let mut lines = String::with_capacity(text.len());
        for ch in text.trim().chars() {
            if ch == '\n' {
                if !lines.chars().last().is_some_and(|prev| ",;:.!?".contains(prev)) { lines.push('.'); }
                lines.push(' ');
            } else { lines.push(ch); }
        }
        let text = lines.to_lowercase().replace('—', "–").replace('‑', "-");
        let lang = speaker.split('_').next().unwrap_or(speaker);
        let mut converted = text;
        if let Some(mapping) = self.transliteration.get(lang) {
            if converted.chars().any(|ch| ch.is_alphabetic() && !self.alphabet.contains(&ch)) {
                converted = converted.chars().map(|ch| mapping.get(&ch).cloned().unwrap_or_else(|| ch.to_string())).collect();
            }
        }
        let filtered: String = converted.chars().filter(|ch| self.symbols.contains(ch)).collect();
        let text = filtered.split_whitespace().collect::<Vec<_>>().join(" ");
        ensure!(text.chars().any(|ch| ch.is_alphabetic() && self.alphabet.contains(&ch)), "text empty after V5 normalization");
        let mut result = Vec::with_capacity(text.chars().count()+2);
        result.push(self.sos);
        for ch in text.chars() { result.push(*self.symbol_ids.get(&ch.to_string()).with_context(|| format!("unknown V5 symbol {ch:?}"))? as i64); }
        result.push(self.eos);
        Ok(result)
    }

    pub(crate) fn warmup_input(&self) -> Result<(String, String)> {
        let speaker = self.speakers.keys().find(|s| s.as_str() != "random").context("no V5 speaker")?.clone();
        let letter = self.alphabet.iter().filter(|ch| ch.is_alphabetic() && self.symbols.contains(ch)).min_by_key(|&&ch| (!('а'..='я').contains(&ch), ch)).context("no V5 warmup symbol")?;
        Ok((letter.to_string().repeat(3), speaker))
    }

    pub(crate) fn infer_text(&self, text: &str, speaker: &str, rate: u32) -> Result<Vec<f32>> {
        ensure!(speaker != "random", "random voice generation is not implemented");
        let id = *self.speakers.get(speaker).context("unknown V5 speaker")? as i64;
        ensure!(self.sample_rates().contains(&rate), "unsupported V5 sample rate");
        longform::infer_text_chunks(text, rate, |fragment| {
            let tokens = self.encode(fragment, speaker)?;
            self.infer_token_ids(&tokens, id, rate)?.into_data().try_to_vec::<f32>().map_err(Into::into)
        })
    }

    pub(crate) fn infer_token_ids(&self, tokens: &[i64], speaker: i64, rate: u32) -> Result<Tensor<2>> {
        ensure!(self.speakers.values().any(|&id| id as i64 == speaker), "unknown V5 speaker ID");
        ensure!(tokens.iter().all(|id| self.symbol_ids.values().any(|&v| v as i64 == *id)), "unknown V5 token ID");
        let (mel, _) = if self.standalone { self.acoustic.infer_standalone_v5(tokens,speaker)? } else { self.acoustic.infer(tokens, speaker)? };
        self.synthesizer.forward(mel, rate)
    }
}
