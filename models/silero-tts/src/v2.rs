use crate::{english_vocoder::V2Vocoder, longform, store::TensorStore, tacotron::Tacotron2};
use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Tensor};
use silero_package::{V2Checkpoint, encode_v2};
use std::collections::BTreeMap;

pub(crate) struct V2Tts {
    symbols: String,
    encoded_symbols: String,
    ascii_symbols: BTreeMap<char, String>,
    recurrent_accent: Option<crate::recurrent_accent::RecurrentAccent>,
    pub(crate) speakers: BTreeMap<String, usize>,
    pub(crate) rates: Vec<u32>,
    acoustic: Tacotron2,
    vocoder: V2Vocoder,
}
impl V2Tts {
    pub(crate) fn load(checkpoint: V2Checkpoint, device: &Device) -> Result<Self> {
        let recurrent_accent = checkpoint.recurrent_accent.map(crate::recurrent_accent::RecurrentAccent::load).transpose()?;
        let weights = TensorStore::native(checkpoint.weights);
        Ok(Self {
            symbols: checkpoint.symbols,
            encoded_symbols: checkpoint.encoded_symbols,
            ascii_symbols: checkpoint.ascii_symbols,
            recurrent_accent,
            speakers: checkpoint.speakers,
            rates: checkpoint.sample_rates.clone(),
            acoustic: Tacotron2::load(
                &weights,
                checkpoint.multi,
                checkpoint.max_steps,
                checkpoint.gate_threshold,
                device,
            )?,
            vocoder: V2Vocoder::load(&weights, &checkpoint.sample_rates, device)?,
        })
    }
    pub(crate) fn warmup_input(&self) -> Result<(String, String)> {
        let c = self
            .symbols
            .chars()
            .find(|c| c.is_alphabetic())
            .context("no V2 warmup symbol")?;
        Ok((
            c.to_string().repeat(3),
            self.speakers
                .keys()
                .next()
                .context("no V2 speaker")?
                .clone(),
        ))
    }
    pub(crate) fn infer_text(&self, text: &str, speaker: &str, rate: u32) -> Result<Vec<f32>> {
        let id = *self.speakers.get(speaker).context("unknown V2 speaker")? as i64;
        ensure!(self.rates.contains(&rate), "unsupported V2 sample rate");
        let mut output = Vec::new();
        for part in longform::split_text(text, 140)? {
            let ids = prepare_tokens(&self.symbols, &self.encoded_symbols, &self.ascii_symbols, self.recurrent_accent.as_ref(), &part)?;
            let pcm = self
                .infer_token_ids(&ids, id, rate)?
                .into_data()
                .try_to_vec::<f32>()?;
            longform::append_audio(&mut output, pcm, rate)?;
        }
        Ok(output)
    }
    pub(crate) fn infer_token_ids(
        &self,
        tokens: &[i64],
        speaker: i64,
        rate: u32,
    ) -> Result<Tensor<2>> {
        ensure!(
            self.rates.contains(&rate) && self.speakers.values().any(|&id| id as i64 == speaker),
            "unsupported V2 sample rate or speaker"
        );
        self.vocoder
            .forward(self.acoustic.infer(tokens, speaker)?, rate)
    }
}


pub(crate) fn prepare_tokens(symbols: &str, encoded_symbols: &str, ascii_symbols: &BTreeMap<char,String>, accent: Option<&crate::recurrent_accent::RecurrentAccent>, text: &str) -> Result<Vec<i64>> {
    if let Some(accent) = accent {
        let mut normalized = text.to_lowercase().chars().filter(|c| symbols.chars().skip(2).any(|s|s==*c)).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
        ensure!(!normalized.is_empty(), "empty V2 text after normalization");
        if !normalized.ends_with(['.', '!', '?']) { normalized.push('.'); }
        let ascii: String = normalized.chars().map(|c| ascii_symbols.get(&c).cloned().unwrap_or_else(||c.to_string())).collect();
        let accented = accent.accentuate(&ascii, true)?;
        encode_v2(encoded_symbols, &accented)
    } else { encode_v2(symbols, text) }
}
