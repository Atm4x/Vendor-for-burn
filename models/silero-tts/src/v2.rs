use crate::{english_vocoder::V2Vocoder, longform, store::TensorStore, tacotron::Tacotron2};
use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Tensor};
use silero_package::{V2Checkpoint, encode_v2};
use std::collections::BTreeMap;

pub(crate) struct V2Tts {
    symbols: String,
    pub(crate) speakers: BTreeMap<String, usize>,
    pub(crate) rates: Vec<u32>,
    acoustic: Tacotron2,
    vocoder: V2Vocoder,
}
impl V2Tts {
    pub(crate) fn load(checkpoint: V2Checkpoint, device: &Device) -> Result<Self> {
        let weights = TensorStore::native(checkpoint.weights);
        Ok(Self {
            symbols: checkpoint.symbols,
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
            let ids = encode_v2(&self.symbols, &part)?;
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
