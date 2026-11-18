//! Silero 48/24/8 kHz audio reconstruction after ConvNeXt spectral prediction.

use anyhow::{Result, bail};
use burn::tensor::{Device, Tensor, module::conv1d, ops::ConvOptions};
use crate::{spectral::SameIstft, store::TensorStore, vocoder::Vocoder};

pub struct Synthesizer {
    vocoder: Vocoder,
    istft: SameIstft,
    pqmf2: Tensor<3>,
    pqmf6: Tensor<3>,
}

impl Synthesizer {
    pub fn load(tts: &TensorStore, pqmf: &TensorStore, device: &Device) -> Result<Self> {
        let vocoder = Vocoder::load(tts, device)?;
        let window = tts.float_data("vocoder.head.istft.window")?;
        let istft = SameIstft::new(window, 600, device)?;
        let pqmf2 = pqmf.float("analysis_2", device)?;
        let pqmf6 = pqmf.float("analysis_6", device)?;
        Ok(Self { vocoder, istft, pqmf2, pqmf6 })
    }

    /// Mel must have shape `[batch, 80, frames]`. Returns PCM `[batch, samples]`.
    pub fn forward(&self, mel: Tensor<3>, sample_rate: u32) -> Result<Tensor<2>> {
        if !matches!(sample_rate, 8000 | 24000 | 48000) {
            bail!("Silero supports 8000, 24000, 48000 Hz only");
        }
        let (magnitude, cosine, sine) = self.vocoder.forward(mel);
        let waveform = self.istft.forward(magnitude.clone() * cosine, magnitude * sine)?;
        if sample_rate == 48000 {
            return Ok(waveform);
        }
        let kernel = if sample_rate == 24000 { self.pqmf2.clone() } else { self.pqmf6.clone() };
        let bands = if sample_rate == 24000 { 2 } else { 6 };
        let audio = conv1d(waveform.unsqueeze::<3>(), kernel, None,
            ConvOptions::new([bands], [31], [1], 1));
        let limit: f32 = audio.clone().abs().max().into_scalar();
        let audio = if limit <= 1.0 { audio } else { audio.tanh() };
        let [batch, _, length] = audio.dims();
        Ok(audio.slice([0..batch, 0..1, 0..length]).reshape([batch, length]))
    }
}
