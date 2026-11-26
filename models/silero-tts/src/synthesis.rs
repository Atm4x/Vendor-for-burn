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

    /// Mel must have shape `[batch, 192, frames]`. Returns PCM `[batch, samples]`.
    pub fn forward(&self, mel: Tensor<3>, sample_rate: u32) -> Result<Tensor<2>> {
        if !matches!(sample_rate, 8000 | 24000 | 48000) {
            bail!("Silero supports 8000, 24000, 48000 Hz only");
        }
        let timed = std::env::var_os("SILERO_TIMING").is_some();
        let start = std::time::Instant::now();
        let (magnitude, cosine, sine) = self.vocoder.forward(mel);
        if timed { eprintln!("[silero-stage] vocoder_ms={:.3}", start.elapsed().as_secs_f64()*1000.0); }
        let start = std::time::Instant::now();
        let waveform = self.istft.forward(magnitude.clone() * cosine, magnitude * sine)?;
        if timed { eprintln!("[silero-stage] istft_ms={:.3}", start.elapsed().as_secs_f64()*1000.0); }
        if sample_rate == 48000 {
            return Ok(waveform);
        }
        let kernel = if sample_rate == 24000 { self.pqmf2.clone() } else { self.pqmf6.clone() };
        let bands = if sample_rate == 24000 { 2 } else { 6 };
        let start = std::time::Instant::now();
        let audio = conv1d(waveform.unsqueeze::<3>(), kernel, None,
            ConvOptions::new([bands], [31], [1], 1));
        let limit: f32 = audio.clone().abs().max().into_scalar();
        let audio = if limit <= 1.0 { audio } else { audio.tanh() };
        let [batch, _, length] = audio.dims();
        if timed { eprintln!("[silero-stage] pqmf_ms={:.3}", start.elapsed().as_secs_f64()*1000.0); }
        Ok(audio.slice([0..batch, 0..1, 0..length]).reshape([batch, length]))
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn actual_silero_v5_pcm_parity_all_sample_rates() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let root = Path::new(&dir);
        let device = Device::flex();
        let weights = TensorStore::open(&root.join("tts.safetensors")).unwrap();
        let filters = TensorStore::open(&root.join("pqmf.safetensors")).unwrap();
        let reference = TensorStore::open(&root.join("parity.safetensors")).unwrap();
        let model = Synthesizer::load(&weights, &filters, &device).unwrap();
        let mel = reference.float("mel", &device).unwrap();
        for sr in [48000, 24000, 8000] {
            let got = model.forward(mel.clone(), sr).unwrap()
                .into_data().try_to_vec::<f32>().unwrap();
            let expected = reference.float::<2>(&format!("wave{sr}"), &device).unwrap()
                .into_data().try_to_vec::<f32>().unwrap();
            assert_eq!(got.len(),expected.len(), "sample count differs for {sr} Hz");
            let max_abs = got.iter().zip(expected.iter())
                .map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
            println!("PCM {sr} Hz max_abs={max_abs:.9}");
            assert!(max_abs < 1.0e-3, "PCM at {sr} Hz max_abs={max_abs:.9}");
        }
    }
}
