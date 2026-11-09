use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::{
    checkpoint::RvcCheckpoint,
    chunked::{ChunkConfig, ChunkProgress, infer_chunked_with_progress},
    decoder::NsfWaveformDecoder,
    faiss_index::IvfFlatIndex,
    flow::ResidualCouplingBlock,
    hubert_model::HubertFeatureModel,
    nsf_source::SourceModuleHnNSF,
    preprocess::highpass_48hz_zero_phase,
    rmvpe_mel::RmvpeAudioModel,
    text_encoder::TextEncoder,
};

const RVC_SAMPLE_RATE: usize = 40_000;
const RVC_HOP: usize = 400;
const CONTENT_CHANNELS: usize = 768;

#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub index_rate: f32,
    pub protect: f32,
    pub pitch_shift_semitones: f32,
    pub rmvpe_threshold: f32,
    pub speaker_id: usize,
    pub latent_noise_scale: f32,
    pub seed: u64,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            index_rate: 0.75,
            protect: 0.33,
            pitch_shift_semitones: 0.0,
            rmvpe_threshold: 0.03,
            speaker_id: 0,
            latent_noise_scale: 0.66666,
            seed: 0x53_74_61_67_65_32_32,
        }
    }
}

pub struct RvcV2Inferencer {
    device: Device,
    hubert: HubertFeatureModel,
    rmvpe: RmvpeAudioModel,
    encoder: TextEncoder,
    flow: ResidualCouplingBlock,
    decoder: NsfWaveformDecoder,
    source: SourceModuleHnNSF,
    speaker_embeddings: Vec<f32>,
    speaker_count: usize,
    speaker_channels: usize,
    index: Option<IvfFlatIndex>,
}

impl RvcV2Inferencer {
    pub fn load(
        voice_path: &str,
        hubert_path: &str,
        rmvpe_path: &str,
        index_path: Option<&str>,
        device: Device,
    ) -> Result<Self, String> {
        let voice = RvcCheckpoint::open(voice_path)?;
        let (shape, speaker_embeddings) = voice.tensor_f32("emb_g.weight")?;
        let [speaker_count, speaker_channels] = shape.as_slice() else {
            return Err(format!(
                "RVC speaker embedding must be rank-2, got {shape:?}"
            ));
        };
        if *speaker_count == 0 || *speaker_channels == 0 {
            return Err("RVC has no speaker embeddings".into());
        }
        let encoder = TextEncoder::load(&voice, &device)?;
        let flow = ResidualCouplingBlock::load(&voice, &device)?;
        let decoder = NsfWaveformDecoder::load(&voice, &[10, 10, 2, 2], 3, [1, 3, 5], &device)?;
        let source = SourceModuleHnNSF::load(&voice, RVC_SAMPLE_RATE, 0, &device)?;
        drop(voice);

        let hubert_checkpoint = RvcCheckpoint::open_model(hubert_path)?;
        let hubert = HubertFeatureModel::load(&hubert_checkpoint, 12, 12, &device)?;
        drop(hubert_checkpoint);

        let rmvpe_checkpoint = RvcCheckpoint::open_root(rmvpe_path)?;
        let rmvpe = RmvpeAudioModel::load(&rmvpe_checkpoint, &device)?;
        drop(rmvpe_checkpoint);

        let index = index_path.map(IvfFlatIndex::open).transpose()?;
        if let Some(index) = &index {
            if index.dimension() != CONTENT_CHANNELS {
                return Err(format!(
                    "RVC v2 requires a {CONTENT_CHANNELS}-dimensional FAISS index, got {}",
                    index.dimension()
                ));
            }
        }
        Ok(Self {
            device,
            hubert,
            rmvpe,
            encoder,
            flow,
            decoder,
            source,
            speaker_embeddings,
            speaker_count: *speaker_count,
            speaker_channels: *speaker_channels,
            index,
        })
    }

    pub fn infer_long_16khz(
        &self,
        pcm: &[f32],
        options: &InferenceOptions,
        config: ChunkConfig,
        progress: impl FnMut(ChunkProgress),
    ) -> Result<Vec<f32>, String> {
        let filtered = highpass_48hz_zero_phase(pcm)?;
        infer_chunked_with_progress(
            &filtered,
            config,
            |segment, chunk_index| {
                let mut settings = options.clone();
                settings.seed = options
                    .seed
                    .wrapping_add((chunk_index as u64).wrapping_mul(0x9e3779b97f4a7c15));
                self.infer_16khz(segment, &settings)
            },
            progress,
        )
    }

    pub fn infer_16khz(&self, pcm: &[f32], options: &InferenceOptions) -> Result<Vec<f32>, String> {
        if pcm.len() < 3_200 || pcm.len() > 160_000 || pcm.iter().any(|x| !x.is_finite()) {
            return Err("RVC inference requires 0.2–10 seconds of finite 16-kHz mono PCM".into());
        }
        if options.speaker_id >= self.speaker_count {
            return Err(format!("invalid RVC speaker id {}", options.speaker_id));
        }
        if !(0.0..=1.0).contains(&options.index_rate)
            || !(0.0..=1.0).contains(&options.protect)
            || !options.pitch_shift_semitones.is_finite()
            || !options.latent_noise_scale.is_finite()
            || options.latent_noise_scale < 0.0
        {
            return Err("invalid RVC index, protection, pitch shift or noise settings".into());
        }
        let waveform = Tensor::from_data(
            TensorData::new(pcm.to_vec(), [1, 1, pcm.len()]),
            &self.device,
        );
        let content = self.hubert.forward(waveform, None);
        let [batch, content_time, channels] = content.dims();
        if batch != 1 || channels != CONTENT_CHANNELS {
            return Err(format!(
                "HuBERT result must be [1, T, {CONTENT_CHANNELS}], got {:?}",
                content.dims()
            ));
        }
        let raw = content
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|e| e.to_string())?;
        let mut f0 = self
            .rmvpe
            .infer_f0(pcm, &self.device, options.rmvpe_threshold)?;
        let frames = (content_time * 2).min(f0.len());
        if frames == 0 {
            return Err("HuBERT/RMVPE returned no aligned frames".into());
        }
        f0.truncate(frames);
        let mut retrieved = raw.clone();
        if options.index_rate > 0.0 {
            let index = self
                .index
                .as_ref()
                .ok_or("FAISS retrieval requested but no index was loaded")?;
            index.blend_features(&mut retrieved, options.index_rate)?;
        }
        let duplicate_frames = |source: &[f32]| -> Vec<f32> {
            source
                .chunks_exact(channels)
                .flat_map(|frame| (0..2).flat_map(move |_| frame.iter().copied()))
                .take(frames * channels)
                .collect()
        };
        let mut features = duplicate_frames(&retrieved);
        if options.index_rate > 0.0 && options.protect < 0.5 {
            let original = duplicate_frames(&raw);
            for (frame, &f0_value) in f0.iter().enumerate() {
                if f0_value < 1.0 {
                    let offset = frame * channels;
                    for (value, unmodified) in features[offset..offset + channels]
                        .iter_mut()
                        .zip(&original[offset..offset + channels])
                    {
                        *value = *value * options.protect + *unmodified * (1.0 - options.protect);
                    }
                }
            }
        }
        let pitch_factor = 2.0_f32.powf(options.pitch_shift_semitones / 12.0);
        for value in &mut f0 {
            *value *= pitch_factor;
        }
        let pitch = coarse_f0(&f0);
        let phone = Tensor::from_data(
            TensorData::new(features, [1, frames, channels]),
            &self.device,
        );
        let pitch_tensor =
            Tensor::<2, Int>::from_data(TensorData::new(pitch, [1, frames]), &self.device);
        let encoded = self
            .encoder
            .forward(phone, Some(pitch_tensor), &[frames], None);
        let [_, latent_channels, latent_frames] = encoded.mean.dims();
        let mut random = Gaussian::new(options.seed);
        let gaussian: Vec<f32> = (0..latent_channels * latent_frames)
            .map(|_| random.sample() * options.latent_noise_scale)
            .collect();
        let latent_noise = Tensor::from_data(
            TensorData::new(gaussian, [1, latent_channels, latent_frames]),
            &self.device,
        );
        let latent = (encoded.mean + encoded.log_scale.exp() * latent_noise) * encoded.mask.clone();
        let start = options.speaker_id * self.speaker_channels;
        let speaker = Tensor::from_data(
            TensorData::new(
                self.speaker_embeddings[start..start + self.speaker_channels].to_vec(),
                [1, self.speaker_channels, 1],
            ),
            &self.device,
        );
        let latent = self
            .flow
            .forward(latent, encoded.mask, Some(speaker.clone()), true);
        let initial_phase = Tensor::zeros([1, 1, 1], &self.device);
        let sine_noise: Vec<f32> = (0..frames * RVC_HOP).map(|_| random.sample()).collect();
        let sine_noise = Tensor::from_data(
            TensorData::new(sine_noise, [1, frames * RVC_HOP, 1]),
            &self.device,
        );
        let f0_tensor = Tensor::from_data(TensorData::new(f0, [1, frames]), &self.device);
        let output = self.decoder.forward_from_f0(
            latent,
            f0_tensor,
            Some(speaker),
            &self.source,
            initial_phase,
            sine_noise,
        )?;
        let samples = output
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|e| e.to_string())?;
        if samples.iter().any(|x| !x.is_finite()) {
            return Err("RVC generated non-finite waveform".into());
        }
        Ok(samples)
    }
}

fn coarse_f0(values: &[f32]) -> Vec<i64> {
    let min = 1127.0_f32 * (1.0 + 50.0_f32 / 700.0).ln();
    let max = 1127.0_f32 * (1.0 + 1100.0_f32 / 700.0).ln();
    values
        .iter()
        .map(|&hz| {
            if hz <= 0.0 {
                return 1;
            }
            let mel = 1127.0 * (1.0 + hz / 700.0).ln();
            ((mel - min) * 254.0 / (max - min) + 1.0)
                .round()
                .clamp(1.0, 255.0) as i64
        })
        .collect()
}

struct Gaussian {
    state: u64,
    spare: Option<f32>,
}

impl Gaussian {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.state;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        (x >> 11) as f64 / (1_u64 << 53) as f64
    }
    fn sample(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some((radius * theta.sin()) as f32);
        (radius * theta.cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f0_bins_and_noise_are_well_formed() {
        assert_eq!(coarse_f0(&[0.0, 50.0, 1100.0, 11000.0]), [1, 1, 255, 255]);
        let mut a = Gaussian::new(77);
        let mut b = Gaussian::new(77);
        for _ in 0..100 {
            assert_eq!(a.sample(), b.sample());
        }
    }
}
