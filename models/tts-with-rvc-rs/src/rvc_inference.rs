use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::{
    checkpoint::RvcCheckpoint,
    chunked::{ChunkConfig, ChunkProgress, infer_chunked_batched_with_progress},
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
        self.infer_long_16khz_batched(pcm, options, config, 1, progress)
    }

    /// True tensor batching through HuBERT, RMVPE, TextEncoder, Flow and NSF.
    /// Equal-length inputs avoid artificial padding changing HuBERT's normalization.
    pub fn infer_long_16khz_batched(
        &self,
        pcm: &[f32],
        options: &InferenceOptions,
        config: ChunkConfig,
        batch_size: usize,
        progress: impl FnMut(ChunkProgress),
    ) -> Result<Vec<f32>, String> {
        let filtered = highpass_48hz_zero_phase(pcm)?;
        self.infer_long_filtered_16khz_batched(&filtered, options, config, batch_size, progress)
    }

    /// Run chunked inference on PCM already high-pass filtered at 16 kHz.
    /// The caller owns the filtered buffer and may reuse it for output RMS matching.
    pub fn infer_long_filtered_16khz_batched(
        &self,
        filtered_pcm: &[f32],
        options: &InferenceOptions,
        config: ChunkConfig,
        batch_size: usize,
        progress: impl FnMut(ChunkProgress),
    ) -> Result<Vec<f32>, String> {
        infer_chunked_batched_with_progress(
            filtered_pcm,
            config,
            batch_size,
            |segments, indices| {
                let settings: Vec<_> = indices.iter().map(|&i| {
                    let mut settings = options.clone();
                    settings.seed = options.seed.wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15));
                    settings
                }).collect();
                self.infer_16khz_batch(segments, &settings)
            },
            progress,
        )
    }

    pub fn infer_16khz(&self, pcm: &[f32], options: &InferenceOptions) -> Result<Vec<f32>, String> {
        self.infer_16khz_batch(&[pcm.to_vec()], std::slice::from_ref(options))?
            .into_iter().next().ok_or("empty RVC inference result".into())
    }

    pub fn infer_16khz_batch(&self, audios: &[Vec<f32>], options: &[InferenceOptions]) -> Result<Vec<Vec<f32>>, String> {
        let batch = audios.len();
        if batch == 0 || batch != options.len() || batch > 32 {
            return Err("RVC batch inputs/options must match and size must be 1..=32".into());
        }
        let audio_len = audios[0].len();
        if !(3_200..=160_000).contains(&audio_len) || audios.iter().any(|pcm| {
            pcm.len() != audio_len || pcm.iter().any(|v| !v.is_finite())
        }) {
            return Err("RVC tensor batching requires 0.2–10 s equal-length finite 16-kHz PCM".into());
        }
        for setting in options {
            if setting.speaker_id >= self.speaker_count {
                return Err(format!("invalid RVC speaker id {}", setting.speaker_id));
            }
            if !(0.0..=1.0).contains(&setting.index_rate)
                || !(0.0..=1.0).contains(&setting.protect)
                || !setting.pitch_shift_semitones.is_finite()
                || !setting.latent_noise_scale.is_finite()
                || setting.latent_noise_scale < 0.0
            {
                return Err("invalid RVC index, protection, pitch shift or noise settings".into());
            }
        }
        let profile = std::env::var_os("RVC_PROFILE").is_some();
        let mut phase = std::time::Instant::now();
        let samples: Vec<f32> = audios.iter().flat_map(|pcm| pcm.iter().copied()).collect();
        let waveform = Tensor::from_data(TensorData::new(samples, [batch, 1, audio_len]), &self.device);
        let content = self.hubert.forward(waveform, None);
        let [result_batch, content_time, channels] = content.dims();
        if result_batch != batch || channels != CONTENT_CHANNELS {
            return Err(format!("HuBERT expected [{batch}, T, {CONTENT_CHANNELS}], got {:?}", content.dims()));
        }
        let raw = content.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        if profile { eprintln!("profile hubert B={batch} {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let thresholds_uniform = options.iter().all(|x| x.rmvpe_threshold == options[0].rmvpe_threshold);
        let mut f0_batch = if thresholds_uniform {
            self.rmvpe.infer_f0_batch(audios, &self.device, options[0].rmvpe_threshold)?
        } else {
            let (salience, frames) = self.rmvpe.forward_pcm_batch(audios, &self.device)?;
            let values = salience.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
            values.chunks_exact(frames * crate::rmvpe::F0_BINS).zip(options)
                .map(|(row, opts)| crate::rmvpe::salience_to_f0(row, opts.rmvpe_threshold))
                .collect::<Result<Vec<_>,_>>()?
        };
        if profile { eprintln!("profile rmvpe B={batch} {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let frames = (content_time * 2).min(f0_batch[0].len());
        if frames == 0 || f0_batch.iter().any(|row| row.len() < frames) {
            return Err("HuBERT/RMVPE returned no aligned frames".into());
        }
        let mut all_features = Vec::with_capacity(batch * frames * channels);
        let mut all_pitch = Vec::with_capacity(batch * frames);
        let mut all_f0 = Vec::with_capacity(batch * frames);
        let mut random_sources = Vec::with_capacity(batch);
        for (b, (f0, opts)) in f0_batch.iter_mut().zip(options).enumerate() {
            f0.truncate(frames);
            let original = &raw[b * content_time * channels..(b + 1) * content_time * channels];
            let mut retrieved = original.to_vec();
            if opts.index_rate > 0.0 {
                let index = self.index.as_ref().ok_or("FAISS retrieval requested but no index was loaded")?;
                index.blend_features(&mut retrieved, opts.index_rate)?;
            }
            let pitch_factor = 2.0_f32.powf(opts.pitch_shift_semitones / 12.0);
            for (frame, value) in f0.iter_mut().enumerate() {
                let frame_src = frame / 2 * channels;
                let data = &retrieved[frame_src..frame_src + channels];
                if opts.index_rate > 0.0 && opts.protect < 0.5 && *value < 1.0 {
                    let source = &original[frame_src..frame_src + channels];
                    for (&retrieved, &raw) in data.iter().zip(source) {
                        all_features.push(retrieved * opts.protect + raw * (1.0 - opts.protect));
                    }
                } else {
                    all_features.extend_from_slice(data);
                }
                *value *= pitch_factor;
            }
            all_pitch.extend(coarse_f0(f0));
            all_f0.extend(f0.iter().copied());
            random_sources.push(Gaussian::new(opts.seed));
        }
        if profile { eprintln!("profile retrieval B={batch} {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let phone = Tensor::from_data(TensorData::new(all_features, [batch, frames, channels]), &self.device);
        let pitch_tensor = Tensor::<2, Int>::from_data(TensorData::new(all_pitch, [batch, frames]), &self.device);
        let encoded = self.encoder.forward(phone, Some(pitch_tensor), &vec![frames; batch], None);
        if profile { eprintln!("profile encoder B={batch} {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let [_, latent_channels, latent_frames] = encoded.mean.dims();
        let mut noise = Vec::with_capacity(batch * latent_channels * latent_frames);
        for (random, setting) in random_sources.iter_mut().zip(options) {
            noise.extend((0..latent_channels * latent_frames).map(|_| random.sample() * setting.latent_noise_scale));
        }
        let latent_noise = Tensor::from_data(
            TensorData::new(noise, [batch, latent_channels, latent_frames]), &self.device);
        let latent = (encoded.mean + encoded.log_scale.exp() * latent_noise) * encoded.mask.clone();
        let mut speaker_embeddings = Vec::with_capacity(batch * self.speaker_channels);
        for setting in options {
            let start = setting.speaker_id * self.speaker_channels;
            speaker_embeddings.extend_from_slice(&self.speaker_embeddings[start..start + self.speaker_channels]);
        }
        let speaker = Tensor::from_data(TensorData::new(speaker_embeddings, [batch, self.speaker_channels, 1]), &self.device);
        let latent = self.flow.forward(latent, encoded.mask, Some(speaker.clone()), true);
        if profile { eprintln!("profile flow B={batch} {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let initial_phase = Tensor::zeros([1, 1, 1], &self.device);
        let mut gaussian_noise = Vec::with_capacity(batch * frames * RVC_HOP);
        for random in &mut random_sources {
            gaussian_noise.extend((0..frames * RVC_HOP).map(|_| random.sample()));
        }
        let sine_noise = Tensor::from_data(TensorData::new(gaussian_noise, [batch, frames * RVC_HOP, 1]), &self.device);
        let f0_tensor = Tensor::from_data(TensorData::new(all_f0, [batch, frames]), &self.device);
        let output = self.decoder.forward_from_f0(latent, f0_tensor, Some(speaker), &self.source, initial_phase, sine_noise)?;
        let [wave_batch, channel_count, sample_count] = output.dims();
        if wave_batch != batch || channel_count != 1 {
            return Err(format!("NSF decoder returned invalid batch shape {:?}", output.dims()));
        }
        let samples = output.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        if profile { eprintln!("profile nsf B={batch} {:.3}s", phase.elapsed().as_secs_f64()); }
        if samples.iter().any(|x| !x.is_finite()) {
            return Err("RVC generated non-finite waveform".into());
        }
        Ok(samples.chunks_exact(sample_count).map(|row| row.to_vec()).collect())
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
