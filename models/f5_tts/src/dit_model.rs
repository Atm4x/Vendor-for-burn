use burn::tensor::{Device, Int, Tensor};

use crate::{
    F5Error, Result,
    backend::cubecl::DiTVelocity,
    checkpoint::Checkpoint,
    dit::{DiTBackbone, RotaryPositions, TimeEmbedding},
    dit_input::{InputEmbedding, TextEmbedding},
    model::DiTConfig,
};

struct RequestCache {
    frames: usize,
    batch: usize,
    conditional_text: Tensor<3>,
    unconditional_text: Tensor<3>,
    rope: RotaryPositions,
}

pub struct F5DiT {
    time: TimeEmbedding,
    text: TextEmbedding,
    input: InputEmbedding,
    backbone: DiTBackbone,
    config: DiTConfig,
    cache: Option<RequestCache>,
}

impl F5DiT {
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        vocab_size: usize,
        device: &Device,
    ) -> Result<Self> {
        checkpoint.validate_dit_v1()?;
        let config = DiTConfig::V1_BASE;
        config.verify_v1()?;
        Self::from_validated_config(checkpoint, config, vocab_size, device)
    }

    fn from_validated_config(
        checkpoint: &Checkpoint,
        config: DiTConfig,
        vocab_size: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            time: TimeEmbedding::from_checkpoint(checkpoint, config.dim, device)?,
            text: TextEmbedding::from_checkpoint(
                checkpoint,
                vocab_size,
                config.text_dim,
                config.conv_layers,
                device,
            )?,
            input: InputEmbedding::from_checkpoint(
                checkpoint,
                config.dim,
                config.mel_channels,
                config.text_dim,
                device,
            )?,
            backbone: DiTBackbone::from_checkpoint(
                checkpoint,
                config.dim,
                config.heads,
                config.depth,
                config.ff_mult,
                config.mel_channels,
                device,
            )?,
            config,
            cache: None,
        })
    }

    fn prepare_cache(
        &mut self,
        tokens: Tensor<2, Int>,
        frames: usize,
        batch: usize,
        device: &Device,
    ) -> Result<()> {
        if self.cache.is_some() {
            let cache = self.cache.as_ref().unwrap();
            if cache.frames == frames && cache.batch == batch {
                return Ok(());
            }
            return Err(F5Error::InvalidShape(
                "F5 request shape changed without clearing cached text",
            ));
        }
        let conditional_text = self.text.forward(tokens.clone(), frames, false)?;
        let unconditional_text = self.text.forward(tokens, frames, true)?;
        let rope = RotaryPositions::new(frames, self.config.head_dim, device)?;
        self.cache = Some(RequestCache {
            frames,
            batch,
            conditional_text,
            unconditional_text,
            rope,
        });
        Ok(())
    }

    pub fn forward_cfg(
        &mut self,
        noisy: Tensor<3>,
        reference: Tensor<3>,
        tokens: Tensor<2, Int>,
        t: f32,
        cfg: f32,
    ) -> Result<Tensor<3>> {
        let [batch, frames, channels] = noisy.dims();
        if reference.dims() != [batch, frames, channels]
            || channels != self.config.mel_channels
            || tokens.dims()[0] != batch
        {
            return Err(F5Error::InvalidShape(
                "F5 DiT expects aligned mel and token batch dimensions",
            ));
        }
        if !t.is_finite() || !cfg.is_finite() || cfg < 0.0 {
            return Err(F5Error::InvalidConfiguration(
                "F5 timestep and CFG must be finite and nonnegative",
            ));
        }
        let device = noisy.device();
        self.prepare_cache(tokens, frames, batch, &device)?;
        let cached = self.cache.as_ref().unwrap();
        let doubled = if noisy.dtype() == burn::tensor::DType::F16 {
            let silence = Tensor::zeros([batch, frames, channels], &device);
            self.input.forward(
                Tensor::cat(vec![noisy.clone(), noisy], 0),
                Tensor::cat(vec![reference, silence], 0),
                Tensor::cat(
                    vec![
                        cached.conditional_text.clone(),
                        cached.unconditional_text.clone(),
                    ],
                    0,
                ),
                false,
            )?
        } else {
            let conditional = self.input.forward(
                noisy.clone(),
                reference.clone(),
                cached.conditional_text.clone(),
                false,
            )?;
            let unconditional =
                self.input
                    .forward(noisy, reference, cached.unconditional_text.clone(), true)?;
            Tensor::cat(vec![conditional, unconditional], 0)
        };
        let time = self.time.forward(&vec![t; batch], &device)?;
        let time = Tensor::cat(vec![time.clone(), time], 0);
        let predicted = self.backbone.forward(doubled, time, &cached.rope)?;
        let conditional = predicted.clone().slice([0..batch, 0..frames, 0..channels]);
        let unconditional = predicted.slice([batch..(2 * batch), 0..frames, 0..channels]);
        Ok(conditional.clone() + (conditional - unconditional) * cfg)
    }
}

impl DiTVelocity for F5DiT {
    fn predict(
        &mut self,
        noisy_mel: Tensor<3>,
        reference_mel: Tensor<3>,
        text_tokens: Tensor<2, Int>,
        time: f32,
        cfg_strength: f32,
    ) -> Result<Tensor<3>> {
        self.forward_cfg(noisy_mel, reference_mel, text_tokens, time, cfg_strength)
    }

    fn release_request_cache(&mut self) {
        self.cache = None;
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod numerical_tests {
    use super::*;
    use burn::tensor::TensorData;
    use std::path::Path;

    #[test]
    fn full_dit_cfg_matches_independent_pytorch_from_text_to_output() {
        let device = Device::cpu();
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let checkpoint = Checkpoint::open(fixture_dir.join("dit_full_small.safetensors")).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/dit_full_expected.json")).unwrap();
        let floats = |field: &str| {
            expected[field]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let config = DiTConfig {
            dim: 16,
            depth: 2,
            heads: 4,
            head_dim: 4,
            text_dim: 8,
            ff_mult: 2,
            conv_layers: 1,
            mel_channels: 2,
            sample_rate: 24_000,
            hop_length: 256,
        };
        let mut model = F5DiT::from_validated_config(&checkpoint, config, 8, &device).unwrap();
        let tokens: Vec<i64> = expected["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, 3]), &device);
        let noisy = Tensor::from_data(TensorData::new(floats("noisy"), [1, 4, 2]), &device);
        let reference = Tensor::from_data(TensorData::new(floats("reference"), [1, 4, 2]), &device);
        let check = |label: &str, actual: Tensor<3>, reference: Vec<f32>| {
            let actual = actual.into_data().try_to_vec::<f32>().unwrap();
            assert_eq!(actual.len(), reference.len());
            let max_abs = actual
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_abs <= 2e-4, "{label} PyTorch parity max_abs={max_abs}");
        };
        let text_cond = model.text.forward(tokens.clone(), 4, false).unwrap();
        let text_uncond = model.text.forward(tokens.clone(), 4, true).unwrap();
        check(
            "conditional text",
            text_cond.clone(),
            floats("conditional_text"),
        );
        check(
            "unconditional text",
            text_uncond.clone(),
            floats("unconditional_text"),
        );
        check(
            "conditional input",
            model
                .input
                .forward(noisy.clone(), reference.clone(), text_cond, false)
                .unwrap(),
            floats("conditional_input"),
        );
        check(
            "unconditional input",
            model
                .input
                .forward(noisy.clone(), reference.clone(), text_uncond, true)
                .unwrap(),
            floats("unconditional_input"),
        );
        let result = model
            .forward_cfg(
                noisy,
                reference,
                tokens,
                expected["timestep"].as_f64().unwrap() as f32,
                expected["cfg_strength"].as_f64().unwrap() as f32,
            )
            .unwrap();
        check("complete two-branch DiT CFG", result, floats("cfg_output"));
        model.release_request_cache();
        assert!(model.cache.is_none());
    }
    #[test]
    fn full_dit_dynamic_shapes_and_cfg_match_pytorch() {
        let device = Device::cpu();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let weights = Checkpoint::open(root.join("dit_full_small.safetensors")).unwrap();
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/dit_full_grid_expected.json"
        ))
        .unwrap();
        let config = DiTConfig {
            dim: 16,
            depth: 2,
            heads: 4,
            head_dim: 4,
            text_dim: 8,
            ff_mult: 2,
            conv_layers: 1,
            mel_channels: 2,
            sample_rate: 24_000,
            hop_length: 256,
        };
        let all = fixtures.as_array().unwrap();
        assert_eq!(all.len(), 5);
        for (index, expected) in all.iter().enumerate() {
            if let Ok(only) = std::env::var("F5_DIT_CASE") {
                if only.parse::<usize>().ok() != Some(index) {
                    continue;
                }
            }
            eprintln!("F5 full DiT parity case {index} begin");
            let batch = expected["batch"].as_u64().unwrap() as usize;
            let frames = expected["frames"].as_u64().unwrap() as usize;
            let token_length = expected["token_length"].as_u64().unwrap() as usize;
            let data = |key: &str| -> Vec<f32> {
                expected[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            };
            let tokens: Vec<i64> = expected["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let tokens = Tensor::<2, Int>::from_data(
                TensorData::new(tokens, [batch, token_length]),
                &device,
            );
            let noisy =
                Tensor::from_data(TensorData::new(data("noisy"), [batch, frames, 2]), &device);
            let reference = Tensor::from_data(
                TensorData::new(data("reference"), [batch, frames, 2]),
                &device,
            );
            let model = F5DiT::from_validated_config(&weights, config, 8, &device).unwrap();
            check_grid(
                &format!("{index} conditional text"),
                model.text.forward(tokens.clone(), frames, false).unwrap(),
                data("conditional_text"),
            );
            check_grid(
                &format!("{index} unconditional text"),
                model.text.forward(tokens.clone(), frames, true).unwrap(),
                data("unconditional_text"),
            );
            let mut model = model;
            let output = model
                .forward_cfg(
                    noisy.clone(),
                    reference.clone(),
                    tokens.clone(),
                    expected["timestep"].as_f64().unwrap() as f32,
                    expected["cfg_strength"].as_f64().unwrap() as f32,
                )
                .unwrap();
            check_grid(&format!("{index} full DiT"), output, data("cfg_output"));
            model.release_request_cache();

            if let Some(steps) = expected["sampling_steps"].as_u64() {
                let steps = steps as usize;
                let cond = Tensor::from_data(
                    TensorData::new(data("conditioned_reference"), [batch, frames, 2]),
                    &device,
                );
                let reference_frames = expected["reference_frames"].as_u64().unwrap() as usize;
                let timeline: Vec<f32> = data("trajectory_timesteps");
                let mut state = noisy.clone();
                for step in 0..steps {
                    eprintln!("F5 case {index} step {step} start");
                    let velocity = model
                        .forward_cfg(
                            state.clone(),
                            cond.clone(),
                            tokens.clone(),
                            timeline[step],
                            expected["cfg_strength"].as_f64().unwrap() as f32,
                        )
                        .unwrap();
                    state = state + velocity * (timeline[step + 1] - timeline[step]);
                    let target = expected["trajectory_states"][step]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect();
                    check_grid(&format!("{index} Euler step {step}"), state.clone(), target);
                }
                model.release_request_cache();
                let integrated = crate::backend::cubecl::integrate_with_channels(
                    &mut model,
                    noisy,
                    cond,
                    tokens,
                    reference_frames,
                    crate::cfm::SamplingConfig {
                        steps,
                        cfg_strength: expected["cfg_strength"].as_f64().unwrap() as f32,
                        sway_sampling_coef: None,
                        empirically_pruned_steps: false,
                    },
                    2,
                )
                .unwrap();
                eprintln!("F5 case {index} CFM integration complete");
                check_grid(
                    &format!("{index} inpainted CFM"),
                    integrated,
                    data("trajectory_final"),
                );
                assert!(model.cache.is_none());
            }
        }
    }

    fn check_grid(label: &str, actual: Tensor<3>, expected: Vec<f32>) {
        let received = actual.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(received.len(), expected.len(), "{label} size");
        let max_abs = received
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 2e-4, "{label}: max_abs={max_abs}");
    }
    #[test]
    fn twenty_two_dit_blocks_match_independent_pytorch_layer_by_layer() {
        let device = Device::cpu();
        let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let checkpoint = Checkpoint::open(folder.join("dit_full_22_blocks.safetensors")).unwrap();
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/dit_full_22_expected.json"))
                .unwrap();
        let data = |name: &str| -> Vec<f32> {
            oracle[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_f64().unwrap() as f32)
                .collect()
        };
        let config = DiTConfig {
            dim: 16,
            depth: 22,
            heads: 4,
            head_dim: 4,
            text_dim: 8,
            ff_mult: 2,
            conv_layers: 1,
            mel_channels: 2,
            sample_rate: 24_000,
            hop_length: 256,
        };
        let model = F5DiT::from_validated_config(&checkpoint, config, 8, &device).unwrap();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(vec![1_i64, 2], [1, 2]), &device);
        let noisy = Tensor::from_data(TensorData::new(data("noisy"), [1, 3, 2]), &device);
        let reference = Tensor::from_data(TensorData::new(data("reference"), [1, 3, 2]), &device);
        let text_cond = model.text.forward(tokens.clone(), 3, false).unwrap();
        let text_uncond = model.text.forward(tokens, 3, true).unwrap();
        let embedded_cond = model
            .input
            .forward(noisy.clone(), reference.clone(), text_cond, false)
            .unwrap();
        let embedded_uncond = model
            .input
            .forward(noisy, reference, text_uncond, true)
            .unwrap();
        let embedding = Tensor::cat(vec![embedded_cond, embedded_uncond], 0);
        let timestep = oracle["timestep"].as_f64().unwrap() as f32;
        let time = model.time.forward(&[timestep], &device).unwrap();
        let time = Tensor::cat(vec![time.clone(), time], 0);
        let positions = RotaryPositions::new(3, config.head_dim, &device).unwrap();
        let (predicted, states) = model
            .backbone
            .forward_traced(embedding, time, &positions)
            .unwrap();
        assert_eq!(states.len(), 22);
        for (index, state) in states.into_iter().enumerate() {
            let expected = oracle["layer_states"][index]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_f64().unwrap() as f32)
                .collect();
            check_grid(&format!("22-block layer {index}"), state, expected);
        }
        let conditional = predicted.clone().slice([0..1, 0..3, 0..2]);
        let unconditional = predicted.slice([1..2, 0..3, 0..2]);
        let cfg = oracle["cfg_strength"].as_f64().unwrap() as f32;
        check_grid(
            "22-block DiT CFG output",
            conditional.clone() + (conditional - unconditional) * cfg,
            data("cfg_output"),
        );
    }

    fn check_e2e_fixture(json: &str) {
        use crate::{
            backend::cubecl::integrate,
            cfm::SamplingConfig,
            vocos::{VocosConfig, VocosDecoder},
            wav::mono_pcm16,
        };

        let oracle: serde_json::Value = serde_json::from_str(json).unwrap();
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let device = Device::cpu();
        let dit_checkpoint = Checkpoint::open(directory.join("e2e_dit_small.safetensors")).unwrap();
        let vocos_checkpoint =
            Checkpoint::open(directory.join("vocos_decoder_small.safetensors")).unwrap();
        let config = DiTConfig {
            dim: 16,
            depth: 2,
            heads: 4,
            head_dim: 4,
            text_dim: 8,
            ff_mult: 2,
            conv_layers: 1,
            mel_channels: 100,
            sample_rate: 24_000,
            hop_length: 256,
        };
        let floats = |field: &str| -> Vec<f32> {
            oracle[field]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_f64().unwrap() as f32)
                .collect()
        };
        let compare = |stage: &str, actual: &[f32], expected: &[f32], tolerance: f32| {
            assert_eq!(actual.len(), expected.len(), "{stage} length");
            let max_abs = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("{stage} PyTorch max_abs={max_abs:.9}");
            assert!(max_abs <= tolerance, "{stage} max_abs={max_abs}");
        };

        let mut dit = F5DiT::from_validated_config(&dit_checkpoint, config, 8, &device).unwrap();
        let tokens = oracle["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect::<Vec<_>>();
        let frames = oracle["frames"].as_u64().unwrap() as usize;
        let reference_frames = oracle["reference_frames"].as_u64().unwrap() as usize;
        let token_count = tokens.len();
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, token_count]), &device);
        let noise = Tensor::from_data(TensorData::new(floats("noise"), [1, frames, 100]), &device);
        let condition = Tensor::from_data(
            TensorData::new(floats("conditioned_reference"), [1, frames, 100]),
            &device,
        );
        let result = integrate(
            &mut dit,
            noise,
            condition,
            tokens,
            reference_frames,
            SamplingConfig {
                steps: oracle["steps"].as_u64().unwrap() as usize,
                cfg_strength: oracle["cfg_strength"].as_f64().unwrap() as f32,
                sway_sampling_coef: oracle["sway_sampling_coef"].as_f64().map(|x| x as f32),
                empirically_pruned_steps: oracle["empirically_pruned_steps"].as_bool().unwrap(),
            },
        )
        .unwrap();
        compare(
            "DiT/CFM/inpainting full mel",
            &result.clone().into_data().try_to_vec::<f32>().unwrap(),
            &floats("full_mel"),
            2e-4,
        );
        let generated = result.slice([0..1, reference_frames..frames, 0..100]);
        compare(
            "generated mel",
            &generated.clone().into_data().try_to_vec::<f32>().unwrap(),
            &floats("generated_mel"),
            2e-4,
        );
        let vocos = VocosDecoder::from_checkpoint(
            &vocos_checkpoint,
            VocosConfig {
                mel_channels: 100,
                dim: 8,
                intermediate_dim: 16,
                layers: 2,
            },
            &device,
        )
        .unwrap();
        compare(
            "Vocos spectrum head",
            &vocos
                .spectrogram_parameters(generated.clone())
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            &floats("vocos_head"),
            2e-4,
        );
        let gain = oracle["restoration_gain"].as_f64().unwrap() as f32;
        let waveform = vocos.decode(generated, gain).unwrap();
        compare("Vocos/ISTFT PCM float", &waveform, &floats("wave"), 2e-4);
        let wav = mono_pcm16(&waveform).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + waveform.len() * 2);
        let expected_pcm = oracle["pcm16"].as_array().unwrap();
        let max_pcm_error = wav[44..]
            .chunks_exact(2)
            .zip(expected_pcm)
            .map(|(bytes, expected)| {
                (i32::from(i16::from_le_bytes([bytes[0], bytes[1]]))
                    - expected.as_i64().unwrap() as i32)
                    .abs()
            })
            .max()
            .unwrap();
        eprintln!("WAV PCM16 max quantization error={max_pcm_error}");
        assert!(max_pcm_error <= 7);
    }

    #[test]
    fn full_dit_cfm_vocos_pcm_matches_independent_pytorch() {
        check_e2e_fixture(include_str!(
            "../tests/fixtures/e2e_audio_pytorch_expected.json"
        ));
    }

    #[test]
    fn full_dit_cfm_vocos_epss_sway_pcm_matches_pytorch() {
        check_e2e_fixture(include_str!(
            "../tests/fixtures/e2e_audio_sway_pytorch_expected.json"
        ));
    }
}
