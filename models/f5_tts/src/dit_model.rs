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
        let conditional = self.input.forward(
            noisy.clone(),
            reference.clone(),
            cached.conditional_text.clone(),
            false,
        )?;
        let unconditional =
            self.input
                .forward(noisy, reference, cached.unconditional_text.clone(), true)?;
        let doubled = Tensor::cat(vec![conditional, unconditional], 0);
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
