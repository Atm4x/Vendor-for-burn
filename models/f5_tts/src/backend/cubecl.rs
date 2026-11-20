use burn::tensor::{Device, Int, Tensor};

use crate::{F5Error, Result, cfm::SamplingConfig};

pub trait DiTVelocity {
    fn predict(
        &mut self,
        noisy_mel: Tensor<3>,
        reference_mel: Tensor<3>,
        text_tokens: Tensor<2, Int>,
        time: f32,
        cfg_strength: f32,
    ) -> Result<Tensor<3>>;

    fn release_request_cache(&mut self);
}

pub fn integrate<N: DiTVelocity>(
    network: &mut N,
    mut noise: Tensor<3>,
    reference_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    sampling: SamplingConfig,
) -> Result<Tensor<3>> {
    let result = (|| {
        let timeline = sampling.timeline()?;
        let shape = noise.dims();
        if reference_mel.dims() != shape || shape[2] != 100 {
            return Err(F5Error::InvalidShape(
                "CFM tensors must have matching [B, frames, 100] shapes",
            ));
        }
        if text_tokens.dims()[0] != shape[0] {
            return Err(F5Error::InvalidShape("Text and mel batch sizes must match"));
        }
        for times in timeline.windows(2) {
            let velocity = network.predict(
                noise.clone(),
                reference_mel.clone(),
                text_tokens.clone(),
                times[0],
                sampling.cfg_strength,
            )?;
            if velocity.dims() != shape {
                return Err(F5Error::InvalidShape("DiT must output [B, frames, 100]"));
            }
            noise = noise + velocity * (times[1] - times[0]);
        }
        Ok(noise)
    })();
    network.release_request_cache();
    result
}

#[cfg(feature = "cubecl-cuda")]
pub fn cuda_device(index: usize) -> Device {
    Device::cuda(index)
}

#[cfg(feature = "cubecl-wgpu")]
pub fn wgpu_device() -> Device {
    Device::wgpu(Default::default())
}

#[cfg(feature = "cubecl-cpu")]
pub fn cpu_device() -> Device {
    Device::cpu()
}
