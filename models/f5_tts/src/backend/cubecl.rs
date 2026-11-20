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
    noise: Tensor<3>,
    reference_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    reference_frames: usize,
    sampling: SamplingConfig,
) -> Result<Tensor<3>> {
    integrate_with_channels(
        network,
        noise,
        reference_mel,
        text_tokens,
        reference_frames,
        sampling,
        100,
    )
}

pub(crate) fn integrate_with_channels<N: DiTVelocity>(
    network: &mut N,
    mut noise: Tensor<3>,
    reference_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    reference_frames: usize,
    sampling: SamplingConfig,
    mel_channels: usize,
) -> Result<Tensor<3>> {
    let result = (|| {
        let timeline = sampling.timeline()?;
        let shape = noise.dims();
        if reference_mel.dims() != shape || shape[2] != mel_channels {
            return Err(F5Error::InvalidShape(
                "CFM tensors must have matching mel channels and batch dimensions",
            ));
        }
        if reference_frames == 0 || reference_frames >= shape[1] {
            return Err(F5Error::InvalidShape(
                "CFM reference length must be between zero and total frames",
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
        let reference_prefix = reference_mel.slice([0..shape[0], 0..reference_frames, 0..shape[2]]);
        let generated_suffix = noise.slice([0..shape[0], reference_frames..shape[1], 0..shape[2]]);
        Ok(Tensor::cat(vec![reference_prefix, generated_suffix], 1))
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

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    struct ConstantVelocity {
        calls: usize,
        released: bool,
    }

    impl DiTVelocity for ConstantVelocity {
        fn predict(
            &mut self,
            noisy: Tensor<3>,
            _: Tensor<3>,
            _: Tensor<2, Int>,
            _: f32,
            _: f32,
        ) -> Result<Tensor<3>> {
            self.calls += 1;
            Ok(Tensor::ones(noisy.dims(), &noisy.device()))
        }

        fn release_request_cache(&mut self) {
            self.released = true;
        }
    }

    #[test]
    fn cpu_cfm_keeps_reference_and_integrates_generated_suffix() {
        let device = Device::cpu();
        let mut model = ConstantVelocity {
            calls: 0,
            released: false,
        };
        let noise = Tensor::zeros([1, 6, 100], &device);
        let mut condition = vec![0.0; 600];
        condition[..200].fill(3.5);
        let condition = Tensor::from_data(TensorData::new(condition, [1, 6, 100]), &device);
        let text = Tensor::<2, Int>::zeros([1, 4], &device);
        let sampling = SamplingConfig {
            steps: 5,
            cfg_strength: 2.0,
            sway_sampling_coef: None,
            empirically_pruned_steps: false,
        };
        let output = integrate(&mut model, noise, condition, text, 2, sampling).unwrap();
        let values = output.into_data().try_to_vec::<f32>().unwrap();
        assert!(values[..200].iter().all(|&v| (v - 3.5).abs() < 1e-6));
        assert!(values[200..].iter().all(|&v| (v - 1.0).abs() < 1e-5));
        assert_eq!(model.calls, 5);
        assert!(model.released);
    }
}
