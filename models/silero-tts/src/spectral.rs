//! Exact-length real inverse FFT and same-padded inverse STFT using Burn tensors.
//!
//! This first functional CubeCL implementation evaluates the DFT through dense
//! tensor contractions. It preserves the exact transform length (2400), unlike
//! replacing it by a padded power-of-two FFT. An optimized CubeCL kernel is a
//! separate performance task and must preserve this contract.

use anyhow::{Result, bail};
use burn::tensor::{Device, Tensor, TensorData};

pub struct RealInverseDft {
    n_fft: usize,
    cos_basis: Tensor<2>,
    sin_basis: Tensor<2>,
}

impl RealInverseDft {
    pub fn new(n_fft: usize, device: &Device) -> Result<Self> {
        if n_fft < 2 || n_fft % 2 != 0 {
            bail!("Real inverse DFT requires an even FFT length >= 2");
        }
        let bins = n_fft / 2 + 1;
        let mut cos = vec![0.0f32; bins * n_fft];
        let mut sin = vec![0.0f32; bins * n_fft];
        for k in 0..bins {
            let scale = if k == 0 || k == n_fft / 2 { 1.0 } else { 2.0 } / n_fft as f64;
            for t in 0..n_fft {
                let angle = std::f64::consts::TAU * (k * t) as f64 / n_fft as f64;
                cos[k * n_fft + t] = (scale * angle.cos()) as f32;
                sin[k * n_fft + t] = (scale * angle.sin()) as f32;
            }
        }
        Ok(Self {
            n_fft,
            cos_basis: Tensor::from_data(TensorData::new(cos, [bins, n_fft]), device),
            sin_basis: Tensor::from_data(TensorData::new(sin, [bins, n_fft]), device),
        })
    }

    /// Input: `[batch, frames, bins]` for each real and imaginary component.
    /// Output: `[batch, frames, n_fft]` in frame-major order.
    pub fn forward(&self, real: Tensor<3>, imag: Tensor<3>) -> Result<Tensor<3>> {
        let [batch, frames, bins] = real.dims();
        if imag.dims() != [batch, frames, bins] || bins != self.n_fft / 2 + 1 {
            bail!("Inverse FFT requires matched spectra with n_fft/2+1 bins");
        }
        let real = real.reshape([batch * frames, bins]);
        let imag = imag.reshape([batch * frames, bins]);
        let result = real.matmul(self.cos_basis.clone()) - imag.matmul(self.sin_basis.clone());
        Ok(result.reshape([batch, frames, self.n_fft]))
    }
}

pub struct SameIstft {
    fft: RealInverseDft,
    window: Tensor<2>,
    window_squared: Vec<f32>,
    hop: usize,
    n_fft: usize,
    device: Device,
}

impl SameIstft {
    pub fn new(window: Vec<f32>, hop: usize, device: &Device) -> Result<Self> {
        let n_fft = window.len();
        if hop == 0 || hop >= n_fft || (n_fft - hop) % 2 != 0 {
            bail!("Unsupported same ISTFT window/hop configuration");
        }
        let window_squared = window.iter().map(|v| v*v).collect();
        Ok(Self {
            fft: RealInverseDft::new(n_fft, device)?,
            window: Tensor::from_data(TensorData::new(window, [1, n_fft]), device),
            window_squared,
            hop,
            n_fft,
            device: device.clone(),
        })
    }

    /// Spectra use `[batch, frequency, frame]`, exactly like Silero's `ISTFT`.
    pub fn forward(&self, real: Tensor<3>, imag: Tensor<3>) -> Result<Tensor<2>> {
        let [batch, bins, frames] = real.dims();
        if imag.dims() != [batch, bins, frames] || bins != self.n_fft / 2 + 1 || frames == 0 {
            bail!("Invalid inverse STFT spectral dimensions");
        }
        let segments = self.fft.forward(real.swap_dims(1,2), imag.swap_dims(1,2))?;
        let total = (frames - 1) * self.hop + self.n_fft;
        let mut result = Tensor::<2>::zeros([batch, total], &self.device);
        let mut envelope = vec![0.0f32; total];
        for frame in 0..frames {
            let start = frame * self.hop;
            let previous = result.clone().slice([0..batch, start..start+self.n_fft]);
            let next = segments.clone().slice([0..batch, frame..frame+1])
                .reshape([batch, self.n_fft]) * self.window.clone();
            result = result.slice_assign([0..batch, start..start+self.n_fft], previous + next);
            for j in 0..self.n_fft {
                envelope[start+j] += self.window_squared[j];
            }
        }
        let trim = (self.n_fft - self.hop) / 2;
        let length = total - trim * 2;
        if envelope[trim..trim+length].iter().any(|v| *v <= 1.0e-11) {
            bail!("ISTFT envelope contains zero-valued overlap positions");
        }
        let normalized = result.slice([0..batch, trim..trim+length]);
        let env = Tensor::<2>::from_data(
            TensorData::new(envelope[trim..trim+length].to_vec(), [1, length]),
            &self.device,
        );
        Ok(normalized / env)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn irfft_dc_and_nyquist() {
        #[cfg(feature = "reference-flex")]
        let device = Device::flex();
        #[cfg(all(not(feature = "reference-flex"), feature = "cubecl-cpu"))]
        let device = Device::cpu();
        let dft = RealInverseDft::new(8, &device).unwrap();
        let real = Tensor::from_data(TensorData::new(vec![8., 0., 0., 0., 8.], [1,1,5]), &device);
        let imag = Tensor::<3>::zeros([1,1,5], &device);
        let output = dft.forward(real, imag).unwrap().into_data().try_to_vec::<f32>().unwrap();
        for (i, v) in output.iter().enumerate() {
            let reference = if i % 2 == 0 { 2.0 } else { 0.0 };
            assert!((v-reference).abs() < 1.0e-5, "sample {i}: {v} != {reference}");
        }
    }
}
