//! Exact-length ISTFT: native 2400-point FFT on Flex CPU, tensor DFT on CubeCL.

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
    fft: Option<RealInverseDft>,
    window: Option<Tensor<2>>,
    #[cfg(feature = "reference-flex")]
    window_host: Vec<f32>,
    #[cfg(feature = "reference-flex")]
    inverse_fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
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
        #[cfg(feature = "reference-flex")]
        let use_native_fft = device == &Device::flex();
        #[cfg(not(feature = "reference-flex"))]
        let use_native_fft = false;
        Ok(Self {
            fft: if use_native_fft { None } else { Some(RealInverseDft::new(n_fft, device)?) },
            window: if use_native_fft { None } else { Some(Tensor::from_data(TensorData::new(window.clone(), [1, n_fft]), device)) },
            #[cfg(feature = "reference-flex")]
            window_host: window,
            #[cfg(feature = "reference-flex")]
            inverse_fft: rustfft::FftPlanner::new().plan_fft_inverse(n_fft),
            window_squared,
            hop,
            n_fft,
            device: device.clone(),
        })
    }

    /// Spectra use `[batch, frequency, frame]`, exactly like Silero's `ISTFT`.
    pub fn forward(&self, real: Tensor<3>, imag: Tensor<3>) -> Result<Tensor<2>> {
        #[cfg(feature = "reference-flex")]
        if self.device == Device::flex() {
            return self.forward_native_fft(real, imag);
        }
        self.forward_tensor_dft(real, imag)
    }

    #[cfg(feature = "reference-flex")]
    fn forward_native_fft(&self, real: Tensor<3>, imag: Tensor<3>) -> Result<Tensor<2>> {
        use rustfft::num_complex::Complex32;
        let [batch, bins, frames] = real.dims();
        if imag.dims() != [batch, bins, frames] || bins != self.n_fft / 2 + 1 || frames == 0 {
            bail!("Invalid inverse STFT spectral dimensions");
        }
        let real = real.into_data().try_to_vec::<f32>()?;
        let imag = imag.into_data().try_to_vec::<f32>()?;
        let n_fft = self.n_fft;
        let total = (frames - 1) * self.hop + n_fft;
        let mut envelope = vec![0.0f32; total];
        for frame in 0..frames {
            let start = frame * self.hop;
            for (dst, value) in envelope[start..start+n_fft].iter_mut().zip(&self.window_squared) {
                *dst += value;
            }
        }
        let trim = (n_fft - self.hop) / 2;
        let length = total - trim * 2;
        if envelope[trim..trim+length].iter().any(|v| *v <= 1.0e-11) {
            bail!("ISTFT envelope contains zero-valued overlap positions");
        }
        let mut output = vec![0.0f32; batch * length];
        let mut spectrum = vec![Complex32::new(0., 0.); n_fft];
        let mut scratch = vec![Complex32::new(0., 0.); self.inverse_fft.get_inplace_scratch_len()];
        let scale = 1.0f32 / n_fft as f32;
        for b in 0..batch {
            let mut signal = vec![0.0f32; total];
            for frame in 0..frames {
                for k in 0..bins {
                    let index = (b * bins + k) * frames + frame;
                    spectrum[k] = Complex32::new(real[index], imag[index]);
                }
                for k in bins..n_fft {
                    spectrum[k] = spectrum[n_fft-k].conj();
                }
                self.inverse_fft.process_with_scratch(&mut spectrum, &mut scratch);
                let start = frame * self.hop;
                for j in 0..n_fft {
                    signal[start+j] += spectrum[j].re * scale * self.window_host[j];
                }
            }
            for j in 0..length {
                output[b * length + j] = signal[trim+j] / envelope[trim+j];
            }
        }
        Ok(Tensor::from_data(TensorData::new(output, [batch, length]), &self.device))
    }

    fn forward_tensor_dft(&self, real: Tensor<3>, imag: Tensor<3>) -> Result<Tensor<2>> {
        let [batch, bins, frames] = real.dims();
        if imag.dims() != [batch, bins, frames] || bins != self.n_fft / 2 + 1 || frames == 0 {
            bail!("Invalid inverse STFT spectral dimensions");
        }
        let segments = self.fft.as_ref().expect("Tensor DFT is unavailable on Flex")
            .forward(real.swap_dims(1,2), imag.swap_dims(1,2))?;
        let total = (frames - 1) * self.hop + self.n_fft;
        let mut result = Tensor::<2>::zeros([batch, total], &self.device);
        let mut envelope = vec![0.0f32; total];
        for frame in 0..frames {
            let start = frame * self.hop;
            let previous = result.clone().slice([0..batch, start..start+self.n_fft]);
            let next = segments.clone().slice([0..batch, frame..frame+1])
                .reshape([batch, self.n_fft]) * self.window.as_ref().expect("Missing tensor window").clone();
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

    #[cfg(feature = "reference-flex")]
    #[test]
    fn native_fft_istft_matches_dense_dft_for_multiple_batches() {
        let device = Device::flex();
        let n_fft = 24usize;
        let hop = 6usize;
        let frames = 4usize;
        let batch = 2usize;
        let bins = n_fft / 2 + 1;
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / n_fft as f32).cos()))
            .collect();
        let mut real = vec![0.0f32; batch * bins * frames];
        let mut imag = real.clone();
        for b in 0..batch {
            for k in 0..bins {
                for t in 0..frames {
                    let i = (b * bins + k) * frames + t;
                    real[i] = ((b * 19 + k * 13 + t * 7) % 31) as f32 / 31.0 - 0.5;
                    if k != 0 && k != n_fft / 2 {
                        imag[i] = ((b * 11 + k * 17 + t * 5) % 29) as f32 / 29.0 - 0.5;
                    }
                }
            }
        }
        let real = Tensor::from_data(TensorData::new(real, [batch, bins, frames]), &device);
        let imag = Tensor::from_data(TensorData::new(imag, [batch, bins, frames]), &device);
        let dense = RealInverseDft::new(n_fft, &device).unwrap()
            .forward(real.clone().swap_dims(1, 2), imag.clone().swap_dims(1, 2)).unwrap()
            .into_data().try_to_vec::<f32>().unwrap();
        let got = SameIstft::new(window.clone(), hop, &device).unwrap()
            .forward(real, imag).unwrap().into_data().try_to_vec::<f32>().unwrap();
        let trim = (n_fft - hop) / 2;
        let total = (frames - 1) * hop + n_fft;
        let length = total - 2 * trim;
        let mut envelope = vec![0.0f32; total];
        for frame in 0..frames {
            for j in 0..n_fft {
                envelope[frame * hop + j] += window[j] * window[j];
            }
        }
        for b in 0..batch {
            let mut overlap = vec![0.0f32; total];
            for frame in 0..frames {
                for j in 0..n_fft {
                    overlap[frame * hop + j] += dense[(b * frames + frame) * n_fft + j] * window[j];
                }
            }
            for j in 0..length {
                let expected = overlap[j + trim] / envelope[j + trim];
                let actual = got[b * length + j];
                assert!((actual - expected).abs() <= 1e-5,
                    "batch={b} sample={j}: got={actual} expected={expected}");
            }
        }
    }

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
