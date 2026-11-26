//! Device-executed Fish Speech 1.5 LogMel with reflect/Hann, radix-2 FFT and Slaney mel filters.
//! The CPU path is used only for preprocessing constants and numerical parity tests.

use cubecl::prelude::*;
use cubecl::server::Handle;
use fish_speech_core::{
    audio::{HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE},
    error::{FishError, Result},
    firefly_cpu::AudioTensor,
};
use std::f64::consts::PI;

const FREQS: usize = N_FFT / 2 + 1;
const LEFT_PAD: usize = (N_FFT - HOP_LENGTH) / 2;
const MAX_BYTES: usize = 512 * 1024 * 1024;

#[cube(launch)]
fn reflect_hann_kernel(
    waveform: &[f32],
    bit_reverse: &[i32],
    hann: &[f32],
    output: &mut [f32],
    #[comptime] samples: usize,
 ) {
    if ABSOLUTE_POS < output.len() / 2 {
        let frame = ABSOLUTE_POS / N_FFT;
        let n = usize::cast_from(bit_reverse[ABSOLUTE_POS % N_FFT]);
        let pos = i32::cast_from(frame * HOP_LENGTH + n) - i32::cast_from(LEFT_PAD);
        let ix = if pos < 0 {
            usize::cast_from(-pos)
        } else if pos >= i32::cast_from(samples) {
            usize::cast_from(2 * i32::cast_from(samples) - pos - 2)
        } else {
            usize::cast_from(pos)
        };
        output[2 * ABSOLUTE_POS] = waveform[ix] * hann[n];
        output[2 * ABSOLUTE_POS + 1] = 0.0f32;
    }
}

#[cube(launch)]
fn fft_stage_kernel(
    input: &[f32],
    twiddles: &[f32],
    output: &mut [f32],
    #[comptime] half: usize,
) {
    if ABSOLUTE_POS < output.len() / 4 {
        let k = ABSOLUTE_POS % half;
        let group = ABSOLUTE_POS / half;
        let frame = group / (N_FFT / (2 * half));
        let block = group % (N_FFT / (2 * half));
        let left = frame * N_FFT + block * (2 * half) + k;
        let right = left + half;
        let w = 2 * (half - 1 + k);
        let wr = twiddles[w];
        let wi = twiddles[w + 1];
        let ar = input[2 * left];
        let ai = input[2 * left + 1];
        let br = input[2 * right];
        let bi = input[2 * right + 1];
        let tr = br * wr - bi * wi;
        let ti = br * wi + bi * wr;
        output[2 * left] = ar + tr;
        output[2 * left + 1] = ai + ti;
        output[2 * right] = ar - tr;
        output[2 * right + 1] = ai - ti;
    }
}

#[cube(launch)]
fn magnitude_kernel(fft: &[f32], magnitudes: &mut [f32]) {
    if ABSOLUTE_POS < magnitudes.len() {
        let frame = ABSOLUTE_POS / FREQS;
        let bin = ABSOLUTE_POS % FREQS;
        let offset = 2 * (frame * N_FFT + bin);
        let re = fft[offset];
        let im = fft[offset + 1];
        magnitudes[ABSOLUTE_POS] = (re * re + im * im + 1e-6f32).sqrt();
    }
}

#[cube(launch)]
fn mel_projection_kernel(
    magnitudes: &[f32],
    filters: &[f32],
    output: &mut [f32],
    #[comptime] frames: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let mel = ABSOLUTE_POS / frames;
        let frame = ABSOLUTE_POS % frames;
        let mut amplitude = 0.0f32;
        for bin in 0..FREQS {
            amplitude += filters[mel * FREQS + bin] * magnitudes[frame * FREQS + bin];
        }
        output[ABSOLUTE_POS] = if amplitude < 1e-5f32 { (1e-5f32).ln() } else { amplitude.ln() };
    }
}

fn hz_to_mel(freq: f64) -> f64 {
    if freq < 1000.0 { freq / (200.0 / 3.0) } else { 15.0 + (freq / 1000.0).ln() / (6.4_f64.ln() / 27.0) }
}
fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 { mel * (200.0 / 3.0) } else { 1000.0 * ((mel - 15.0) * (6.4_f64.ln() / 27.0)).exp() }
}

pub struct CubeClLogMel {
    client: Client,
    bit_reverse: Handle,
    hann: Handle,
    twiddles: Handle,
    filters: Handle,
}

pub(crate) struct DeviceLogMel {
    pub handle: Handle,
    pub retained: Vec<Handle>,
    pub frames: usize,
}

impl CubeClLogMel {
    pub fn new(client: Client) -> Self {
        let bits = N_FFT.trailing_zeros();
        let bit_reverse: Vec<i32> = (0..N_FFT)
            .map(|n| (n.reverse_bits() >> (usize::BITS - bits)) as i32)
            .collect();
        let hann: Vec<f32> = (0..N_FFT)
            .map(|n| (0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        let mut twiddles = Vec::with_capacity((N_FFT - 1) * 2);
        let mut half = 1;
        while half < N_FFT {
            for k in 0..half {
                let angle = -PI * k as f64 / half as f64;
                twiddles.push(angle.cos() as f32);
                twiddles.push(angle.sin() as f32);
            }
            half *= 2;
        }
        let max_mel = hz_to_mel((SAMPLE_RATE / 2) as f64);
        let points: Vec<f64> = (0..N_MELS + 2)
            .map(|i| mel_to_hz(max_mel * i as f64 / (N_MELS + 1) as f64))
            .collect();
        let mut filters = vec![0.0f32; N_MELS * FREQS];
        for mel in 0..N_MELS {
            let (left, center, right) = (points[mel], points[mel + 1], points[mel + 2]);
            let scale = 2.0 / (right - left);
            for bin in 0..FREQS {
                let hz = bin as f64 * (SAMPLE_RATE / 2) as f64 / (FREQS - 1) as f64;
                let factor = ((hz - left) / (center - left)).min((right - hz) / (right - center)).max(0.0);
                filters[mel * FREQS + bin] = (factor * scale) as f32;
            }
        }
        Self {
            bit_reverse: client.create_from_slice(bytemuck::cast_slice(&bit_reverse)),
            hann: client.create_from_slice(bytemuck::cast_slice(&hann)),
            twiddles: client.create_from_slice(bytemuck::cast_slice(&twiddles)),
            filters: client.create_from_slice(bytemuck::cast_slice(&filters)),
            client,
        }
    }

    pub(crate) fn transform_device(&self, audio: &[f32]) -> Result<DeviceLogMel> {
        if audio.len() <= LEFT_PAD || audio.len() > i32::MAX as usize || audio.iter().any(|x| !x.is_finite()) {
            return Err(FishError::InvalidShape("invalid CubeCL LogMel waveform"));
        }
        let frames = audio.len() / HOP_LENGTH;
        let fft_len = frames.checked_mul(N_FFT).and_then(|v| v.checked_mul(2))
            .ok_or(FishError::InvalidShape("CubeCL FFT extent overflow"))?;
        if fft_len * 4 > MAX_BYTES {
            return Err(FishError::InvalidShape("CubeCL FFT memory budget exceeded"));
        }
        let client = &self.client;
        let waveform = client.create_from_slice(bytemuck::cast_slice(audio));
        let mut retained = vec![waveform.clone()];
        let mut current = client.empty(fft_len * 4);
        retained.push(current.clone());
        reflect_hann_kernel::launch(
            client,
            CubeCount::Static((frames * N_FFT).div_ceil(128) as u32, 1, 1),
            CubeDim::new_1d(128),
            unsafe { BufferArg::from_raw_parts(waveform, audio.len()) },
            unsafe { BufferArg::from_raw_parts(self.bit_reverse.clone(), N_FFT) },
            unsafe { BufferArg::from_raw_parts(self.hann.clone(), N_FFT) },
            unsafe { BufferArg::from_raw_parts(current.clone(), fft_len) },
            audio.len(),
        );
        let mut half = 1;
        while half < N_FFT {
            let result = client.empty(fft_len * 4);
            fft_stage_kernel::launch(
                client,
                CubeCount::Static((frames * (N_FFT / 2)).div_ceil(128) as u32, 1, 1),
                CubeDim::new_1d(128),
                unsafe { BufferArg::from_raw_parts(current, fft_len) },
                unsafe { BufferArg::from_raw_parts(self.twiddles.clone(), (N_FFT - 1) * 2) },
                unsafe { BufferArg::from_raw_parts(result.clone(), fft_len) },
                half,
            );
            retained.push(result.clone());
            current = result;
            half *= 2;
        }
        let magnitude_len = frames * FREQS;
        let magnitudes = client.empty(magnitude_len * 4);
        magnitude_kernel::launch(
            client,
            CubeCount::Static(magnitude_len.div_ceil(128) as u32, 1, 1),
            CubeDim::new_1d(128),
            unsafe { BufferArg::from_raw_parts(current, fft_len) },
            unsafe { BufferArg::from_raw_parts(magnitudes.clone(), magnitude_len) },
        );
        retained.push(magnitudes.clone());
        let mel_len = frames * N_MELS;
        let mel = client.empty(mel_len * 4);
        mel_projection_kernel::launch(
            client,
            CubeCount::Static(mel_len.div_ceil(128) as u32, 1, 1),
            CubeDim::new_1d(128),
            unsafe { BufferArg::from_raw_parts(magnitudes, magnitude_len) },
            unsafe { BufferArg::from_raw_parts(self.filters.clone(), N_MELS * FREQS) },
            unsafe { BufferArg::from_raw_parts(mel.clone(), mel_len) },
            frames,
        );
        retained.push(mel.clone());
        Ok(DeviceLogMel {handle: mel, retained, frames})
    }

    pub fn transform_mono(&self, audio: &[f32]) -> Result<AudioTensor> {
        let device = self.transform_device(audio)?;
        let bytes = self.client.read_one(device.handle)
            .map_err(|_| FishError::InvalidTransition("CubeCL LogMel readback failed"))?;
        AudioTensor::new(1, N_MELS, device.frames,
            bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect())
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use cubecl::Device;
    use fish_speech_core::spectrogram_cpu::FishLogMel;

    #[test]
    fn native_fft_mel_matches_core_reference() {
        let gpu = CubeClLogMel::new(Device::Cpu(Default::default()).client());
        for n in [1024, 2048, 3073] {
            let input: Vec<f32> = (0..n).map(|i| {
                (2.0 * PI * i as f64 * 431.0 / 44100.0).sin() as f32 * 0.35 + (i % 71) as f32 / 500.0
            }).collect();
            let actual = gpu.transform_mono(&input).unwrap();
            let expected = FishLogMel::new().transform_mono(&input).unwrap();
            assert_eq!((actual.batch,actual.channels,actual.frames), (expected.batch,expected.channels,expected.frames));
            for (i, (&a, &b)) in actual.data.iter().zip(expected.data.iter()).enumerate() {
                assert!((a-b).abs() <= 0.025, "n={n}, index={i}, actual={a}, expected={b}");
            }
        }
    }

    #[test]
    fn rejects_invalid_audio_without_dispatch() {
        let gpu = CubeClLogMel::new(Device::Cpu(Default::default()).client());
        assert!(gpu.transform_mono(&vec![0.0; 768]).is_err());
        let mut nonfinite = vec![0.0; 1024];
        nonfinite[17] = f32::INFINITY;
        assert!(gpu.transform_mono(&nonfinite).is_err());
    }
}
