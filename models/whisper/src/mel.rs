use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 80;
pub const CHUNK_SECONDS: usize = 30;
pub const N_SAMPLES: usize = SAMPLE_RATE * CHUNK_SECONDS;
pub const N_FRAMES: usize = N_SAMPLES / HOP;
const N_FREQS: usize = N_FFT / 2 + 1;

fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f64).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * logstep).exp()
    } else {
        mel * F_SP
    }
}

pub fn mel_filters(n_mels: usize) -> Vec<Vec<f64>> {
    let fft_freqs: Vec<f64> = (0..N_FREQS)
        .map(|k| k as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
        .collect();

    let (m_min, m_max) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fdiff: Vec<f64> = mel_f.windows(2).map(|w| w[1] - w[0]).collect();

    (0..n_mels)
        .map(|i| {
            let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
            fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - mel_f[i]) / fdiff[i];
                    let upper = (mel_f[i + 2] - f) / fdiff[i + 1];
                    lower.min(upper).max(0.0) * enorm
                })
                .collect()
        })
        .collect()
}

pub fn pad_or_trim(samples: &[f32]) -> Vec<f32> {
    let mut out = samples[..samples.len().min(N_SAMPLES)].to_vec();
    out.resize(N_SAMPLES, 0.0);
    out
}

fn reflect_pad(x: &[f64], pad: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| x[i]));
    out.extend_from_slice(x);
    out.extend((1..=pad).map(|i| x[n - 1 - i]));
    out
}

struct SparseFilter {
    start: usize,
    w: Vec<f64>,
}

fn sparsify(filters: &[Vec<f64>]) -> Vec<SparseFilter> {
    filters
        .iter()
        .map(|f| {
            let start = f.iter().position(|&x| x != 0.0).unwrap_or(0);
            let end = f.iter().rposition(|&x| x != 0.0).map_or(start, |e| e + 1);
            SparseFilter {
                start,
                w: f[start..end].to_vec(),
            }
        })
        .collect()
}

fn power_spectrogram(padded: &[f64]) -> Vec<[f64; N_FREQS]> {
    let window: Vec<f64> = (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos())
        .collect();

    let fft = FftPlanner::<f64>::new().plan_fft_forward(N_FFT);
    let mut power = vec![[0.0f64; N_FREQS]; N_FRAMES];
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(16);
    let per_thread = N_FRAMES.div_ceil(threads);
    std::thread::scope(|scope| {
        for (ci, frames) in power.chunks_mut(per_thread).enumerate() {
            let (fft, window) = (&fft, &window);
            scope.spawn(move || {
                let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
                for (i, frame_power) in frames.iter_mut().enumerate() {
                    let start = (ci * per_thread + i) * HOP;
                    for n in 0..N_FFT {
                        buf[n] = Complex::new(padded[start + n] * window[n], 0.0);
                    }
                    fft.process(&mut buf);
                    for (k, p) in frame_power.iter_mut().enumerate() {
                        *p = buf[k].norm_sqr();
                    }
                }
            });
        }
    });
    power
}

pub fn log_mel_spectrogram(samples: &[f32], n_mels: usize) -> Vec<f32> {
    let audio: Vec<f64> = pad_or_trim(samples).iter().map(|&s| s as f64).collect();
    let padded = reflect_pad(&audio, N_FFT / 2);

    let power = power_spectrogram(&padded);
    let filters = sparsify(&mel_filters(n_mels));

    let mut log_spec = vec![0.0f64; n_mels * N_FRAMES];
    for (m, filt) in filters.iter().enumerate() {
        for t in 0..N_FRAMES {
            let bins = &power[t][filt.start..filt.start + filt.w.len()];
            let e: f64 = filt.w.iter().zip(bins).map(|(f, p)| f * p).sum();
            log_spec[m * N_FRAMES + t] = e.max(1e-10).log10();
        }
    }

    let max = log_spec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    log_spec
        .iter()
        .map(|&v| ((v.max(max - 8.0) + 4.0) / 4.0) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal() -> Vec<f32> {
        (0..N_SAMPLES)
            .map(|i| {
                let t = i as f64 / SAMPLE_RATE as f64;
                (0.5 * (2.0 * PI * 440.0 * t).sin() + 0.2 * (2.0 * PI * 3100.0 * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft() {
        let audio: Vec<f64> = test_signal().iter().map(|&s| s as f64).collect();
        let padded = reflect_pad(&audio, N_FFT / 2);
        let power = power_spectrogram(&padded);
        for &frame in &[0usize, 1, 1234, 2999] {
            let start = frame * HOP;
            for k in (0..N_FREQS).step_by(7) {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for n in 0..N_FFT {
                    let w = 0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos();
                    let a = -2.0 * PI * (k * n) as f64 / N_FFT as f64;
                    re += padded[start + n] * w * a.cos();
                    im += padded[start + n] * w * a.sin();
                }
                let want = re * re + im * im;
                let got = power[frame][k];
                assert!(
                    (got - want).abs() <= 1e-9 * (1.0 + want),
                    "frame {frame} bin {k}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn filterbank_properties() {
        for n_mels in [80usize, 128] {
            let f = mel_filters(n_mels);
            assert_eq!((f.len(), f[0].len()), (n_mels, N_FREQS));
            for (i, row) in f.iter().enumerate() {
                assert!(
                    row.iter().all(|&w| w >= 0.0),
                    "negative weight in filter {i}"
                );
                assert!(row.iter().any(|&w| w > 0.0), "empty filter {i}");
                if i >= n_mels * 3 / 4 {
                    let area: f64 = row.iter().sum::<f64>() * (SAMPLE_RATE as f64 / N_FFT as f64);
                    assert!((area - 1.0).abs() < 0.1, "filter {i} area {area}");
                }
            }
            let centre = |row: &Vec<f64>| {
                row.iter()
                    .enumerate()
                    .fold((0.0, 0), |b, (k, &w)| if w > b.0 { (w, k) } else { b })
                    .1
            };
            let c: Vec<usize> = f.iter().map(centre).collect();
            assert!(c.windows(2).all(|w| w[0] <= w[1]));
        }
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9);
        assert!((mel_to_hz(hz_to_mel(5234.5)) - 5234.5).abs() < 1e-6);
    }

    #[test]
    fn log_mel_shape_range_and_pitch() {
        let m = log_mel_spectrogram(&test_signal(), N_MELS);
        assert_eq!(m.len(), N_MELS * N_FRAMES);
        let max = m.iter().cloned().fold(f32::MIN, f32::max);
        let min = m.iter().cloned().fold(f32::MAX, f32::min);
        assert!(m.iter().all(|v| v.is_finite()));
        assert!(
            min >= max - 2.0 - 1e-4,
            "dynamic range must be clamped to 8 decades (2.0 after /4)"
        );
        let t = 1500;
        let best = (0..N_MELS)
            .max_by(|&a, &b| m[a * N_FRAMES + t].total_cmp(&m[b * N_FRAMES + t]))
            .unwrap();
        assert!(best < 12, "440 Hz peak landed in mel band {best}");
    }

    #[test]
    fn silence_is_constant() {
        let m = log_mel_spectrogram(&vec![0.0; 16_000], N_MELS);
        assert!(m.iter().all(|&v| (v - m[0]).abs() < 1e-6));
    }
}
