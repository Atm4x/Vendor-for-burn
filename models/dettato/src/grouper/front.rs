use crate::hcqt::{Features, Hcqt};

pub const HD_ATTACK_OFFSETS: [isize; 10] = [-2, 0, 2, 4, 8, 12, 16, 24, 32, 48];
pub const GROUPER_PAD0_SECONDS: f32 = 0.5;
pub const GROUPER_FPS: f32 = 100.0;
const PCEN_LENGTH: usize = 200;

#[derive(Clone, Debug)]
pub struct AudioFeatures {
    pub spectrogram: Features,
    pub extended_bins: usize,
    pub power: Vec<f32>,
    pub low_shift: isize,
}

impl AudioFeatures {
    pub fn power_at(&self, bin: usize, frame: usize) -> f32 {
        self.power[bin * self.spectrogram.frames + frame]
    }
}

pub fn extract(audio16k: &[f32]) -> Result<AudioFeatures, String> {
    if audio16k.len() < 160 { return Err("Grouper needs at least one 10 ms audio frame".into()); }
    let front = Hcqt::default_16k();
    let mut spectrogram = front.extract(audio16k);
    let (extended_bins, low_shift, power) = front.extended_power(audio16k);
    pcen(&mut spectrogram, 0.025, 0.98, 2.0, 0.5, 1e-6);
    standardise(&mut spectrogram);
    Ok(AudioFeatures { spectrogram, extended_bins, power, low_shift })
}

/// Finite, causal 200-tap PCEN smoother: exactly the truncated geometric kernel
/// used by the reference FFT convolution, without allocating a temporal FFT.
pub fn pcen(x: &mut Features, s: f32, alpha: f32, delta: f32, r: f32, eps: f32) {
    let t = x.frames;
    if t == 0 { return; }
    let delta_r = delta.powf(r);
    let q = 1.0 - s;
    let trailing = s * q.powi(PCEN_LENGTH as i32);
    for channel in 0..x.channels {
        for bin in 0..x.bins {
            let mut m = 0.0f32;
            let mut history = [0.0f32; PCEN_LENGTH];
            for frame in 0..t {
                let ix = (channel * t + frame) * x.bins + bin;
                let e = x.data[ix].exp();
                m = s * e + q * m;
                let slot = frame % PCEN_LENGTH;
                if frame >= PCEN_LENGTH { m -= trailing * history[slot]; }
                history[slot] = e;
                x.data[ix] = ((e / (eps + m.max(0.0)).powf(alpha) + delta).powf(r) - delta_r + 1e-4).ln();
            }
        }
    }
}

pub fn standardise(x: &mut Features) {
    let frames = x.frames;
    let bins = x.bins;
    for channel in 0..x.channels {
        let lo = channel * frames * bins;
        let slice = &mut x.data[lo..lo + frames * bins];
        let peak = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let threshold = peak - 12.0;
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for &v in slice.iter() {
            if v > threshold { count += 1; sum += f64::from(v); }
        }
        let mean = sum / count.max(1) as f64;
        let variance = slice.iter().filter(|&&v| v > threshold)
            .map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / count.max(1) as f64;
        let std = variance.sqrt().max(0.5);
        for v in slice { *v = ((f64::from(*v) - mean) / std) as f32; }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NoteFrame {
    pub on: usize,
    pub off: usize,
    pub lane: usize,
}

pub fn note_frame(note: &[f32; 12], frames: usize) -> NoteFrame {
    let on = ((note[0] + GROUPER_PAD0_SECONDS) * GROUPER_FPS).round()
        .clamp(0.0, frames.saturating_sub(1) as f32) as usize;
    let off = (((note[1] + GROUPER_PAD0_SECONDS) * GROUPER_FPS).round().max((on + 1) as f32))
        .clamp(0.0, frames as f32) as usize;
    let lane = ((note[2].round() as isize) - 21).clamp(0, 87) as usize;
    NoteFrame { on, off, lane }
}

/// PyTorch HDemix.raw: shared harmonic occupancy and per-note logarithmic energy.
/// The `prior` argument contains the learned *unconstrained* checkpoint weights.
pub fn harmonic_features(
    front: &AudioFeatures,
    notes: &[[f32; 12]],
    prior: &[f32],
    sustain_points: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let h = prior.len();
    let t = front.spectrogram.frames;
    let ne = front.extended_bins;
    if h == 0 || sustain_points == 0 || t == 0 { return Err("invalid harmonic feature dimensions".into()); }
    let mut delta = vec![0.0f32; ne * (t + 1)];
    let mut bins = vec![vec![None; h]; notes.len()];
    let mut frames = Vec::with_capacity(notes.len());
    let claims = prior.iter().map(|&v| if v > 20.0 { v } else { (1.0 + v.exp()).ln() }).collect::<Vec<_>>();
    for (i, note) in notes.iter().enumerate() {
        let frame = note_frame(note, t);
        for harmonic in 1..=h {
            let shift = (36.0 * (harmonic as f64).log2()).round() as isize;
            let raw_bin = 3 * frame.lane as isize - front.low_shift + shift;
            if raw_bin < ne as isize {
                let bin = raw_bin.clamp(0, (ne - 1) as isize) as usize;
                bins[i][harmonic - 1] = Some(bin);
                delta[bin * (t + 1) + frame.on] += claims[harmonic - 1];
                delta[bin * (t + 1) + frame.off] -= claims[harmonic - 1];
            }
        }
        frames.push(frame);
    }
    let mut occupancy = vec![0.0f32; ne * t];
    for bin in 0..ne {
        let mut acc = 0.0f32;
        for frame in 0..t {
            acc += delta[bin * (t + 1) + frame];
            occupancy[bin * t + frame] = acc;
        }
    }
    let points = HD_ATTACK_OFFSETS.len() + sustain_points;
    let mut result = vec![vec![0.0f32; h * (points + 1)]; notes.len()];
    for (i, frame) in frames.iter().enumerate() {
        let duration = (frame.off - frame.on) as f32;
        let mut sample_frames = Vec::with_capacity(points);
        for &offset in &HD_ATTACK_OFFSETS {
            sample_frames.push(((frame.on as isize + offset).clamp(0, t as isize - 1)) as usize);
        }
        for k in 0..sustain_points {
            let fraction = if sustain_points == 4 { [0.25, 0.5, 0.75, 0.95][k] }
                else { 0.25 + 0.7 * k as f32 / (sustain_points - 1).max(1) as f32 };
            sample_frames.push((frame.on + (duration * fraction) as usize).min(t - 1));
        }
        let mut max_log = if bins[i].iter().any(Option::is_none) { 0.0 } else { f32::NEG_INFINITY };
        let mut energies = vec![0.0f32; h * points];
        for harmonic in 0..h {
            let Some(bin) = bins[i][harmonic] else { continue; };
            for (point, &ft) in sample_frames.iter().enumerate() {
                let active = ft >= frame.on && ft < frame.off;
                let own = claims[harmonic];
                let denominator = occupancy[bin * t + ft] + if active { 0.0 } else { own };
                let share = (own / denominator.max(1e-6)).min(1.0);
                let energy = (front.power_at(bin, ft) * share + 1e-7).ln();
                energies[harmonic * points + point] = energy;
                max_log = max_log.max(energy);
            }
            result[i][h * points + harmonic] = 1.0;
        }
        if !max_log.is_finite() { max_log = 0.0; }
        for harmonic in 0..h {
            if bins[i][harmonic].is_some() {
                for point in 0..points {
                    result[i][harmonic * points + point] =
                        ((energies[harmonic * points + point] - max_log).max(-20.0)) / 10.0;
                }
            } else {
                for point in 0..points {
                    result[i][harmonic * points + point] = (-max_log).max(-20.0) / 10.0;
                }
            }
        }
    }
    Ok(result)
}

/// Profile statistics in the note's bin and its immediately higher log-frequency bin.
pub fn profile_features(x: &Features, notes: &[[f32; 12]]) -> Vec<Vec<f32>> {
    let t = x.frames;
    let mut output = Vec::with_capacity(notes.len());
    for note in notes {
        let frame = note_frame(note, t);
        let mut features = Vec::with_capacity(4 * x.channels);
        for off in 0..2 {
            let bin = (3 * frame.lane + off).min(x.bins - 1);
            for stat in 0..2 {
                for channel in 0..x.channels {
                    let mut sum = 0.0f64;
                    let mut squares = 0.0f64;
                    for ti in frame.on..frame.off {
                        let v = f64::from(x.get(channel, ti, bin));
                        sum += v;
                        squares += v * v;
                    }
                    let n = (frame.off - frame.on).max(1) as f64;
                    let mean = sum / n;
                    features.push(if stat == 0 { mean as f32 } else { ((squares / n - mean * mean).max(0.0) + 1e-4).sqrt() as f32 });
                }
            }
        }
        output.push(features);
    }
    output
}

/// Occupancy rasters at 100 / stride frames per second, channels: onset,
/// sustain, onset confidence, second-voice onset.
pub fn note_rasters(notes: &[[f32; 12]], frames: usize, lanes: usize, stride: usize) -> Vec<f32> {
    let mut rasters = vec![0.0f32; 4 * (frames + 1) * lanes];
    let fps = GROUPER_FPS / stride as f32;
    for note in notes {
        let on = ((note[0] + GROUPER_PAD0_SECONDS) * fps).trunc().clamp(0.0, frames.saturating_sub(1) as f32) as usize;
        let off = (((note[1] + GROUPER_PAD0_SECONDS) * fps).trunc() as usize).clamp(on + 1, frames);
        let lane = ((note[2].round() as isize) - 21).clamp(0, lanes as isize - 1) as usize;
        let idx = |c: usize, ti: usize| (c * (frames + 1) + ti) * lanes + lane;
        rasters[idx(0, on)] += 1.0;
        rasters[idx(1, on)] += 1.0;
        rasters[idx(1, off)] -= 1.0;
        let confidence = if note[3] > 0.5 { note[5] } else { note[4] };
        rasters[idx(2, on)] += confidence;
        rasters[idx(3, on)] += if note[3] > 0.5 { 1.0 } else { 0.0 };
    }
    let mut out = vec![0.0; 4 * frames * lanes];
    for channel in 0..4 {
        for lane in 0..lanes {
            let mut accum = 0.0;
            for frame in 0..frames {
                let pos = (channel * (frames + 1) + frame) * lanes + lane;
                if channel == 1 { accum += rasters[pos]; out[(channel * frames + frame) * lanes + lane] = accum; }
                else { out[(channel * frames + frame) * lanes + lane] = rasters[pos]; }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_finite_and_shapes_are_consistent() {
        let audio = vec![0.0f32; 1600];
        let front = extract(&audio).unwrap();
        assert_eq!(front.spectrogram.frames, 10);
        assert_eq!(front.spectrogram.bins, 288);
        assert_eq!(front.extended_bins, 288 + 101 + 36);
        assert!(front.spectrogram.data.iter().all(|v| v.is_finite()));
        assert!(front.power.iter().all(|v| *v >= 0.0 && v.is_finite()));
    }

    #[test]
    fn harmonic_occupancy_shares_spectrum_between_simultaneous_notes() {
        let audio = vec![0.05f32; 1600];
        let front = extract(&audio).unwrap();
        let mut note = [0.0f32; 12];
        note[0] = 0.0; note[1] = 0.07; note[2] = 60.0;
        let single = harmonic_features(&front, &[note], &[0.0; 10], 4).unwrap();
        let double = harmonic_features(&front, &[note, note], &[0.0; 10], 4).unwrap();
        assert_eq!(single[0].len(), 150);
        assert_eq!(double[0], double[1]);
        assert!(double[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn raster_retains_two_independent_voice_onsets() {
        let mut note = [0.0f32; 12];
        note[0] = 0.2; note[1] = 0.5; note[2] = 60.0; note[3] = 1.0; note[5] = 0.8;
        let raster = note_rasters(&[note], 20, 96, 8);
        let lane = 60 - 21;
        let on = ((0.2f32 + 0.5) * 12.5) as usize;
        assert_eq!(raster[(3 * 20 + on) * 96 + lane], 1.0);
    }
}
