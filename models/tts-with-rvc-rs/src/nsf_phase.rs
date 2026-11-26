use crate::chunked::reflect_index;

/// Deterministic fundamental-oscillator phase in cycles (modulo one).
///
/// Only the excitation phase is global. This does not make independently
/// decoded NSF chunks equivalent to a single uninterrupted NSF forward pass.
#[derive(Debug, Clone)]
pub struct GlobalNsfPhase {
    f0: Vec<f32>,
    prefix: Vec<f64>,
    pitch_factor: f32,
    sample_rate: f32,
    upsample: usize,
}

impl GlobalNsfPhase {
    pub fn new(
        f0: &[f32],
        pitch_shift_semitones: f32,
        sample_rate: usize,
        upsample: usize,
    ) -> Result<Self, String> {
        if f0.is_empty() || sample_rate == 0 || upsample == 0
            || !pitch_shift_semitones.is_finite()
            || f0.iter().any(|&value| !value.is_finite() || value < 0.0)
        {
            return Err("global NSF phase requires finite nonnegative F0 and valid sampling parameters".into());
        }
        let pitch_factor = 2.0_f32.powf(pitch_shift_semitones / 12.0);
        if !pitch_factor.is_finite() {
            return Err("global NSF phase pitch factor is not finite".into());
        }
        let mut result = Self {
            f0: f0.to_vec(),
            prefix: Vec::with_capacity(f0.len() + 1),
            pitch_factor,
            sample_rate: sample_rate as f32,
            upsample,
        };
        result.prefix.push(0.0);
        for &hz in f0 {
            let next = (result.prefix.last().copied().unwrap_or(0.0)
                + result.frame_advance(hz)).rem_euclid(1.0);
            result.prefix.push(next);
        }
        Ok(result)
    }

    /// Phase before the first sample in the frame at this absolute position.
    /// Negative positions match the same reflected F0 used by chunk slicing.
    pub fn phase_at(&self, absolute_frame: i64) -> Result<f32, String> {
        let frames = self.f0.len();
        let start = i64::try_from(frames).map_err(|_| "global F0 timeline is too long".to_owned())?;
        let phase = if (0..=start).contains(&absolute_frame) {
            self.prefix[absolute_frame as usize]
        } else if absolute_frame < 0 {
            let mut phase = 0.0f64;
            for frame in absolute_frame..0 {
                let hz = self.f0[reflect_index(frame, frames)];
                phase = (phase - self.frame_advance(hz)).rem_euclid(1.0);
            }
            phase
        } else {
            let mut phase = self.prefix[frames];
            for frame in start..absolute_frame {
                let hz = self.f0[reflect_index(frame, frames)];
                phase = (phase + self.frame_advance(hz)).rem_euclid(1.0);
            }
            phase
        };
        Ok(phase as f32)
    }

    fn frame_advance(&self, hz: f32) -> f64 {
        // Match the RVC SineGen sample-rate and per-frame wrapped increment.
        let final_step = (hz * self.pitch_factor / self.sample_rate) * self.upsample as f32;
        let wrapped = (final_step + 0.5).rem_euclid(1.0) - 0.5;
        (wrapped as f64).rem_euclid(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify_local_against_global(f0: &[f32], semitones: f32, start: i64, context: usize) {
        let phase = GlobalNsfPhase::new(f0, semitones, 40_000, 400).unwrap();
        let left = start - context as i64;
        let mut local = phase.phase_at(left).unwrap() as f64;
        for frame in left..(start + 100) {
            let global = phase.phase_at(frame).unwrap() as f64;
            let difference = (local - global).rem_euclid(1.0);
            let circular_error = difference.min(1.0 - difference);
            assert!(circular_error < 3e-7, "start={start} frame={frame} error={circular_error}");
            let hz = f0[reflect_index(frame, f0.len())];
            local = (local + phase.frame_advance(hz)).rem_euclid(1.0);
        }
    }

    #[test]
    fn global_and_segment_fundamental_phases_agree_for_reflected_contexts() {
        let f0: Vec<f32> = (0..2_000)
            .map(|i| if i % 17 < 3 { 0.0 } else { 110.0 + (i * 73 % 481) as f32 * 0.2 })
            .collect();
        for semitones in [-12.0, 0.0, 7.0, 12.0] {
            for start in [0, 1, 41, 88, 304, 879, 940, 1_799, 1_999] {
                verify_local_against_global(&f0, semitones, start, 40);
            }
        }
    }

    #[test]
    fn global_phase_is_periodic_for_silence_and_exact_integer_cycles() {
        let phase = GlobalNsfPhase::new(&[0.0; 1_000], 0.0, 40_000, 400).unwrap();
        assert_eq!(phase.phase_at(0).unwrap(), 0.0);
        assert_eq!(phase.phase_at(980).unwrap(), 0.0);
        assert_eq!(phase.phase_at(-40).unwrap(), 0.0);
        let phase = GlobalNsfPhase::new(&[100.0; 500], 0.0, 40_000, 400).unwrap();
        for idx in [-40, 0, 1, 100, 500, 540] {
            assert!(phase.phase_at(idx).unwrap().abs() < 1e-6);
        }
    }

    #[test]
    fn per_chunk_offsets_work_independently_of_chunk_geometry() {
        let f0: Vec<f32> = (0..8_994)
            .map(|i| 210.0 + ((i * 31) % 43) as f32)
            .collect();
        let phase = GlobalNsfPhase::new(&f0, 0.0, 40_000, 400).unwrap();
        for core in [300, 600, 900] {
            let mut start = 0;
            while start < f0.len() {
                verify_local_against_global(&f0, 0.0, start as i64, 40);
                start += core;
            }
            assert!(phase.phase_at(start.min(f0.len()) as i64).unwrap().is_finite());
        }
    }

    #[test]
    fn invalid_f0_and_pitch_are_rejected() {
        for f0 in [&[][..], &[f32::NAN][..], &[-20.0][..], &[f32::INFINITY][..]] {
            assert!(GlobalNsfPhase::new(f0, 0.0, 40_000, 400).is_err());
        }
        assert!(GlobalNsfPhase::new(&[200.0], f32::NAN, 40_000, 400).is_err());
        assert!(GlobalNsfPhase::new(&[200.0], 0.0, 0, 400).is_err());
        assert!(GlobalNsfPhase::new(&[200.0], 0.0, 40_000, 0).is_err());
    }
}
