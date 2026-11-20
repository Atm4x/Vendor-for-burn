use crate::Note;

pub const DRUM_SAMPLE_RATE: u32 = 44_100;
pub const MIN_SOURCE_SAMPLE_RATE: u32 = 40_000;
pub const DRUM_FPS: f64 = 50.0;
pub const DRUM_KEYS: usize = 61;
pub const TIMPANI_KEYS: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrumModel { Kit, Orchestral }

#[derive(Clone, Copy, Debug)]
pub struct DrumThresholds {
    pub kit: f32,
    pub orchestral: f32,
    pub timpani: f32,
    pub router: f32,
}

impl DrumThresholds {
    pub fn validate(self) -> Result<(), String> {
        for p in [self.kit, self.orchestral, self.timpani, self.router] {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err("drum probabilities require thresholds in [0,1]".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DrumProbabilities {
    pub router: f32,
    pub frames: usize,
    pub kit: Vec<f32>,
    pub kit_velocity: Vec<f32>,
    pub orchestral: Vec<f32>,
    pub timpani: Vec<f32>,
    pub orchestral_velocity: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DrumOutput {
    pub model: DrumModel,
    pub router: f32,
    pub hits: Vec<Note>,
}

pub fn peaks(probs: &[f32], frames: usize, lanes: usize, threshold: f32, gap: usize)
    -> Result<Vec<(usize, usize)>, String>
{
    if lanes == 0 || frames.checked_mul(lanes) != Some(probs.len())
        || probs.iter().any(|p| !p.is_finite()) || !threshold.is_finite()
    { return Err("invalid drum onset probabilities".into()); }
    let mut last = vec![None::<usize>; lanes];
    let mut found = Vec::new();
    for t in 0..frames {
        for lane in 0..lanes {
            let val = probs[t * lanes + lane];
            let left = probs[t.saturating_sub(1) * lanes + lane];
            let right = probs[(t + 1).min(frames - 1) * lanes + lane];
            if val < threshold || val < left || val < right { continue; }
            if last[lane].is_some_and(|previous| t - previous < gap) { continue; }
            found.push((t, lane));
            last[lane] = Some(t);
        }
    }
    Ok(found)
}

fn hit(time: usize, pitch: u8, velocity: f32) -> Result<Note, String> {
    let on = (time as f64 / DRUM_FPS * 10_000.0).round_ties_even() / 10_000.0;
    let off = ((on + 0.1) * 10_000.0).round_ties_even() / 10_000.0;
    let v = velocity.clamp(0.05, 1.0);
    let strength = (f64::from(v) * 127.0).round_ties_even().clamp(1.0, 127.0) as u8;
    Note::new(on, off, pitch, strength)
}

pub fn decode(probs: &DrumProbabilities, thresholds: DrumThresholds) -> Result<DrumOutput, String> {
    thresholds.validate()?;
    if !probs.router.is_finite() || probs.frames == 0
        || probs.kit.len() != probs.frames * DRUM_KEYS
        || probs.kit_velocity.len() != probs.frames * DRUM_KEYS
        || probs.orchestral.len() != probs.frames * DRUM_KEYS
        || probs.timpani.len() != probs.frames * TIMPANI_KEYS
        || probs.orchestral_velocity.len() != probs.frames * DRUM_KEYS
    { return Err("invalid Dettato drum probabilities".into()); }
    let model = if probs.router < thresholds.router { DrumModel::Kit } else { DrumModel::Orchestral };
    let mut hits = Vec::new();
    match model {
        DrumModel::Kit => {
            for (time, lane) in peaks(&probs.kit, probs.frames, DRUM_KEYS, thresholds.kit, 3)? {
                hits.push(hit(time, 27 + lane as u8, probs.kit_velocity[time * DRUM_KEYS + lane])?);
            }
        }
        DrumModel::Orchestral => {
            for (time, lane) in peaks(&probs.orchestral, probs.frames, DRUM_KEYS, thresholds.orchestral, 3)? {
                hits.push(hit(time, 27 + lane as u8, probs.orchestral_velocity[time * DRUM_KEYS + lane])?);
            }
            for (time, lane) in peaks(&probs.timpani, probs.frames, TIMPANI_KEYS, thresholds.timpani, 3)? {
                let pitch = 41 + lane as u8;
                let onset = time as f64 / DRUM_FPS;
                if hits.iter().any(|x| x.pitch == pitch && (x.onset - onset).abs() <= 0.04) {
                    continue;
                }
                let strength = probs.timpani[time * TIMPANI_KEYS + lane].clamp(0.3, 1.0);
                hits.push(hit(time, pitch, strength)?);
            }
            hits.sort_by(|a, b| a.onset.total_cmp(&b.onset));
        }
    }
    Ok(DrumOutput { model, router: probs.router, hits })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn empty(frames: usize) -> DrumProbabilities {
        DrumProbabilities {
            router: 0.1, frames,
            kit: vec![0.0; frames * DRUM_KEYS],
            kit_velocity: vec![0.8; frames * DRUM_KEYS],
            orchestral: vec![0.0; frames * DRUM_KEYS],
            timpani: vec![0.0; frames * TIMPANI_KEYS],
            orchestral_velocity: vec![0.7; frames * DRUM_KEYS],
        }
    }
    fn th() -> DrumThresholds {
        DrumThresholds { kit: 0.5, orchestral: 0.5, timpani: 0.5, router: 0.5 }
    }
    #[test]
    fn kit_hit_has_correct_general_midi_pitch_and_duration() {
        let mut p = empty(16);
        p.kit[5 * DRUM_KEYS + (36 - 27)] = 0.9;
        let got = decode(&p, th()).unwrap();
        assert_eq!(got.model, DrumModel::Kit);
        assert_eq!(got.hits.len(), 1);
        assert_eq!(got.hits[0].pitch, 36);
        assert!((got.hits[0].offset - got.hits[0].onset - 0.1).abs() < 1e-7);
    }
    #[test]
    fn orchestral_timpani_near_duplicate_is_rejected() {
        let mut p = empty(16);
        p.router = 0.6;
        p.orchestral[5 * DRUM_KEYS + (41 - 27)] = 0.8;
        p.timpani[6 * TIMPANI_KEYS] = 0.95;
        let got = decode(&p, th()).unwrap();
        assert_eq!(got.model, DrumModel::Orchestral);
        assert_eq!(got.hits.len(), 1);
    }
    #[test]
    fn greedy_peaks_respect_three_frame_spacing() {
        let mut p = vec![0.0; 10];
        p[2] = 0.7; p[4] = 0.9; p[5] = 0.0; p[7] = 0.9;
        assert_eq!(peaks(&p, 10, 1, 0.5, 3).unwrap(), [(2, 0), (7, 0)]);
    }
}
