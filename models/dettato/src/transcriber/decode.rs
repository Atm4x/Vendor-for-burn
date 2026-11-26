use crate::{PITCH_COUNT, PITCH_LO};

pub const HEAD_NAMES: [&str; 6] = ["onset", "frame", "offset", "onset2", "frame2", "offset2"];

#[derive(Clone, Debug)]
pub struct DecodeConfig {
    pub on_th: f32,
    pub v2_th: Option<f32>,
    pub off_th: f32,
    pub fr_th: f32,
    pub fr_low_frames: usize,
    pub on_nms: usize,
    pub min_dur: f64,
    pub off_min_frames: usize,
    pub hop_seconds: f64,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            on_th: 0.25,
            v2_th: None,
            off_th: 0.3,
            fr_th: 0.4,
            fr_low_frames: 2,
            on_nms: 3,
            min_dur: 0.02,
            off_min_frames: 2,
            hop_seconds: 0.02,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedNote {
    pub onset: f64,
    pub offset: f64,
    pub pitch: u8,
    pub confidence: f32,
    pub slot: u8,
}

#[derive(Clone, Debug)]
pub struct FrameMaps {
    frames: usize,
    pub heads: [Vec<f32>; 6],
}

impl FrameMaps {
    pub fn new(frames: usize, heads: [Vec<f32>; 6]) -> Result<Self, String> {
        let elements = frames.checked_mul(PITCH_COUNT).ok_or("frame map dimension overflow")?;
        if heads.iter().any(|head| head.len() != elements) {
            return Err("each of the 6 maps must have shape [frames,88]".into());
        }
        if heads.iter().flatten().any(|x| !x.is_finite() || !(0.0..=1.0).contains(x)) {
            return Err("frame probabilities must be finite and in [0,1]".into());
        }
        Ok(Self { frames, heads })
    }

    pub fn from_logits(frames: usize, logits: [Vec<f32>; 6]) -> Result<Self, String> {
        if logits.iter().flatten().any(|x| !x.is_finite()) {
            return Err("non-finite transcriber logits".into());
        }
        let heads = logits.map(|head| head.into_iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect());
        Self::new(frames, heads)
    }

    pub fn frames(&self) -> usize { self.frames }
}

fn peak_at(map: &[f32], time: usize, pitch: usize, frames: usize, threshold: f32) -> bool {
    let i = time * PITCH_COUNT + pitch;
    let current = map[i];
    let left = if time == 0 { -1.0 } else { map[i - PITCH_COUNT] };
    let right = if time + 1 == frames { -1.0 } else { map[i + PITCH_COUNT] };
    current >= threshold && current > left && current >= right
}

fn refine(map: &[f32], time: usize, pitch: usize, frames: usize) -> f64 {
    let i = time * PITCH_COUNT + pitch;
    let c = map[i] as f64;
    let a = if time > 0 { map[i - PITCH_COUNT] as f64 } else { c };
    let b = if time + 1 < frames { map[i + PITCH_COUNT] as f64 } else { c };
    let den = 2.0 * (c - a.min(b));
    if den <= 1e-6 { 0.0 } else { ((b - a) / den).clamp(-0.5, 0.5) }
}

fn decode_voice(
    maps: &FrameMaps,
    cfg: &DecodeConfig,
    slot: u8,
    onset_map: usize,
    on_th: f32,
    output: &mut Vec<DecodedNote>,
) {
    let frames = maps.frames;
    let onset = &maps.heads[onset_map];
    let frame = &maps.heads[onset_map + 1];
    let offset = &maps.heads[onset_map + 2];
    for pitch in 0..PITCH_COUNT {
        let mut onsets = (0..frames)
            .filter(|&t| peak_at(onset, t, pitch, frames, on_th))
            .collect::<Vec<_>>();
        if onsets.is_empty() { continue; }
        if cfg.on_nms > 0 && onsets.len() > 1 {
            let mut ranked = (0..onsets.len()).collect::<Vec<_>>();
            ranked.sort_by(|&a, &b| {
                onset[onsets[b] * PITCH_COUNT + pitch]
                    .total_cmp(&onset[onsets[a] * PITCH_COUNT + pitch])
            });
            let mut keep = vec![true; onsets.len()];
            for index in ranked {
                if !keep[index] { continue; }
                for j in 0..onsets.len() {
                    if j != index && onsets[index].abs_diff(onsets[j]) <= cfg.on_nms {
                        keep[j] = false;
                    }
                }
            }
            onsets = onsets.into_iter().zip(keep)
                .filter_map(|(t, keep)| keep.then_some(t)).collect();
        }
        let offsets = (0..frames)
            .filter(|&t| peak_at(offset, t, pitch, frames, cfg.off_th))
            .collect::<Vec<_>>();
        let lows = (0..frames).filter(|&t| {
            (0..cfg.fr_low_frames)
                .all(|k| t + k >= frames || frame[(t + k) * PITCH_COUNT + pitch] < cfg.fr_th)
        }).collect::<Vec<_>>();
        for (i, &t0) in onsets.iter().enumerate() {
            let u0 = t0 as f64 + refine(onset, t0, pitch, frames);
            let next = onsets.get(i + 1).copied().unwrap_or(frames);
            let first_low = lows.partition_point(|&t| t < t0 + 1);
            let end_frame = lows.get(first_low).copied().unwrap_or(frames);
            let first_off = offsets.partition_point(|&t| t < t0.saturating_add(cfg.off_min_frames));
            let end_offset = offsets.get(first_off).copied().unwrap_or(frames);
            let u1 = if end_offset < frames && end_offset <= end_frame.min(next) {
                end_offset as f64 + refine(offset, end_offset, pitch, frames)
            } else if end_frame <= next {
                end_frame as f64 - 0.5
            } else {
                next as f64 + if next < frames { refine(onset, next, pitch, frames) } else { 0.0 }
            };
            let on_sec = u0.max(0.0) * cfg.hop_seconds;
            let off_sec = (u1 * cfg.hop_seconds).max(on_sec + cfg.min_dur);
            let frame_index = ((on_sec / cfg.hop_seconds).round() as isize)
                .clamp(0, frames as isize - 1) as usize;
            output.push(DecodedNote {
                onset: on_sec,
                offset: off_sec,
                pitch: PITCH_LO + pitch as u8,
                confidence: onset[frame_index * PITCH_COUNT + pitch],
                slot,
            });
        }
    }
}

pub fn decode(maps: &FrameMaps, cfg: &DecodeConfig) -> Result<Vec<DecodedNote>, String> {
    if !cfg.hop_seconds.is_finite() || cfg.hop_seconds <= 0.0
        || !cfg.min_dur.is_finite() || cfg.min_dur <= 0.0
        || cfg.fr_low_frames == 0
    {
        return Err("invalid transcriber decoder configuration".into());
    }
    if maps.frames == 0 { return Ok(Vec::new()); }
    let mut notes = Vec::new();
    decode_voice(maps, cfg, 1, 0, cfg.on_th, &mut notes);
    if let Some(th) = cfg.v2_th {
        decode_voice(maps, cfg, 2, 3, th, &mut notes);
    }
    notes.sort_by(|a, b| {
        a.onset.total_cmp(&b.onset)
            .then_with(|| a.offset.total_cmp(&b.offset))
            .then_with(|| a.pitch.cmp(&b.pitch))
    });
    Ok(notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps(n: usize) -> FrameMaps {
        FrameMaps::new(n, std::array::from_fn(|_| vec![0.0; n * PITCH_COUNT])).unwrap()
    }

    #[test]
    fn decodes_both_voice_slots_without_merging_them() {
        let mut m = maps(12);
        let p = 60 - PITCH_LO as usize;
        for (o, f, x) in [(0, 1, 2), (3, 4, 5)] {
            m.heads[o][3 * PITCH_COUNT + p] = 0.9;
            m.heads[x][8 * PITCH_COUNT + p] = 0.9;
            for t in 3..=8 { m.heads[f][t * PITCH_COUNT + p] = 0.95; }
        }
        let cfg = DecodeConfig { v2_th: Some(0.3), ..Default::default() };
        let notes = decode(&m, &cfg).unwrap();
        assert_eq!(notes.len(), 2);
        assert_eq!((notes[0].slot, notes[1].slot), (1, 2));
        assert_eq!(notes[0].pitch, 60);
        assert_eq!(notes[1].pitch, 60);
        assert!(notes.iter().all(|n| n.offset > n.onset));
    }

    #[test]
    fn suppressed_onset_and_frame_end_follow_python_decoder() {
        let mut m = maps(25);
        let p = 40;
        m.heads[0][4 * PITCH_COUNT + p] = 0.8;
        m.heads[0][6 * PITCH_COUNT + p] = 0.7;
        m.heads[0][15 * PITCH_COUNT + p] = 0.9;
        for t in 4..12 { m.heads[1][t * PITCH_COUNT + p] = 0.9; }
        let notes = decode(&m, &DecodeConfig::default()).unwrap();
        assert_eq!(notes.len(), 2);
        assert!((notes[0].onset - 0.08).abs() <= 0.011);
        assert!((notes[0].offset - 0.23).abs() < 0.02);
    }

    #[test]
    fn first_frame_peak_cannot_start_before_the_audio() {
        let mut m = maps(8);
        m.heads[0][0] = 0.9;
        m.heads[0][PITCH_COUNT] = 0.2;
        let notes = decode(&m, &DecodeConfig::default()).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].onset, 0.0);
        assert!(notes[0].offset > notes[0].onset);
        assert!(crate::evidence::features::note_frames(&notes, 8, 0.02).is_ok());
    }

    #[test]
    fn shape_and_probability_validation() {
        assert!(FrameMaps::new(2, std::array::from_fn(|_| vec![0.0; 1])).is_err());
        assert!(FrameMaps::from_logits(1, std::array::from_fn(|_| vec![f32::NAN; 88])).is_err());
        assert!(decode(&maps(0), &DecodeConfig::default()).unwrap().is_empty());
    }
}
