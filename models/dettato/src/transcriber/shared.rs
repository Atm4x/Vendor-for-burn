use super::decode::{decode, DecodeConfig, DecodedNote, FrameMaps};
use crate::{PITCH_COUNT, PITCH_LO, evidence::features::{self, EVIDENCE_FEATURES, HIDDEN_CHANNELS}};

const HEAD_COUNT: usize = 6;
const COARSE_TAP_COUNT: usize = 3;
const CHANNELS: usize = HIDDEN_CHANNELS;

pub(crate) struct SharedRun {
    maps: FrameMaps,
    coarse_maps: FrameMaps,
    hidden: Vec<f32>,
    coarse_hidden: Vec<f32>,
    coarse_stem: Vec<f32>,
}

pub(crate) struct SharedAccumulator {
    frames: usize,
    window_frames: usize,
    fine_heads: [Vec<f32>; HEAD_COUNT],
    coarse_heads: [Vec<f32>; HEAD_COUNT],
    fine_weight: Vec<f32>,
    coarse_weight: Vec<f32>,
    hidden: Vec<f32>,
    coarse_hidden: Vec<f32>,
    coarse_stem: Vec<f32>,
}

impl SharedAccumulator {
    pub(crate) fn new(frames: usize, window_frames: usize) -> Result<Self, String> {
        if frames == 0 || window_frames == 0 { return Err("invalid shared Transcriber frame grid".into()); }
        let map_len = frames.checked_mul(PITCH_COUNT).ok_or("shared map size overflow")?;
        let tap_len = map_len.checked_mul(CHANNELS).ok_or("shared tap size overflow")?;
        Ok(Self {
            frames,
            window_frames,
            fine_heads: std::array::from_fn(|_| vec![0.0; map_len]),
            coarse_heads: std::array::from_fn(|_| vec![0.0; map_len]),
            fine_weight: vec![0.0; frames],
            coarse_weight: vec![0.0; frames],
            hidden: vec![0.0; tap_len],
            coarse_hidden: vec![0.0; tap_len],
            coarse_stem: vec![0.0; tap_len],
        })
    }

    pub(crate) fn add(
        &mut self,
        first_frame: isize,
        coarse: bool,
        maps: &FrameMaps,
        hidden: &[f32],
        stem: &[f32],
        stem_lanes: usize,
    ) -> Result<(), String> {
        let input_len = self.window_frames.checked_mul(PITCH_COUNT)
            .and_then(|v| v.checked_mul(CHANNELS)).ok_or("shared input size overflow")?;
        let stem_len = self.window_frames.checked_mul(stem_lanes)
            .and_then(|v| v.checked_mul(CHANNELS)).ok_or("shared stem size overflow")?;
        if maps.frames() != self.window_frames || hidden.len() != input_len
            || stem_lanes < PITCH_COUNT || stem.len() != stem_len
        { return Err("incompatible shared Transcriber window shapes".into()); }
        for local in 0..self.window_frames {
            let global = first_frame + local as isize;
            if global < 0 || global >= self.frames as isize { continue; }
            let global = global as usize;
            let w = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * (local + 1) as f64
                / (self.window_frames + 1) as f64).cos()) as f32;
            self.fine_weight[global] += w;
            if coarse { self.coarse_weight[global] += w; }
            for head in 0..HEAD_COUNT {
                for pitch in 0..PITCH_COUNT {
                    let from = local * PITCH_COUNT + pitch;
                    let to = global * PITCH_COUNT + pitch;
                    self.fine_heads[head][to] += maps.heads[head][from] * w;
                    if coarse { self.coarse_heads[head][to] += maps.heads[head][from] * w; }
                }
            }
            for channel in 0..CHANNELS {
                let source_base = (channel * self.window_frames + local) * PITCH_COUNT;
                let stem_base = (channel * self.window_frames + local) * stem_lanes;
                for pitch in 0..PITCH_COUNT {
                    let to = (global * PITCH_COUNT + pitch) * CHANNELS + channel;
                    self.hidden[to] += hidden[source_base + pitch] * w;
                    if coarse {
                        self.coarse_hidden[to] += hidden[source_base + pitch] * w;
                        self.coarse_stem[to] += stem[stem_base + pitch] * w;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<SharedRun, String> {
        if self.fine_weight.iter().any(|&w| w <= 0.0)
            || self.coarse_weight.iter().any(|&w| w <= 0.0)
        { return Err("shared Transcriber windows failed to cover all frames".into()); }
        for frame in 0..self.frames {
            let wf = self.fine_weight[frame].max(1e-6);
            let wc = self.coarse_weight[frame].max(1e-6);
            for head in 0..HEAD_COUNT {
                for pitch in 0..PITCH_COUNT {
                    let i = frame * PITCH_COUNT + pitch;
                    self.fine_heads[head][i] /= wf;
                    self.coarse_heads[head][i] /= wc;
                }
            }
            for pitch in 0..PITCH_COUNT {
                let i = (frame * PITCH_COUNT + pitch) * CHANNELS;
                for channel in 0..CHANNELS {
                    self.hidden[i + channel] /= wf;
                    self.coarse_hidden[i + channel] /= wc;
                    self.coarse_stem[i + channel] /= wc;
                }
            }
        }
        Ok(SharedRun {
            maps: FrameMaps::new(self.frames, self.fine_heads)?,
            coarse_maps: FrameMaps::new(self.frames, self.coarse_heads)?,
            hidden: self.hidden,
            coarse_hidden: self.coarse_hidden,
            coarse_stem: self.coarse_stem,
        })
    }
}

impl SharedRun {
    pub(crate) fn notes_with_evidence(&self, decoder: &DecodeConfig)
        -> Result<(Vec<DecodedNote>, Vec<[f32; EVIDENCE_FEATURES]>), String>
    {
        let notes = decode(&self.maps, decoder)?;
        let evidence = features::evidence_from_taps(
            &self.maps, &self.hidden, &notes, decoder.hop_seconds,
        )?;
        Ok((notes, evidence))
    }

    pub(crate) fn grouper_features(&self, notes: &[DecodedNote], decoder: &DecodeConfig)
        -> Result<(Vec<[f32; 12]>, Vec<[[f32; CHANNELS]; COARSE_TAP_COUNT]>), String>
    {
        let decoded = decode(&self.coarse_maps, decoder)?;
        let (meta, intervals) = grouper_metadata(&self.coarse_maps, &decoded, notes, decoder.hop_seconds)?;
        let mut taps = Vec::with_capacity(intervals.len());
        for (t0, end, pitch) in intervals {
            let mut v = [[0.0f32; CHANNELS]; COARSE_TAP_COUNT];
            for (tap, limit, source) in [
                (0usize, t0+1, &self.coarse_hidden),
                (1usize, end.min(t0+50), &self.coarse_hidden),
                (2usize, end.min(t0+10), &self.coarse_stem),
            ] {
                let n = limit - t0;
                for time in t0..limit {
                    let start = (time * PITCH_COUNT + pitch) * CHANNELS;
                    for (out, &value) in v[tap].iter_mut().zip(&source[start..start + CHANNELS]) {
                        *out += value / n as f32;
                    }
                }
            }
            taps.push(v);
        }
        Ok((meta, taps))
    }
}

pub(crate) fn grouper_metadata(
    maps: &FrameMaps, decoded: &[DecodedNote], notes: &[DecodedNote], hop: f64,
) -> Result<(Vec<[f32; 12]>, Vec<(usize, usize, usize)>), String> {
    let total = maps.frames();
    if total == 0 || !hop.is_finite() || hop <= 0.0 {
        return Err("invalid Grouper frame grid".into());
    }
    let mut sorted = notes.to_vec();
    sorted.sort_by(|a,b| a.onset.total_cmp(&b.onset).then_with(|| a.pitch.cmp(&b.pitch)));
    let mut meta = Vec::with_capacity(sorted.len());
    let mut intervals = Vec::with_capacity(sorted.len());
    for note in sorted {
        if !note.onset.is_finite() || !note.offset.is_finite()
            || note.onset < 0.0 || note.offset <= note.onset
        { return Err("invalid Grouper note".into()); }
        let pitch = (note.pitch as isize - PITCH_LO as isize)
            .clamp(0, PITCH_COUNT as isize - 1) as usize;
        let t0 = ((note.onset / hop).round() as isize)
            .clamp(0, total as isize - 1) as usize;
        let end = ((note.offset.max(note.onset + 0.02) / hop).ceil() as isize)
            .clamp(t0 as isize + 1, total as isize) as usize;
        let slot = decoded.iter().filter(|n| n.pitch == note.pitch)
            .min_by(|a,b| (a.onset - note.onset).abs().total_cmp(&(b.onset - note.onset).abs()))
            .filter(|n| (n.onset - note.onset).abs() <= 0.1)
            .map(|n| f32::from(n.slot.saturating_sub(1))).unwrap_or(0.0);
        let mut row = [0.0f32; 12];
        row[..4].copy_from_slice(&[
            note.onset as f32, note.offset.max(note.onset + 0.02) as f32,
            f32::from(note.pitch), slot,
        ]);
        for (voice, base) in [0usize, 3].iter().enumerate() {
            row[4 + voice] = (t0.saturating_sub(1)..=(t0+1).min(total-1))
                .map(|t| maps.heads[*base][t*PITCH_COUNT + pitch])
                .fold(0.0f32, f32::max);
            row[6 + voice] = maps.heads[*base+1][t0*PITCH_COUNT + pitch];
            row[8 + voice] = (t0..end).map(|t| maps.heads[*base+1][t*PITCH_COUNT + pitch])
                .sum::<f32>() / (end - t0) as f32;
            row[10 + voice] = ((end-1).saturating_sub(1)..=end.min(total-1))
                .map(|t| maps.heads[*base+2][t*PITCH_COUNT + pitch])
                .fold(0.0f32, f32::max);
        }
        meta.push(row);
        intervals.push((t0,end,pitch));
    }
    Ok((meta, intervals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fine_and_coarse_hann_accumulators_preserve_window_selection() {
        let mut sum = SharedAccumulator::new(2, 2).unwrap();
        let input = |value| FrameMaps::new(2, std::array::from_fn(|_| vec![value; 2*PITCH_COUNT])).unwrap();
        let tap = |value| vec![value; 2*PITCH_COUNT*CHANNELS];
        sum.add(0, true, &input(0.2), &tap(2.0), &tap(3.0), PITCH_COUNT).unwrap();
        sum.add(0, false, &input(0.8), &tap(8.0), &tap(9.0), PITCH_COUNT).unwrap();
        let result = sum.finish().unwrap();
        assert!((result.maps.heads[0][0] - 0.5).abs() < 1e-6);
        assert!((result.coarse_maps.heads[0][0] - 0.2).abs() < 1e-6);
        assert!((result.hidden[0] - 5.0).abs() < 1e-6);
        assert!((result.coarse_hidden[0] - 2.0).abs() < 1e-6);
        assert!((result.coarse_stem[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn grouper_metadata_clamps_spectral_lane_but_preserves_midi_pitch() {
        let maps = FrameMaps::new(4, std::array::from_fn(|_| vec![0.5; 4 * PITCH_COUNT])).unwrap();
        let notes = [
            DecodedNote {onset: 0.02, offset: 0.06, pitch: 4, confidence: 0.5, slot: 1},
            DecodedNote {onset: 0.03, offset: 0.07, pitch: 127, confidence: 0.5, slot: 1},
        ];
        let (meta, intervals) = grouper_metadata(&maps, &[], &notes, 0.02).unwrap();
        assert_eq!(meta[0][2], 4.0);
        assert_eq!(meta[1][2], 127.0);
        assert_eq!(intervals[0].2, 0);
        assert_eq!(intervals[1].2, PITCH_COUNT - 1);
    }

    #[test]
    fn shared_grouper_features_preserve_onset_and_span_taps() {
        let mut sum = SharedAccumulator::new(4, 4).unwrap();
        let input = FrameMaps::new(4, std::array::from_fn(|_| vec![0.5; 4*PITCH_COUNT])).unwrap();
        let taps = vec![2.0f32; 4*PITCH_COUNT*CHANNELS];
        sum.add(0, true, &input, &taps, &taps, PITCH_COUNT).unwrap();
        let result = sum.finish().unwrap();
        let notes = [DecodedNote {onset: 0.02, offset: 0.08, pitch: 60, confidence: 0.5, slot: 1}];
        let (meta, taps) = result.grouper_features(&notes, &DecodeConfig::default()).unwrap();
        assert_eq!(meta.len(), 1);
        assert!((taps[0][0][0] - 2.0).abs() < 1e-6);
        assert!((taps[0][1][0] - 2.0).abs() < 1e-6);
        assert!((taps[0][2][0] - 2.0).abs() < 1e-6);
    }
}
