use crate::{PITCH_COUNT, PITCH_LO, transcriber::{DecodedNote, FrameMaps}};

pub const HIDDEN_CHANNELS: usize = 368;
pub const MAP_FEATURES: usize = 31;
pub const EVIDENCE_FEATURES: usize = MAP_FEATURES + 2 + 2 * HIDDEN_CHANNELS;
pub const QUANT_SCALE: f32 = 32.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoteFrames {
    pub pitch_index: usize,
    pub start: usize,
    pub end: usize,
    pub slot: u8,
}

pub fn note_frames(notes: &[DecodedNote], frames: usize, hop: f64) -> Result<Vec<NoteFrames>, String> {
    if frames == 0 || !hop.is_finite() || hop <= 0.0 {
        return Err("invalid evidence frame grid".into());
    }
    notes.iter().map(|note| {
        if note.pitch < PITCH_LO || usize::from(note.pitch - PITCH_LO) >= PITCH_COUNT
            || !note.onset.is_finite() || !note.offset.is_finite()
            || note.onset < 0.0 || note.offset <= note.onset || !matches!(note.slot, 1 | 2)
        {
            return Err("invalid evidence note".into());
        }
        let start = ((note.onset / hop).round() as isize).clamp(0, frames as isize - 1) as usize;
        let end = ((note.offset / hop).round() as isize).clamp(start as isize + 1, frames as isize) as usize;
        Ok(NoteFrames { pitch_index: usize::from(note.pitch - PITCH_LO), start, end, slot: note.slot })
    }).collect()
}

fn sample(map: &[f32], frames: usize, t: isize, pitch: usize) -> f32 {
    let t = t.clamp(0, frames as isize - 1) as usize;
    map[t * PITCH_COUNT + pitch]
}

pub fn map_evidence(maps: &FrameMaps, notes: &[NoteFrames]) -> Result<Vec<[f32; MAP_FEATURES]>, String> {
    let frames = maps.frames();
    if frames == 0 { return Err("empty evidence maps".into()); }
    let mut result = Vec::with_capacity(notes.len());
    for note in notes {
        if note.pitch_index >= PITCH_COUNT || note.start >= frames || note.end > frames
            || note.end <= note.start || !matches!(note.slot, 1 | 2)
        { return Err("invalid evidence frame interval".into()); }
        let own = if note.slot == 1 { 0 } else { 3 };
        let other = if note.slot == 1 { 3 } else { 0 };
        let (l, a, b) = (note.pitch_index, note.start as isize, note.end as isize);
        let span = (b - a).max(1);
        let (on, fr, off) = (&maps.heads[own], &maps.heads[own + 1], &maps.heads[own + 2]);
        let (o_on, o_fr) = (&maps.heads[other], &maps.heads[other + 1]);
        let mut row = [0.0f32; MAP_FEATURES];
        let mut i = 0usize;
        for offset in [-2, -1, 0, 1, 2] {
            row[i] = sample(on, frames, a + offset, l); i += 1;
        }
        for j in 0..8 {
            row[i] = sample(fr, frames, a + (j * span) / 8, l); i += 1;
        }
        for offset in [1, 3, 6, 12] {
            row[i] = sample(fr, frames, b - 1 + offset, l); i += 1;
        }
        row[i] = (1..=5).filter(|&k| a - k >= 0)
            .map(|k| sample(fr, frames, a - k, l)).fold(0.0f32, f32::max); i += 1;
        let offset_start = (a + 1) as usize;
        let offset_end = (b + 12).min(frames as isize) as usize;
        if offset_start < offset_end {
            let (rel, maximum) = (offset_start..offset_end)
                .map(|t| (t - offset_start, off[t * PITCH_COUNT + l]))
                .max_by(|(ia, va), (ib, vb)| va.total_cmp(vb).then_with(|| ib.cmp(ia)))
                .unwrap();
            row[i] = maximum;
            row[i + 1] = ((rel + 1) as f32 / span as f32).min(2.0);
        }
        i += 2;
        row[i] = sample(off, frames, b - 1, l); i += 1;
        row[i] = [-1, 0, 1].into_iter()
            .map(|d| sample(o_on, frames, a + d, l)).fold(0.0f32, f32::max); i += 1;
        row[i] = sample(o_fr, frames, a, l); i += 1;
        row[i] = (0..4).map(|j| sample(o_fr, frames, a + (j * span) / 4, l))
            .sum::<f32>() / 4.0; i += 1;
        row[i] = if note.slot == 2 { 1.0 } else { 0.0 }; i += 1;
        for offset in [-1, 1, -12, 12, 19, 24] {
            let nearby = l as isize + offset;
            row[i] = if (0..PITCH_COUNT as isize).contains(&nearby) {
                [-1, 0, 1].into_iter().map(|d| sample(&maps.heads[0], frames, a + d, nearby as usize))
                    .fold(0.0f32, f32::max)
            } else { 0.0 };
            i += 1;
        }
        debug_assert_eq!(i, MAP_FEATURES);
        result.push(row);
    }
    Ok(result)
}

pub fn gather_hidden(
    hidden: &[f32], frames: usize, channels: usize, notes: &[NoteFrames]
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    if channels == 0 || hidden.len() != frames.checked_mul(PITCH_COUNT)
        .and_then(|n| n.checked_mul(channels)).ok_or("hidden size overflow")?
    { return Err("hidden state must have shape [frames, 88, channels]".into()); }
    let mut at = Vec::with_capacity(notes.len());
    let mut mean = Vec::with_capacity(notes.len());
    for note in notes {
        if note.pitch_index >= PITCH_COUNT || note.start >= frames
            || note.end > frames || note.start >= note.end
        { return Err("hidden note interval outside song".into()); }
        let offset = (note.start * PITCH_COUNT + note.pitch_index) * channels;
        at.push(hidden[offset..offset + channels].to_vec());
        let mut avg = vec![0.0f64; channels];
        for t in note.start..note.end {
            let offset = (t * PITCH_COUNT + note.pitch_index) * channels;
            for (a, &x) in avg.iter_mut().zip(&hidden[offset..offset + channels]) {
                *a += f64::from(x);
            }
        }
        let n = (note.end - note.start) as f64;
        mean.push(avg.into_iter().map(|x| (x / n) as f32).collect());
    }
    Ok((at, mean))
}

fn standardise_and_quantise(x: &[f32]) -> Result<(Vec<f32>, f32), String> {
    if x.len() != HIDDEN_CHANNELS || x.iter().any(|v| !v.is_finite()) {
        return Err("evidence hidden vector must contain 368 finite values".into());
    }
    let mean = x.iter().copied().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let sd = var.sqrt() + 1e-6;
    let quant = x.iter().map(|&v| {
        let z = ((v - mean) / sd).clamp(-3.96, 3.96);
        ((z * QUANT_SCALE) as f64).round_ties_even() as f32 / QUANT_SCALE
    }).collect();
    Ok((quant, sd.ln()))
}

pub fn assemble(
    maps: &[[f32; MAP_FEATURES]], at: &[Vec<f32>], mean: &[Vec<f32>]
) -> Result<Vec<[f32; EVIDENCE_FEATURES]>, String> {
    if maps.len() != at.len() || maps.len() != mean.len() {
        return Err("evidence rows must align with notes".into());
    }
    let mut out = Vec::with_capacity(maps.len());
    for ((row, a), b) in maps.iter().zip(at).zip(mean) {
        let (qa, la) = standardise_and_quantise(a)?;
        let (qb, lb) = standardise_and_quantise(b)?;
        let mut joined = [0.0f32; EVIDENCE_FEATURES];
        for i in 0..MAP_FEATURES { joined[i] = half::f16::from_f32(row[i]).to_f32(); }
        joined[MAP_FEATURES] = half::f16::from_f32(la).to_f32();
        joined[MAP_FEATURES + 1] = half::f16::from_f32(lb).to_f32();
        joined[MAP_FEATURES + 2..MAP_FEATURES + 2 + HIDDEN_CHANNELS].copy_from_slice(&qa);
        joined[MAP_FEATURES + 2 + HIDDEN_CHANNELS..].copy_from_slice(&qb);
        out.push(joined);
    }
    Ok(out)
}

pub fn evidence_from_taps(
    maps: &FrameMaps, hidden: &[f32], notes: &[DecodedNote], hop: f64,
) -> Result<Vec<[f32; EVIDENCE_FEATURES]>, String> {
    let refs = note_frames(notes, maps.frames(), hop)?;
    let mf = map_evidence(maps, &refs)?;
    let (at, mean) = gather_hidden(hidden, maps.frames(), HIDDEN_CHANNELS, &refs)?;
    assemble(&mf, &at, &mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps() -> FrameMaps {
        let heads = std::array::from_fn(|h| (0..(28 * PITCH_COUNT))
            .map(|idx| ((idx / PITCH_COUNT * (h + 1) + idx % PITCH_COUNT) % 97) as f32 / 100.0)
            .collect());
        FrameMaps::new(28, heads).unwrap()
    }

    #[test]
    fn feature_shape_and_boundary_clamping() {
        let note = NoteFrames { pitch_index: 0, start: 0, end: 3, slot: 2 };
        let features = map_evidence(&maps(), &[note]).unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].len(), 31);
        assert_eq!(features[0][24], 1.0);
        assert_eq!(features[0][25], 0.0);
        assert_eq!(features[0][27], 0.0);
    }

    #[test]
    fn zero_hidden_vectors_produce_finite_evidence() {
        let note = NoteFrames { pitch_index: 5, start: 1, end: 4, slot: 1 };
        let h = vec![0.0; 28 * PITCH_COUNT * HIDDEN_CHANNELS];
        let (at, mean) = gather_hidden(&h, 28, HIDDEN_CHANNELS, &[note]).unwrap();
        let assembled = assemble(&map_evidence(&maps(), &[note]).unwrap(), &at, &mean).unwrap();
        assert_eq!(assembled[0].len(), EVIDENCE_FEATURES);
        assert!(assembled[0].iter().all(|x| x.is_finite()));
        assert!(assembled[0][33..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn map_and_quantization_match_original_python_reference() {
        let reference: serde_json::Value = serde_json::from_str(
            include_str!("../../tests/fixtures/evidence_reference.json")
        ).unwrap();
        let maps = maps_for_reference();
        let frames = reference["note_frames"].as_array().unwrap().iter().map(|f| NoteFrames {
            pitch_index: f["pitch_index"].as_u64().unwrap() as usize,
            start: f["start"].as_u64().unwrap() as usize,
            end: f["end"].as_u64().unwrap() as usize,
            slot: f["slot"].as_u64().unwrap() as u8,
        }).collect::<Vec<_>>();
        let mf = map_evidence(&maps, &frames).unwrap();
        for (row, expected) in mf.iter().zip(reference["map_features"].as_array().unwrap()) {
            for (&value, expected) in row.iter().zip(expected.as_array().unwrap()) {
                assert!((f64::from(value) - expected.as_f64().unwrap()).abs() < 1e-5);
            }
        }
        let vectors = |key: &str| reference[key].as_array().unwrap().iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let evidence = assemble(&mf, &vectors("at"), &vectors("mean")).unwrap();
        for (row, expected) in evidence.iter().zip(reference["expected_evidence"].as_array().unwrap()) {
            for (&value, expected) in row.iter().zip(expected.as_array().unwrap()) {
                assert!((f64::from(value) - expected.as_f64().unwrap()).abs() < 0.034,
                    "mismatch in quantized evidence: {value} vs {expected}");
            }
        }
    }

    fn maps_for_reference() -> FrameMaps {
        let heads = std::array::from_fn(|h| (0..(35 * PITCH_COUNT))
            .map(|idx| (((idx / PITCH_COUNT) * (h + 1) + (idx % PITCH_COUNT) * 3 + h) % 97) as f32 / 100.0)
            .collect::<Vec<_>>());
        FrameMaps::new(35, heads).unwrap()
    }
}
