use super::repair::{RepairCandidate, RepairEdit, RepairInput, RepairKind, WINDOW_SECONDS, MAX_WINDOW_NOTES};
use super::features::EVIDENCE_FEATURES;

pub const ONSET_BINS: usize = 201;
pub const DURATION_RATIO_BINS: usize = 161;
pub const INSERTION_TIME_BINS: usize = 401;
pub const INSERTION_PITCH_BINS: usize = 49;
pub const ABSOLUTE_DURATION_BINS: usize = 128;
pub const INSERTION_SLOTS: usize = 2;

#[derive(Clone, Debug)]
pub struct InsertionLogits {
    pub exists: f32,
    pub pitch: Vec<f32>,
    pub onset: Vec<f32>,
    pub duration: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct NoteLogits {
    pub keep: f32,
    pub onset: Vec<f32>,
    pub duration_ratio: Vec<f32>,
    pub insertion: [InsertionLogits; INSERTION_SLOTS],
}

#[derive(Clone, Debug)]
pub struct CandidateLogits {
    pub notes: Vec<NoteLogits>,
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn probabilities(logits: &[f32]) -> Result<Vec<f32>, String> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err("non-finite or empty Evidence logits".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits.iter().map(|&x| (x - max).exp()).collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    Ok(exps.into_iter().map(|x| x / sum).collect())
}

pub fn box_argmax(probs: &[f32], half: usize) -> usize {
    let mut prefix = vec![0.0f64; probs.len() + 1];
    for (i, &x) in probs.iter().enumerate() { prefix[i + 1] = prefix[i] + f64::from(x); }
    (0..probs.len()).max_by(|&i, &j| {
        let score = |k: usize| prefix[(k + half + 1).min(probs.len())] - prefix[k.saturating_sub(half)];
        score(i).total_cmp(&score(j)).then_with(|| j.cmp(&i))
    }).unwrap_or(0)
}

pub fn duration_centres() -> [f32; ABSOLUTE_DURATION_BINS] {
    let ln_lo = 0.01f64.ln();
    let ln_hi = 16.0f64.ln();
    std::array::from_fn(|i| {
        ((ln_lo + (ln_hi - ln_lo) * (i as f64 + 0.5) / ABSOLUTE_DURATION_BINS as f64).exp()) as f32
    })
}

pub fn decode_candidates(
    notes: &[RepairInput], output: &CandidateLogits, window_start: f64,
) -> Result<(Vec<RepairEdit>, Vec<RepairCandidate>), String> {
    if notes.len() != output.notes.len() || !window_start.is_finite() {
        return Err("Evidence output/note batch mismatch".into());
    }
    let centres = duration_centres();
    let mut edits = Vec::with_capacity(notes.len());
    let mut insertions = Vec::with_capacity(notes.len() * INSERTION_SLOTS);
    for (input, predictions) in notes.iter().zip(&output.notes) {
        if !predictions.keep.is_finite() || predictions.onset.len() != ONSET_BINS
            || predictions.duration_ratio.len() != DURATION_RATIO_BINS
        { return Err("invalid Evidence keep/onset/duration head dimensions".into()); }
        let on = box_argmax(&probabilities(&predictions.onset)?, 10);
        let duration = box_argmax(&probabilities(&predictions.duration_ratio)?, 3);
        let corrected_onset = input.onset + on as f64 * 0.005 - 0.5;
        let corrected_duration = (input.offset - input.onset).max(1e-4)
            * ((duration as f64) * 0.05 - 4.0).exp();
        let local_on = corrected_onset - window_start;
        let score = if (0.0..WINDOW_SECONDS).contains(&local_on) {
            sigmoid(predictions.keep)
        } else { -1.0 };
        edits.push(RepairEdit {
            score,
            onset: corrected_onset,
            offset: corrected_onset + corrected_duration,
        });
        for ins in &predictions.insertion {
            if !ins.exists.is_finite() || ins.pitch.len() != INSERTION_PITCH_BINS
                || ins.onset.len() != INSERTION_TIME_BINS || ins.duration.len() != ABSOLUTE_DURATION_BINS
            { return Err("invalid Evidence insertion head dimensions".into()); }
            let pitch = probabilities(&ins.pitch)?.iter().enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap().0;
            let frame = box_argmax(&probabilities(&ins.onset)?, 5);
            let duration_probs = probabilities(&ins.duration)?;
            let duration = (0..ABSOLUTE_DURATION_BINS)
                .map(|j| {
                    let tol = (0.2f32 * centres[j]).max(0.05);
                    let score = (0..ABSOLUTE_DURATION_BINS)
                        .filter(|&k| (centres[k] - centres[j]).abs() <= tol)
                        .map(|k| duration_probs[k]).sum::<f32>();
                    (j, score)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap().0;
            let onset = input.onset + frame as f64 * 0.01 - 1.0;
            if !(0.0..WINDOW_SECONDS).contains(&(onset - window_start)) { continue; }
            let pitch = (i32::from(input.pitch) + pitch as i32 - 24).clamp(0, 127) as u8;
            insertions.push(RepairCandidate {
                onset, offset: onset + f64::from(centres[duration]), pitch,
                score: sigmoid(ins.exists), kind: RepairKind::Inserted,
            });
        }
    }
    Ok((edits, insertions))
}

pub fn infer_edits<F>(
    notes: &[RepairInput], evidence: &[[f32; EVIDENCE_FEATURES]], mut predict: F,
) -> Result<(Vec<Option<RepairEdit>>, Vec<RepairCandidate>), String>
where F: FnMut(&[RepairInput], &[[f32; EVIDENCE_FEATURES]], f64) -> Result<CandidateLogits, String>
{
    if notes.len() != evidence.len() || notes.windows(2).any(|w| w[0].onset > w[1].onset) {
        return Err("Evidence inputs must be aligned and sorted by onset".into());
    }
    let mut edits = vec![None; notes.len()];
    let mut insertions = Vec::new();
    if notes.is_empty() { return Ok((edits, insertions)); }
    let end = notes.iter().map(|n| n.offset).fold(0.0, f64::max);
    let mut start = 0.0f64;
    while start < end + WINDOW_SECONDS {
        let lo = notes.partition_point(|n| n.onset < start);
        let hi = notes.partition_point(|n| n.onset < start + WINDOW_SECONDS);
        if hi > lo && hi - lo <= MAX_WINDOW_NOTES {
            let output = predict(&notes[lo..hi], &evidence[lo..hi], start)?;
            let (changed, inserted) = decode_candidates(&notes[lo..hi], &output, start)?;
            for (i, edit) in changed.into_iter().enumerate() { edits[lo + i] = Some(edit); }
            insertions.extend(inserted);
        }
        start += WINDOW_SECONDS;
    }
    Ok((edits, insertions))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn peaked(n: usize, index: usize) -> Vec<f32> {
        (0..n).map(|i| if i == index { 25.0 } else { -25.0 }).collect()
    }
    fn insertion() -> InsertionLogits {
        InsertionLogits {
            exists: 2.0,
            pitch: peaked(INSERTION_PITCH_BINS, 24),
            onset: peaked(INSERTION_TIME_BINS, 100),
            duration: peaked(ABSOLUTE_DURATION_BINS, 63),
        }
    }
    #[test]
    fn zero_shift_evidence_candidate() {
        let note = RepairInput { onset: 2.0, offset: 2.2, pitch: 64, velocity: 95, original_index: 0 };
        let logits = CandidateLogits { notes: vec![NoteLogits {
            keep: 3.0,
            onset: peaked(ONSET_BINS, 100),
            duration_ratio: peaked(DURATION_RATIO_BINS, 80),
            insertion: [insertion(), insertion()],
        }]};
        let (edits, inserted) = decode_candidates(&[note], &logits, 0.0).unwrap();
        assert!((edits[0].onset - note.onset).abs() < 1e-7);
        assert!((edits[0].offset - note.offset).abs() < 1e-6);
        assert_eq!(inserted.len(), 2);
        assert_eq!(inserted[0].pitch, 64);
        assert!((inserted[0].onset - 2.0).abs() < 1e-6);
    }
    #[test]
    fn box_argmax_picks_leftmost_equal_peak() {
        assert_eq!(box_argmax(&[0.0, 0.1, 0.9, 0.9, 0.1, 0.0], 0), 2);
    }
}
