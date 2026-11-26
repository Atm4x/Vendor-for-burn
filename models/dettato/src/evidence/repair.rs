use crate::transcriber::DecodedNote;

pub const WINDOW_SECONDS: f64 = 8.0;
pub const MAX_WINDOW_NOTES: usize = 4096;
pub const DEDUP_SECONDS: f64 = 0.03;
pub const ONSET_THRESHOLD: f32 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepairKind { Kept, Inserted }

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepairCandidate {
    pub onset: f64,
    pub offset: f64,
    pub pitch: u8,
    pub score: f32,
    pub kind: RepairKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepairInput {
    pub onset: f64,
    pub offset: f64,
    pub pitch: u8,
    pub velocity: u8,
    pub original_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepairEdit {
    pub score: f32,
    pub onset: f64,
    pub offset: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepairConfig {
    pub keep_threshold: f32,
    pub insert_threshold: f32,
    pub dedup_seconds: f64,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self { keep_threshold: 0.4, insert_threshold: 0.6, dedup_seconds: DEDUP_SECONDS }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepairInfo {
    pub transcribed: usize,
    pub kept: usize,
    pub inserted: usize,
    pub dropped: usize,
}

fn round_four(x: f64) -> f64 { (x * 10_000.0).round_ties_even() / 10_000.0 }

pub fn input_notes(notes: &[DecodedNote]) -> Result<Vec<RepairInput>, String> {
    let mut result = Vec::with_capacity(notes.len());
    for (i, note) in notes.iter().enumerate() {
        if !note.onset.is_finite() || !note.offset.is_finite() || !note.confidence.is_finite()
            || note.onset < 0.0 || note.offset <= note.onset || note.pitch > 127
        { return Err("invalid transcriber note for evidence repair".into()); }
        let onset = round_four(note.onset);
        let offset = round_four(note.offset).max(onset + 0.001);
        let conf = (f64::from(note.confidence) - f64::from(ONSET_THRESHOLD))
            / (1.0 - f64::from(ONSET_THRESHOLD));
        let velocity = (1.0 + 126.0 * conf).round_ties_even().clamp(1.0, 127.0) as u8;
        result.push(RepairInput { onset, offset, pitch: note.pitch, velocity, original_index: i });
    }
    result.sort_by(|a, b| a.onset.total_cmp(&b.onset).then_with(|| a.pitch.cmp(&b.pitch)));
    Ok(result)
}

fn remove_duplicate_insertions(candidates: &mut Vec<RepairCandidate>, threshold: f64) {
    let mut kept_by_pitch = vec![Vec::<f64>::new(); 128];
    for x in candidates.iter().filter(|x| x.kind == RepairKind::Kept) {
        kept_by_pitch[x.pitch as usize].push(x.onset);
    }
    let mut insertions = candidates.iter().filter(|x| x.kind == RepairKind::Inserted).copied()
        .filter(|x| kept_by_pitch[x.pitch as usize].iter().all(|&on| (on - x.onset).abs() >= threshold))
        .collect::<Vec<_>>();
    insertions.sort_by(|a, b| a.pitch.cmp(&b.pitch).then_with(|| a.onset.total_cmp(&b.onset)));
    let mut alive = vec![true; insertions.len()];
    for i in 0..insertions.len().saturating_sub(1) {
        if !alive[i] { continue; }
        let left = insertions[i];
        let right = insertions[i + 1];
        if left.pitch == right.pitch && right.onset - left.onset < threshold {
            if left.score >= right.score { alive[i + 1] = false; } else { alive[i] = false; }
        }
    }
    candidates.retain(|x| x.kind == RepairKind::Kept);
    candidates.extend(insertions.into_iter().zip(alive)
        .filter_map(|(x, keep)| keep.then_some(x)));
}

pub fn apply_repair(
    sorted_notes: &[RepairInput], edits: &[Option<RepairEdit>],
    insertions: &[RepairCandidate], config: RepairConfig,
) -> Result<(Vec<RepairCandidate>, RepairInfo), String> {
    if edits.len() != sorted_notes.len() || !config.keep_threshold.is_finite()
        || !config.insert_threshold.is_finite() || !config.dedup_seconds.is_finite()
        || !(0.0..=1.0).contains(&config.keep_threshold)
        || !(0.0..=1.0).contains(&config.insert_threshold) || config.dedup_seconds < 0.0
    { return Err("invalid repair input or thresholds".into()); }
    let mut rows = Vec::with_capacity(sorted_notes.len() + insertions.len());
    for (note, edit) in sorted_notes.iter().zip(edits) {
        let (onset, offset, score) = if let Some(edit) = edit {
            if !edit.score.is_finite() || edit.score < config.keep_threshold { continue; }
            (edit.onset, edit.offset, edit.score)
        } else { (note.onset, note.offset, 1.0) };
        if !onset.is_finite() || !offset.is_finite() || onset < 0.0 {
            return Err("non-finite or negative repaired onset".into());
        }
        rows.push(RepairCandidate { onset, offset, pitch: note.pitch, score, kind: RepairKind::Kept });
    }
    for &insertion in insertions {
        if insertion.kind != RepairKind::Inserted {
            return Err("insertion list contains kept note".into());
        }
        if !insertion.score.is_finite() || !insertion.onset.is_finite() || !insertion.offset.is_finite()
            || insertion.pitch > 127
        { return Err("invalid insertion proposal".into()); }
        if insertion.score >= config.insert_threshold && insertion.onset >= 0.0 {
            rows.push(insertion);
        }
    }
    remove_duplicate_insertions(&mut rows, config.dedup_seconds);
    for row in &mut rows {
        row.onset = round_four(row.onset);
        row.offset = round_four(row.offset.max(row.onset + 0.01));
    }
    rows.sort_by(|a, b| a.onset.total_cmp(&b.onset).then_with(|| a.pitch.cmp(&b.pitch)));
    let kept = rows.iter().filter(|x| x.kind == RepairKind::Kept).count();
    let inserted = rows.len() - kept;
    let info = RepairInfo {
        transcribed: sorted_notes.len(), kept, inserted,
        dropped: sorted_notes.len().saturating_sub(kept),
    };
    Ok((rows, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input(onset: f64, pitch: u8) -> RepairInput {
        RepairInput { onset, offset: onset + 0.2, pitch, velocity: 100, original_index: 0 }
    }
    fn insert(onset: f64, pitch: u8, score: f32) -> RepairCandidate {
        RepairCandidate { onset, offset: onset + 0.1, pitch, score, kind: RepairKind::Inserted }
    }

    #[test]
    fn kept_polyphonic_note_wins_over_insertion_even_if_score_lower() {
        let notes = [input(0.1, 60), input(0.101, 60)];
        let insertions = [insert(0.105, 60, 0.99), insert(0.3, 60, 0.7)];
        let (result, info) = apply_repair(&notes, &[None, None], &insertions, RepairConfig::default()).unwrap();
        assert_eq!(info, RepairInfo { transcribed: 2, kept: 2, inserted: 1, dropped: 0 });
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn insertions_are_deduplicated_by_score_and_pitch() {
        let insertions = [insert(0.1, 60, 0.65), insert(0.12, 60, 0.8), insert(0.19, 60, 0.75)];
        let (result, info) = apply_repair(&[], &[], &insertions, RepairConfig::default()).unwrap();
        assert_eq!(info.inserted, 2);
        assert_eq!(result[0].score, 0.8);
    }

    #[test]
    fn pass_through_on_skipped_window_and_drop_low_confidence() {
        let notes = [input(0.1, 60), input(1.0, 61)];
        let edits = [None, Some(RepairEdit { onset: 1.0, offset: 1.4, score: 0.25 })];
        let (result, info) = apply_repair(&notes, &edits, &[], RepairConfig::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(info.dropped, 1);
        assert_eq!(result[0].score, 1.0);
    }
}
