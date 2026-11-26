use super::{decode::InstrumentGroup, instruments::INSTRUMENTS};
use crate::{Note, Track};

pub fn render_tracks(
    groups: &[InstrumentGroup], notes: &[(f64, f64, u8)], confidence: &[f32],
    time_scale: f64, label_separate_tracks: bool,
) -> Result<Vec<Track>, String> {
    if !time_scale.is_finite() || time_scale <= 0.0 || notes.len() != confidence.len() {
        return Err("invalid Grouper track rendering arguments".into());
    }
    let mut sorted = groups.iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| b.note_indices.len().cmp(&a.note_indices.len()));
    let mut counts = vec![0usize; INSTRUMENTS.len()];
    let mut tracks = Vec::with_capacity(groups.len());
    for group in sorted {
        let instrument = INSTRUMENTS.get(group.instrument_index).ok_or("unknown instrument index")?;
        let mut indices = group.note_indices.clone();
        if indices.iter().any(|&i| i >= notes.len()) { return Err("invalid group note id".into()); }
        indices.sort_by(|&a, &b| notes[a].2.cmp(&notes[b].2)
            .then_with(|| notes[a].0.total_cmp(&notes[b].0)));
        let mut kept = Vec::new();
        let mut previous = None::<usize>;
        for id in indices {
            if let Some(prev) = previous {
                if notes[prev].2 == notes[id].2 && notes[id].0 - notes[prev].0 <= 0.03 {
                    continue;
                }
            }
            previous = Some(id);
            kept.push(id);
        }
        kept.sort_by(|&a, &b| notes[a].0.total_cmp(&notes[b].0));
        let mut midi = Vec::with_capacity(kept.len());
        for id in kept {
            let (on, off, pitch) = notes[id];
            if !on.is_finite() || !off.is_finite() || on < 0.0 || off < on
                || !confidence[id].is_finite()
            { return Err("invalid classified note".into()); }
            let velocity = (20.0 + 107.0 * confidence[id]).clamp(1.0, 127.0) as u8;
            midi.push(Note::new(on * time_scale, off.max(on + 0.02) * time_scale, pitch, velocity)?);
        }
        counts[group.instrument_index] += 1;
        let mut name = instrument.name.replace('_', " ");
        if label_separate_tracks && counts[group.instrument_index] > 1 {
            name.push_str(&format!(" #{}", counts[group.instrument_index]));
        }
        tracks.push(Track { name, program: instrument.program, is_drum: false, notes: midi });
    }
    Ok(tracks)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keeps_distinct_same_instrument_tracklets() {
        let groups = [
            InstrumentGroup { instrument_index: 0, tracklet_index: 1, note_indices: vec![0, 1] },
            InstrumentGroup { instrument_index: 0, tracklet_index: 2, note_indices: vec![2] },
        ];
        let notes = [(0.1, 0.3, 60), (0.11, 0.31, 60), (0.2, 0.4, 64)];
        let tracks = render_tracks(&groups, &notes, &[0.8, 0.9, 0.5], 1.0, true).unwrap();
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].notes.len(), 1);
        assert_eq!(tracks[0].name, "piano");
        assert_eq!(tracks[1].name, "piano #2");
    }
}
