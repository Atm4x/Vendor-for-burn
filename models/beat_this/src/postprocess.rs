#[derive(Debug, Clone, serde::Serialize)]
pub struct BeatTimes {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
}

fn find_peaks(logits: &[f32]) -> Vec<f64> {
    let mut peaks = Vec::new();
    for (i, &value) in logits.iter().enumerate() {
        if value <= 0.0 {
            continue;
        }
        let begin = i.saturating_sub(3);
        let end = (i + 4).min(logits.len());
        if logits[begin..end].iter().any(|&other| other > value) {
            continue;
        }
        peaks.push(i as f64);
    }
    let mut deduplicated = Vec::new();
    let mut candidates = peaks.into_iter();
    if let Some(mut average) = candidates.next() {
        let mut count = 1.0;
        for next in candidates {
            if next - average <= 1.0 {
                count += 1.0;
                average += (next - average) / count;
            } else {
                deduplicated.push(average / 50.0);
                average = next;
                count = 1.0;
            }
        }
        deduplicated.push(average / 50.0);
    }
    deduplicated
}

pub fn postprocess(beat_logits: &[f32], downbeat_logits: &[f32]) -> BeatTimes {
    let beats = find_peaks(beat_logits);
    let mut downbeats: Vec<f64> = find_peaks(downbeat_logits)
        .into_iter()
        .map(|downbeat| {
            beats
                .iter()
                .copied()
                .min_by(|a, b| (a - downbeat).abs().total_cmp(&(b - downbeat).abs()))
                .unwrap_or(downbeat)
        })
        .collect();
    downbeats.sort_by(f64::total_cmp);
    downbeats.dedup();
    BeatTimes { beats, downbeats }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimal_peaks_and_downbeat_snapping() {
        let mut beat = vec![-1.0; 100];
        let mut downbeat = vec![-1.0; 100];
        beat[20] = 3.0;
        beat[21] = 2.0;
        beat[60] = 5.0;
        downbeat[61] = 2.0;
        let out = postprocess(&beat, &downbeat);
        assert_eq!(out.beats, [0.4, 1.2]);
        assert_eq!(out.downbeats, [1.2]);
    }

    #[test]
    fn adjacent_plateau_matches_reference_running_mean() {
        let mut beat = vec![-1.0; 64];
        beat[10] = 1.0;
        beat[11] = 1.0;
        beat[12] = 1.0;
        beat[13] = 1.0;
        let out = postprocess(&beat, &vec![-1.0; 64]);
        assert_eq!(out.beats, [10.5 / 50.0, 12.5 / 50.0]);
    }

    #[test]
    fn adjacent_pair_keeps_subframe_average() {
        let mut beat = vec![-1.0; 48];
        beat[20] = 2.0;
        beat[21] = 2.0;
        let out = postprocess(&beat, &vec![-1.0; 48]);
        assert_eq!(out.beats, [20.5 / 50.0]);
    }
}
