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
    let mut current = 0;
    while current < peaks.len() {
        let mut average = peaks[current];
        let mut count = 1;
        let mut last = peaks[current];
        while current + 1 < peaks.len() && peaks[current + 1] - last <= 1.0 {
            current += 1;
            count += 1;
            last = peaks[current];
            average += (last - average) / count as f64;
        }
        deduplicated.push(average.trunc() / 50.0);
        current += 1;
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
}
