pub mod audio;
pub mod checkpoint;
pub mod model;
pub mod postprocess;

use model::BeatThisModel;
use postprocess::{BeatTimes, postprocess};

pub const CHUNK_SIZE: usize = 1500;
pub const BORDER: usize = 6;

#[derive(Debug, serde::Serialize)]
pub struct FramePrediction {
    pub beat: Vec<f32>,
    pub downbeat: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkWindow {
    pub start: isize,
    pub length: usize,
}

impl ChunkWindow {
    fn extract(&self, spectrogram: &[[f32; 128]]) -> Vec<[f32; 128]> {
        let full_size = spectrogram.len();
        let end = self.start + CHUNK_SIZE as isize;
        let source_start = self.start.max(0) as usize;
        let source_end = end.max(0).min(full_size as isize) as usize;
        let left = (-self.start).max(0) as usize;
        let right = (end - full_size as isize).clamp(0, BORDER as isize) as usize;
        let mut frames = Vec::with_capacity(self.length);
        frames.extend(std::iter::repeat_n([0f32; 128], left));
        if source_end > source_start {
            frames.extend_from_slice(&spectrogram[source_start..source_end]);
        }
        frames.extend(std::iter::repeat_n([0f32; 128], right));
        frames
    }
}

pub fn plan_chunks(frame_count: usize) -> Vec<ChunkWindow> {
    if frame_count == 0 {
        return vec![];
    }
    let step = CHUNK_SIZE - 2 * BORDER;
    let mut starts = Vec::new();
    let mut next = -(BORDER as isize);
    while next < frame_count as isize - BORDER as isize {
        starts.push(next);
        next += step as isize;
    }
    if frame_count > step && starts.len() > 1 {
        *starts.last_mut().unwrap() = frame_count as isize - (CHUNK_SIZE - BORDER) as isize;
    }
    starts
        .into_iter()
        .map(|start| {
            let end = start + CHUNK_SIZE as isize;
            let source_start = start.max(0) as usize;
            let source_end = end.max(0).min(frame_count as isize) as usize;
            let left = (-start).max(0) as usize;
            let right = (end - frame_count as isize).clamp(0, BORDER as isize) as usize;
            ChunkWindow {
                start,
                length: left + source_end.saturating_sub(source_start) + right,
            }
        })
        .collect()
}

pub trait FrameModel {
    fn infer(&self, frames: &[[f32; 128]]) -> Result<(Vec<f32>, Vec<f32>), String>;
}

impl FrameModel for BeatThisModel {
    fn infer(&self, frames: &[[f32; 128]]) -> Result<(Vec<f32>, Vec<f32>), String> {
        BeatThisModel::infer(self, frames)
    }
}

pub fn predict_logits(
    model: &impl FrameModel,
    spectrogram: &[[f32; 128]],
) -> Result<FramePrediction, String> {
    let frames = spectrogram.len();
    if frames == 0 {
        return Err("empty spectrogram".into());
    }
    let mut beat = vec![-1000.0f32; frames];
    let mut downbeat = vec![-1000.0f32; frames];
    for window in plan_chunks(frames).iter().rev() {
        if window.length <= 2 * BORDER {
            return Err(format!("short inference window: {}", window.length));
        }
        let excerpt = window.extract(spectrogram);
        if excerpt.len() != window.length {
            return Err("invalid chunk plan".into());
        }
        let (local_beat, local_downbeat) = model.infer(&excerpt)?;
        for i in BORDER..(window.length - BORDER) {
            let global = window.start + i as isize;
            if (0..frames as isize).contains(&global) {
                let pos = global as usize;
                beat[pos] = local_beat[i];
                downbeat[pos] = local_downbeat[i];
            }
        }
    }
    Ok(FramePrediction { beat, downbeat })
}

pub fn predict_beats(
    model: &impl FrameModel,
    spectrogram: &[[f32; 128]],
) -> Result<BeatTimes, String> {
    let framewise = predict_logits(model, spectrogram)?;
    Ok(postprocess(&framewise.beat, &framewise.downbeat))
}

pub fn export_beats_tsv(times: &BeatTimes) -> String {
    let mut output = String::new();
    let first_downbeat = times
        .downbeats
        .first()
        .and_then(|time| {
            times
                .beats
                .binary_search_by(|beat| beat.total_cmp(time))
                .ok()
        })
        .unwrap_or(0);
    let measure = if times.downbeats.len() >= 2 {
        let second = times
            .beats
            .binary_search_by(|beat| beat.total_cmp(&times.downbeats[1]))
            .unwrap_or(first_downbeat + 4);
        second.saturating_sub(first_downbeat).max(1)
    } else {
        1
    };
    let mut count = if times.downbeats.len() >= 2 && first_downbeat < measure {
        measure - first_downbeat
    } else {
        1
    };
    for &beat in &times.beats {
        if times
            .downbeats
            .binary_search_by(|time| time.total_cmp(&beat))
            .is_ok()
        {
            count = 1;
        } else {
            count += 1;
        }
        output.push_str(&format!("{beat}\t{count}\n"));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_plan_covers_every_frame_and_matches_upstream_boundaries() {
        for length in [
            1, 6, 7, 100, 1487, 1488, 1489, 1490, 1500, 1501, 2970, 3000, 5000, 10000,
        ] {
            let plan = plan_chunks(length);
            let mut coverage = vec![0usize; length];
            for window in &plan {
                for local in BORDER..(window.length - BORDER) {
                    let global = window.start + local as isize;
                    if (0..length as isize).contains(&global) {
                        coverage[global as usize] += 1;
                    }
                }
            }
            assert!(
                coverage.iter().all(|&count| count >= 1),
                "uncovered frames for {length}: {:?}",
                coverage
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| **n == 0)
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            plan_chunks(1501)
                .iter()
                .map(|x| x.start)
                .collect::<Vec<_>>(),
            [-6, 7]
        );
        assert_eq!(
            plan_chunks(3000)
                .iter()
                .map(|x| x.start)
                .collect::<Vec<_>>(),
            [-6, 1482, 1506]
        );
    }

    struct IdentityModel;
    impl FrameModel for IdentityModel {
        fn infer(&self, frames: &[[f32; 128]]) -> Result<(Vec<f32>, Vec<f32>), String> {
            let values = frames.iter().map(|row| row[0]).collect::<Vec<_>>();
            Ok((values.clone(), values))
        }
    }

    #[test]
    fn chunk_aggregate_is_identity_for_long_audio() {
        for frames in [1, 2, 100, 1488, 1489, 1500, 1501, 2980, 3000, 10000] {
            let input = (0..frames)
                .map(|i| {
                    let mut frame = [0.0f32; 128];
                    frame[0] = i as f32 + 1.0;
                    frame
                })
                .collect::<Vec<_>>();
            let output = predict_logits(&IdentityModel, &input).unwrap();
            assert_eq!(output.beat.len(), frames);
            assert_eq!(output.downbeat.len(), frames);
            for (i, value) in output.beat.iter().enumerate() {
                assert_eq!(
                    *value,
                    i as f32 + 1.0,
                    "wrong aggregated frame {i} of {frames}"
                );
            }
        }
    }
}
