use anyhow::{Result, ensure};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcription {
    pub text: String,
    pub token_ids: Vec<usize>,
    pub frames: Vec<usize>,
}

pub fn ctc_greedy(
    labels: &[usize],
    valid_frames: usize,
    vocab: &[String],
) -> Result<Transcription> {
    ensure!(
        valid_frames <= labels.len(),
        "CTC valid_frames exceeds labels length"
    );
    let blank = vocab.len();
    let mut token_ids = Vec::new();
    let mut frames = Vec::new();
    for t in 0..valid_frames {
        let id = labels[t];
        ensure!(id <= blank, "CTC token {id} is outside vocabulary");
        if id != blank && (t == 0 || id != labels[t - 1]) {
            token_ids.push(id);
            frames.push(t);
        }
    }
    let mut text = String::new();
    for &id in &token_ids {
        text.push_str(&vocab[id]);
    }
    Ok(Transcription {
        text,
        token_ids,
        frames,
    })
}

pub fn ctc_argmax(logits: &[f32], time: usize, n_classes: usize) -> Result<Vec<usize>> {
    ensure!(n_classes > 0, "num_classes must be > 0");
    ensure!(logits.len() == time * n_classes, "logits shape mismatch");
    Ok(logits
        .chunks_exact(n_classes)
        .map(|scores| {
            scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ctc_collapse_keeps_repeats_separated_by_blank() {
        let v = vec!["а".to_string(), "б".to_string()];
        let result = ctc_greedy(&[2, 0, 0, 2, 0, 1, 1, 2], 8, &v).unwrap();
        assert_eq!(result.text, "ааб");
        assert_eq!(result.frames, vec![1, 4, 5]);
    }
    #[test]
    fn ctc_uses_previous_frame_across_blank_and_repeats() {
        let v = vec!["z".to_owned()];
        assert_eq!(ctc_greedy(&[0, 0, 1, 0, 0], 5, &v).unwrap().text, "zz");
        assert_eq!(ctc_greedy(&[0, 0, 1, 0, 0], 3, &v).unwrap().text, "z");
    }
}
