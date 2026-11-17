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
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
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

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

pub fn words_from_frames(
    tokenizer: &crate::tokenizer::Tokenizer,
    token_ids: &[usize],
    token_frames: &[usize],
    sample_count: usize,
    encoded_frames: usize,
) -> Result<Vec<Word>> {
    ensure!(encoded_frames > 0, "encoder produced no frames");
    ensure!(
        token_ids.len() == token_frames.len(),
        "token/frame count mismatch"
    );
    let frame_shift =
        sample_count as f64 / crate::audio::SAMPLE_RATE as f64 / encoded_frames as f64;
    let mut result = Vec::new();
    let mut current_pieces = Vec::<&str>::new();
    let mut current_frames = Vec::<usize>::new();

    let commit = |pieces: &mut Vec<&str>, frames: &mut Vec<usize>, words: &mut Vec<Word>| {
        if pieces.is_empty() {
            return;
        }
        let combined = pieces.concat();
        let text = combined.trim();
        if !text.is_empty() {
            words.push(Word {
                text: text.to_owned(),
                start: frames[0] as f64 * frame_shift,
                end: (frames[frames.len() - 1] + 1) as f64 * frame_shift,
            });
        }
        pieces.clear();
        frames.clear();
    };

    for (&token, &frame) in token_ids.iter().zip(token_frames.iter()) {
        ensure!(
            frame < encoded_frames,
            "token frame outside encoder sequence"
        );
        let piece = tokenizer.id_to_str(token)?;
        let piece = if let Some(rest) = piece.strip_prefix('▁') {
            commit(&mut current_pieces, &mut current_frames, &mut result);
            rest
        } else if piece == " " {
            commit(&mut current_pieces, &mut current_frames, &mut result);
            continue;
        } else {
            piece
        };
        current_pieces.push(piece);
        current_frames.push(frame);
    }
    commit(&mut current_pieces, &mut current_frames, &mut result);
    Ok(result)
}

#[cfg(test)]
mod timestamp_tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn character_boundaries_match_python_reference() {
        let tokenizer = Tokenizer::Character(
            ["П", "р", "и", "в", "е", "т", " ", "м", "!"]
                .map(str::to_owned)
                .to_vec(),
        );
        let words = words_from_frames(
            &tokenizer,
            &[0, 1, 2, 3, 4, 5, 6, 7, 2, 1, 8],
            &[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
            16_000,
            50,
        )
        .unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Привет");
        assert_eq!(words[1].text, "мир!");
        assert!((words[1].start - 0.16).abs() < 1e-8);
        assert!((words[1].end - 0.24).abs() < 1e-8);
    }

    #[test]
    fn empty_sentencepiece_boundary_counts_as_a_frame() {
        let tok = Tokenizer::SentencePiece(vec![
            ("▁".to_owned(), 1),
            ("д".to_owned(), 1),
            ("о".to_owned(), 1),
            (" ".to_owned(), 1),
            ("▁кот".to_owned(), 1),
        ]);
        let words =
            words_from_frames(&tok, &[0, 1, 2, 3, 4], &[1, 3, 4, 6, 8], 16_000, 10).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "до");
        assert!((words[0].start - 0.1).abs() < 1e-12);
        assert!((words[0].end - 0.5).abs() < 1e-12);
        assert_eq!(words[1].text, "кот");
        assert!((words[1].start - 0.8).abs() < 1e-12);
    }

    #[test]
    fn repeated_empty_pieces_do_not_leak_into_next_word() {
        let tok = Tokenizer::SentencePiece(vec![
            ("▁".to_owned(), 1),
            (" ".to_owned(), 1),
            ("▁да".to_owned(), 1),
            ("!".to_owned(), 1),
        ]);
        let words =
            words_from_frames(&tok, &[0, 0, 1, 2, 3], &[0, 1, 2, 5, 7], 16_000, 10).unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "да!");
        assert!((words[0].start - 0.5).abs() < 1e-12);
        assert!((words[0].end - 0.8).abs() < 1e-12);
    }

    #[test]
    fn sentencepiece_start_of_word_and_safety() {
        let tokenizer =
            Tokenizer::SentencePiece(vec![("▁Добрый".to_owned(), 1), ("▁день".to_owned(), 1)]);
        let words = words_from_frames(&tokenizer, &[0, 1], &[1, 4], 16_000, 10).unwrap();
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            ["Добрый", "день"]
        );
        assert!(words_from_frames(&tokenizer, &[0], &[11], 16_000, 10).is_err());
    }
}
