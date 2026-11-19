//! Sentence-aware long-form scheduling and bounded acoustic-limit retries.

use std::collections::VecDeque;

use anyhow::{Context, Result, ensure};

use crate::{frontend::ModelConfig, ssml::Segment};

pub(crate) const TARGET_CHARS: usize = 400;
pub(crate) const MAX_INPUT_CHARS: usize = 16_384;
pub(crate) const MAX_CHUNKS: usize = 256;
pub(crate) const MAX_AUDIO_SECONDS: usize = 600;

fn split_boundary(text: &str, max_chars: usize) -> Option<usize> {
    let mut sentence = None;
    let mut clause = None;
    let mut whitespace = None;
    let min_sentence = max_chars / 3;
    let min_clause = max_chars / 2;
    for (count, (pos, ch)) in text.char_indices().enumerate() {
        if count >= max_chars {
            break;
        }
        let end = pos + ch.len_utf8();
        let next = text[end..].chars().next();
        if ch.is_whitespace() {
            whitespace = Some(end);
        }
        if (".!?…".contains(ch) && next.is_none_or(char::is_whitespace)) && count >= min_sentence
        {
            sentence = Some(end);
        } else if (",;:".contains(ch) && next.is_none_or(char::is_whitespace))
            && count >= min_clause
        {
            clause = Some(end);
        }
    }
    sentence.or(clause).or(whitespace)
}

pub(crate) fn split_text(text: &str, max_chars: usize) -> Result<Vec<String>> {
    ensure!(
        max_chars >= 2,
        "Text chunk limit must be at least two characters"
    );
    ensure!(
        text.chars().count() <= MAX_INPUT_CHARS,
        "Silero text exceeds long-form input limit"
    );
    let mut remaining = text.trim();
    ensure!(!remaining.is_empty(), "Silero text is empty");
    let mut chunks = Vec::new();
    while !remaining.is_empty() {
        if remaining.chars().count() <= max_chars {
            chunks.push(remaining.to_owned());
            break;
        }
        let boundary = split_boundary(remaining, max_chars).context(
            "Cannot safely chunk Silero text: a token has no whitespace or sentence boundary",
        )?;
        let (part, rest) = remaining.split_at(boundary);
        let part = part.trim();
        ensure!(
            !part.is_empty(),
            "Silero long-form splitter produced an empty segment"
        );
        chunks.push(part.to_owned());
        ensure!(
            chunks.len() < MAX_CHUNKS,
            "Silero long-form segment count exceeds limit"
        );
        remaining = rest.trim_start();
    }
    Ok(chunks)
}

pub(crate) fn is_acoustic_limit(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let msg = cause.to_string();
        msg.contains("positional encoding limit")
    })
}

pub(crate) fn append_audio(out: &mut Vec<f32>, data: Vec<f32>, sample_rate: u32) -> Result<()> {
    let max = (sample_rate as usize)
        .checked_mul(MAX_AUDIO_SECONDS)
        .context("Invalid Silero audio size limit")?;
    let count = out
        .len()
        .checked_add(data.len())
        .context("Silero long-form sample count overflow")?;
    ensure!(count <= max, "Silero long-form output exceeds 10 minutes");
    out.extend(data);
    Ok(())
}

pub(crate) fn infer_text_chunks(
    text: &str,
    sample_rate: u32,
    mut infer_one: impl FnMut(&str) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let mut pending: VecDeque<String> = split_text(text, TARGET_CHARS)?.into();
    let mut completed = 0;
    let mut result = Vec::new();
    while let Some(chunk) = pending.pop_front() {
        match infer_one(&chunk) {
            Ok(pcm) => {
                append_audio(&mut result, pcm, sample_rate)?;
                completed += 1;
            }
            Err(error) if is_acoustic_limit(&error) => {
                let length = chunk.chars().count();
                let halves = split_text(&chunk, (length / 2).max(2)).with_context(|| {
                    format!("Acoustic duration limit reached in a {length}-character segment")
                })?;
                if halves.len() <= 1 {
                    return Err(error.context("Cannot split segment below acoustic limit"));
                }
                ensure!(
                    completed + pending.len() + halves.len() <= MAX_CHUNKS,
                    "Silero long-form segment count exceeds limit"
                );
                for half in halves.into_iter().rev() {
                    pending.push_front(half);
                }
            }
            Err(error) => {
                return Err(error.context(format!(
                    "Silero long-form inference failed in segment {}",
                    completed + 1,
                )));
            }
        }
    }
    Ok(result)
}

pub(crate) fn split_ssml_segment(
    config: &ModelConfig,
    item: &Segment,
    max_chars: usize,
) -> Result<Vec<Segment>> {
    let parts = split_text(&item.text, max_chars)?;
    let count = parts.len();
    parts
        .into_iter()
        .enumerate()
        .map(|(index, text)| {
            let clean_text = config.prepare_text_fragment(&text)?.clean_sentence;
            Ok(Segment {
                text,
                clean_text,
                rate: item.rate,
                pitch: item.pitch,
                break_frames: if index + 1 == count {
                    item.break_frames
                } else {
                    None
                },
            })
        })
        .collect()
}

pub(crate) fn group_ssml_segments(
    config: &ModelConfig,
    segments: Vec<Segment>,
) -> Result<Vec<Vec<Segment>>> {
    let mut groups = Vec::new();
    let mut current = Vec::new();
    let mut current_chars = 0;
    for segment in segments {
        for part in split_ssml_segment(config, &segment, TARGET_CHARS)? {
            let length = part.text.chars().count();
            if !current.is_empty() && current_chars + length + 1 > TARGET_CHARS {
                groups.push(std::mem::take(&mut current));
                current_chars = 0;
            }
            current_chars += length + usize::from(!current.is_empty());
            current.push(part);
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    ensure!(
        groups.len() <= MAX_CHUNKS,
        "Silero SSML segment count exceeds limit"
    );
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn short_text_remains_exactly_one_chunk() {
        assert_eq!(
            split_text("  На двери замок.  ", TARGET_CHARS).unwrap(),
            ["На двери замок."]
        );
    }

    #[test]
    fn splits_long_cyrillic_at_natural_boundaries_without_losing_words() {
        let source = "На двери висит замок. Привет, мир! ".repeat(90);
        let result = split_text(&source, TARGET_CHARS).unwrap();
        assert!(result.len() > 2);
        assert!(
            result
                .iter()
                .all(|part| part.chars().count() <= TARGET_CHARS)
        );
        let actual = result.join(" ");
        assert_eq!(
            actual.split_whitespace().collect::<Vec<_>>(),
            source.split_whitespace().collect::<Vec<_>>()
        );
    }

    #[test]
    fn refuses_to_split_inside_extremely_long_word_or_accent_marker() {
        assert!(split_text(&"а".repeat(500), TARGET_CHARS).is_err());
        let stressed = "м+олоко ".repeat(90);
        let chunks = split_text(&stressed, TARGET_CHARS).unwrap();
        assert!(
            chunks
                .iter()
                .all(|text| !text.ends_with('+') && !text.starts_with('+'))
        );
    }

    #[test]
    fn dynamically_retries_only_model_length_failures() {
        let source = "На двери замок. ".repeat(36);
        let mut calls = 0;
        let result = infer_text_chunks(&source, 8000, |part| {
            calls += 1;
            if part.chars().count() > 100 {
                anyhow::bail!(
                    "Expanded Silero utterance exceeds acoustic positional encoding limit"
                );
            }
            Ok(vec![part.chars().count() as f32])
        })
        .unwrap();
        assert!(calls > result.len());
        assert!(result.iter().all(|length| *length <= 100.0));
        assert!(result.len() > 2);
    }

    #[test]
    fn ssml_chunks_keep_prosody_and_attach_break_only_to_last_fragment() {
        use std::collections::BTreeMap;
        let config = ModelConfig {
            format_version: 1,
            symbols: "_~|!+,-.:;?абвгдежзийклмнопрстуфхцчшщъыьэюяё–… ".into(),
            alphabet: "абвгдежзийклмнопрстуфхцчшщъыьэюяё+".into(),
            sos: "|".into(),
            eos: "~".into(),
            speakers: BTreeMap::new(),
            sample_rates: vec![8000],
        };
        let original = Segment {
            text: "Привет, мир! ".repeat(95),
            clean_text: String::new(),
            break_frames: Some(24),
            rate: 0.8,
            pitch: 1.15,
        };
        let groups = group_ssml_segments(&config, vec![original]).unwrap();
        assert!(groups.len() > 2);
        let parts = groups.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(parts.iter().filter(|x| x.break_frames.is_some()).count(), 1);
        assert_eq!(parts.last().unwrap().break_frames, Some(24));
        assert!(parts.iter().all(|s| s.rate == 0.8 && s.pitch == 1.15));
        assert!(parts.iter().all(|s| s.text.chars().count() <= TARGET_CHARS));
    }

    #[test]
    fn propagates_unrelated_errors_without_retries() {
        let mut calls = 0;
        let result = infer_text_chunks("Привет", 8000, |_| {
            calls += 1;
            Err(anyhow!("model checkpoint missing"))
        });
        assert!(result.is_err());
        assert_eq!(calls, 1);
    }
}
