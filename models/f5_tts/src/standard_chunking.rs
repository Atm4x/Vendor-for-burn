use crate::{F5Error, Result, text_frontend::standard_total_frames};

const SAMPLE_RATE: f64 = 24_000.0;
const HOP_LENGTH: f64 = 256.0;
const MAX_TOTAL_SECONDS: f64 = 22.0;

#[derive(Debug, Clone, PartialEq)]
pub struct StandardChunk {
    pub text: String,
    pub total_frames: usize,
    pub fixed_seconds: Option<f32>,
}

fn is_cjk_delimiter(ch: char) -> bool {
    matches!(ch, '；' | '：' | '，' | '。' | '！' | '？')
}

fn is_ascii_delimiter(ch: char) -> bool {
    matches!(ch, ';' | ':' | ',' | '.' | '!' | '?')
}

pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut iter = text.char_indices().peekable();
    while let Some((index, ch)) = iter.next() {
        let after = index + ch.len_utf8();
        if is_cjk_delimiter(ch) {
            if after > start {
                segments.push(&text[start..after]);
            }
            start = after;
            continue;
        }
        if is_ascii_delimiter(ch) && iter.peek().is_some_and(|(_, next)| next.is_whitespace()) {
            if after > start {
                segments.push(&text[start..after]);
            }
            while iter.peek().is_some_and(|(_, next)| next.is_whitespace()) {
                let (pos, whitespace) = iter.next().unwrap();
                start = pos + whitespace.len_utf8();
            }
            start = start.max(after);
        }
    }
    if start < text.len() {
        segments.push(&text[start..]);
    }
    segments
}

pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for sentence in split_sentences(text) {
        if sentence.is_empty() {
            continue;
        }
        if current.len().saturating_add(sentence.len()) > max_chars && !current.is_empty() {
            chunks.push(current.trim().to_owned());
            current.clear();
        }
        current.push_str(sentence);
        if sentence.chars().last().is_some_and(|ch| ch.is_ascii()) {
            current.push(' ');
        }
    }
    if !current.is_empty() {
        chunks.push(current.trim().to_owned());
    }
    chunks
}

pub fn plan_standard_chunks(
    generated_text: &str,
    reference_text: &str,
    reference_samples: usize,
    speed: f32,
    fixed_seconds: Option<f32>,
    cross_fade_seconds: f32,
) -> Result<Vec<StandardChunk>> {
    if reference_samples == 0
        || !speed.is_finite()
        || speed <= 0.0
        || !cross_fade_seconds.is_finite()
        || cross_fade_seconds < 0.0
        || fixed_seconds.is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        return Err(F5Error::InvalidConfiguration(
            "Invalid F5 chunking reference, speaking rate or cross-fade",
        ));
    }
    if generated_text.trim().is_empty() {
        return Err(F5Error::InvalidConfiguration("F5 generation text is empty"));
    }
    let ref_seconds = reference_samples as f64 / SAMPLE_RATE;
    let reference_frames = reference_samples / HOP_LENGTH as usize;
    if reference_frames == 0 {
        return Err(F5Error::InvalidShape("F5 reference has no mel frames"));
    }
    let transcript = crate::text_frontend::normalized_reference_transcript(reference_text)?;
    let recommended_max =
        (transcript.len() as f64 / ref_seconds * (MAX_TOTAL_SECONDS - ref_seconds) * speed as f64)
            .trunc()
            .max(0.0) as usize;
    let chunks = chunk_text(generated_text, recommended_max);
    if chunks.is_empty() {
        return Err(F5Error::InvalidConfiguration("No nonempty F5 text chunks"));
    }
    let total_bytes: usize = chunks.iter().map(String::len).sum();
    let allocated_seconds = fixed_seconds.map(|seconds| {
        seconds as f64 - reference_frames as f64 * HOP_LENGTH / SAMPLE_RATE
            + cross_fade_seconds as f64 * chunks.len().saturating_sub(1) as f64
    });
    let mut planned = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let local_fixed = allocated_seconds.map(|seconds| {
            (reference_frames as f64 * HOP_LENGTH / SAMPLE_RATE
                + seconds * chunk.len() as f64 / total_bytes as f64) as f32
        });
        let total_frames =
            standard_total_frames(reference_frames, reference_text, &chunk, speed, local_fixed)?;
        planned.push(StandardChunk {
            text: chunk,
            total_frames,
            fixed_seconds: local_fixed,
        });
    }
    Ok(planned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_ascii_and_cjk_boundaries() {
        assert_eq!(
            chunk_text("Hello! World. Again?", 12),
            ["Hello!", "World.", "Again?"]
        );
        assert_eq!(
            chunk_text("你好，世界。你好！", 9),
            ["你好，", "世界。", "你好！"]
        );
        assert_eq!(
            chunk_text("UnbrokenVeryLongSentence", 2),
            ["UnbrokenVeryLongSentence"]
        );
        assert_eq!(chunk_text("a,b", 4), ["a,b"]);
    }

    #[test]
    fn fixed_duration_is_distributed_not_repeated_per_chunk() {
        let plans = plan_standard_chunks(
            "One sentence. Another sentence. Yet another sentence.",
            "A",
            120_000,
            1.0,
            Some(12.0),
            0.15,
        )
        .unwrap();
        assert!(plans.len() > 1);
        let audio_seconds = plans
            .iter()
            .map(|chunk| (chunk.total_frames as f64 - 468.0) * HOP_LENGTH / SAMPLE_RATE)
            .sum::<f64>()
            - (plans.len() - 1) as f64 * 0.15;
        assert!((audio_seconds - (12.0 - 468.0 * HOP_LENGTH / SAMPLE_RATE)).abs() < 0.1);
    }
}
