use crate::{F5Error, Result};

pub const SPEED_CLASSES: usize = 32;
pub const SPEED_STEP: f32 = 0.25;
pub const MAX_TOTAL_SECONDS: f32 = 22.0;
pub const SAMPLE_RATE: f32 = 24_000.0;
pub const HOP_LENGTH: f32 = 256.0;

pub trait EnglishSyllableCounter {
    fn count_syllables(&self, english_word: &str) -> Result<usize>;
}

pub fn speed_from_label(label: usize) -> Result<f32> {
    if label >= SPEED_CLASSES {
        return Err(F5Error::InvalidConfiguration(
            "SpeedPredictor class is out of range",
        ));
    }
    Ok((label + 1) as f32 * SPEED_STEP)
}

pub fn best_speed_class(logits: &[f32]) -> Result<usize> {
    if logits.len() != SPEED_CLASSES || logits.iter().any(|v| !v.is_finite()) {
        return Err(F5Error::InvalidShape(
            "SpeedPredictor expects 32 finite logits",
        ));
    }
    let (index, _) = logits
        .iter()
        .enumerate()
        .max_by(|(left_id, left), (right_id, right)| {
            left.partial_cmp(right)
                .unwrap()
                .then_with(|| right_id.cmp(left_id))
        })
        .ok_or(F5Error::InvalidShape("Empty SpeedPredictor output"))?;
    Ok(index)
}

pub fn count_units(text: &str, english: &impl EnglishSyllableCounter) -> Result<usize> {
    let mut units = 0.0_f32;
    let mut ascii_word = String::new();

    let flush_word = |word: &mut String, units: &mut f32| -> Result<()> {
        if !word.is_empty() {
            *units += english.count_syllables(&word.to_ascii_lowercase())? as f32;
            word.clear();
        }
        Ok(())
    };

    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '\'' {
            ascii_word.push(ch);
            continue;
        }
        flush_word(&mut ascii_word, &mut units)?;
        if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            units += 1.0;
        }
        units += match ch {
            '.' | '!' | '?' => 1.5,
            ',' => 0.7,
            ';' | ':' => 1.0,
            '。' | '！' | '？' => 1.3,
            '，' => 0.6,
            '；' | '：' => 0.8,
            _ => 0.0,
        };
    }
    flush_word(&mut ascii_word, &mut units)?;
    Ok(units.round_ties_even().max(1.0) as usize)
}

pub fn chunk_text(
    text: &str,
    max_units: usize,
    english: &impl EnglishSyllableCounter,
) -> Result<Vec<String>> {
    if max_units == 0 {
        return Err(F5Error::InvalidConfiguration(
            "Maximum units must be positive",
        ));
    }
    let mut sentences = Vec::new();
    let mut part = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        part.push(ch);
        let ascii_break = matches!(ch, ';' | ':' | ',' | '.' | '!' | '?')
            && chars.peek().is_some_and(|next| next.is_whitespace());
        let cjk_break = matches!(ch, '；' | '：' | '，' | '。' | '！' | '？');
        if ascii_break || cjk_break {
            if !part.trim().is_empty() {
                sentences.push(part.trim().to_owned());
            }
            part.clear();
        }
    }
    if !part.trim().is_empty() {
        sentences.push(part.trim().to_owned());
    }

    let mut result = Vec::new();
    let mut current = String::new();
    for sentence in sentences {
        let candidate = if current.is_empty() {
            sentence.clone()
        } else {
            format!("{current} {sentence}")
        };
        if !current.is_empty() && count_units(&candidate, english)? > max_units {
            result.push(core::mem::take(&mut current));
            current = sentence;
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    Ok(result)
}

pub fn available_units(reference_samples: usize, predicted_speed: f32) -> Result<usize> {
    let speed = predicted_speed.max(0.25);
    if !speed.is_finite() {
        return Err(F5Error::InvalidConfiguration(
            "Speaking speed must be finite",
        ));
    }
    let seconds = (MAX_TOTAL_SECONDS - reference_samples as f32 / SAMPLE_RATE).max(1.0);
    Ok((speed * seconds).floor().max(1.0) as usize)
}

pub fn generated_frames(
    units: usize,
    predicted_speed: f32,
    user_speed: f32,
    fixed_seconds: Option<f32>,
) -> Result<usize> {
    if !predicted_speed.is_finite()
        || !user_speed.is_finite()
        || predicted_speed <= 0.0
        || user_speed <= 0.0
    {
        return Err(F5Error::InvalidConfiguration(
            "Speaking rates must be positive and finite",
        ));
    }
    let local_speed = if units < 4 { 0.5 } else { user_speed };
    let seconds = fixed_seconds.unwrap_or(units as f32 / predicted_speed.max(0.25) / local_speed);
    if !seconds.is_finite() || seconds < 0.0 || seconds > 65_536.0 * HOP_LENGTH / SAMPLE_RATE {
        return Err(F5Error::InvalidConfiguration(
            "Requested generation duration exceeds limit",
        ));
    }
    Ok((seconds * SAMPLE_RATE / HOP_LENGTH).floor() as usize)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossLingualChunk {
    pub text: String,
    pub syllable_units: usize,
    pub total_frames: usize,
}

pub fn plan_cross_lingual_chunks(
    text: &str,
    reference_samples: usize,
    predicted_speed: f32,
    user_speed: f32,
    fixed_seconds: Option<f32>,
    english: &impl EnglishSyllableCounter,
) -> Result<Vec<CrossLingualChunk>> {
    if !predicted_speed.is_finite()
        || predicted_speed <= 0.0
        || !user_speed.is_finite()
        || user_speed <= 0.0
    {
        return Err(F5Error::InvalidConfiguration(
            "Invalid Cross-Lingual speaking rate",
        ));
    }
    if reference_samples == 0 {
        return Err(F5Error::InvalidShape(
            "Empty Cross-Lingual reference waveform",
        ));
    }
    let reference_frames = reference_samples / HOP_LENGTH as usize;
    let max_units = available_units(reference_samples, predicted_speed)?;
    let chunks = chunk_text(text, max_units, english)?;
    chunks
        .into_iter()
        .map(|chunk| {
            let units = count_units(&chunk, english)?;
            let extra_frames = generated_frames(units, predicted_speed, user_speed, None)?;
            let requested_frames = match fixed_seconds {
                Some(seconds) => {
                    if !seconds.is_finite() || seconds <= 0.0 {
                        return Err(F5Error::InvalidConfiguration(
                            "Invalid fixed Cross-Lingual duration",
                        ));
                    }
                    (seconds * SAMPLE_RATE / HOP_LENGTH).floor() as usize
                }
                None => reference_frames
                    .checked_add(extra_frames)
                    .ok_or(F5Error::InvalidShape("Cross-Lingual duration overflow"))?,
            };
            let minimum = reference_frames
                .checked_add(1)
                .ok_or(F5Error::InvalidShape(
                    "Cross-Lingual reference duration overflow",
                ))?;
            if requested_frames < minimum || requested_frames > 65_536 {
                return Err(F5Error::InvalidShape(
                    "Cross-Lingual target duration must exceed reference duration",
                ));
            }
            Ok(CrossLingualChunk {
                text: chunk,
                syllable_units: units,
                total_frames: requested_frames,
            })
        })
        .collect()
}

pub fn cross_fade_waves(waves: &[Vec<f32>], fade_seconds: f32) -> Result<Vec<f32>> {
    if !fade_seconds.is_finite() || fade_seconds < 0.0 {
        return Err(F5Error::InvalidConfiguration(
            "Cross-fade duration must be nonnegative and finite",
        ));
    }
    let Some(first) = waves.first() else {
        return Err(F5Error::InvalidShape(
            "Cannot cross-fade an empty waveform list",
        ));
    };
    if waves
        .iter()
        .any(|wave| wave.is_empty() || wave.iter().any(|sample| !sample.is_finite()))
    {
        return Err(F5Error::InvalidShape(
            "Cross-fade needs nonempty finite waveforms",
        ));
    }
    let mut joined = first.clone();
    let overlap_samples = (fade_seconds * SAMPLE_RATE).floor() as usize;
    for next_wave in &waves[1..] {
        let overlap = overlap_samples.min(joined.len()).min(next_wave.len());
        if overlap == 0 {
            joined.extend_from_slice(next_wave);
            continue;
        }
        let start = joined.len() - overlap;
        for i in 0..overlap {
            let alpha = if overlap == 1 {
                0.0
            } else {
                i as f32 / (overlap - 1) as f32
            };
            joined[start + i] = joined[start + i] * (1.0 - alpha) + next_wave[i] * alpha;
        }
        joined.extend_from_slice(&next_wave[overlap..]);
    }
    Ok(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct English;
    impl EnglishSyllableCounter for English {
        fn count_syllables(&self, english_word: &str) -> Result<usize> {
            Ok(if english_word == "hello" { 2 } else { 1 })
        }
    }

    #[test]
    fn linear_cross_fade_matches_numpy_linspace_endpoints() {
        let waves = vec![vec![1.0; 6], vec![3.0; 5]];
        let result = cross_fade_waves(&waves, 3.0 / SAMPLE_RATE).unwrap();
        assert_eq!(result, vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
        assert_eq!(cross_fade_waves(&waves, 0.0).unwrap().len(), 11);
        assert!(cross_fade_waves(&[], 0.0).is_err());
    }

    #[test]
    fn cross_lingual_duration_counts_reference_frames_and_fixed_seconds_are_total() {
        let chunks = plan_cross_lingual_chunks("Hello!", 24_000, 2.0, 1.0, None, &English).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].syllable_units, 4);
        assert_eq!(chunks[0].total_frames, 93 + 187);

        let fixed =
            plan_cross_lingual_chunks("Hello!", 24_000, 2.0, 1.0, Some(4.0), &English).unwrap();
        assert_eq!(fixed[0].total_frames, 375);
        assert!(
            plan_cross_lingual_chunks("Hello!", 24_000, 2.0, 1.0, Some(0.5), &English).is_err()
        );
    }

    #[test]
    fn mapper_matches_syllables_checkpoint() {
        assert_eq!(speed_from_label(0).unwrap(), 0.25);
        assert_eq!(speed_from_label(31).unwrap(), 8.0);
        assert!(speed_from_label(32).is_err());
    }

    #[test]
    fn punctuation_and_chinese_match_the_python_rule() {
        assert_eq!(count_units("Hello, 世界!", &English).unwrap(), 6);
    }

    #[test]
    fn argmax_breaks_ties_in_favor_of_first_class() {
        let mut logits = [0.0; SPEED_CLASSES];
        logits[1] = 2.0;
        logits[3] = 2.0;
        assert_eq!(best_speed_class(&logits).unwrap(), 1);
        assert_eq!(
            speed_from_label(best_speed_class(&logits).unwrap()).unwrap(),
            0.5
        );
    }

    #[test]
    fn scheduler_uses_reference_window() {
        assert_eq!(available_units(24_000 * 20, 2.0).unwrap(), 4);
        assert_eq!(generated_frames(4, 2.0, 1.0, None).unwrap(), 187);
        assert_eq!(generated_frames(3, 2.0, 1.0, None).unwrap(), 281);
    }

    #[test]
    fn chunker_preserves_sentence_and_punctuation() {
        let chunks = chunk_text("Hello, world. 再见。", 4, &English).unwrap();
        assert_eq!(chunks, vec!["Hello,", "world.", "再见。"]);
    }
}
