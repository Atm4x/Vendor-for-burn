use std::sync::LazyLock;

use jieba_rs::Jieba;

use crate::{F5Error, Result, vocab::Vocabulary};

static CHINESE_SEGMENTER: LazyLock<Jieba> = LazyLock::new(Jieba::new);

pub trait PinyinProvider {
    /// Returns one tone-marked pinyin syllable per input Han character.
    fn syllables(&self, han: &str) -> Result<Vec<String>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoPinyin;

impl PinyinProvider for NoPinyin {
    fn syllables(&self, _: &str) -> Result<Vec<String>> {
        Err(F5Error::InvalidConfiguration(
            "Chinese text needs a tone-sandhi-aware pinyin provider",
        ))
    }
}

pub struct F5TextFrontend<'a, P> {
    vocabulary: &'a Vocabulary,
    pinyin: P,
}

impl<'a, P: PinyinProvider> F5TextFrontend<'a, P> {
    pub fn new(vocabulary: &'a Vocabulary, pinyin: P) -> Self {
        Self { vocabulary, pinyin }
    }

    pub fn standard_tokens(&self, reference_text: &str, generated_text: &str) -> Result<Vec<i32>> {
        let reference = effective_reference_transcript(reference_text)?;
        if generated_text.trim().is_empty() {
            return Err(F5Error::InvalidConfiguration(
                "Generation text must not be empty",
            ));
        }
        self.encode(&format!("{reference}{generated_text}"))
    }

    pub fn cross_lingual_tokens(&self, generated_text: &str) -> Result<Vec<i32>> {
        if generated_text.trim().is_empty() {
            return Err(F5Error::InvalidConfiguration(
                "Generation text must not be empty",
            ));
        }
        self.encode(generated_text)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let normalized = normalize_punctuation(text);
        let segments = CHINESE_SEGMENTER.cut(&normalized, true);
        let words = segments.iter().map(|item| item.word).collect::<Vec<_>>();
        self.encode_segments(&words)
    }

    pub fn encode_segments(&self, segments: &[&str]) -> Result<Vec<i32>> {
        let mut normalized = Vec::<String>::new();
        for raw in segments {
            let segment = normalize_punctuation(raw);
            let chars = segment.chars().count();
            if chars == 0 {
                continue;
            }
            if segment.is_ascii() {
                if segment.len() > 1
                    && !normalized.is_empty()
                    && !matches!(
                        normalized.last().map(String::as_str),
                        Some(" " | ":" | "'" | "\"")
                    )
                {
                    normalized.push(" ".to_owned());
                }
                normalized.extend(segment.chars().map(|ch| ch.to_string()));
            } else if segment.len() == chars * 3 {
                self.append_han_and_punctuation(&segment, &mut normalized)?;
            } else {
                for ch in segment.chars() {
                    if is_han(ch) {
                        let char_str = ch.to_string();
                        let result = self.pinyin.syllables(&char_str)?;
                        self.append_pinyin(&char_str, result, &mut normalized)?;
                    } else {
                        normalized.push(ch.to_string());
                    }
                }
            }
        }
        if normalized.is_empty() || normalized.len() > 8192 {
            return Err(F5Error::InvalidShape(
                "F5 normalized text must contain 1..8192 tokens",
            ));
        }
        Ok(normalized
            .iter()
            .map(|token| self.vocabulary.token_id(token))
            .collect())
    }

    fn append_han_and_punctuation(&self, segment: &str, target: &mut Vec<String>) -> Result<()> {
        let mut chars = segment.chars().peekable();
        while let Some(ch) = chars.next() {
            if !is_han(ch) {
                target.push(ch.to_string());
                continue;
            }
            let mut word = String::from(ch);
            while chars.peek().is_some_and(|next| is_han(*next)) {
                word.push(chars.next().unwrap());
            }
            let syllables = self.pinyin.syllables(&word)?;
            self.append_pinyin(&word, syllables, target)?;
        }
        Ok(())
    }

    fn append_pinyin(
        &self,
        han: &str,
        syllables: Vec<String>,
        target: &mut Vec<String>,
    ) -> Result<()> {
        if syllables.len() != han.chars().count()
            || syllables
                .iter()
                .any(|word| word.is_empty() || !word.is_ascii())
        {
            return Err(F5Error::InvalidConfiguration(
                "Pinyin provider must return one ASCII syllable per Han character",
            ));
        }
        for syllable in syllables {
            target.push(" ".to_owned());
            target.push(syllable);
        }
        Ok(())
    }
}

fn normalize_punctuation(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            ';' => ',',
            '“' | '”' => '"',
            '‘' | '’' => '\'',
            other => other,
        })
        .collect()
}

fn is_han(ch: char) -> bool {
    ('\u{3100}'..='\u{9fff}').contains(&ch)
}

pub fn normalized_reference_transcript(text: &str) -> Result<String> {
    let mut reference = text.trim().to_owned();
    if reference.is_empty() {
        return Err(F5Error::InvalidConfiguration(
            "Standard F5 requires an explicit reference transcript",
        ));
    }
    if !reference.ends_with(". ") && !reference.ends_with('。') {
        if reference.ends_with('.') {
            reference.push(' ');
        } else {
            reference.push_str(". ");
        }
    }
    Ok(reference)
}

pub fn effective_reference_transcript(text: &str) -> Result<String> {
    let mut transcript = normalized_reference_transcript(text)?;
    if transcript.chars().last().is_some_and(|ch| ch.is_ascii()) {
        transcript.push(' ');
    }
    Ok(transcript)
}

pub fn standard_total_frames(
    reference_frames: usize,
    reference_text: &str,
    generated_text: &str,
    speed: f32,
    fixed_seconds: Option<f32>,
) -> Result<usize> {
    if reference_frames == 0 || !speed.is_finite() || speed <= 0.0 {
        return Err(F5Error::InvalidConfiguration(
            "Invalid standard F5 reference or speed",
        ));
    }
    let reference = effective_reference_transcript(reference_text)?;
    let ref_len = reference.len();
    let gen_len = generated_text.len();
    if gen_len == 0 || generated_text.trim().is_empty() {
        return Err(F5Error::InvalidConfiguration(
            "Generation text must not be empty",
        ));
    }
    let frames = if let Some(seconds) = fixed_seconds {
        if !seconds.is_finite() {
            return Err(F5Error::InvalidConfiguration(
                "Fixed duration must be finite",
            ));
        }
        (seconds as f64 * 24_000.0 / 256.0).floor()
    } else {
        let local_speed = if gen_len < 10 {
            speed as f64 * 0.3
        } else {
            speed as f64
        };
        reference_frames as f64
            + ((reference_frames as f64 / ref_len as f64) * gen_len as f64 / local_speed).floor()
    };
    if !frames.is_finite() || frames > 65_536.0 {
        return Err(F5Error::InvalidShape(
            "Generated frames exceed model limits",
        ));
    }
    Ok(frames.max(0.0) as usize)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceSegment {
    pub voice: String,
    pub text: String,
}

/// Parses original F5 `[voice]` markup without binding the neural runtime to a voice registry.
pub fn parse_voice_segments(input: &str) -> Vec<VoiceSegment> {
    let mut sections = Vec::new();
    let mut current_voice = "main".to_owned();
    let mut start = 0;
    let mut position = 0;
    while position < input.len() {
        let rest = &input[position..];
        if rest.starts_with('[') {
            if let Some(end_relative) = rest.find(']') {
                let name = &rest[1..end_relative];
                if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    let segment = input[start..position].trim();
                    if !segment.is_empty() {
                        sections.push(VoiceSegment {
                            voice: current_voice.clone(),
                            text: segment.to_owned(),
                        });
                    }
                    current_voice = name.to_owned();
                    position += end_relative + 1;
                    start = position;
                    continue;
                }
            }
        }
        position += rest.chars().next().unwrap().len_utf8();
    }
    let last = input[start..].trim();
    if !last.is_empty() {
        sections.push(VoiceSegment {
            voice: current_voice,
            text: last.to_owned(),
        });
    }
    sections
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakePinyin;
    impl PinyinProvider for FakePinyin {
        fn syllables(&self, han: &str) -> Result<Vec<String>> {
            han.chars()
                .map(|ch| match ch {
                    '你' => Ok("ni3".to_owned()),
                    '好' => Ok("hao3".to_owned()),
                    _ => Err(F5Error::InvalidConfiguration(
                        "Unexpected fixture Han character",
                    )),
                })
                .collect()
        }
    }

    #[test]
    fn original_reference_punctuation_and_duration_rules() {
        assert_eq!(normalized_reference_transcript("Hello").unwrap(), "Hello. ");
        assert_eq!(
            normalized_reference_transcript("Hello.").unwrap(),
            "Hello. "
        );
        assert_eq!(normalized_reference_transcript("你好。").unwrap(), "你好。");
        assert_eq!(
            standard_total_frames(120, "Hello", "some speech", 1.0, None).unwrap(),
            120 + (120 * 11 / 8)
        );
        assert_eq!(
            standard_total_frames(120, "Hello", "Hi", 1.0, None).unwrap(),
            220
        );
        assert_eq!(
            standard_total_frames(120, "Hello", "Hi", 1.0, Some(2.0)).unwrap(),
            187
        );
        assert_eq!(
            standard_total_frames(120, "Hello", "Hi", 1.0, Some(0.5)).unwrap(),
            46
        );
    }

    #[test]
    fn normalized_english_and_chinese_path() {
        let alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'\"!?";
        let vocab = Vocabulary::from_lines(
            alphabet
                .chars()
                .map(|c| c.to_string())
                .chain(["ni3".to_owned(), "hao3".to_owned()]),
        )
        .unwrap();
        let english = F5TextFrontend::new(&vocab, NoPinyin);
        let full = english.standard_tokens("Hello", "world").unwrap();
        assert_eq!(
            full,
            vocab.encode_tokens(
                "Hello.  world"
                    .chars()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .iter()
                    .map(String::as_str)
            )
        );
        assert!(english.cross_lingual_tokens("你好").is_err());
        let multilingual = F5TextFrontend::new(&vocab, FakePinyin);
        let romanized = multilingual.cross_lingual_tokens("你好!").unwrap();
        let expected = vocab.encode_tokens([" ", "ni3", " ", "hao3", "!"].into_iter());
        assert_eq!(romanized, expected);
        let mixed = multilingual.cross_lingual_tokens("你好world!").unwrap();
        let target = vocab.encode_tokens(
            [" ", "ni3", " ", "hao3", " ", "w", "o", "r", "l", "d", "!"].into_iter(),
        );
        assert_eq!(mixed, target);
        let rjieba_segments = multilingual
            .encode_segments(&["你好", "world", "!"])
            .unwrap();
        assert_eq!(mixed, rjieba_segments);

        assert_eq!(
            english.encode("he;“yo”").unwrap(),
            english.encode("he,\"yo\"").unwrap()
        );
    }

    #[test]
    fn speaker_markers_match_f5_segment_boundaries() {
        assert_eq!(
            parse_voice_segments("one [alice] hello [bob] world"),
            vec![
                VoiceSegment {
                    voice: "main".into(),
                    text: "one".into()
                },
                VoiceSegment {
                    voice: "alice".into(),
                    text: "hello".into()
                },
                VoiceSegment {
                    voice: "bob".into(),
                    text: "world".into()
                },
            ]
        );
        assert_eq!(parse_voice_segments("[a]hi[b]there").len(), 2);
        assert_eq!(parse_voice_segments(""), Vec::<VoiceSegment>::new());
    }
}
