use std::{collections::HashMap, path::Path};

use crate::{F5Error, Result, cross_lingual::EnglishSyllableCounter};

pub struct EnglishHyphenator {
    patterns: HashMap<Vec<u8>, Vec<u8>>,
    longest: usize,
    left: usize,
    right: usize,
}

impl EnglishHyphenator {
    pub fn us_english() -> Result<Self> {
        Self::from_dictionary(include_str!("../assets/hyph_en_US.dic"))
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| F5Error::InvalidCheckpoint(error.to_string()))?;
        Self::from_dictionary(&text)
    }

    pub fn from_dictionary(dictionary: &str) -> Result<Self> {
        let mut patterns = HashMap::new();
        let mut longest = 0;
        for (index, line) in dictionary.lines().enumerate() {
            let pattern = line.trim();
            if index == 0 || pattern.is_empty() || pattern.starts_with('#') || pattern.contains(' ')
            {
                continue;
            }
            let mut letters = Vec::new();
            let mut priorities = vec![0_u8];
            for byte in pattern.bytes() {
                if byte.is_ascii_digit() {
                    *priorities.last_mut().unwrap() = byte - b'0';
                } else {
                    letters.push(byte.to_ascii_lowercase());
                    priorities.push(0);
                }
            }
            if letters.is_empty() || priorities.iter().all(|&value| value == 0) {
                continue;
            }
            longest = longest.max(letters.len());
            patterns.insert(letters, priorities);
        }
        if patterns.is_empty() {
            return Err(F5Error::InvalidCheckpoint(
                "Hyphenation dictionary contains no patterns".to_owned(),
            ));
        }
        Ok(Self {
            patterns,
            longest,
            left: 2,
            right: 2,
        })
    }

    pub fn syllables(&self, word: &str) -> Result<usize> {
        if !word.is_ascii() || !word.chars().all(|c| c.is_ascii_alphabetic() || c == '\'') {
            return Err(F5Error::InvalidConfiguration(
                "English syllable counter accepts ASCII letters and apostrophes",
            ));
        }
        let word = word.to_ascii_lowercase();
        let wrapped = format!(".{word}.");
        let bytes = wrapped.as_bytes();
        let mut priorities = vec![0_u8; bytes.len() + 1];
        for start in 0..bytes.len() {
            let last = (start + self.longest).min(bytes.len());
            for end in (start + 1)..=last {
                if let Some(pattern) = self.patterns.get(&bytes[start..end]) {
                    for (offset, &priority) in pattern.iter().enumerate() {
                        priorities[start + offset] = priorities[start + offset].max(priority);
                    }
                }
            }
        }
        let mut count = 1;
        let rightmost = word.len().saturating_sub(self.right);
        for (position, priority) in priorities.iter().enumerate() {
            let word_position = position.saturating_sub(1);
            if position >= 1
                && word_position >= self.left
                && word_position <= rightmost
                && priority % 2 == 1
            {
                count += 1;
            }
        }
        Ok(count)
    }
}

impl EnglishSyllableCounter for EnglishHyphenator {
    fn count_syllables(&self, english_word: &str) -> Result<usize> {
        self.syllables(english_word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_pyphen_us_dictionary() {
        let en = EnglishHyphenator::us_english().unwrap();
        let cases = [
            ("hello", 2),
            ("world", 1),
            ("crosslingual", 2),
            ("beautiful", 3),
            ("transformer", 2),
            ("I", 1),
            ("don't", 1),
            ("testing", 2),
            ("neural", 2),
            ("synthesis", 3),
            ("characteristic", 5),
        ];
        for (word, expected) in cases {
            assert_eq!(en.syllables(word).unwrap(), expected, "word={word}");
        }
    }

    #[test]
    fn parity_with_independent_pyphen_500_word_corpus() {
        let en = EnglishHyphenator::us_english().unwrap();
        let mut checked = 0;
        for row in include_str!("../tests/fixtures/hyphenation_pyphen_us_expected.tsv").lines() {
            let (word, units) = row.split_once('\t').unwrap();
            assert_eq!(
                en.syllables(word).unwrap(),
                units.parse::<usize>().unwrap(),
                "word={word}"
            );
            checked += 1;
        }
        assert_eq!(checked, 552);
    }

    #[test]
    fn pattern_priorities_and_left_right_match_pyphen() {
        let en = EnglishHyphenator::from_dictionary("UTF-8\n.ab1cd\n2b3c\n").unwrap();
        assert_eq!(en.syllables("abcd").unwrap(), 2);
        assert!(en.syllables("漢字").is_err());
    }
}
