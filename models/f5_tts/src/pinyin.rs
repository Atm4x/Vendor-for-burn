use std::{collections::HashMap, sync::Arc};

use crate::{F5Error, Result, text_frontend::PinyinProvider};

const CHAR_PRONUNCIATIONS: &str = include_str!("../assets/han_pinyin_unihan.tsv");
const WORD_PRONUNCIATIONS: &str = include_str!("../assets/han_pinyin_words.tsv");
const PYPINYIN_CHAR_PRONUNCIATIONS: &str = include_str!("../assets/pypinyin_chars.tsv");
const PYPINYIN_WORD_PRONUNCIATIONS: &str = include_str!("../assets/pypinyin_phrases.tsv");

/// Model-specific Chinese text-to-pinyin lexicon. No Python modules or network requests at runtime.
/// The bundled character table is based on Unihan, not the polyphonic pypinyin dictionary.
#[derive(Debug, Clone)]
pub struct ChinesePinyin {
    characters: HashMap<char, String>,
    words: HashMap<String, Vec<String>>,
    exact_words: HashMap<String, Vec<String>>,
    longest_word: usize,
}

impl ChinesePinyin {
    pub fn bundled() -> Result<Self> {
        let mut dictionary = Self::from_tsv(CHAR_PRONUNCIATIONS, WORD_PRONUNCIATIONS)?;
        dictionary
            .load_exact_dictionary(PYPINYIN_CHAR_PRONUNCIATIONS, PYPINYIN_WORD_PRONUNCIATIONS)?;
        Ok(dictionary)
    }

    /// Word pronunciations override individual character readings before tone sandhi.
    /// Provide an rjieba-compatible word lexicon to improve polyphonic word coverage.
    pub fn from_tsv(characters: &str, words: &str) -> Result<Self> {
        let mut dictionary = HashMap::new();
        for (number, row) in characters.lines().enumerate() {
            if row.trim().is_empty() || row.starts_with('#') {
                continue;
            }
            let (han, pronunciation) = row.split_once('\t').ok_or_else(|| {
                F5Error::InvalidCheckpoint(format!("Invalid Han reading at line {}", number + 1))
            })?;
            let mut chars = han.chars();
            let letter = chars.next().ok_or_else(|| {
                F5Error::InvalidCheckpoint("Empty Han dictionary key".to_string())
            })?;
            if chars.next().is_some() || !is_han(letter) || !valid_pinyin(pronunciation) {
                return Err(F5Error::InvalidCheckpoint(format!(
                    "Invalid Han pinyin record at line {}",
                    number + 1
                )));
            }
            if dictionary
                .insert(letter, pronunciation.to_owned())
                .is_some()
            {
                return Err(F5Error::InvalidCheckpoint(format!(
                    "Duplicate Han pronunciation for {han}"
                )));
            }
        }
        if dictionary.is_empty() {
            return Err(F5Error::InvalidCheckpoint(
                "Empty Han pinyin dictionary".to_string(),
            ));
        }
        let mut result = Self {
            characters: dictionary,
            words: HashMap::new(),
            exact_words: HashMap::new(),
            longest_word: 1,
        };
        result.extend_words(words)?;
        Ok(result)
    }

    /// Override or extend word pronunciations without changing the character table.
    pub fn extend_words(&mut self, words: &str) -> Result<()> {
        for (index, row) in words.lines().enumerate() {
            if row.trim().is_empty() || row.starts_with('#') {
                continue;
            }
            let (han, readings) = row.split_once('\t').ok_or_else(|| {
                F5Error::InvalidCheckpoint(format!("Invalid pinyin phrase at line {}", index + 1))
            })?;
            let symbols = han.chars().collect::<Vec<_>>();
            let pronunciation = readings.split_whitespace().collect::<Vec<_>>();
            if symbols.len() < 2
                || symbols.len() > 64
                || !symbols.iter().all(|&c| is_han(c))
                || symbols.len() != pronunciation.len()
                || !pronunciation.iter().all(|&reading| valid_pinyin(reading))
            {
                return Err(F5Error::InvalidCheckpoint(format!(
                    "Invalid pinyin phrase shape at line {}",
                    index + 1
                )));
            }
            self.longest_word = self.longest_word.max(symbols.len());
            let readings = pronunciation
                .iter()
                .map(|value| (*value).to_string())
                .collect();
            if self.exact_words.is_empty() {
                self.words.insert(han.to_string(), readings);
            } else {
                self.exact_words.insert(han.to_string(), readings);
            }
        }
        Ok(())
    }

    pub fn load_exact_dictionary(&mut self, characters: &str, words: &str) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for (index, row) in characters.lines().enumerate() {
            if row.is_empty() || row.starts_with('#') {
                continue;
            }
            let (symbol, reading) = row.split_once('\t').ok_or_else(|| {
                F5Error::InvalidCheckpoint(format!(
                    "Invalid exact pinyin character at line {}",
                    index + 1
                ))
            })?;
            let mut chars = symbol.chars();
            let ch = chars.next().ok_or_else(|| {
                F5Error::InvalidCheckpoint("Empty exact pinyin character".to_string())
            })?;
            if chars.next().is_some() || !is_han(ch) || !valid_pinyin(reading) || !seen.insert(ch) {
                return Err(F5Error::InvalidCheckpoint(format!(
                    "Invalid or duplicate exact character at line {}",
                    index + 1
                )));
            }
            self.characters.insert(ch, reading.to_owned());
        }
        let mut seen = std::collections::HashSet::new();
        for (index, row) in words.lines().enumerate() {
            if row.is_empty() || row.starts_with('#') {
                continue;
            }
            let (word, readings) = row.split_once('\t').ok_or_else(|| {
                F5Error::InvalidCheckpoint(format!(
                    "Invalid exact pinyin phrase at line {}",
                    index + 1
                ))
            })?;
            let width = word.chars().count();
            let syllables: Vec<_> = readings.split_whitespace().collect();
            if !(2..=64).contains(&width)
                || !word.chars().all(is_han)
                || syllables.len() != width
                || !syllables.iter().all(|item| valid_pinyin(item))
                || !seen.insert(word.to_string())
            {
                return Err(F5Error::InvalidCheckpoint(format!(
                    "Invalid or duplicate exact phrase at line {}",
                    index + 1
                )));
            }
            self.longest_word = self.longest_word.max(width);
            self.exact_words.insert(
                word.to_owned(),
                syllables.iter().map(|item| (*item).to_owned()).collect(),
            );
        }
        Ok(())
    }

    pub fn exact_word_count(&self) -> usize {
        self.exact_words.len()
    }

    pub fn character_count(&self) -> usize {
        self.characters.len()
    }

    pub fn word_count(&self) -> usize {
        self.words.len() + self.exact_words.len()
    }

    pub fn syllables_for_han(&self, han: &str) -> Result<Vec<String>> {
        if han.is_empty() || !han.chars().all(is_han) {
            return Err(F5Error::InvalidConfiguration(
                "Chinese pinyin expects a nonempty Han-only segment",
            ));
        }
        let chars = han.chars().collect::<Vec<_>>();
        let mut syllables = Vec::with_capacity(chars.len());
        let mut offset = 0;
        while offset < chars.len() {
            let mut matched = None;
            let maximum = (chars.len() - offset).min(self.longest_word);
            for width in (2..=maximum).rev() {
                let candidate = chars[offset..offset + width].iter().collect::<String>();
                if let Some(readings) = self.exact_words.get(&candidate) {
                    matched = Some((width, readings, true));
                    break;
                }
                if let Some(readings) = self.words.get(&candidate) {
                    matched = Some((width, readings, false));
                    break;
                }
            }
            if let Some((width, readings, exact)) = matched {
                let mut word_readings = readings.clone();
                if !exact {
                    sandhi_word(&chars[offset..offset + width], &mut word_readings);
                }
                syllables.extend(word_readings);
                offset += width;
            } else {
                let reading = self.characters.get(&chars[offset]).ok_or_else(|| {
                    F5Error::InvalidCheckpoint(format!(
                        "No native pinyin for Unicode character U+{:04X}",
                        chars[offset] as u32
                    ))
                })?;
                syllables.push(reading.clone());
                offset += 1;
            }
        }
        Ok(syllables)
    }
}

impl PinyinProvider for ChinesePinyin {
    fn syllables(&self, han: &str) -> Result<Vec<String>> {
        self.syllables_for_han(han)
    }
}

impl PinyinProvider for &ChinesePinyin {
    fn syllables(&self, han: &str) -> Result<Vec<String>> {
        self.syllables_for_han(han)
    }
}

impl PinyinProvider for Arc<ChinesePinyin> {
    fn syllables(&self, han: &str) -> Result<Vec<String>> {
        self.syllables_for_han(han)
    }
}

fn is_han(value: char) -> bool {
    ('\u{3100}'..='\u{9fff}').contains(&value)
}

fn valid_pinyin(value: &str) -> bool {
    if value.is_empty() || !value.is_ascii() {
        return false;
    }
    let base = value.trim_end_matches(|ch: char| ch.is_ascii_digit());
    let suffix = &value[base.len()..];
    !base.is_empty()
        && base.bytes().all(|ch| ch.is_ascii_lowercase())
        && (suffix.is_empty() || suffix.len() == 1 && matches!(suffix, "1" | "2" | "3" | "4" | "5"))
}

fn tone(value: &str) -> Option<char> {
    value.chars().last().filter(|ch| ('1'..='5').contains(ch))
}

fn replace_tone(value: &mut String, next: char) {
    if tone(value).is_some() {
        value.pop();
    }
    value.push(next);
}

/// Apply bounded word-local tone sandhi. Does not claim to match every pypinyin
/// multiword / neutral-tone exception; those need lexical overrides.
fn sandhi_word(word: &[char], readings: &mut [String]) {
    for index in 0..readings.len() {
        if word[index] == '不' && readings.get(index + 1).and_then(|next| tone(next)) == Some('4')
        {
            replace_tone(&mut readings[index], '2');
        }
        if word[index] == '一' && index > 0 && word[index - 1] == '第' {
            replace_tone(&mut readings[index], '1');
        } else if word[index] == '一' {
            match readings.get(index + 1).and_then(|next| tone(next)) {
                Some('4') => replace_tone(&mut readings[index], '2'),
                Some('1' | '2' | '3') => replace_tone(&mut readings[index], '4'),
                _ => {}
            }
        }
    }
    for index in 0..readings.len().saturating_sub(1) {
        if tone(&readings[index]) == Some('3') && tone(&readings[index + 1]) == Some('3') {
            replace_tone(&mut readings[index], '2');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_table_has_unihan_coverage() {
        let pinyin = ChinesePinyin::bundled().unwrap();
        assert!(pinyin.character_count() >= 26_676);
        assert!(pinyin.exact_word_count() >= 40_000);
        assert!(pinyin.word_count() >= 20);
        assert_eq!(
            pinyin.syllables_for_han("中国").unwrap(),
            ["zhong1", "guo2"]
        );
    }

    #[test]
    fn word_readings_and_tone_sandhi() {
        let pinyin = ChinesePinyin::bundled().unwrap();
        assert_eq!(pinyin.syllables_for_han("你好").unwrap(), ["ni2", "hao3"]);
        assert_eq!(pinyin.syllables_for_han("银行").unwrap(), ["yin2", "hang2"]);
        assert_eq!(
            pinyin.syllables_for_han("重庆").unwrap(),
            ["chong2", "qing4"]
        );
        assert_eq!(pinyin.syllables_for_han("一共").unwrap(), ["yi1", "gong4"]);
        assert_eq!(pinyin.syllables_for_han("不要").unwrap(), ["bu2", "yao4"]);
        assert_eq!(pinyin.syllables_for_han("第一").unwrap(), ["di4", "yi1"]);
    }

    #[test]
    fn word_overrides_have_priority_and_are_replaceable() {
        let mut pinyin = ChinesePinyin::bundled().unwrap();
        pinyin.extend_words("银行\tyin2 xing2\n").unwrap();
        assert_eq!(pinyin.syllables_for_han("银行").unwrap(), ["yin2", "xing2"]);
    }

    #[test]
    fn full_text_frontend_handles_chinese_without_python() {
        use crate::{text_frontend::F5TextFrontend, vocab::Vocabulary};
        let pinyin = ChinesePinyin::bundled().unwrap();
        let vocab = Vocabulary::from_lines(
            [" ", "ni2", "hao3", "yin2", "hang2", "O", "K", "!", "?", "x"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();
        let frontend = F5TextFrontend::new(&vocab, &pinyin);
        assert_eq!(
            frontend.cross_lingual_tokens("你好").unwrap(),
            vec![0, 1, 0, 2]
        );
        assert_eq!(
            frontend.cross_lingual_tokens("银行").unwrap(),
            vec![0, 3, 0, 4]
        );
        assert_eq!(
            frontend.cross_lingual_tokens("OK 你好!").unwrap(),
            vec![5, 6, 0, 0, 1, 0, 2, 7]
        );
    }

    #[test]
    fn entire_bundled_table_is_decodable() {
        let pinyin = ChinesePinyin::bundled().unwrap();
        let mut count = 0;
        for row in PYPINYIN_CHAR_PRONUNCIATIONS.lines() {
            let (han, expected) = row.split_once('\t').unwrap();
            let actual = pinyin.syllables_for_han(han).unwrap();
            assert_eq!(actual, [expected], "{han}");
            count += 1;
        }
        assert_eq!(count, 26_703);
    }

    #[test]
    fn invalid_lexicons_and_unknown_han_are_not_silently_accepted() {
        assert!(ChinesePinyin::from_tsv("你\tni3\n你\tni3", "").is_err());
        assert!(ChinesePinyin::from_tsv("你\tni3", "银行\tyin2").is_err());
        let pinyin = ChinesePinyin::bundled().unwrap();
        assert!(pinyin.syllables_for_han("你好!").is_err());
    }
}
