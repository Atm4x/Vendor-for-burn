use anyhow::{Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub format_version: u32,
    pub symbols: String,
    pub alphabet: String,
    pub sos: String,
    pub eos: String,
    pub speakers: BTreeMap<String, usize>,
    pub sample_rates: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanText {
    pub sentence: String,
    pub clean_sentence: String,
}

impl ModelConfig {
    pub fn validate_speaker(&self, speaker: &str) -> Result<usize> {
        self.speakers.get(speaker).copied().ok_or_else(|| {
            anyhow::anyhow!("Unsupported Silero speaker: {speaker}")
        })
    }

    pub fn validate_sample_rate(&self, sample_rate: u32) -> Result<()> {
        if self.sample_rates.contains(&sample_rate) {
            Ok(())
        } else {
            bail!("Unsupported Silero sample rate: {sample_rate}")
        }
    }

    /// Reproduces the letter-mode `prepare_text_input` rules before accentuation.
    pub fn prepare_text_fragment(&self, text: &str) -> Result<CleanText> {
        let text = text.to_lowercase().replace(['—', '–'], "–").replace('‑', "-");
        let allowed = self.symbols.chars().skip(3).chain(std::iter::once('^'))
            .collect::<std::collections::HashSet<_>>();
        let filtered: String = text.chars().filter(|ch| allowed.contains(ch)).collect();
        let spaces = Regex::new(r"\s+")?;
        let mut sentence = spaces.replace_all(&filtered.replace('^', " ^"), " ").trim().to_owned();
        sentence = sentence.replace("^ ", "^");
        let carets = Regex::new(r"\^+")?;
        sentence = carets.replace_all(&sentence, "^").trim().to_owned();
        if sentence.ends_with('^') {
            sentence.pop();
            sentence = sentence.trim().to_owned();
        }
        let clean_sentence: String = sentence.chars().filter(|ch| {
            ('а'..='я').contains(ch) || *ch == '-' || *ch == ' '
        }).collect();
        Ok(CleanText { sentence, clean_sentence })
    }

    pub fn prepare_text(&self, text: &str) -> Result<CleanText> {
        let cleaned = self.prepare_text_fragment(text)?;
        if !cleaned.clean_sentence.chars().any(|ch| ('а'..='я').contains(&ch)) {
            bail!("Text is empty after Silero normalization");
        }
        Ok(cleaned)
    }

    /// `preprocess_tacotron`: accentuation must be applied before this call.
    pub fn encode_accented(&self, accented: &str) -> Result<(Vec<i64>, Vec<bool>)> {
        let text = format!("{}{}{}", self.sos, accented, self.eos);
        let mut ids = Vec::with_capacity(text.chars().count());
        let mut word_mask = Vec::with_capacity(ids.capacity());
        for symbol in text.chars() {
            let idx = self.symbols.chars().position(|ch| ch == symbol)
                .ok_or_else(|| anyhow::anyhow!("Unknown Silero symbol: {symbol:?}"))?;
            ids.push(idx as i64);
            word_mask.push(self.alphabet.contains(symbol));
        }
        Ok((ids, word_mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> ModelConfig {
        ModelConfig {
            format_version: 1,
            symbols: "_~|!+,-.:;?абвгдежзийклмнопрстуфхцчшщъыьэюяё–… ".into(),
            alphabet: "абвгдежзийклмнопрстуфхцчшщъыьэюяё+".into(),
            sos: "|".into(), eos: "~".into(),
            speakers: [("xenia".to_string(),4)].into(),
            sample_rates: vec![8000,24000,48000],
        }
    }

    #[test]
    fn normalizes_russian_text() {
        let actual = config().prepare_text("  ПРИВЕТ — МИР!  ").unwrap();
        assert_eq!(actual.sentence, "привет – мир!");
        assert_eq!(actual.clean_sentence, "привет  мир");
    }

    #[test]
    fn maps_symbols_to_original_ids() {
        let (ids,mask) = config().encode_accented("мир").unwrap();
        assert_eq!(ids, [2, 23, 19, 27, 1]);
        assert_eq!(mask, [false, true, true, true, false]);
    }

    #[test]
    fn rejects_non_russian_text() {
        assert!(config().prepare_text("English 123").is_err());
    }
}
