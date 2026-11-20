use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{F5Error, Result};

#[derive(Debug, Clone)]
pub struct Vocabulary {
    ids: HashMap<String, i32>,
}

impl Vocabulary {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let reader = BufReader::new(
            File::open(path).map_err(|error| F5Error::InvalidCheckpoint(error.to_string()))?,
        );
        let mut ids = HashMap::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line.map_err(|error| F5Error::InvalidCheckpoint(error.to_string()))?;
            let id = i32::try_from(index)
                .map_err(|_| F5Error::InvalidConfiguration("Vocabulary exceeds i32 index range"))?;
            if ids.insert(line.clone(), id).is_some() {
                return Err(F5Error::InvalidCheckpoint(format!(
                    "Duplicate token in vocabulary: {line}"
                )));
            }
        }
        if ids.is_empty() {
            return Err(F5Error::InvalidCheckpoint("Vocabulary is empty".to_owned()));
        }
        if ids.get(" ") != Some(&0) {
            return Err(F5Error::InvalidCheckpoint(
                "F5-TTS expects the space token at vocabulary index 0".to_owned(),
            ));
        }
        Ok(Self { ids })
    }

    pub fn from_lines(lines: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut ids = HashMap::new();
        for (index, line) in lines.into_iter().enumerate() {
            let id = i32::try_from(index)
                .map_err(|_| F5Error::InvalidConfiguration("Vocabulary exceeds i32 index range"))?;
            if ids.insert(line.clone(), id).is_some() {
                return Err(F5Error::InvalidCheckpoint(format!(
                    "Duplicate token in vocabulary: {line}"
                )));
            }
        }
        if ids.get(" ") != Some(&0) {
            return Err(F5Error::InvalidCheckpoint(
                "F5-TTS expects space at vocabulary index 0".to_owned(),
            ));
        }
        Ok(Self { ids })
    }

    pub fn size(&self) -> usize {
        self.ids.len()
    }

    pub fn encode_tokens<'a>(&self, tokens: impl IntoIterator<Item = &'a str>) -> Vec<i32> {
        tokens
            .into_iter()
            .map(|token| *self.ids.get(token).unwrap_or(&0))
            .collect()
    }

    pub fn encode_plain_ascii(&self, text: &str) -> Result<Vec<i32>> {
        if !text.is_ascii() {
            return Err(F5Error::InvalidConfiguration(
                "Raw CJK/Unicode must first pass the F5-TTS pinyin normalizer",
            ));
        }
        Ok(text
            .chars()
            .map(|token| *self.ids.get(&token.to_string()).unwrap_or(&0))
            .collect())
    }

    pub fn pad_batch(&self, sequences: &[Vec<i32>]) -> Result<(Vec<i32>, usize)> {
        if sequences.is_empty() || sequences.iter().any(Vec::is_empty) {
            return Err(F5Error::InvalidShape(
                "Token batch must contain nonempty sequences",
            ));
        }
        let max_len = sequences.iter().map(Vec::len).max().unwrap();
        let mut flat = vec![-1; sequences.len() * max_len];
        for (batch, sequence) in sequences.iter().enumerate() {
            flat[batch * max_len..batch * max_len + sequence.len()].copy_from_slice(sequence);
        }
        Ok((flat, max_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_ids_follow_pytorch_unknown_and_padding_rules() {
        let vocab = Vocabulary::from_lines([" ", "h", "i"].into_iter().map(str::to_owned)).unwrap();
        let encoded = vocab.encode_plain_ascii("hi?").unwrap();
        assert_eq!(encoded, vec![1, 2, 0]);
        let (flat, width) = vocab.pad_batch(&[encoded, vec![2]]).unwrap();
        assert_eq!(width, 3);
        assert_eq!(flat, vec![1, 2, 0, 2, -1, -1]);
    }

    #[test]
    fn unsupported_raw_unicode_requires_explicit_normalization() {
        let vocab = Vocabulary::from_lines([" ", "h"].into_iter().map(str::to_owned)).unwrap();
        assert!(vocab.encode_plain_ascii("你好").is_err());
    }
}
