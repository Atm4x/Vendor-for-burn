//! BERT BasicTokenizer + greedy WordPiece for Silero HomoSolver.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::{collections::{HashMap, HashSet}, fs, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct TokenizerConfig {
    pub unk_token: String,
    pub cls_token: String,
    pub sep_token: String,
    pub pad_token: String,
    pub homo_start_id: usize,
    pub homo_end_id: usize,
    pub never_split: Vec<String>,
    pub tokenize_chinese_chars: bool,
    pub model_max_length: usize,
}

pub struct WordPieceTokenizer {
    vocab: HashMap<String, usize>,
    config: TokenizerConfig,
    never_split: HashSet<String>,
}

impl WordPieceTokenizer {
    pub fn open(folder: &Path) -> Result<Self> {
        let json: serde_json::Value = serde_json::from_slice(&fs::read(folder.join("config.json"))?)?;
        let config: TokenizerConfig = serde_json::from_value(json["tokenizer"].clone())?;
        let vocab = serde_json::from_slice(&fs::read(folder.join("vocab.json"))?)?;
        let never_split = config.never_split.iter().cloned().collect();
        Ok(Self { vocab, config, never_split })
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    pub fn from_parts(config: TokenizerConfig, vocab: HashMap<String, usize>) -> Self {
        let never_split = config.never_split.iter().cloned().collect();
        Self { config, vocab, never_split }
    }

    fn is_chinese(ch: char) -> bool {
        let c = ch as u32;
        (0x4e00..=0x9fff).contains(&c)
            || (0x3400..=0x4dbf).contains(&c)
            || (0x20000..=0x2a6df).contains(&c)
            || (0x2a700..=0x2b73f).contains(&c)
            || (0x2b740..=0x2b81f).contains(&c)
            || (0x2b820..=0x2ceaf).contains(&c)
            || (0xf900..=0xfaff).contains(&c)
            || (0x2f800..=0x2fa1f).contains(&c)
    }

    fn is_punctuation(ch: char) -> bool {
        ch.is_ascii_punctuation() || matches!(ch,
            '«'|'»'|'„'|'“'|'”'|'‘'|'’'|'—'|'–'|'…'|'。'|'、'|'！'|'？'|'：'|'；'|'，'|'（'|'）'|'【'|'】'|'『'|'』'|'「'|'」')
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut cleaned = String::with_capacity(text.len());
        for ch in text.chars() {
            if ch == '\0' || ch == '\u{fffd}' || (ch.is_control() && !ch.is_whitespace()) {
                continue;
            }
            if ch.is_whitespace() {
                cleaned.push(' ');
            } else if self.config.tokenize_chinese_chars && Self::is_chinese(ch) {
                cleaned.push(' ');
                cleaned.push(ch);
                cleaned.push(' ');
            } else {
                cleaned.push(ch);
            }
        }
        let mut output = Vec::new();
        for part in cleaned.split_whitespace() {
            if self.never_split.contains(part) {
                output.push(part.to_string());
                continue;
            }
            let mut basic = Vec::new();
            let mut current = String::new();
            for ch in part.chars() {
                if Self::is_punctuation(ch) {
                    if !current.is_empty() {
                        basic.push(std::mem::take(&mut current));
                    }
                    basic.push(ch.to_string());
                } else {
                    current.push(ch);
                }
            }
            if !current.is_empty() {
                basic.push(current);
            }
            for item in basic {
                let chars: Vec<char> = item.chars().collect();
                if chars.len() > 100 {
                    output.push(self.config.unk_token.clone());
                    continue;
                }
                let mut subwords = Vec::new();
                let mut start = 0;
                let mut failed = false;
                while start < chars.len() {
                    let mut found = None;
                    for end in (start+1..=chars.len()).rev() {
                        let mut candidate: String = chars[start..end].iter().collect();
                        if start != 0 {
                            candidate.insert_str(0, "##");
                        }
                        if self.vocab.contains_key(&candidate) {
                            found = Some((end, candidate));
                            break;
                        }
                    }
                    if let Some((end, token)) = found {
                        subwords.push(token);
                        start = end;
                    } else {
                        failed = true;
                        break;
                    }
                }
                if failed {
                    output.push(self.config.unk_token.clone());
                } else {
                    output.extend(subwords);
                }
            }
        }
        output
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let tokens = std::iter::once(self.config.cls_token.clone())
            .chain(self.tokenize(text))
            .chain(std::iter::once(self.config.sep_token.clone()));
        tokens.map(|t| self.vocab.get(&t).copied().with_context(|| {
            format!("token {t:?} missing from exported vocab")
        })).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn splits_greedy_wordpieces() {
        let config = TokenizerConfig {
            unk_token: "[UNK]".into(), cls_token: "[CLS]".into(), sep_token: "[SEP]".into(),
            pad_token: "[PAD]".into(), homo_start_id: 4, homo_end_id: 5,
            never_split: vec!["[HOMO]".into()], tokenize_chinese_chars: true,
            model_max_length: 2048,
        };
        let vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[HOMO]", "[/HOMO]", "при", "##вет", "!", "中"]
            .iter().enumerate().map(|(id,token)| (token.to_string(), id)).collect();
        let tokenizer = WordPieceTokenizer::from_parts(config, vocab);
        assert_eq!(tokenizer.tokenize("привет!"), ["при", "##вет", "!"]);
        assert_eq!(tokenizer.encode("привет!").unwrap(), [1, 6, 7, 8, 2]);
    }
}
