use base64::{Engine as _, engine::general_purpose::STANDARD};
use fish_speech_core::conversation::TokenEncoder;
use fish_speech_core::error::{FishError, Result as FishResult};
use fish_speech_core::tokenizer::{BASE_SPECIAL_TOKEN_COUNT, SEMANTIC_TOKEN_COUNT, SpecialTokens};
use std::collections::HashSet;
use std::path::Path;
use tiktoken_rs::CoreBPE;

const MAX_ENCODE_CHARS: usize = 400_000;
const FISH_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    "|",
    r"\p{P}",
    "|",
    r"[^\r\n\p{L}\p{N}]?\p{L}+",
    "|",
    r"\p{N}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+(\?!\S)",
    "|",
    r"\s+",
);

#[derive(Debug, thiserror::Error)]
pub enum FishTokenizerError {
    #[error("reading tokenizer: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid tokenizer BPE file: {0}")]
    InvalidBpe(String),
    #[error("tokenizer encoding error: {0}")]
    Encoding(String),
}

#[derive(Clone)]
pub struct FishBpeTokenizer {
    bpe: CoreBPE,
    special: SpecialTokens,
}

impl FishBpeTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> std::result::Result<Self, FishTokenizerError> {
        let source = std::fs::read_to_string(path)?;
        Self::from_tiktoken_text(&source)
    }

    pub fn from_tiktoken_text(source: &str) -> std::result::Result<Self, FishTokenizerError> {
        let mut ranks = Vec::new();
        let mut rank_ids = HashSet::new();
        let mut bytes_set = HashSet::new();
        for (i, line) in source.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let words: Vec<_> = line.split_ascii_whitespace().collect();
            if words.len() != 2 {
                return Err(FishTokenizerError::InvalidBpe(format!(
                    "line {} does not have token and rank",
                    i + 1
                )));
            }
            let bytes = STANDARD
                .decode(words[0])
                .map_err(|e| FishTokenizerError::InvalidBpe(format!("line {}: {e}", i + 1)))?;
            let rank = words[1]
                .parse::<u32>()
                .map_err(|e| FishTokenizerError::InvalidBpe(format!("line {}: {e}", i + 1)))?;
            if bytes.is_empty() || !bytes_set.insert(bytes.clone()) || !rank_ids.insert(rank) {
                return Err(FishTokenizerError::InvalidBpe(format!(
                    "duplicate or empty BPE token at line {}",
                    i + 1
                )));
            }
            ranks.push((bytes, rank));
        }
        if ranks.is_empty()
            || ranks.len()
                > (u32::MAX as usize)
                    .saturating_sub(BASE_SPECIAL_TOKEN_COUNT + SEMANTIC_TOKEN_COUNT)
        {
            return Err(FishTokenizerError::InvalidBpe("invalid rank count".into()));
        }
        if rank_ids.len() != ranks.len()
            || (0..ranks.len()).any(|i| !rank_ids.contains(&(i as u32)))
        {
            return Err(FishTokenizerError::InvalidBpe(
                "BPE ranks must be contiguous from zero".into(),
            ));
        }
        for byte in 0..=255_u8 {
            if !bytes_set.contains(&vec![byte]) {
                return Err(FishTokenizerError::InvalidBpe(format!(
                    "missing base byte {byte}"
                )));
            }
        }
        let special = SpecialTokens::new(ranks.len())
            .map_err(|e| FishTokenizerError::InvalidBpe(e.to_string()))?;
        let first = u32::try_from(ranks.len())
            .map_err(|e| FishTokenizerError::InvalidBpe(e.to_string()))?;
        let mut names = vec![
            "<|begin_of_text|>".to_string(),
            "<|end_of_text|>".to_string(),
            "<|pad|>".to_string(),
            "<|im_start|>".to_string(),
            "<|im_end|>".to_string(),
        ];
        for i in 0..4 {
            names.push(format!("<|placeholder:{i}|>"));
        }
        names.extend(["<|text|>", "<|voice|>", "<|interleave|>"].map(str::to_string));
        for i in 0..SEMANTIC_TOKEN_COUNT {
            names.push(format!("<|semantic:{i}|>"));
        }
        debug_assert_eq!(names.len(), BASE_SPECIAL_TOKEN_COUNT + SEMANTIC_TOKEN_COUNT);
        let special_names = names
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, first + i as u32))
            .collect();
        let bpe = CoreBPE::new(ranks.into_iter().collect(), special_names, FISH_PATTERN)
            .map_err(|e| FishTokenizerError::InvalidBpe(e.to_string()))?;
        Ok(Self { bpe, special })
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    pub fn decode(&self, tokens: &[u32]) -> std::result::Result<String, FishTokenizerError> {
        self.bpe
            .decode(tokens)
            .map_err(|e| FishTokenizerError::Encoding(e.to_string()))
    }

    pub fn encode_checked(&self, value: &str) -> std::result::Result<Vec<u32>, FishTokenizerError> {
        let allowed = self.bpe.special_tokens();
        let mut tokens = Vec::new();
        let mut chunk_start = 0;
        let mut count = 0;
        for (offset, _) in value.char_indices() {
            if count == MAX_ENCODE_CHARS {
                tokens.extend(
                    self.bpe
                        .encode(&value[chunk_start..offset], &allowed)
                        .map_err(|e| FishTokenizerError::Encoding(e.to_string()))?
                        .0,
                );
                chunk_start = offset;
                count = 0;
            }
            count += 1;
        }
        if chunk_start < value.len() {
            tokens.extend(
                self.bpe
                    .encode(&value[chunk_start..], &allowed)
                    .map_err(|e| FishTokenizerError::Encoding(e.to_string()))?
                    .0,
            );
        }
        Ok(tokens)
    }
}

impl TokenEncoder for FishBpeTokenizer {
    fn encode(&self, value: &str) -> FishResult<Vec<u32>> {
        self.encode_checked(value)
            .map_err(|_| FishError::InvalidShape("invalid Fish BPE text"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fish_speech_core::prompt::encode_tokens;

    fn source_bytes() -> String {
        let mut source = String::new();
        for i in 0_u16..=255 {
            source.push_str(&format!("{} {i}\n", STANDARD.encode([i as u8])));
        }
        for (rank, text) in ["he", "hel", "hell", "hello"].iter().enumerate() {
            source.push_str(&format!("{} {}\n", STANDARD.encode(text), 256 + rank));
        }
        source
    }

    #[test]
    fn special_ids_and_custom_fish_regex() {
        let t = FishBpeTokenizer::from_tiktoken_text(&source_bytes()).unwrap();
        assert_eq!(t.special_tokens().mergeable_ranks_count, 260);
        assert_eq!(t.encode_checked("hello").unwrap(), vec![259]);
        assert_eq!(t.encode_checked("<|im_start|>").unwrap(), vec![263]);
        assert_eq!(t.encode_checked("<|voice|>").unwrap(), vec![270]);
        assert_eq!(t.encode_checked("<|semantic:1023|>").unwrap(), vec![1295]);
        assert_eq!(
            t.encode_checked("🤖").unwrap(),
            "🤖"
                .as_bytes()
                .iter()
                .map(|x| *x as u32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            t.decode(&t.encode_checked("hello 🤖").unwrap()).unwrap(),
            "hello 🤖"
        );
        let prompt = encode_tokens(&t, t.special_tokens(), 2, "hello", None).unwrap();
        assert_eq!(prompt.channels.len(), 3);
        assert!(prompt.channels[0].contains(&259));
        assert_eq!(prompt.channels[0].last(), Some(&270));
    }

    #[test]
    fn corrupted_rank_file_is_rejected_without_panic() {
        let mut t = source_bytes();
        t.push_str("YWJj 0\n");
        assert!(FishBpeTokenizer::from_tiktoken_text(&t).is_err());
        assert!(FishBpeTokenizer::from_tiktoken_text("AA== 0\n").is_err());
        assert!(FishBpeTokenizer::from_tiktoken_text("not-base64 0\n").is_err());
    }
}
