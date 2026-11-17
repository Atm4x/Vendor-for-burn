use anyhow::{Context, Result, ensure};
use prost::Message;
use std::path::Path;

#[derive(Clone, PartialEq, Message)]
struct SentencePieceModel {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<SentencePiece>,
}

#[derive(Clone, PartialEq, Message)]
struct SentencePiece {
    #[prost(string, tag = "1")]
    piece: String,
    #[prost(float, tag = "2")]
    score: f32,
    #[prost(int32, tag = "3")]
    kind: i32,
}

pub enum Tokenizer {
    Character(Vec<String>),
    SentencePiece(Vec<(String, i32)>),
}

impl Tokenizer {
    pub fn load(vocabulary: Vec<String>, sentencepiece: Option<&Path>) -> Result<Self> {
        match sentencepiece {
            Some(path) => {
                let data = std::fs::read(path)
                    .with_context(|| format!("reading SentencePiece model {}", path.display()))?;
                let model = SentencePieceModel::decode(data.as_slice())
                    .context("decoding SentencePiece ModelProto")?;
                ensure!(!model.pieces.is_empty(), "empty SentencePiece vocabulary");
                let pieces: Vec<_> = model
                    .pieces
                    .into_iter()
                    .map(|v| (v.piece, v.kind))
                    .collect();
                Ok(Self::SentencePiece(pieces))
            }
            None => {
                ensure!(
                    !vocabulary.is_empty(),
                    "character vocabulary is empty (or provide sentencepiece_model)"
                );
                Ok(Self::Character(vocabulary))
            }
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Character(v) => v.len(),
            Self::SentencePiece(v) => v.len(),
        }
    }
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let mut bytes = Vec::<u8>::new();
        let mut trim_dummy_prefix = false;
        let mut first_piece_seen = false;
        for &id in ids {
            match self {
                Self::Character(v) => bytes
                    .extend_from_slice(v.get(id).context("token ID out of vocabulary")?.as_bytes()),
                Self::SentencePiece(v) => {
                    let (piece, kind) = v
                        .get(id)
                        .context("SentencePiece token ID out of vocabulary")?;
                    match *kind {
                        3 | 5 => continue,
                        _ => {}
                    }
                    if !first_piece_seen {
                        trim_dummy_prefix = piece.starts_with('▁') && *kind != 2;
                        first_piece_seen = true;
                    }
                    match *kind {
                        2 => bytes.extend_from_slice(" ⁇ ".as_bytes()),
                        6 => {
                            let hex = piece
                                .strip_prefix("<0x")
                                .and_then(|x| x.strip_suffix('>'))
                                .context("invalid SentencePiece byte token")?;
                            ensure!(hex.len() == 2, "invalid SentencePiece byte token");
                            bytes.push(u8::from_str_radix(hex, 16)?);
                        }
                        _ => bytes.extend_from_slice(piece.as_bytes()),
                    }
                }
            }
        }
        let out = String::from_utf8_lossy(&bytes).replace('▁', " ");
        if matches!(self, Self::SentencePiece(_)) && trim_dummy_prefix {
            Ok(out.strip_prefix(' ').unwrap_or(&out).to_owned())
        } else {
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sentencepiece_decoder_supports_unicode_bytes_and_control() {
        let model = SentencePieceModel {
            pieces: vec![
                SentencePiece {
                    piece: "<unk>".into(),
                    score: 0.,
                    kind: 2,
                },
                SentencePiece {
                    piece: "▁Привет".into(),
                    score: 0.,
                    kind: 1,
                },
                SentencePiece {
                    piece: "▁мир".into(),
                    score: 0.,
                    kind: 1,
                },
                SentencePiece {
                    piece: "<s>".into(),
                    score: 0.,
                    kind: 3,
                },
                SentencePiece {
                    piece: "<0x21>".into(),
                    score: 0.,
                    kind: 6,
                },
            ],
        };
        let bytes = model.encode_to_vec();
        let loaded = SentencePieceModel::decode(bytes.as_slice()).unwrap();
        let tok = Tokenizer::SentencePiece(
            loaded
                .pieces
                .into_iter()
                .map(|v| (v.piece, v.kind))
                .collect(),
        );
        assert_eq!(tok.decode(&[3, 1, 2, 4]).unwrap(), "Привет мир!");
        assert_eq!(tok.decode(&[0]).unwrap(), " ⁇ ");
    }
    #[test]
    fn character_tokenizer_keeps_all_spaces() {
        let tok = Tokenizer::Character(vec![" А".into(), "Б".into()]);
        assert_eq!(tok.decode(&[0, 1]).unwrap(), " АБ");
    }
}
