use crate::error::{FishError, Result};
use crate::tokenizer::SpecialTokens;

pub trait TokenEncoder {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Part {
    Text(String),
    Vq(Vec<Vec<u32>>),
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
    pub add_im_start: bool,
    pub add_im_end: bool,
    pub modality: Option<&'static str>,
}

impl Message {
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            parts: vec![Part::Text(text.into())],
            add_im_start: true,
            add_im_end: true,
            modality: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncodedConversation {
    pub channels: Vec<Vec<u32>>,
    pub vq_positions: Vec<bool>,
}

impl EncodedConversation {
    pub fn length(&self) -> usize {
        self.channels[0].len()
    }
}

pub fn encode_for_inference<T: TokenEncoder>(
    messages: &[Message],
    tokenizer: &T,
    special: &SpecialTokens,
    num_codebooks: usize,
) -> Result<EncodedConversation> {
    if messages.is_empty() || num_codebooks == 0 {
        return Err(FishError::InvalidShape(
            "conversation must have messages and codebooks",
        ));
    }
    let mut channels = vec![Vec::new(); num_codebooks + 1];
    let mut mask = Vec::new();
    for msg in messages {
        if msg.add_im_start {
            let modality = msg
                .modality
                .map(|x| match x {
                    "text" => "<|text|>",
                    "voice" => "<|voice|>",
                    "interleave" => "<|interleave|>",
                    _ => "",
                })
                .unwrap_or("");
            append_text(
                &format!("<|im_start|>{}\n{modality}", msg.role.as_str()),
                tokenizer,
                &mut channels,
                &mut mask,
            )?;
        }
        for part in &msg.parts {
            match part {
                Part::Text(text) => append_text(text, tokenizer, &mut channels, &mut mask)?,
                Part::Vq(codes) => {
                    if codes.len() != num_codebooks
                        || codes[0].is_empty()
                        || codes.iter().any(|v| v.len() != codes[0].len())
                    {
                        return Err(FishError::InvalidShape(
                            "VQ codes should be [num_codebooks,seq_len]",
                        ));
                    }
                    for t in 0..codes[0].len() {
                        let semantic = special.semantic_token(codes[0][t] as usize)?;
                        channels[0].push(semantic as u32);
                        for i in 0..num_codebooks {
                            channels[i + 1].push(codes[i][t]);
                        }
                        mask.push(true);
                    }
                }
            }
        }
        if msg.add_im_end {
            append_text("<|im_end|>", tokenizer, &mut channels, &mut mask)?;
        }
    }
    if channels[0].is_empty() || channels.iter().any(|c| c.len() != channels[0].len()) {
        return Err(FishError::InvalidShape("inconsistent conversation length"));
    }
    Ok(EncodedConversation {
        channels,
        vq_positions: mask,
    })
}

fn append_text<T: TokenEncoder>(
    text: &str,
    tokenizer: &T,
    channels: &mut [Vec<u32>],
    mask: &mut Vec<bool>,
) -> Result<()> {
    let ids = tokenizer.encode(text)?;
    for id in ids {
        channels[0].push(id);
        for channel in channels.iter_mut().skip(1) {
            channel.push(0);
        }
        mask.push(false);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixtureEncoder {
        special: SpecialTokens,
    }
    impl TokenEncoder for FixtureEncoder {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            let mut out = Vec::new();
            let mut rest = text;
            while !rest.is_empty() {
                if rest.starts_with("<|im_start|>") {
                    out.push(self.special.im_start as u32);
                    rest = &rest[12..];
                } else if rest.starts_with("<|im_end|>") {
                    out.push(self.special.im_end as u32);
                    rest = &rest[10..];
                } else if rest.starts_with("<|voice|>") {
                    out.push(self.special.voice as u32);
                    rest = &rest[9..];
                } else {
                    let ch = rest.chars().next().unwrap();
                    out.push(ch as u32);
                    rest = &rest[ch.len_utf8()..];
                }
            }
            Ok(out)
        }
    }

    #[test]
    fn plain_text_fills_only_semantic_channel() {
        let tokens = SpecialTokens::new(200).unwrap();
        let encoded = encode_for_inference(
            &[Message::text(Role::System, "Speak.")],
            &FixtureEncoder { special: tokens },
            &tokens,
            2,
        )
        .unwrap();
        assert_eq!(encoded.channels.len(), 3);
        assert_eq!(encoded.channels[0][0], tokens.im_start as u32);
        assert_eq!(*encoded.channels[0].last().unwrap(), tokens.im_end as u32);
        assert!(encoded
            .channels
            .iter()
            .skip(1)
            .all(|c| c.iter().all(|&x| x == 0)));
        assert!(encoded.vq_positions.iter().all(|&x| !x));
    }
    #[test]
    fn vq_semantic_ids_and_codebook_offsets() {
        let tokens = SpecialTokens::new(200).unwrap();
        let msg = Message {
            role: Role::Assistant,
            parts: vec![
                Part::Text("<|voice|>".into()),
                Part::Vq(vec![vec![5, 7], vec![2, 3]]),
            ],
            add_im_start: true,
            add_im_end: true,
            modality: None,
        };
        let encoded =
            encode_for_inference(&[msg], &FixtureEncoder { special: tokens }, &tokens, 2).unwrap();
        let positions: Vec<usize> = encoded
            .vq_positions
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(positions.len(), 2);
        assert_eq!(
            encoded.channels[0][positions[0]],
            (tokens.semantic_begin + 5) as u32
        );
        assert_eq!(encoded.channels[1][positions[1]], 7);
        assert_eq!(encoded.channels[2][positions[1]], 3);
    }
    #[test]
    fn assistant_prompt_without_im_end() {
        let tokens = SpecialTokens::new(200).unwrap();
        let msg = Message {
            role: Role::Assistant,
            parts: vec![Part::Text("<|voice|>".into())],
            add_im_start: true,
            add_im_end: false,
            modality: None,
        };
        let encoded =
            encode_for_inference(&[msg], &FixtureEncoder { special: tokens }, &tokens, 8).unwrap();
        assert_eq!(*encoded.channels[0].last().unwrap(), tokens.voice as u32);
    }
    #[test]
    fn reject_bad_vq_layout() {
        let tokens = SpecialTokens::new(200).unwrap();
        let msg = Message {
            role: Role::User,
            parts: vec![Part::Vq(vec![vec![1, 2], vec![2]])],
            add_im_start: true,
            add_im_end: true,
            modality: None,
        };
        assert!(
            encode_for_inference(&[msg], &FixtureEncoder { special: tokens }, &tokens, 2).is_err()
        );
    }
}
