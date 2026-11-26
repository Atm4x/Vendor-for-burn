use crate::conversation::{
    encode_for_inference, EncodedConversation, Message, Part, Role, TokenEncoder,
};
use crate::error::{FishError, Result};
use crate::text::{clean_text, split_text};
use crate::tokenizer::SpecialTokens;

const SYSTEM_PROMPT: &str = "Speak out the provided text.";

#[derive(Debug, Clone)]
pub struct ReferenceExample {
    pub text: String,
    pub acoustic_codes: Vec<Vec<u32>>,
}

pub fn encode_tokens<T: TokenEncoder>(
    tokenizer: &T,
    special: &SpecialTokens,
    num_codebooks: usize,
    text: &str,
    reference: Option<&[Vec<u32>]>,
) -> Result<EncodedConversation> {
    let user = Message::text(Role::User, clean_text(text));
    let mut assistant = Message::text(Role::Assistant, "<|voice|>");
    if let Some(codes) = reference {
        if codes.len() < num_codebooks {
            return Err(FishError::InvalidShape(
                "reference must contain all acoustic codebooks",
            ));
        }
        assistant
            .parts
            .push(Part::Vq(codes[..num_codebooks].to_vec()));
    } else {
        assistant.add_im_end = false;
    }
    encode_for_inference(&[user, assistant], tokenizer, special, num_codebooks)
}

fn concatenate_encoded(inputs: &[&EncodedConversation]) -> Result<EncodedConversation> {
    let count = inputs
        .first()
        .ok_or(FishError::InvalidShape("empty prompt"))?
        .channels
        .len();
    let mut channels = vec![Vec::new(); count];
    let mut vq_positions = Vec::new();
    for encoded in inputs {
        if encoded.channels.len() != count || encoded.channels.is_empty() {
            return Err(FishError::InvalidShape("prompt codebook count differs"));
        }
        let frames = encoded.channels[0].len();
        if encoded.channels.iter().any(|c| c.len() != frames)
            || encoded.vq_positions.len() != frames
        {
            return Err(FishError::InvalidShape("invalid prompt channel lengths"));
        }
        for (dst, src) in channels.iter_mut().zip(&encoded.channels) {
            dst.extend_from_slice(src);
        }
        vq_positions.extend_from_slice(&encoded.vq_positions);
    }
    Ok(EncodedConversation {
        channels,
        vq_positions,
    })
}

#[derive(Debug, Clone)]
pub struct PromptChunk {
    pub index: usize,
    pub text: String,
    pub encoded: EncodedConversation,
}

#[derive(Debug, Clone)]
pub struct GeneratedChunk {
    pub text: String,
    pub acoustic_codes: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct LongPromptPlanner {
    texts: Vec<String>,
    encoded_texts: Vec<EncodedConversation>,
    reference_prompts: Vec<EncodedConversation>,
    history: Vec<EncodedConversation>,
    next_index: usize,
    pending: bool,
    max_context_length: usize,
    num_codebooks: usize,
}

impl LongPromptPlanner {
    pub fn new<T: TokenEncoder>(
        tokenizer: &T,
        special: &SpecialTokens,
        num_codebooks: usize,
        text: &str,
        iterative_prompt: bool,
        chunk_length: usize,
        max_context_length: usize,
        references: &[ReferenceExample],
    ) -> Result<Self> {
        if num_codebooks == 0 || max_context_length == 0 || (iterative_prompt && chunk_length == 0)
        {
            return Err(FishError::InvalidConfig("invalid long prompt parameters"));
        }
        let texts = if iterative_prompt {
            split_text(text, chunk_length)?
        } else {
            vec![text.to_string()]
        };
        let encoded_texts = texts
            .iter()
            .map(|t| encode_tokens(tokenizer, special, num_codebooks, t, None))
            .collect::<Result<Vec<_>>>()?;
        let mut reference_prompts = Vec::new();
        if !references.is_empty() {
            reference_prompts.push(encode_for_inference(
                &[Message::text(Role::System, SYSTEM_PROMPT)],
                tokenizer,
                special,
                num_codebooks,
            )?);
            for r in references {
                reference_prompts.push(encode_tokens(
                    tokenizer,
                    special,
                    num_codebooks,
                    &r.text,
                    Some(&r.acoustic_codes),
                )?);
            }
        }
        Ok(Self {
            texts,
            encoded_texts,
            reference_prompts,
            history: Vec::new(),
            next_index: 0,
            pending: false,
            max_context_length,
            num_codebooks,
        })
    }

    pub fn total_chunks(&self) -> usize {
        self.texts.len()
    }
    pub fn completed_chunks(&self) -> usize {
        self.next_index
    }
    pub fn is_complete(&self) -> bool {
        !self.pending && self.next_index == self.texts.len()
    }

    pub fn begin_next_chunk(&mut self) -> Result<Option<PromptChunk>> {
        if self.pending {
            return Err(FishError::InvalidTransition(
                "previous chunk has no generated output",
            ));
        }
        if self.next_index == self.texts.len() {
            return Ok(None);
        }
        let mut all = self.history.iter().collect::<Vec<_>>();
        all.push(&self.encoded_texts[self.next_index]);
        let reserved: usize = self
            .reference_prompts
            .iter()
            .map(EncodedConversation::length)
            .sum();
        let threshold = self
            .max_context_length
            .saturating_sub(1024)
            .saturating_sub(reserved);
        let mut cumulative = 0_usize;
        let mut index = 0;
        for (i, seg) in all.iter().rev().enumerate() {
            cumulative = cumulative.saturating_add(seg.length());
            index = i;
            if cumulative.saturating_add(seg.length()) > threshold {
                break;
            }
        }
        if index != 0 && index % 2 == 0 {
            index -= 1;
        }
        let selected = if index < all.len().saturating_sub(2) {
            let mut selection = all[..2].to_vec();
            selection.extend_from_slice(if index == 0 {
                &all[..]
            } else {
                &all[all.len() - index..]
            });
            selection
        } else {
            all
        };
        let mut prompt_parts = self.reference_prompts.iter().collect::<Vec<_>>();
        prompt_parts.extend(selected);
        let encoded = concatenate_encoded(&prompt_parts)?;
        self.pending = true;
        Ok(Some(PromptChunk {
            index: self.next_index,
            text: self.texts[self.next_index].clone(),
            encoded,
        }))
    }

    pub fn accept_generated(&mut self, generated: Vec<Vec<u32>>) -> Result<GeneratedChunk> {
        if !self.pending {
            return Err(FishError::InvalidTransition("no pending chunk to finish"));
        }
        if generated.len() != self.num_codebooks + 1 || generated[0].is_empty() {
            return Err(FishError::InvalidShape(
                "generated output must have semantic and acoustic channels",
            ));
        }
        let frames = generated[0].len();
        if generated.iter().any(|c| c.len() != frames) {
            return Err(FishError::InvalidShape(
                "inconsistent generated code lengths",
            ));
        }
        let acoustic_codes = generated.iter().skip(1).map(|c| c[1..].to_vec()).collect();
        self.history
            .push(self.encoded_texts[self.next_index].clone());
        self.history.push(EncodedConversation {
            channels: generated,
            vq_positions: vec![true; frames],
        });
        let text = self.texts[self.next_index].clone();
        self.next_index += 1;
        self.pending = false;
        Ok(GeneratedChunk {
            text,
            acoustic_codes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UtfEncoder;
    impl TokenEncoder for UtfEncoder {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.chars().map(|c| c as u32).collect())
        }
    }

    fn spec() -> SpecialTokens {
        SpecialTokens::new(100).unwrap()
    }

    #[test]
    fn reference_free_prompt_keeps_assistant_open() {
        let value = encode_tokens(&UtfEncoder, &spec(), 2, "  Hi,,,, ‘Fish’  ", None).unwrap();
        assert_eq!(value.channels.len(), 3);
        assert_eq!(value.channels[0].last().copied(), Some('>' as u32));
        assert!(value.vq_positions.iter().all(|v| !v));
        assert!(
            value.channels[0]
                .windows(3)
                .any(|v| v == ['H' as u32, 'i' as u32, '<' as u32])
                == false
        );
    }

    #[test]
    fn reference_prompt_has_vq_and_closes_message() {
        let value = encode_tokens(
            &UtfEncoder,
            &spec(),
            2,
            "reference",
            Some(&[vec![2, 3], vec![4, 5]]),
        )
        .unwrap();
        assert_eq!(value.channels[0].last().copied(), Some('>' as u32));
        assert_eq!(value.vq_positions.iter().filter(|x| **x).count(), 2);
        assert!(value.channels[0].contains(&((spec().semantic_begin + 2) as u32)));
        assert!(encode_tokens(&UtfEncoder, &spec(), 2, "x", Some(&[vec![1]])).is_err());
    }

    #[test]
    fn long_prompt_lifecycle_and_first_generated_acoustic_column() {
        let references = [ReferenceExample {
            text: "reference".into(),
            acoustic_codes: vec![vec![1, 2], vec![3, 4]],
        }];
        let mut planner = LongPromptPlanner::new(
            &UtfEncoder,
            &spec(),
            2,
            "Alpha. Beta. Gamma.",
            true,
            7,
            2048,
            &references,
        )
        .unwrap();
        assert!(planner.total_chunks() > 1);
        let first = planner.begin_next_chunk().unwrap().unwrap();
        assert_eq!(first.index, 0);
        assert!(first.encoded.vq_positions.iter().any(|v| *v));
        assert!(planner.begin_next_chunk().is_err());
        assert!(planner.accept_generated(vec![vec![1, 2], vec![3]]).is_err());
        let result = planner
            .accept_generated(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
            .unwrap();
        assert_eq!(result.acoustic_codes, vec![vec![5, 6], vec![8, 9]]);
        assert_eq!(planner.completed_chunks(), 1);
        let second = planner.begin_next_chunk().unwrap().unwrap();
        assert!(second.encoded.length() > first.encoded.length());
    }

    #[test]
    fn no_reference_does_not_insert_system_prompt() {
        let mut planner =
            LongPromptPlanner::new(&UtfEncoder, &spec(), 2, "Hey", true, 150, 2048, &[]).unwrap();
        let first = planner.begin_next_chunk().unwrap().unwrap();
        let expected = encode_tokens(&UtfEncoder, &spec(), 2, "Hey", None).unwrap();
        assert_eq!(first.encoded.channels, expected.channels);
        planner
            .accept_generated(vec![vec![1], vec![0], vec![0]])
            .unwrap();
        assert!(planner.is_complete());
        assert!(planner.begin_next_chunk().unwrap().is_none());
    }
}
