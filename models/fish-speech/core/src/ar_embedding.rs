use crate::error::{FishError, Result};
use crate::tokenizer::SEMANTIC_TOKEN_COUNT;

pub fn semantic_membership_lut(vocabulary_size: usize, ids: &[i32]) -> Result<Vec<i32>> {
    if vocabulary_size == 0 || ids.len() != SEMANTIC_TOKEN_COUNT {
        return Err(FishError::InvalidShape(
            "invalid semantic vocabulary or token ID count",
        ));
    }
    let mut lut = vec![0i32; vocabulary_size];
    for &id in ids {
        let id = usize::try_from(id)
            .map_err(|_| FishError::InvalidShape("negative semantic token ID"))?;
        let entry = lut.get_mut(id).ok_or(FishError::InvalidShape(
            "semantic token ID outside vocabulary",
        ))?;
        if *entry != 0 {
            return Err(FishError::InvalidShape("duplicate semantic token ID"));
        }
        *entry = 1;
    }
    Ok(lut)
}

pub fn fish_embedding_cpu(
    tokens: &[i32],
    batch: usize,
    frames: usize,
    num_codebooks: usize,
    codebook_size: usize,
    word_weights: &[f32],
    codebook_weights: &[f32],
    embedding_dim: usize,
    membership_lut: &[i32],
    share_codebook_embeddings: bool,
) -> Result<Vec<f32>> {
    let vocab_size = membership_lut.len();
    let check_length = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or(FishError::InvalidShape("embedding shape overflow"))
    };
    if vocab_size == 0
        || num_codebooks == 0
        || codebook_size == 0
        || embedding_dim == 0
        || tokens.len() != check_length(check_length(batch, num_codebooks + 1)?, frames)?
        || word_weights.len() != check_length(vocab_size, embedding_dim)?
        || codebook_weights.len()
            != check_length(check_length(num_codebooks, codebook_size)?, embedding_dim)?
        || membership_lut.iter().any(|&x| x != 0 && x != 1)
    {
        return Err(FishError::InvalidShape(
            "FishEmbedding input dimensions or gate invalid",
        ));
    }
    let mut output = vec![0f32; check_length(check_length(batch, frames)?, embedding_dim)?];
    for b in 0..batch {
        for t in 0..frames {
            let word = usize::try_from(tokens[(b * (num_codebooks + 1)) * frames + t])
                .map_err(|_| FishError::InvalidShape("negative semantic token"))?;
            if word >= vocab_size {
                return Err(FishError::InvalidShape("semantic token outside vocabulary"));
            }
            let o = (b * frames + t) * embedding_dim;
            output[o..o + embedding_dim]
                .copy_from_slice(&word_weights[word * embedding_dim..(word + 1) * embedding_dim]);
            if membership_lut[word] == 1 {
                for c in 0..num_codebooks {
                    let code =
                        usize::try_from(tokens[(b * (num_codebooks + 1) + c + 1) * frames + t])
                            .map_err(|_| FishError::InvalidShape("negative acoustic token"))?;
                    if code >= codebook_size {
                        return Err(FishError::InvalidShape("acoustic token outside codebook"));
                    }
                    let row = code
                        + if share_codebook_embeddings {
                            c * codebook_size
                        } else {
                            0
                        };
                    for (dst, src) in output[o..o + embedding_dim]
                        .iter_mut()
                        .zip(&codebook_weights[row * embedding_dim..(row + 1) * embedding_dim])
                    {
                        *dst += *src;
                    }
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_rejects_duplicate_out_of_range_and_negative_ids() {
        let mut ids: Vec<i32> = (0..1024).collect();
        let lut = semantic_membership_lut(1030, &ids).unwrap();
        assert_eq!(lut.iter().filter(|&&v| v == 1).count(), 1024);
        assert_eq!(lut[1029], 0);
        ids[1023] = ids[0];
        assert!(semantic_membership_lut(1030, &ids).is_err());
        ids[1023] = -1;
        assert!(semantic_membership_lut(1030, &ids).is_err());
        ids[1023] = 1030;
        assert!(semantic_membership_lut(1030, &ids).is_err());
    }

    #[test]
    fn gates_acoustic_embeddings_and_respects_shared_offsets() {
        let tokens = [5, 9, 1, 1, 1, 0];
        let lut = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let words: Vec<f32> = (0..10).map(|v| v as f32).collect();
        let cb = [0., 10., 20., 30.];
        for (share, first) in [(false, 25.0), (true, 45.0)] {
            assert_eq!(
                fish_embedding_cpu(&tokens, 1, 2, 2, 2, &words, &cb, 1, &lut, share).unwrap(),
                [first, 9.]
            );
        }
        let mut invalid = tokens;
        invalid[2] = 2;
        assert!(fish_embedding_cpu(&invalid, 1, 2, 2, 2, &words, &cb, 1, &lut, true).is_err());
    }
}
