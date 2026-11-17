use crate::error::{FishError, Result};

pub const SEMANTIC_TOKEN_COUNT: usize = 1024;
pub const BASE_SPECIAL_TOKEN_COUNT: usize = 12;
pub const SPECIAL_TOKEN_COUNT: usize = BASE_SPECIAL_TOKEN_COUNT + SEMANTIC_TOKEN_COUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokens {
    pub mergeable_ranks_count: usize,
    pub im_start: usize,
    pub im_end: usize,
    pub text: usize,
    pub voice: usize,
    pub interleave: usize,
    pub semantic_begin: usize,
    pub semantic_end: usize,
}

impl SpecialTokens {
    pub fn new(mergeable_ranks_count: usize) -> Result<Self> {
        let base = mergeable_ranks_count;
        if base.checked_add(SPECIAL_TOKEN_COUNT).is_none() {
            return Err(FishError::InvalidConfig("token ID range overflow"));
        }
        Ok(Self {
            mergeable_ranks_count: base,
            im_start: base + 3,
            im_end: base + 4,
            text: base + 9,
            voice: base + 10,
            interleave: base + 11,
            semantic_begin: base + BASE_SPECIAL_TOKEN_COUNT,
            semantic_end: base + BASE_SPECIAL_TOKEN_COUNT + SEMANTIC_TOKEN_COUNT - 1,
        })
    }

    pub fn semantic_token(&self, acoustic_index: usize) -> Result<usize> {
        if acoustic_index >= SEMANTIC_TOKEN_COUNT {
            return Err(FishError::InvalidShape("semantic code index exceeds 1023"));
        }
        Ok(self.semantic_begin + acoustic_index)
    }

    pub fn semantic_to_acoustic(&self, semantic_token: usize) -> usize {
        semantic_token.saturating_sub(self.semantic_begin)
    }

    pub fn dual_ar_first_codebook(
        &self,
        semantic_token: usize,
        codebook_size: usize,
    ) -> Result<usize> {
        let code = self.semantic_to_acoustic(semantic_token);
        if code >= codebook_size {
            return Err(FishError::InvalidShape(
                "semantic-derived codebook index out of embedding range",
            ));
        }
        Ok(code)
    }

    pub fn is_semantic(&self, id: usize) -> bool {
        (self.semantic_begin..=self.semantic_end).contains(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fish_special_token_order() {
        let x = SpecialTokens::new(200).unwrap();
        assert_eq!(x.im_start, 203);
        assert_eq!(x.im_end, 204);
        assert_eq!(x.voice, 210);
        assert_eq!(x.semantic_begin, 212);
        assert_eq!(x.semantic_end, 1235);
    }
    #[test]
    fn semantic_index_clamps_negative() {
        let x = SpecialTokens::new(200).unwrap();
        assert_eq!(x.semantic_to_acoustic(3), 0);
        assert_eq!(x.semantic_to_acoustic(x.semantic_begin + 9), 9);
    }
    #[test]
    fn first_dual_ar_code_is_from_semantic_not_random() {
        let x = SpecialTokens::new(200).unwrap();
        assert_eq!(
            x.dual_ar_first_codebook(x.semantic_begin + 42, 1024)
                .unwrap(),
            42
        );
        assert_eq!(x.dual_ar_first_codebook(x.im_end, 1024).unwrap(), 0);
        assert!(x.dual_ar_first_codebook(x.semantic_end + 1, 1024).is_err());
    }
    #[test]
    fn semantic_bounds() {
        let x = SpecialTokens::new(200).unwrap();
        assert_eq!(x.semantic_token(1023).unwrap(), x.semantic_end);
        assert!(x.semantic_token(1024).is_err());
        assert!(!x.is_semantic(x.im_end));
    }
}
