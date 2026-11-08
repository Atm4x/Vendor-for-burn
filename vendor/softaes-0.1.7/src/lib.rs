//! Software implementation of the AES round function.

#![no_std]

use core::{cmp, ops};

pub mod key_schedule;

/// An AES block, stored as two 64-bit words.
#[derive(Copy, Clone, Debug, Default)]
pub struct Block {
    lo: u64,
    hi: u64,
}

impl cmp::PartialEq for Block {
    #[inline(never)]
    fn eq(&self, other: &Block) -> bool {
        let z = self ^ other;
        z.lo | z.hi == 0
    }
}

impl cmp::Eq for Block {}

impl Block {
    #[inline(always)]
    pub fn from_bytes(input: &[u8; 16]) -> Block {
        let (lo, hi) = input.split_at(8);
        Block {
            lo: u64::from_le_bytes(lo.try_into().unwrap()),
            hi: u64::from_le_bytes(hi.try_into().unwrap()),
        }
    }

    /// Reads a block from the first 16 bytes of `input`.
    #[inline(always)]
    pub fn from_slice(input: &[u8]) -> Block {
        debug_assert!(input.len() == 16);
        Block::from_bytes(input[..16].try_into().unwrap())
    }

    #[inline(always)]
    pub fn from64x2(a: u64, b: u64) -> Block {
        Block { lo: b, hi: a }
    }

    #[inline(always)]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.lo.to_le_bytes());
        out[8..16].copy_from_slice(&self.hi.to_le_bytes());
        out
    }

    #[inline(always)]
    pub fn xor(&self, other: &Block) -> Block {
        Block {
            lo: self.lo ^ other.lo,
            hi: self.hi ^ other.hi,
        }
    }

    #[inline(always)]
    pub fn and(&self, other: &Block) -> Block {
        Block {
            lo: self.lo & other.lo,
            hi: self.hi & other.hi,
        }
    }
}

impl ops::BitAnd for Block {
    type Output = Block;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(&rhs)
    }
}

impl ops::BitAnd for &Block {
    type Output = Block;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}

impl ops::BitXor for Block {
    type Output = Block;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(&rhs)
    }
}

impl ops::BitXor for &Block {
    type Output = Block;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}

mod bitsliced;

/// Constant-time software AES implementation.
///
/// It never uses secret data as a memory index, so it runs in constant time on
/// every platform.
pub struct SoftAes;

impl SoftAes {
    /// AES forward round function.
    /// `rk` is the round key.
    #[inline]
    pub fn block_encrypt(block: &Block, rk: &Block) -> Block {
        bitsliced::block_encrypt(block, rk)
    }

    /// AES decryption round function.
    /// `rk` is the round key from the inverse key schedule.
    #[inline]
    pub fn block_decrypt(block: &Block, rk: &Block) -> Block {
        bitsliced::block_decrypt(block, rk)
    }

    /// AES forward round function for the last round.
    /// `rk` is the round key.
    #[inline]
    pub fn block_encrypt_last(block: &Block, rk: &Block) -> Block {
        bitsliced::block_encrypt_last(block, rk)
    }

    /// AES final decryption round.
    /// `rk` is the round key from the inverse key schedule.
    #[inline]
    pub fn block_decrypt_last(block: &Block, rk: &Block) -> Block {
        bitsliced::block_decrypt_last(block, rk)
    }
}

/// Constant-time software AES implementation (formerly the paranoid stride-16 variant)
pub type SoftAesSlow = SoftAes;

/// Constant-time software AES implementation (formerly the practical stride-64 variant)
pub type SoftAesModerate = SoftAes;

/// Constant-time software AES implementation (formerly the minimal-protection variant)
pub type SoftAesFast = SoftAes;

/// Fastest software AES implementation, but with no protection against side channels
pub mod unprotected;
