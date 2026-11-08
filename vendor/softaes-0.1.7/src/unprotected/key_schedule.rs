use super::Block;
use crate::bitsliced;
use crate::key_schedule::{RCON, S_BOX};

pub(crate) const S_BOX_INV: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
];

#[inline(always)]
fn sbox(input: u8) -> u8 {
    S_BOX[input as usize]
}

#[inline(always)]
fn sub_word(word: u32) -> u32 {
    let b0 = sbox(((word >> 24) & 0xFF) as u8) as u32;
    let b1 = sbox(((word >> 16) & 0xFF) as u8) as u32;
    let b2 = sbox(((word >> 8) & 0xFF) as u8) as u32;
    let b3 = sbox((word & 0xFF) as u8) as u32;
    (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
}

#[inline(always)]
fn rot_word(word: u32) -> u32 {
    word.rotate_left(8)
}

/// Applies the inverse MixColumns transformation to an entire Block.
pub fn inv_mix_block(block: Block) -> Block {
    let b = block.to_bytes();
    let (lo, hi) = b.split_at(8);
    let lo = bitsliced::inv_mix_columns(u64::from_le_bytes(lo.try_into().unwrap()));
    let hi = bitsliced::inv_mix_columns(u64::from_le_bytes(hi.try_into().unwrap()));
    Block::from64x2(hi, lo)
}

/// Packs four key expansion words into a round key.
fn pack_round_key(w: &[u32], j: usize) -> Block {
    let mut bytes = [0u8; 16];
    for (i, chunk) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *chunk = w[j + i].to_be_bytes();
    }
    Block::from_bytes(&bytes)
}

pub fn key_expansion_128(key: &[u8; 16]) -> [Block; 11] {
    const NK: usize = 4;
    const TOTAL_WORDS: usize = 44;
    let mut w = [0u32; TOTAL_WORDS];

    for (i, chunk) in key.as_chunks::<4>().0.iter().take(NK).enumerate() {
        w[i] = u32::from_be_bytes(*chunk);
    }
    for i in NK..TOTAL_WORDS {
        let mut temp = w[i - 1];
        if i % NK == 0 {
            temp = sub_word(rot_word(temp)) ^ ((RCON[i / NK] as u32) << 24);
        }
        w[i] = w[i - NK] ^ temp;
    }
    let mut blocks = [Block::default(); 11];
    for (i, block) in blocks.iter_mut().enumerate() {
        let j = 4 * i;
        *block = pack_round_key(&w, j);
    }
    blocks
}

pub fn key_expansion_192(key: &[u8; 24]) -> [Block; 13] {
    const NK: usize = 6;
    const TOTAL_WORDS: usize = 52;
    let mut w = [0u32; TOTAL_WORDS];

    for (i, chunk) in key.as_chunks::<4>().0.iter().take(NK).enumerate() {
        w[i] = u32::from_be_bytes(*chunk);
    }
    for i in NK..TOTAL_WORDS {
        let mut temp = w[i - 1];
        if i % NK == 0 {
            temp = sub_word(rot_word(temp)) ^ ((RCON[i / NK] as u32) << 24);
        }
        w[i] = w[i - NK] ^ temp;
    }
    let mut blocks = [Block::default(); 13];
    for (i, block) in blocks.iter_mut().enumerate() {
        let j = 4 * i;
        *block = pack_round_key(&w, j);
    }
    blocks
}

pub fn key_expansion_256(key: &[u8; 32]) -> [Block; 15] {
    const NK: usize = 8;
    const TOTAL_WORDS: usize = 60;
    let mut w = [0u32; TOTAL_WORDS];

    for (i, chunk) in key.as_chunks::<4>().0.iter().take(NK).enumerate() {
        w[i] = u32::from_be_bytes(*chunk);
    }
    for i in NK..TOTAL_WORDS {
        let mut temp = w[i - 1];
        if i % NK == 0 {
            temp = sub_word(rot_word(temp)) ^ ((RCON[i / NK] as u32) << 24);
        } else if i % NK == 4 {
            temp = sub_word(temp);
        }
        w[i] = w[i - NK] ^ temp;
    }
    let mut blocks = [Block::default(); 15];
    for (i, block) in blocks.iter_mut().enumerate() {
        let j = 4 * i;
        *block = pack_round_key(&w, j);
    }
    blocks
}

#[inline(always)]
pub fn inverse_key_schedule_128(enc: &[Block; 11]) -> [Block; 11] {
    let mut dec = [Block::default(); 11];
    dec[0] = enc[10];
    for i in 1..10 {
        dec[i] = inv_mix_block(enc[10 - i]);
    }
    dec[10] = enc[0];
    dec
}

#[inline(always)]
pub fn inverse_key_schedule_192(enc: &[Block; 13]) -> [Block; 13] {
    let mut dec = [Block::default(); 13];
    dec[0] = enc[12];
    for i in 1..12 {
        dec[i] = inv_mix_block(enc[12 - i]);
    }
    dec[12] = enc[0];
    dec
}

#[inline(always)]
pub fn inverse_key_schedule_256(enc: &[Block; 15]) -> [Block; 15] {
    let mut dec = [Block::default(); 15];
    dec[0] = enc[14];
    for i in 1..14 {
        dec[i] = inv_mix_block(enc[14 - i]);
    }
    dec[14] = enc[0];
    dec
}
