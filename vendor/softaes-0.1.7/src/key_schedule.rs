use core::sync::atomic::{self, Ordering};

use super::{bitsliced, Block};

// Standard AES S-box.
pub(crate) const S_BOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

// Round constants (Rcon); index 0 is unused.
pub(crate) const RCON: [u8; 11] = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36,
];

/// Constant-time equality: returns 0xFF if a == b, else 0.
fn ct_eq(a: u8, b: u8) -> u8 {
    let diff = (a as u32) ^ (b as u32);
    (((diff.wrapping_sub(1)) >> 31) & 1) as u8 * 0xFF
}

/// Constant-time S-box lookup.
fn sbox_ct(input: u8) -> u8 {
    let mut res = 0;
    let mut i = 0;
    while i < 256 {
        let mask = ct_eq(input, i as u8);
        res |= S_BOX[i] & mask;
        i += 1;
    }
    atomic::compiler_fence(Ordering::Acquire);
    res
}

/// Applies the S-box substitution to each byte of a 32-bit word in constant time.
fn sub_word_ct(word: u32) -> u32 {
    let mut result = 0u32;
    let mut i = 0;
    while i < 4 {
        let shift = 24 - i * 8;
        let byte = ((word >> shift) & 0xFF) as u8;
        let sub = sbox_ct(byte) as u32;
        result |= sub << shift;
        i += 1;
    }
    result
}

/// Rotates a 32-bit word left by 8 bits.
fn rot_word(word: u32) -> u32 {
    word.rotate_left(8)
}

/// Applies the inverse MixColumns transformation to an entire Block.
pub fn inv_mix_block(block: Block) -> Block {
    Block {
        lo: bitsliced::inv_mix_columns(block.lo),
        hi: bitsliced::inv_mix_columns(block.hi),
    }
}

/// Packs four key expansion words into a round key.
fn pack_round_key(w: &[u32], j: usize) -> Block {
    let mut bytes = [0u8; 16];
    for (i, chunk) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *chunk = w[j + i].to_be_bytes();
    }
    Block::from_bytes(&bytes)
}

/// Expands a 16-byte AES-128 key into 11 128-bit blocks.
pub fn key_expansion_128(key: &[u8; 16]) -> [Block; 11] {
    const NK: usize = 4;
    const TOTAL_WORDS: usize = 44;
    let mut w = [0u32; TOTAL_WORDS];

    // Copy key into the first NK words.
    let mut i = 0;
    while i < NK {
        let j = i * 4;
        w[i] = ((key[j] as u32) << 24)
            | ((key[j + 1] as u32) << 16)
            | ((key[j + 2] as u32) << 8)
            | (key[j + 3] as u32);
        i += 1;
    }
    while i < TOTAL_WORDS {
        let mut temp = w[i - 1];
        if i % NK == 0 {
            temp = sub_word_ct(rot_word(temp)) ^ ((RCON[i / NK] as u32) << 24);
        }
        w[i] = w[i - NK] ^ temp;
        i += 1;
    }
    let mut blocks = [Block::default(); 11];
    let mut i = 0;
    while i < 11 {
        let j = i * 4;
        blocks[i] = pack_round_key(&w, j);
        i += 1;
    }
    blocks
}

/// Expands a 24-byte AES-192 key into 13 128-bit blocks.
pub fn key_expansion_192(key: &[u8; 24]) -> [Block; 13] {
    const NK: usize = 6;
    const TOTAL_WORDS: usize = 52;
    let mut w = [0u32; TOTAL_WORDS];
    let mut i = 0;
    while i < NK {
        let j = i * 4;
        w[i] = ((key[j] as u32) << 24)
            | ((key[j + 1] as u32) << 16)
            | ((key[j + 2] as u32) << 8)
            | (key[j + 3] as u32);
        i += 1;
    }
    while i < TOTAL_WORDS {
        let mut temp = w[i - 1];
        if i % NK == 0 {
            temp = sub_word_ct(rot_word(temp)) ^ ((RCON[i / NK] as u32) << 24);
        }
        w[i] = w[i - NK] ^ temp;
        i += 1;
    }
    let mut blocks = [Block::default(); 13];
    let mut i = 0;
    while i < 13 {
        let j = i * 4;
        blocks[i] = pack_round_key(&w, j);
        i += 1;
    }
    blocks
}

/// Expands a 32-byte AES-256 key into 15 128-bit blocks.
pub fn key_expansion_256(key: &[u8; 32]) -> [Block; 15] {
    const NK: usize = 8;
    const TOTAL_WORDS: usize = 60;
    let mut w = [0u32; TOTAL_WORDS];
    let mut i = 0;
    while i < NK {
        let j = i * 4;
        w[i] = ((key[j] as u32) << 24)
            | ((key[j + 1] as u32) << 16)
            | ((key[j + 2] as u32) << 8)
            | (key[j + 3] as u32);
        i += 1;
    }
    while i < TOTAL_WORDS {
        let mut temp = w[i - 1];
        if i % NK == 0 {
            temp = sub_word_ct(rot_word(temp)) ^ ((RCON[i / NK] as u32) << 24);
        } else if i % NK == 4 {
            temp = sub_word_ct(temp);
        }
        w[i] = w[i - NK] ^ temp;
        i += 1;
    }
    let mut blocks = [Block::default(); 15];
    let mut i = 0;
    while i < 15 {
        let j = i * 4;
        blocks[i] = pack_round_key(&w, j);
        i += 1;
    }
    blocks
}

/// Inverts an AES-128 encryption key schedule (11 blocks) to produce the
/// decryption key schedule.
///
/// For decryption the round keys are applied in reverse order, and every key
/// except the first and last is passed through InvMixColumns.
pub fn inverse_key_schedule_128(enc: &[Block; 11]) -> [Block; 11] {
    let mut dec = [Block::default(); 11];
    dec[0] = enc[10]; // Last round key (no InvMixColumns)
    let mut i = 1;
    while i < 10 {
        dec[i] = inv_mix_block(enc[10 - i]);
        i += 1;
    }
    dec[10] = enc[0]; // First round key (no InvMixColumns)
    dec
}

/// Inverts an AES-192 encryption key schedule (13 blocks) to produce the decryption key schedule.
pub fn inverse_key_schedule_192(enc: &[Block; 13]) -> [Block; 13] {
    let mut dec = [Block::default(); 13];
    dec[0] = enc[12];
    let mut i = 1;
    while i < 12 {
        dec[i] = inv_mix_block(enc[12 - i]);
        i += 1;
    }
    dec[12] = enc[0];
    dec
}

/// Inverts an AES-256 encryption key schedule (15 blocks) to produce the decryption key schedule.
pub fn inverse_key_schedule_256(enc: &[Block; 15]) -> [Block; 15] {
    let mut dec = [Block::default(); 15];
    dec[0] = enc[14];
    let mut i = 1;
    while i < 14 {
        dec[i] = inv_mix_block(enc[14 - i]);
        i += 1;
    }
    dec[14] = enc[0];
    dec
}
