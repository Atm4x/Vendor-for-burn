//! Constant-time AES round functions.
//!
//! Only SubBytes is bitsliced: the block is split into eight words, one per bit
//! position, and the S-box is computed with logic gates on those.
//!
//! The other steps work directly on the bytes, stored as two 64-bit words.
//!
//! Nothing here branches on secret data or uses it as a memory index.

use super::Block;

/// The S-box constant 0x63 in every byte.
const C63: u64 = 0x6363_6363_6363_6363;

/// The bit positions used by a bit plane: one bit out of every four.
const LANES: u64 = 0x1111_1111_1111_1111;

/// Moves the top bit of each byte's upper nibble into the other word, so that
/// each word only holds four of the eight bit positions.
///
/// Applying it twice gives back the input.
#[inline(always)]
fn swap_index_bits_6_2(lo: u64, hi: u64) -> (u64, u64) {
    let t = ((lo >> 4) ^ hi) & 0x0f0f_0f0f_0f0f_0f0f;
    (lo ^ (t << 4), hi ^ t)
}

// Each 32-bit half of a word is a column, and its byte `r` is row `r`.
const ROW0: u64 = 0x0000_00ff_0000_00ff;
const ROW2: u64 = 0x00ff_0000_00ff_0000;
// Rows 1 and 3, split by which rotated word they are taken from.
const ROWS_A: u64 = 0xff00_0000_0000_ff00;
const ROWS_B: u64 = 0x0000_ff00_ff00_0000;

/// ShiftRows, or InvShiftRows when the two masks are swapped.
///
/// Row 2 is taken from the other word. Rows 1 and 3 are taken from copies of
/// the words with their two columns swapped.
#[inline(always)]
fn permute_rows(lo: u64, hi: u64, mx: u64, my: u64) -> (u64, u64) {
    let x = lo.rotate_left(32);
    let y = hi.rotate_left(32);
    (
        (lo & ROW0) | (hi & ROW2) | (x & mx) | (y & my),
        (hi & ROW0) | (lo & ROW2) | (y & mx) | (x & my),
    )
}

/// Splits the bytes into eight bit planes, most significant bit first.
#[inline(always)]
fn load(lo: u64, hi: u64) -> [u64; 8] {
    let (lo, hi) = swap_index_bits_6_2(lo, hi);
    [
        (hi >> 3) & LANES,
        (hi >> 2) & LANES,
        (hi >> 1) & LANES,
        hi & LANES,
        (lo >> 3) & LANES,
        (lo >> 2) & LANES,
        (lo >> 1) & LANES,
        lo & LANES,
    ]
}

/// Puts the bit planes back together into bytes. Inverse of `load`.
#[inline(always)]
fn store(p: &[u64; 8]) -> (u64, u64) {
    let lo = p[7] | (p[6] << 1) | (p[5] << 2) | (p[4] << 3);
    let hi = p[3] | (p[2] << 1) | (p[1] << 2) | (p[0] << 3);
    swap_index_bits_6_2(lo, hi)
}

/// The AES S-box on bit planes, most significant bit first, without the final
/// XOR with 0x63. The caller adds that constant on the bytes.
#[inline(always)]
fn sbox_core(planes: &mut [u64; 8]) {
    let s0 = planes[1] ^ planes[4];
    let s1 = planes[5] ^ planes[7];
    let s2 = planes[3] ^ s0;
    let s3 = planes[0] ^ planes[2];
    let q0 = s1 ^ s2;
    let s4 = planes[0] ^ planes[6];
    let s5 = planes[2] ^ planes[6];
    let s6 = planes[3] ^ s1;
    let s7 = planes[5] ^ s3;
    let q1 = s1 ^ s5;
    let q2 = planes[2] ^ q0;
    let q3 = s4 ^ s2;
    let q4 = s3 ^ q0;
    let s8 = planes[4] ^ s3;
    let q5 = s6 ^ s8;
    let q6 = planes[2] ^ planes[3];
    let q7 = planes[6] ^ s2;
    let s9 = planes[6] ^ s0;
    let q8 = s3 ^ s9;
    let q9 = s4 ^ s6;
    let q10 = s0 ^ s5;
    let q12 = planes[7] ^ s2;
    let q13 = planes[1] ^ s7;
    let q14 = planes[7] ^ s3;
    let q15 = s2 ^ s7;
    let q16 = planes[1] ^ s1;
    let q17 = planes[1] ^ planes[7];
    let q11 = planes[5];

    let t20 = q6 & q12;
    let t21 = q3 & q14;
    let t22 = q1 & q16;
    let t23 = q2 & q17;
    let x0 = ((q3 | q14) ^ (q0 & q7)) ^ (t20 ^ t22);
    let x1 = ((q4 | q13) ^ (q10 & q11)) ^ (t21 ^ t20);
    let x2 = ((q2 | q17) ^ (q5 & q9)) ^ (t21 ^ t22);
    let x3 = ((q8 | q15) ^ t23) ^ (t21 ^ (q4 & q13));

    let a = x1 & !x3;
    let b = x0 & !x3;
    let c = x3 & !x1;
    let d = x2 & !x1;
    let e = x0 ^ a;
    let y0 = x3 ^ (x2 & !e);
    let f = x1 ^ b;
    let y1 = c ^ (x2 & f);
    let g = x2 ^ c;
    let y2 = x1 ^ (x0 & !g);
    let h = x3 ^ d;
    let y3 = a ^ (x0 & h);
    let y02 = y2 ^ y0;
    let y13 = y3 ^ y1;
    let y23 = y3 ^ y2;
    let y01 = y1 ^ y0;
    let y00 = y02 ^ y13;

    let a0 = y01 & q11;
    let a1 = y0 & q12;
    let a2 = y1 & q0;
    let a3 = y23 & q17;
    let a4 = y2 & q5;
    let a5 = y3 & q15;
    let a6 = y13 & q14;
    let a7 = y00 & q16;
    let a8 = y02 & q13;
    let a9 = y01 & q7;
    let a10 = y0 & q10;
    let a11 = y1 & q6;
    let a12 = y23 & q2;
    let a13 = y2 & q9;
    let a14 = y3 & q8;
    let a15 = y13 & q3;
    let a16 = y00 & q1;
    let a17 = y02 & q4;

    let r0 = a1 ^ a5;
    let r1 = a9 ^ a15;
    let r2 = a4 ^ r0;
    let r3 = a2 ^ a10;
    let r4 = a11 ^ a17;
    let r5 = a8 ^ r1;
    let r6 = a0 ^ a16;
    let r7 = a7 ^ a13;
    let r8 = a11 ^ a14;
    let r9 = r3 ^ r4;
    let r10 = r5 ^ r6;
    let r11 = r2 ^ r9;
    let r12 = a3 ^ r0;
    let r13 = r7 ^ r8;
    let r14 = r12 ^ r13;
    planes[0] = r10 ^ r14;
    let r15 = a6 ^ a10;
    let r16 = r15 ^ r2;
    planes[1] = r10 ^ r16;
    planes[2] = a2 ^ r2;
    let r17 = a12 ^ a13;
    let r18 = a15 ^ r17;
    planes[3] = r18 ^ r11;
    let r19 = a1 ^ a14;
    let r20 = a17 ^ r3;
    let r21 = r7 ^ r19;
    let r22 = r5 ^ r20;
    planes[4] = r21 ^ r22;
    let r23 = a9 ^ a12;
    planes[5] = r8 ^ r23;
    planes[6] = r1 ^ r4;
    planes[7] = a16 ^ r11;
}

/// The inverse of the linear map used inside the S-box, without its constant.
#[inline(always)]
fn inv_affine_linear(q: &mut [u64; 8]) {
    *q = [
        q[6] ^ q[3] ^ q[1],
        q[7] ^ q[4] ^ q[2],
        q[0] ^ q[5] ^ q[3],
        q[1] ^ q[6] ^ q[4],
        q[2] ^ q[7] ^ q[5],
        q[3] ^ q[0] ^ q[6],
        q[4] ^ q[1] ^ q[7],
        q[5] ^ q[2] ^ q[0],
    ];
}

/// The inverse S-box, reusing the forward circuit.
///
/// The input bytes must already have been XORed with 0x63.
#[inline(always)]
fn inv_sbox_core(planes: &mut [u64; 8]) {
    inv_affine_linear(planes);
    sbox_core(planes);
    inv_affine_linear(planes);
}

/// Row `r` of each column gets row `r + 1`.
#[inline(always)]
fn rot_rows_1(x: u64) -> u64 {
    ((x >> 8) & 0x00ff_ffff_00ff_ffff) | ((x << 24) & 0xff00_0000_ff00_0000)
}

/// Row `r` of each column gets row `r + 2`.
#[inline(always)]
fn rot_rows_2(x: u64) -> u64 {
    ((x >> 16) & 0x0000_ffff_0000_ffff) | ((x << 16) & 0xffff_0000_ffff_0000)
}

/// Multiplies every byte by 2 in GF(2^8), without a multiplication.
#[inline(always)]
fn mul2(x: u64) -> u64 {
    let m = x & 0x8080_8080_8080_8080;
    ((x ^ m) << 1) ^ (((m >> 1) - (m >> 7)) & 0x1b1b_1b1b_1b1b_1b1b)
}

/// Multiplies every byte by 4 in GF(2^8), without a multiplication.
#[inline(always)]
fn mul4(x: u64) -> u64 {
    let m = x & 0xc0c0_c0c0_c0c0_c0c0;
    let u = m ^ (m >> 1);
    ((x ^ m) << 2) ^ (u >> 2) ^ (u >> 5)
}

/// MixColumns on the two columns of a word.
#[inline(always)]
fn mix_columns(x: u64) -> u64 {
    let n = rot_rows_1(x);
    let s = x ^ n;
    mul2(s) ^ n ^ rot_rows_2(s)
}

/// InvMixColumns on the two columns of a word.
///
/// InvMixColumns is MixColumns after a cheaper step, which saves work.
#[inline(always)]
pub(crate) fn inv_mix_columns(x: u64) -> u64 {
    let t = x ^ rot_rows_2(x);
    mix_columns(x ^ mul4(t))
}

/// SubBytes followed by ShiftRows.
#[inline(always)]
fn sub_bytes_shift_rows(block: &Block) -> (u64, u64) {
    let (lo, hi) = permute_rows(block.lo, block.hi, ROWS_A, ROWS_B);
    let mut planes = load(lo, hi);
    sbox_core(&mut planes);
    let (lo, hi) = store(&planes);
    (lo ^ C63, hi ^ C63)
}

/// InvShiftRows followed by InvSubBytes.
#[inline(always)]
fn inv_sub_bytes_shift_rows(block: &Block) -> (u64, u64) {
    let (lo, hi) = permute_rows(block.lo ^ C63, block.hi ^ C63, ROWS_B, ROWS_A);
    let mut planes = load(lo, hi);
    inv_sbox_core(&mut planes);
    store(&planes)
}

/// AES forward round (SubBytes, ShiftRows, MixColumns, AddRoundKey).
#[inline]
pub fn block_encrypt(block: &Block, rk: &Block) -> Block {
    let (lo, hi) = sub_bytes_shift_rows(block);
    Block {
        lo: mix_columns(lo),
        hi: mix_columns(hi),
    }
    .xor(rk)
}

/// AES final forward round (SubBytes, ShiftRows, AddRoundKey, no MixColumns).
#[inline]
pub fn block_encrypt_last(block: &Block, rk: &Block) -> Block {
    let (lo, hi) = sub_bytes_shift_rows(block);
    Block { lo, hi }.xor(rk)
}

/// AES inverse round (InvShiftRows, InvSubBytes, InvMixColumns, AddRoundKey).
///
/// The round key must come from `key_schedule::inverse_key_schedule_*`.
#[inline]
pub fn block_decrypt(block: &Block, rk: &Block) -> Block {
    let (lo, hi) = inv_sub_bytes_shift_rows(block);
    Block {
        lo: inv_mix_columns(lo),
        hi: inv_mix_columns(hi),
    }
    .xor(rk)
}

/// AES final inverse round (InvShiftRows, InvSubBytes, AddRoundKey, no
/// InvMixColumns).
#[inline]
pub fn block_decrypt_last(block: &Block, rk: &Block) -> Block {
    let (lo, hi) = inv_sub_bytes_shift_rows(block);
    Block { lo, hi }.xor(rk)
}
