use crate::error::{FishError, Result};

#[derive(Debug, Clone)]
pub struct InterleavedRope {
    pub pair_indices: Vec<i32>,
    pub cosine: Vec<f32>,
    pub signed_sine: Vec<f32>,
    pub frames: usize,
    pub head_dim: usize,
}

pub fn bf16_frequency(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return f32::from_bits(bits & 0xffff_0000);
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

pub fn fish_rope_coefficients(
    first_position: usize,
    frames: usize,
    head_dim: usize,
    base: f32,
) -> Result<InterleavedRope> {
    if frames == 0
        || head_dim == 0
        || head_dim % 2 != 0
        || first_position.checked_add(frames).is_none()
        || head_dim > i32::MAX as usize
        || !base.is_finite()
        || base <= 0.0
    {
        return Err(FishError::InvalidShape("invalid Fish 1.5 rotary profile"));
    }
    let elements = frames
        .checked_mul(head_dim)
        .ok_or(FishError::InvalidShape("RoPE coefficient size overflow"))?;
    let mut cosine = Vec::with_capacity(elements);
    let mut signed_sine = Vec::with_capacity(elements);
    let mut pair_indices = Vec::with_capacity(head_dim);
    for pair in 0..head_dim / 2 {
        pair_indices.push((2 * pair + 1) as i32);
        pair_indices.push((2 * pair) as i32);
    }
    for frame in 0..frames {
        for pair in 0..head_dim / 2 {
            let freq = 1.0 / base.powf(2.0 * pair as f32 / head_dim as f32);
            let angle = (first_position + frame) as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let cosine_bf16 = bf16_frequency(cos);
            let sine_bf16 = bf16_frequency(sin);
            cosine.extend([cosine_bf16, cosine_bf16]);
            signed_sine.extend([-sine_bf16, sine_bf16]);
        }
    }
    Ok(InterleavedRope {
        pair_indices,
        cosine,
        signed_sine,
        frames,
        head_dim,
    })
}

pub fn rotate_bhtd(
    x: &[f32],
    batch: usize,
    heads: usize,
    first_position: usize,
    frames: usize,
    head_dim: usize,
    base: f32,
) -> Result<Vec<f32>> {
    let plan = fish_rope_coefficients(first_position, frames, head_dim, base)?;
    let extent = batch
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(frames))
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or(FishError::InvalidShape("RoPE input size overflow"))?;
    if x.len() != extent {
        return Err(FishError::InvalidShape("RoPE input extent mismatch"));
    }
    let mut result = Vec::with_capacity(extent);
    for row in x.chunks_exact(frames * head_dim) {
        for frame in 0..frames {
            let frame_base = frame * head_dim;
            for d in 0..head_dim {
                let at = frame_base + d;
                let partner = frame_base + plan.pair_indices[d] as usize;
                result.push(row[at] * plan.cosine[at] + row[partner] * plan.signed_sine[at]);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_uses_nearest_even_not_truncation() {
        assert_eq!(bf16_frequency(1.0), 1.0);
        assert_eq!(bf16_frequency(f32::from_bits(0x3f80_8000)), 1.0);
        assert_eq!(
            bf16_frequency(f32::from_bits(0x3f81_8000)),
            f32::from_bits(0x3f82_0000)
        );
    }

    #[test]
    fn interleaved_rope_has_signed_pair_rotation() {
        let p = fish_rope_coefficients(1, 2, 4, 10_000.).unwrap();
        assert_eq!(p.pair_indices, [1, 0, 3, 2]);
        let out = rotate_bhtd(&[1., 0., 0., 1., 1., 0., 0., 1.], 1, 1, 1, 2, 4, 10_000.).unwrap();
        assert!((out[0] - bf16_frequency(1f32.cos())).abs() < 1e-6);
        assert!((out[1] - bf16_frequency(1f32.sin())).abs() < 1e-6);
        assert_eq!(rotate_bhtd(&[1.], 1, 1, 0, 1, 2, 10_000.).is_err(), true);
    }
}
