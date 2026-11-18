use crate::error::{FishError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16KvLayout {
    pub batch: usize,
    pub kv_heads: usize,
    pub capacity: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvWrite2d {
    pub position: usize,
    pub frames: usize,
    pub rows: usize,
    pub destination_pitch_bytes: usize,
    pub source_pitch_bytes: usize,
    pub destination_offset_bytes: usize,
    pub row_bytes: usize,
    pub total_cache_bytes: usize,
}

impl Bf16KvLayout {
    pub fn write_plan(self, position: usize, frames: usize) -> Result<KvWrite2d> {
        let invalid = || FishError::InvalidShape("invalid BF16 KV strided write parameters");
        if self.batch == 0
            || self.kv_heads == 0
            || self.capacity == 0
            || self.head_dim == 0
            || frames == 0
            || position
                .checked_add(frames)
                .is_none_or(|end| end > self.capacity)
        {
            return Err(invalid());
        }
        let checked = |a: usize, b: usize| a.checked_mul(b).ok_or_else(invalid);
        let rows = checked(self.batch, self.kv_heads)?;
        let destination_pitch_bytes = checked(checked(self.capacity, self.head_dim)?, 2)?;
        let source_pitch_bytes = checked(checked(frames, self.head_dim)?, 2)?;
        let destination_offset_bytes = checked(checked(position, self.head_dim)?, 2)?;
        let total_cache_bytes = checked(rows, destination_pitch_bytes)?;
        Ok(KvWrite2d {
            position,
            frames,
            rows,
            destination_pitch_bytes,
            source_pitch_bytes,
            destination_offset_bytes,
            row_bytes: source_pitch_bytes,
            total_cache_bytes,
        })
    }
}

impl KvWrite2d {
    pub fn apply_bf16_reference(&self, cache: &mut [u16], new_values: &[u16]) -> Result<()> {
        let invalid = || FishError::InvalidShape("KV buffer length or strided address invalid");
        if self.destination_pitch_bytes % 2 != 0
            || self.source_pitch_bytes % 2 != 0
            || self.destination_offset_bytes % 2 != 0
            || self.row_bytes % 2 != 0
            || self.rows == 0
            || self.frames == 0
            || self.row_bytes != self.source_pitch_bytes
            || self
                .destination_offset_bytes
                .checked_add(self.row_bytes)
                .is_none_or(|end| end > self.destination_pitch_bytes)
            || cache.len().checked_mul(2) != Some(self.total_cache_bytes)
            || new_values.len().checked_mul(2) != self.rows.checked_mul(self.source_pitch_bytes)
        {
            return Err(invalid());
        }
        let dst_pitch = self.destination_pitch_bytes / 2;
        let src_pitch = self.source_pitch_bytes / 2;
        let offset = self.destination_offset_bytes / 2;
        for row in 0..self.rows {
            let dst = row * dst_pitch + offset;
            let src = row * src_pitch;
            cache[dst..dst + src_pitch].copy_from_slice(&new_values[src..src + src_pitch]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_only_new_prefix_positions_for_every_batch_and_head() {
        let layout = Bf16KvLayout {
            batch: 2,
            kv_heads: 3,
            capacity: 11,
            head_dim: 4,
        };
        let mut cache = vec![0x7FC1u16; 2 * 3 * 11 * 4];
        let plan = layout.write_plan(5, 2).unwrap();
        assert_eq!(plan.rows, 6);
        assert_eq!(plan.row_bytes, 16);
        assert_eq!(plan.destination_pitch_bytes, 88);
        let values = (0..6 * 2 * 4).map(|i| i as u16).collect::<Vec<_>>();
        plan.apply_bf16_reference(&mut cache, &values).unwrap();
        for row in 0..6 {
            let start = row * 11 * 4;
            assert!(cache[start..start + 5 * 4].iter().all(|v| *v == 0x7FC1));
            assert_eq!(
                &cache[start + 5 * 4..start + 7 * 4],
                &values[row * 8..row * 8 + 8]
            );
            assert!(cache[start + 7 * 4..start + 11 * 4]
                .iter()
                .all(|v| *v == 0x7FC1));
        }
    }

    #[test]
    fn single_token_decode_writes_only_one_frame() {
        let layout = Bf16KvLayout {
            batch: 1,
            kv_heads: 2,
            capacity: 1024,
            head_dim: 64,
        };
        let write = layout.write_plan(999, 1).unwrap();
        assert_eq!(write.row_bytes, 128);
        assert_eq!(write.total_cache_bytes, 1024 * 2 * 64 * 2);
        let mut dst = vec![7u16; 1024 * 2 * 64];
        write
            .apply_bf16_reference(&mut dst, &vec![19u16; 2 * 64])
            .unwrap();
        assert_eq!(dst.iter().filter(|x| **x == 19).count(), 128);
    }

    #[test]
    fn fails_closed_on_overflow_and_incorrect_buffer_geometry() {
        let layout = Bf16KvLayout {
            batch: 1,
            kv_heads: 2,
            capacity: 8,
            head_dim: 3,
        };
        assert!(layout.write_plan(7, 2).is_err());
        assert!(layout.write_plan(0, 0).is_err());
        assert!(Bf16KvLayout {
            head_dim: usize::MAX,
            ..layout
        }
        .write_plan(0, 1)
        .is_err());
        let plan = layout.write_plan(2, 2).unwrap();
        let mut dst = vec![0u16; 2 * 8 * 3];
        assert!(plan.apply_bf16_reference(&mut dst, &[0; 3]).is_err());
        dst.pop();
        assert!(plan.apply_bf16_reference(&mut dst, &[0; 12]).is_err());
    }
}
