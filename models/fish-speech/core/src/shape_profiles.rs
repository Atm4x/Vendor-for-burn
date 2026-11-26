use crate::config::FishModelConfig;
use crate::error::{FishError, Result};
use crate::graph::Stage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeShape {
    pub stage: Stage,
    pub query_len: usize,
    pub active_kv_len: usize,
    pub first_position: usize,
    pub physical_cache_capacity: usize,
}

impl DecodeShape {
    pub fn new(
        stage: Stage,
        query_len: usize,
        active_kv_len: usize,
        capacity: usize,
    ) -> Result<Self> {
        if query_len == 0
            || active_kv_len == 0
            || active_kv_len > capacity
            || query_len > active_kv_len
        {
            return Err(FishError::InvalidShape(
                "invalid decode/prefill query or KV length",
            ));
        }
        if matches!(stage, Stage::SlowDecode | Stage::FastDecode) && query_len != 1 {
            return Err(FishError::InvalidShape(
                "cached decode must have exactly one query token",
            ));
        }
        if stage == Stage::SlowPrefill && query_len != active_kv_len {
            return Err(FishError::InvalidShape(
                "empty-cache prefill needs query=active prefix",
            ));
        }
        if !matches!(
            stage,
            Stage::SlowDecode | Stage::FastDecode | Stage::SlowPrefill
        ) {
            return Err(FishError::Unsupported(
                "KV shape profiles only support AR stages",
            ));
        }
        Ok(Self {
            stage,
            query_len,
            active_kv_len,
            first_position: active_kv_len - query_len,
            physical_cache_capacity: capacity,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLengthBucket {
    pub min: usize,
    pub max: usize,
}

impl KvLengthBucket {
    pub fn contains(&self, length: usize) -> bool {
        (self.min..=self.max).contains(&length)
    }
}

#[derive(Debug, Clone)]
pub struct ArProfileGroup {
    pub stage: Stage,
    pub capacity: usize,
    pub buckets: Vec<KvLengthBucket>,
}

impl ArProfileGroup {
    pub fn new(stage: Stage, capacity: usize, first_bucket: usize) -> Result<Self> {
        if capacity == 0
            || first_bucket == 0
            || !matches!(
                stage,
                Stage::SlowPrefill | Stage::SlowDecode | Stage::FastDecode
            )
        {
            return Err(FishError::InvalidShape("invalid AR bucket profile"));
        }
        let mut buckets = Vec::new();
        let mut end = first_bucket.min(capacity);
        let mut begin = 1usize;
        loop {
            buckets.push(KvLengthBucket {
                min: begin,
                max: end,
            });
            if end == capacity {
                break;
            }
            begin = end + 1;
            end = end.saturating_mul(2).min(capacity);
        }
        Ok(Self {
            stage,
            capacity,
            buckets,
        })
    }

    pub fn locate(&self, query_len: usize, active_kv_len: usize) -> Result<(usize, DecodeShape)> {
        let shape = DecodeShape::new(self.stage, query_len, active_kv_len, self.capacity)?;
        let index = self
            .buckets
            .iter()
            .position(|b| b.contains(shape.active_kv_len))
            .ok_or(FishError::InvalidShape("KV bucket missing logical prefix"))?;
        Ok((index, shape))
    }

    pub fn validate(&self) -> Result<()> {
        if self.buckets.is_empty()
            || self.buckets[0].min != 1
            || self.buckets.last().unwrap().max != self.capacity
        {
            return Err(FishError::InvalidShape("incomplete KV profile coverage"));
        }
        for pair in self.buckets.windows(2) {
            if pair[0].max.checked_add(1) != Some(pair[1].min) {
                return Err(FishError::InvalidShape("KV bucket overlap or gap"));
            }
        }
        if self.buckets.iter().any(|bucket| bucket.min > bucket.max) {
            return Err(FishError::InvalidShape("inverted KV bucket"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ArProfiles {
    pub slow_prefill: ArProfileGroup,
    pub slow_decode: ArProfileGroup,
    pub fast_decode: Option<ArProfileGroup>,
}

impl ArProfiles {
    pub fn from_config(config: &FishModelConfig, first_bucket: usize) -> Result<Self> {
        config.validate()?;
        let slow_prefill =
            ArProfileGroup::new(Stage::SlowPrefill, config.max_seq_len, first_bucket)?;
        let slow_decode = ArProfileGroup::new(Stage::SlowDecode, config.max_seq_len, first_bucket)?;
        let fast_decode = config
            .fast
            .as_ref()
            .map(|_| ArProfileGroup::new(Stage::FastDecode, config.num_codebooks, first_bucket))
            .transpose()?;
        for group in [&slow_prefill, &slow_decode] {
            group.validate()?;
        }
        if let Some(group) = &fast_decode {
            group.validate()?;
        }
        Ok(Self {
            slow_prefill,
            slow_decode,
            fast_decode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_range_is_covered_without_per_token_engine_buckets() {
        let group = ArProfileGroup::new(Stage::SlowDecode, 4096, 32).unwrap();
        group.validate().unwrap();
        assert_eq!(group.buckets.len(), 8);
        for len in 1..=4096 {
            let (index, shape) = group.locate(1, len).unwrap();
            assert_eq!(shape.first_position, len - 1);
            assert_eq!(shape.physical_cache_capacity, 4096);
            assert!(group.buckets[index].contains(len));
        }
        assert_eq!(
            group.locate(1, 33).unwrap().0,
            group.locate(1, 64).unwrap().0
        );
        assert_ne!(
            group.locate(1, 32).unwrap().0,
            group.locate(1, 33).unwrap().0
        );
    }

    #[test]
    fn prefill_uses_real_logical_length_not_bucket_upper_bound() {
        let group = ArProfileGroup::new(Stage::SlowPrefill, 400, 32).unwrap();
        let (index, shape) = group.locate(47, 47).unwrap();
        assert_eq!(group.buckets[index].max, 64);
        assert_eq!(shape.active_kv_len, 47);
        assert_eq!(shape.first_position, 0);
        assert!(group.locate(47, 48).is_err());
    }

    #[test]
    fn fast_and_slow_cache_profiles_are_distinct() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let profiles = ArProfiles::from_config(&c, 32).unwrap();
        assert_eq!(profiles.slow_decode.capacity, c.max_seq_len);
        let fast = profiles.fast_decode.unwrap();
        assert_eq!(fast.capacity, c.num_codebooks);
        assert_eq!(fast.buckets.len(), 1);
        assert_eq!(
            fast.locate(1, c.num_codebooks).unwrap().1.first_position,
            c.num_codebooks - 1
        );
    }

    #[test]
    fn out_of_range_and_invalid_query_sizes_fail() {
        let group = ArProfileGroup::new(Stage::SlowDecode, 64, 8).unwrap();
        assert!(group.locate(2, 4).is_err());
        assert!(group.locate(1, 65).is_err());
        assert!(group.locate(1, 0).is_err());
        assert!(ArProfileGroup::new(Stage::SlowDecode, 100, 0).is_err());
        assert!(DecodeShape::new(Stage::SlowPrefill, 3, 4, 100).is_err());
    }

    #[test]
    fn profile_validation_rejects_gaps() {
        let mut group = ArProfileGroup::new(Stage::SlowPrefill, 100, 8).unwrap();
        group.buckets[1].min += 1;
        assert!(group.validate().is_err());
    }
}
