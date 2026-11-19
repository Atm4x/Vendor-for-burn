use std::collections::VecDeque;

/// Bounded ownership cache for buffers whose backing storage remains allocated while idle.
/// The caller must enforce device-stream completion before consuming a cached buffer.
pub struct BoundedScratchPool<T> {
    capacity_bytes: usize,
    cached_bytes: usize,
    entries: VecDeque<(usize, T)>,
}

impl<T> BoundedScratchPool<T> {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            cached_bytes: 0,
            entries: VecDeque::new(),
        }
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn take(&mut self, bytes: usize) -> Option<T> {
        let position = self.entries.iter().position(|(size, _)| *size == bytes)?;
        let (size, value) = self.entries.remove(position)?;
        self.cached_bytes -= size;
        Some(value)
    }

    pub fn insert(&mut self, bytes: usize, value: T) -> (bool, Vec<T>) {
        if bytes == 0 || bytes > self.capacity_bytes {
            return (false, vec![value]);
        }
        let mut evicted = Vec::new();
        while self.cached_bytes > self.capacity_bytes - bytes {
            let Some((old_size, old)) = self.entries.pop_front() else {
                return (false, vec![value]);
            };
            self.cached_bytes -= old_size;
            evicted.push(old);
        }
        self.entries.push_back((bytes, value));
        self.cached_bytes += bytes;
        (true, evicted)
    }

    pub fn offer(&mut self, bytes: usize, value: T) -> bool {
        self.insert(bytes, value).0
    }

    pub fn drain(&mut self) -> Vec<T> {
        self.cached_bytes = 0;
        self.entries.drain(..).map(|(_, value)| value).collect()
    }

    pub fn clear(&mut self) {
        drop(self.drain());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_exact_size_across_distinct_requests() {
        let mut pool = BoundedScratchPool::new(16);
        assert!(pool.offer(8, String::from("request-one")));
        assert_eq!(pool.cached_bytes(), 8);
        assert_eq!(pool.take(4), None);
        assert_eq!(pool.take(8).as_deref(), Some("request-one"));
        assert_eq!(pool.cached_bytes(), 0);
        assert!(pool.offer(8, String::from("request-two")));
        assert_eq!(pool.take(8).as_deref(), Some("request-two"));
    }

    #[test]
    fn evicts_oldest_to_observe_capacity_and_never_caches_oversize() {
        let mut pool = BoundedScratchPool::new(12);
        assert!(pool.offer(8, 1));
        assert!(pool.offer(8, 2));
        assert_eq!(pool.cached_bytes(), 8);
        assert_eq!(pool.take(8), Some(2));
        assert!(!pool.offer(16, 3));
        assert!(!pool.offer(0, 3));
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn cache_eviction_is_returned_to_owner_for_safe_retirement() {
        let mut pool = BoundedScratchPool::new(8);
        assert!(pool.insert(8, 1).0);
        let (stored, evicted) = pool.insert(8, 2);
        assert!(stored);
        assert_eq!(evicted, vec![1]);
        assert_eq!(pool.take(8), Some(2));
    }

    #[test]
    fn idle_cached_reservations_still_consume_live_device_budget() {
        let budget = crate::device_memory::DeviceMemoryBudget::new(64).unwrap();
        let mut pool = BoundedScratchPool::new(32);
        let lease = budget.reserve(32).unwrap();
        assert!(pool.offer(32, lease));
        assert_eq!(budget.usage().allocated_bytes, 32);
        assert!(budget.reserve(33).is_err());
        let reused = pool.take(32).unwrap();
        assert_eq!(budget.usage().allocated_bytes, 32);
        drop(reused);
        assert_eq!(budget.usage().allocated_bytes, 0);
    }

    #[test]
    fn clear_discards_cached_entries() {
        let mut pool = BoundedScratchPool::new(32);
        pool.offer(16, 42);
        pool.clear();
        assert_eq!(pool.cached_bytes(), 0);
        assert_eq!(pool.take(16), None);
    }
}
