use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryUsage {
    pub limit_bytes: usize,
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryError {
    InvalidSize,
    Exhausted,
}

#[derive(Debug, Clone)]
pub struct DeviceMemoryBudget {
    state: Arc<Mutex<DeviceMemoryUsage>>,
}

#[derive(Debug)]
pub struct DeviceMemoryLease {
    state: Arc<Mutex<DeviceMemoryUsage>>,
    bytes: usize,
}

impl DeviceMemoryBudget {
    pub fn new(limit_bytes: usize) -> Result<Self, DeviceMemoryError> {
        if limit_bytes == 0 {
            return Err(DeviceMemoryError::InvalidSize);
        }
        Ok(Self {
            state: Arc::new(Mutex::new(DeviceMemoryUsage {
                limit_bytes,
                allocated_bytes: 0,
                peak_bytes: 0,
            })),
        })
    }

    pub fn usage(&self) -> DeviceMemoryUsage {
        *self.state.lock().unwrap_or_else(|error| error.into_inner())
    }

    pub fn reserve(&self, bytes: usize) -> Result<DeviceMemoryLease, DeviceMemoryError> {
        if bytes == 0 {
            return Err(DeviceMemoryError::InvalidSize);
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let next = state
            .allocated_bytes
            .checked_add(bytes)
            .ok_or(DeviceMemoryError::Exhausted)?;
        if next > state.limit_bytes {
            return Err(DeviceMemoryError::Exhausted);
        }
        state.allocated_bytes = next;
        state.peak_bytes = state.peak_bytes.max(next);
        Ok(DeviceMemoryLease {
            state: Arc::clone(&self.state),
            bytes,
        })
    }
}

impl Drop for DeviceMemoryLease {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.allocated_bytes = state
            .allocated_bytes
            .checked_sub(self.bytes)
            .expect("device memory lease must be accounted exactly once");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn counts_live_allocations_and_reclaims_last_shared_owner() {
        let budget = DeviceMemoryBudget::new(64).unwrap();
        let leased = Arc::new(budget.reserve(48).unwrap());
        let other = Arc::clone(&leased);
        assert_eq!(budget.usage().allocated_bytes, 48);
        assert_eq!(
            budget.reserve(17).unwrap_err(),
            DeviceMemoryError::Exhausted
        );
        drop(leased);
        assert_eq!(budget.usage().allocated_bytes, 48);
        drop(other);
        assert_eq!(budget.usage().allocated_bytes, 0);
        let _replacement = budget.reserve(64).unwrap();
        assert_eq!(budget.usage().peak_bytes, 64);
    }

    #[test]
    fn rejects_zero_and_overflow_without_changing_usage() {
        assert_eq!(
            DeviceMemoryBudget::new(0).unwrap_err(),
            DeviceMemoryError::InvalidSize
        );
        let budget = DeviceMemoryBudget::new(usize::MAX).unwrap();
        assert_eq!(
            budget.reserve(0).unwrap_err(),
            DeviceMemoryError::InvalidSize
        );
        let _lease = budget.reserve(usize::MAX).unwrap();
        assert_eq!(budget.reserve(1).unwrap_err(), DeviceMemoryError::Exhausted);
        assert_eq!(budget.usage().allocated_bytes, usize::MAX);
    }

    #[test]
    fn concurrent_reservations_never_exceed_budget() {
        let budget = DeviceMemoryBudget::new(1024).unwrap();
        let leases = thread::scope(|scope| {
            let tasks: Vec<_> = (0..16)
                .map(|_| {
                    let budget = budget.clone();
                    scope.spawn(move || budget.reserve(128))
                })
                .collect();
            tasks
                .into_iter()
                .filter_map(|task| task.join().unwrap().ok())
                .collect::<Vec<_>>()
        });
        assert_eq!(leases.len(), 8);
        assert_eq!(budget.usage().allocated_bytes, 1024);
        drop(leases);
        assert_eq!(budget.usage().allocated_bytes, 0);
        assert_eq!(budget.usage().peak_bytes, 1024);
    }
}
