use std::collections::BTreeMap;

use crate::ar_stream_plan::{KvTransfer, KvWriteBarrier};
use crate::error::{FishError, Result};
use crate::kv_device::KvWrite2d;

/// A backend implementation must enqueue allocation users and copies on one ordered GPU stream.
/// It owns both the stream and device allocations, and must keep allocations alive until queued
/// operations complete. This trait does not provide a GPU implementation in the Linux prototype.
pub trait KvStreamDevice {
    type Buffer;

    fn allocate_bf16(&mut self, bytes: usize) -> Result<Self::Buffer>;
    fn copy_2d_async(
        &mut self,
        destination: &Self::Buffer,
        source: &Self::Buffer,
        geometry: KvWrite2d,
    ) -> Result<()>;
    fn synchronize(&mut self) -> Result<()>;
}

struct CacheSlot<B> {
    buffer: B,
    bytes: usize,
    active_len: usize,
}

/// Owns stable-address physical caches across prefill and cached decode.
/// Submission advances logical lengths only after both stream copies have been enqueued.
/// On partial submission failure, the owner is poisoned until explicit synchronization/reset.
pub struct KvDeviceOwner<D: KvStreamDevice> {
    device: D,
    slots: BTreeMap<String, CacheSlot<D::Buffer>>,
    poisoned: bool,
}

impl<D: KvStreamDevice> KvDeviceOwner<D> {
    pub fn new(device: D) -> Self {
        Self {
            device,
            slots: BTreeMap::new(),
            poisoned: false,
        }
    }

    fn ensure_slot(&mut self, write: &KvTransfer) -> Result<()> {
        if let Some(slot) = self.slots.get(&write.cache_name) {
            if slot.bytes != write.geometry.total_cache_bytes {
                return Err(FishError::InvalidShape("KV device cache shape changed"));
            }
            return Ok(());
        }
        let buffer = self
            .device
            .allocate_bf16(write.geometry.total_cache_bytes)?;
        self.slots.insert(
            write.cache_name.clone(),
            CacheSlot {
                buffer,
                bytes: write.geometry.total_cache_bytes,
                active_len: 0,
            },
        );
        Ok(())
    }

    /// Validate an entire sequence of KV writes without allocating device buffers or
    /// submitting network work. The projected lengths include writes in the same plan.
    pub fn preflight_writes<'a>(
        &self,
        writes: impl IntoIterator<Item = &'a KvWriteBarrier>,
    ) -> Result<()> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "KV stream is poisoned; reset required",
            ));
        }
        let mut projected = BTreeMap::<String, (usize, usize)>::new();
        for pair in writes {
            if pair.key.cache_name == pair.value.cache_name
                || pair.key.geometry != pair.value.geometry
                || pair.key.geometry.position != pair.position
            {
                return Err(FishError::InvalidShape(
                    "KV K/V source destination mismatch",
                ));
            }
            for transfer in [&pair.key, &pair.value] {
                let (bytes, active_len) = projected
                    .get(&transfer.cache_name)
                    .copied()
                    .or_else(|| {
                        self.slots
                            .get(&transfer.cache_name)
                            .map(|slot| (slot.bytes, slot.active_len))
                    })
                    .unwrap_or((transfer.geometry.total_cache_bytes, 0));
                if bytes != transfer.geometry.total_cache_bytes {
                    return Err(FishError::InvalidShape("KV device cache shape changed"));
                }
                if active_len != pair.position {
                    return Err(FishError::InvalidTransition(
                        "KV write must append exactly after the active prefix",
                    ));
                }
                let next = pair
                    .position
                    .checked_add(transfer.geometry.frames)
                    .ok_or(FishError::InvalidShape("KV write position overflow"))?;
                projected.insert(transfer.cache_name.clone(), (bytes, next));
            }
        }
        Ok(())
    }

    pub fn ensure_cache(&mut self, write: &KvTransfer) -> Result<()> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "KV stream is poisoned; reset required",
            ));
        }
        self.ensure_slot(write)
    }

    pub fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }

    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    pub fn enqueue_pair(
        &mut self,
        writes: &KvWriteBarrier,
        key_source_bf16: &D::Buffer,
        value_source_bf16: &D::Buffer,
    ) -> Result<()> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "KV stream is poisoned; reset required",
            ));
        }
        if writes.key.cache_name == writes.value.cache_name
            || writes.key.geometry != writes.value.geometry
            || writes.key.geometry.position != writes.position
        {
            return Err(FishError::InvalidShape(
                "KV K/V source destination mismatch",
            ));
        }
        self.ensure_slot(&writes.key)?;
        self.ensure_slot(&writes.value)?;
        for transfer in [&writes.key, &writes.value] {
            let slot = &self.slots[&transfer.cache_name];
            if slot.active_len != writes.position {
                return Err(FishError::InvalidTransition(
                    "KV write must append exactly after the active prefix",
                ));
            }
        }
        let key = &self.slots[&writes.key.cache_name].buffer;
        if let Err(err) = self
            .device
            .copy_2d_async(key, key_source_bf16, writes.key.geometry)
        {
            self.poisoned = true;
            return Err(err);
        }
        let value = &self.slots[&writes.value.cache_name].buffer;
        if let Err(err) = self
            .device
            .copy_2d_async(value, value_source_bf16, writes.value.geometry)
        {
            self.poisoned = true;
            return Err(err);
        }
        let end = writes.position + writes.key.geometry.frames;
        self.slots
            .get_mut(&writes.key.cache_name)
            .unwrap()
            .active_len = end;
        self.slots
            .get_mut(&writes.value.cache_name)
            .unwrap()
            .active_len = end;
        Ok(())
    }

    pub fn cache(&self, name: &str) -> Option<&D::Buffer> {
        self.slots.get(name).map(|slot| &slot.buffer)
    }

    pub fn active_len(&self, name: &str) -> Option<usize> {
        self.slots.get(name).map(|slot| slot.active_len)
    }

    /// Fast AR begins anew for each Slow frame; on the same stream old work remains ordered.
    /// This does not clear memory: only the logical prefix determines cache visibility.
    pub fn reset_fast_prefix(&mut self) -> Result<()> {
        if self.poisoned {
            return Err(FishError::InvalidTransition(
                "KV stream is poisoned; reset required",
            ));
        }
        for (name, slot) in &mut self.slots {
            if name.starts_with("fast_layers.") {
                slot.active_len = 0;
            }
        }
        Ok(())
    }

    /// Request boundary: synchronize once before logical reset or releasing device allocations.
    pub fn reset_request(&mut self) -> Result<()> {
        self.device.synchronize()?;
        for slot in self.slots.values_mut() {
            slot.active_len = 0;
        }
        self.poisoned = false;
        Ok(())
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::*;
    use crate::{
        ar_stream_plan::{ArStreamPhase, ArStreamPlan},
        config::FishModelConfig,
        transformer_ir::TransformerIrFactory,
    };

    #[derive(Default)]
    struct MockDevice {
        allocations: usize,
        copies: usize,
        syncs: usize,
        fail_copy: Option<usize>,
    }

    impl KvStreamDevice for MockDevice {
        type Buffer = Rc<RefCell<Vec<u16>>>;

        fn allocate_bf16(&mut self, bytes: usize) -> Result<Self::Buffer> {
            self.allocations += 1;
            Ok(Rc::new(RefCell::new(vec![0x7fc1; bytes / 2])))
        }

        fn copy_2d_async(
            &mut self,
            destination: &Self::Buffer,
            source: &Self::Buffer,
            geometry: KvWrite2d,
        ) -> Result<()> {
            self.copies += 1;
            if self.fail_copy == Some(self.copies) {
                return Err(FishError::InvalidTransition("injected device copy failure"));
            }
            geometry.apply_bf16_reference(&mut destination.borrow_mut(), &source.borrow())
        }

        fn synchronize(&mut self) -> Result<()> {
            self.syncs += 1;
            Ok(())
        }
    }

    fn first_barrier(prefill: bool) -> KvWriteBarrier {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        let graph = if prefill {
            factory.slow_prefill(4).unwrap()
        } else {
            factory.slow_decode(5).unwrap()
        };
        ArStreamPlan::from_graph(&graph)
            .unwrap()
            .phases
            .into_iter()
            .find_map(|phase| match phase {
                ArStreamPhase::KvWrites(write) => Some(write),
                _ => None,
            })
            .unwrap()
    }

    #[test]
    fn stable_cache_is_reused_across_prefill_and_decode_without_sync() {
        let mut owner = KvDeviceOwner::new(MockDevice::default());
        let prefill = first_barrier(true);
        let key = Rc::new(RefCell::new(vec![
            11u16;
            prefill.key.geometry.rows
                * prefill.key.geometry.frames
                * 8
        ]));
        let value = Rc::new(RefCell::new(vec![
            12u16;
            prefill.value.geometry.rows
                * prefill.value.geometry.frames
                * 8
        ]));
        owner.enqueue_pair(&prefill, &key, &value).unwrap();
        let existing = owner.cache(&prefill.key.cache_name).unwrap().clone();
        assert_eq!(owner.active_len(&prefill.key.cache_name), Some(4));
        let decode = first_barrier(false);
        let next_key = Rc::new(RefCell::new(vec![21u16; decode.key.geometry.rows * 8]));
        let next_value = Rc::new(RefCell::new(vec![22u16; decode.value.geometry.rows * 8]));
        owner.enqueue_pair(&decode, &next_key, &next_value).unwrap();
        assert!(Rc::ptr_eq(
            &existing,
            owner.cache(&prefill.key.cache_name).unwrap()
        ));
        assert_eq!(owner.active_len(&prefill.key.cache_name), Some(5));
        assert_eq!(owner.device.allocations, 2);
        assert_eq!(owner.device.copies, 4);
        assert_eq!(owner.device.syncs, 0);
        owner.reset_request().unwrap();
        assert_eq!(owner.active_len(&prefill.key.cache_name), Some(0));
        assert_eq!(owner.device.syncs, 1);
    }

    #[test]
    fn partial_stream_failure_poisoned_until_synchronized_reset() {
        let mut owner = KvDeviceOwner::new(MockDevice {
            fail_copy: Some(2),
            ..Default::default()
        });
        let prefill = first_barrier(true);
        let size = prefill.key.geometry.rows * prefill.key.geometry.frames * 8;
        let source = Rc::new(RefCell::new(vec![3u16; size]));
        assert!(owner.enqueue_pair(&prefill, &source, &source).is_err());
        assert!(owner.is_poisoned());
        assert!(owner.enqueue_pair(&prefill, &source, &source).is_err());
        owner.reset_request().unwrap();
        assert!(!owner.is_poisoned());
        owner.enqueue_pair(&prefill, &source, &source).unwrap();
    }

    #[test]
    fn fast_kv_restarts_for_new_slow_step_without_synchronizing() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        let plan = ArStreamPlan::from_graph(&factory.fast_decode(0).unwrap()).unwrap();
        let first = plan
            .phases
            .into_iter()
            .find_map(|phase| match phase {
                ArStreamPhase::KvWrites(write) => Some(write),
                _ => None,
            })
            .unwrap();
        let source = Rc::new(RefCell::new(vec![
            4u16;
            first.key.geometry.rows
                * first.key.geometry.row_bytes
                / 2
        ]));
        let mut owner = KvDeviceOwner::new(MockDevice::default());
        owner.enqueue_pair(&first, &source, &source).unwrap();
        let original = owner.cache(&first.key.cache_name).unwrap().clone();
        assert_eq!(owner.active_len(&first.key.cache_name), Some(1));
        owner.reset_fast_prefix().unwrap();
        assert_eq!(owner.active_len(&first.key.cache_name), Some(0));
        assert_eq!(owner.device.syncs, 0);
        owner.enqueue_pair(&first, &source, &source).unwrap();
        assert!(Rc::ptr_eq(
            &original,
            owner.cache(&first.key.cache_name).unwrap()
        ));
        assert_eq!(owner.device.allocations, 2);
    }

    #[test]
    fn zero_position_is_required_after_reset() {
        let mut owner = KvDeviceOwner::new(MockDevice::default());
        let decode = first_barrier(false);
        let source = Rc::new(RefCell::new(vec![0u16; decode.key.geometry.rows * 8]));
        assert!(owner.enqueue_pair(&decode, &source, &source).is_err());
    }
}
