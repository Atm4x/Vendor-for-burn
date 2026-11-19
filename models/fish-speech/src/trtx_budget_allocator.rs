use std::{
    collections::BTreeMap,
    ffi::c_void,
    ptr,
    sync::{Arc, Mutex},
};

use cudarc::driver::{
    CudaContext,
    sys::{self, CUdeviceptr, CUstream},
};
use trtx::interfaces::AllocateGpu;

use crate::{
    cuda_stream::CudaScratchPool,
    device_memory::{DeviceMemoryBudget, DeviceMemoryLease},
};

struct TrackedAllocation {
    base: CUdeviceptr,
    requested: usize,
    _reservation: DeviceMemoryLease,
}

pub struct BudgetedTrtAllocator {
    context: Arc<CudaContext>,
    budget: DeviceMemoryBudget,
    scratch_pool: CudaScratchPool,
    live: Mutex<BTreeMap<usize, TrackedAllocation>>,
}

fn allocation_layout(bytes: u64, alignment: u64) -> Option<(usize, usize)> {
    let size = usize::try_from(bytes).ok()?.max(1);
    let alignment = usize::try_from(alignment).ok()?.max(1);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some((size.checked_add(alignment - 1)?, alignment))
}

impl BudgetedTrtAllocator {
    pub fn new(
        context: Arc<CudaContext>,
        budget: DeviceMemoryBudget,
        scratch_pool: CudaScratchPool,
    ) -> Self {
        Self {
            context,
            budget,
            scratch_pool,
            live: Mutex::new(BTreeMap::new()),
        }
    }

    fn allocate(&self, bytes: u64, alignment: u64) -> *mut c_void {
        let Some((reserved, align)) = allocation_layout(bytes, alignment) else {
            return ptr::null_mut();
        };
        let lease = match self.budget.reserve(reserved) {
            Ok(lease) => lease,
            Err(_) => {
                if self.scratch_pool.clear().is_err() {
                    return ptr::null_mut();
                }
                let Ok(lease) = self.budget.reserve(reserved) else {
                    return ptr::null_mut();
                };
                lease
            }
        };
        if self.context.bind_to_thread().is_err() {
            return ptr::null_mut();
        }
        let mut base = 0u64;
        if unsafe { sys::cuMemAlloc_v2(&mut base, reserved) }
            .result()
            .is_err()
        {
            return ptr::null_mut();
        }
        let Some(aligned) = (base as usize)
            .checked_add(align - 1)
            .map(|n| n & !(align - 1))
        else {
            unsafe { sys::cuMemFree_v2(base) }.result().ok();
            return ptr::null_mut();
        };
        self.live.lock().unwrap_or_else(|e| e.into_inner()).insert(
            aligned,
            TrackedAllocation {
                base,
                requested: usize::try_from(bytes).unwrap_or(0),
                _reservation: lease,
            },
        );
        aligned as *mut c_void
    }

    fn deallocate(&self, memory: *mut c_void, cuda_stream: *mut c_void) -> bool {
        if memory.is_null() {
            return true;
        }
        let address = memory as usize;
        let Some(allocation) = self
            .live
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&address)
        else {
            return false;
        };
        let result = if self.context.bind_to_thread().is_err() {
            false
        } else if cuda_stream.is_null() {
            unsafe { sys::cuCtxSynchronize() }.result().is_ok()
        } else {
            unsafe { sys::cuStreamSynchronize(cuda_stream as CUstream) }
                .result()
                .is_ok()
        };
        if result
            && unsafe { sys::cuMemFree_v2(allocation.base) }
                .result()
                .is_ok()
        {
            true
        } else {
            self.live
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(address, allocation);
            false
        }
    }
}

impl AllocateGpu for BudgetedTrtAllocator {
    unsafe fn allocate_async(
        &self,
        size: u64,
        alignment: u64,
        _flags: u32,
        _cuda_stream: *mut c_void,
    ) -> *mut c_void {
        self.allocate(size, alignment)
    }

    unsafe fn reallocate(&self, memory: *mut c_void, alignment: u64, new_size: u64) -> *mut c_void {
        if memory.is_null() {
            return self.allocate(new_size, alignment);
        }
        if self.context.bind_to_thread().is_err()
            || unsafe { sys::cuCtxSynchronize() }.result().is_err()
        {
            return ptr::null_mut();
        }
        let old_size = match self
            .live
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(memory as usize))
        {
            Some(entry) => entry.requested,
            None => return ptr::null_mut(),
        };
        let new_ptr = self.allocate(new_size, alignment);
        if new_ptr.is_null() {
            return ptr::null_mut();
        }
        let copied = old_size.min(usize::try_from(new_size).unwrap_or(usize::MAX));
        if copied != 0
            && unsafe {
                sys::cuMemcpyDtoD_v2(new_ptr as CUdeviceptr, memory as CUdeviceptr, copied)
            }
            .result()
            .is_err()
        {
            self.deallocate(new_ptr, ptr::null_mut());
            return ptr::null_mut();
        }
        if !self.deallocate(memory, ptr::null_mut()) {
            self.deallocate(new_ptr, ptr::null_mut());
            return ptr::null_mut();
        }
        new_ptr
    }

    unsafe fn deallocate_async(&self, data: *mut c_void, cuda_stream: *mut c_void) -> bool {
        self.deallocate(data, cuda_stream)
    }
}

impl Drop for BudgetedTrtAllocator {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_ok()
            && unsafe { sys::cuCtxSynchronize() }.result().is_ok()
        {
            let entries = std::mem::take(self.live.get_mut().unwrap_or_else(|e| e.into_inner()));
            for (_, allocation) in entries {
                if unsafe { sys::cuMemFree_v2(allocation.base) }
                    .result()
                    .is_err()
                {
                    std::mem::forget(allocation);
                }
            }
        } else {
            let entries = std::mem::take(self.live.get_mut().unwrap_or_else(|e| e.into_inner()));
            std::mem::forget(entries);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_aligned_allocations_and_rejects_invalid_geometry() {
        assert_eq!(allocation_layout(1024, 256), Some((1279, 256)));
        assert_eq!(allocation_layout(1024, 0), Some((1024, 1)));
        assert_eq!(allocation_layout(100, 3), None);
        assert_eq!(allocation_layout(u64::MAX, u64::MAX), None);
    }
}
