use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    sys::{self, CUDA_MEMCPY2D, CUmemorytype},
};
use fish_speech_core::{
    ar_stream_executor::ArStreamDevice,
    error::{FishError, Result},
    graph_ir::TensorType,
    kv_device::KvWrite2d,
    kv_device_owner::KvStreamDevice,
    trtx_blueprint::{Binding, TrtBlueprint},
};

use crate::{
    device_memory::{DeviceMemoryBudget, DeviceMemoryLease, DeviceMemoryUsage},
    scratch_pool::BoundedScratchPool,
};

struct CachedCudaScratch {
    data: CudaSlice<u8>,
    reservation: DeviceMemoryLease,
    completion_stream: Arc<CudaStream>,
}

#[derive(Clone)]
pub struct CudaScratchPool {
    ordinal: usize,
    cached: Arc<Mutex<BoundedScratchPool<CachedCudaScratch>>>,
}

impl CudaScratchPool {
    pub fn new(ordinal: usize, capacity_bytes: usize) -> Self {
        Self {
            ordinal,
            cached: Arc::new(Mutex::new(BoundedScratchPool::new(capacity_bytes))),
        }
    }

    pub fn cached_bytes(&self) -> usize {
        self.cached
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cached_bytes()
    }

    pub fn clear(&self) -> Result<()> {
        let entries = self
            .cached
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain();
        let mut failed = false;
        for entry in entries {
            if entry.completion_stream.synchronize().is_err() {
                failed = true;
                std::mem::forget(entry);
            }
        }
        if failed {
            Err(FishError::InvalidTransition(
                "CUDA scratch retirement failed",
            ))
        } else {
            Ok(())
        }
    }

    fn take(
        &self,
        bytes: usize,
        receiving_stream: &Arc<CudaStream>,
    ) -> Result<Option<CachedCudaScratch>> {
        let old = self
            .cached
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take(bytes);
        if old.as_ref().is_some_and(|cached| {
            !Arc::ptr_eq(&cached.completion_stream, receiving_stream)
                && cached.completion_stream.synchronize().is_err()
        }) {
            std::mem::forget(old);
            return Err(FishError::InvalidTransition(
                "CUDA scratch reuse synchronization failed",
            ));
        }
        Ok(old)
    }

    fn offer(&self, entry: CachedCudaScratch) {
        let bytes = entry.data.num_bytes();
        let (_, evicted) = self
            .cached
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(bytes, entry);
        for old in evicted {
            if old.completion_stream.synchronize().is_err() {
                std::mem::forget(old);
            }
        }
    }
}

pub struct CudaWorkspace {
    data: CudaSlice<u8>,
    _reservation: DeviceMemoryLease,
}

impl CudaWorkspace {
    pub fn allocate(
        stream: &Arc<CudaStream>,
        memory: &DeviceMemoryBudget,
        bytes: usize,
    ) -> Result<Self> {
        let reservation = memory.reserve(bytes).map_err(|_| {
            FishError::InvalidTransition("TensorRT context memory budget exhausted")
        })?;
        let data = unsafe { stream.alloc::<u8>(bytes) }.map_err(cuda_error)?;
        Ok(Self {
            data,
            _reservation: reservation,
        })
    }

    pub fn region(&self, stream: &CudaStream) -> (usize, usize) {
        (
            self.data.device_ptr(stream).0 as usize,
            self.data.num_bytes(),
        )
    }
}

struct ScratchRecycle {
    pool: CudaScratchPool,
    stream: Arc<CudaStream>,
}

struct CudaAllocation {
    data: Mutex<Option<CudaSlice<u8>>>,
    reservation: Option<DeviceMemoryLease>,
    recycle: Option<ScratchRecycle>,
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        if let Some(recycle) = self.recycle.take() {
            let data = self
                .data
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            if let (Some(data), Some(reservation)) = (data, self.reservation.take()) {
                recycle.pool.offer(CachedCudaScratch {
                    data,
                    reservation,
                    completion_stream: recycle.stream,
                });
            }
        }
    }
}

#[derive(Clone)]
pub struct CudaBuffer(Arc<CudaAllocation>);

impl CudaBuffer {
    fn from_allocation(data: CudaSlice<u8>, reservation: DeviceMemoryLease) -> Self {
        Self::from_scratch(data, reservation, None)
    }

    fn from_scratch(
        data: CudaSlice<u8>,
        reservation: DeviceMemoryLease,
        recycle: Option<ScratchRecycle>,
    ) -> Self {
        Self(Arc::new(CudaAllocation {
            data: Mutex::new(Some(data)),
            reservation: Some(reservation),
            recycle,
        }))
    }

    pub fn bytes(&self) -> usize {
        self.0
            .data
            .lock()
            .unwrap()
            .as_ref()
            .expect("live CUDA buffer")
            .num_bytes()
    }

    pub fn address(&self, stream: &CudaStream) -> u64 {
        let guard = self.0.data.lock().unwrap();
        let (address, _record) = guard.as_ref().expect("live CUDA buffer").device_ptr(stream);
        address
    }
}

fn invalid_geometry() -> FishError {
    FishError::InvalidShape("CUDA KV 2D copy geometry invalid")
}

pub fn validate_kv_copy_geometry(
    geometry: KvWrite2d,
    dst_bytes: usize,
    src_bytes: usize,
) -> Result<()> {
    if geometry.rows == 0
        || geometry.frames == 0
        || geometry.row_bytes == 0
        || geometry.row_bytes != geometry.source_pitch_bytes
        || geometry.total_cache_bytes != dst_bytes
        || geometry.rows.checked_mul(geometry.source_pitch_bytes) != Some(src_bytes)
        || geometry
            .destination_offset_bytes
            .checked_add(geometry.row_bytes)
            .is_none_or(|end| end > geometry.destination_pitch_bytes)
        || geometry.rows.checked_mul(geometry.destination_pitch_bytes) != Some(dst_bytes)
        || geometry.destination_offset_bytes % 2 != 0
        || geometry.row_bytes % 2 != 0
    {
        return Err(invalid_geometry());
    }
    Ok(())
}

fn cuda_error(_: cudarc::driver::DriverError) -> FishError {
    FishError::InvalidTransition("CUDA Driver API operation failed")
}

pub struct CudaArDevice<F> {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    dispatch: F,
    memory: DeviceMemoryBudget,
    scratch_pool: Option<CudaScratchPool>,
}

impl<F> CudaArDevice<F> {
    pub fn new(device_ordinal: usize, dispatch: F) -> Result<Self> {
        Self::with_memory_budget(device_ordinal, usize::MAX, dispatch)
    }

    pub fn with_memory_budget(
        device_ordinal: usize,
        limit_bytes: usize,
        dispatch: F,
    ) -> Result<Self> {
        let memory = DeviceMemoryBudget::new(limit_bytes)
            .map_err(|_| FishError::InvalidConfig("CUDA memory budget must be positive"))?;
        Self::with_shared_budget(device_ordinal, memory, dispatch)
    }

    pub fn with_shared_budget(
        device_ordinal: usize,
        memory: DeviceMemoryBudget,
        dispatch: F,
    ) -> Result<Self> {
        let context = CudaContext::new(device_ordinal).map_err(cuda_error)?;
        Self::with_context(context, memory, dispatch)
    }

    pub fn with_context(
        context: Arc<CudaContext>,
        memory: DeviceMemoryBudget,
        dispatch: F,
    ) -> Result<Self> {
        context.bind_to_thread().map_err(cuda_error)?;
        let stream = context.new_stream().map_err(cuda_error)?;
        Self::with_context_stream(context, stream, memory, dispatch)
    }

    pub fn with_context_stream(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        memory: DeviceMemoryBudget,
        dispatch: F,
    ) -> Result<Self> {
        context.bind_to_thread().map_err(cuda_error)?;
        Ok(Self {
            context,
            stream,
            dispatch,
            memory,
            scratch_pool: None,
        })
    }

    pub fn with_scratch_pool(mut self, pool: CudaScratchPool) -> Result<Self> {
        if pool.ordinal != self.context.ordinal() {
            return Err(FishError::InvalidConfig(
                "CUDA scratch pool device mismatch",
            ));
        }
        self.scratch_pool = Some(pool);
        Ok(self)
    }

    pub fn upload_tensor_bytes(&mut self, ty: &TensorType, bytes: &[u8]) -> Result<CudaBuffer> {
        let required = fish_speech_core::ar_stream_executor::tensor_bytes(ty)?;
        if required == 0 || required != bytes.len() {
            return Err(FishError::InvalidShape(
                "CUDA upload tensor byte count mismatch",
            ));
        }
        let reservation = self
            .memory
            .reserve(required)
            .map_err(|_| FishError::InvalidTransition("CUDA device memory budget exhausted"))?;
        let allocation = self.stream.clone_htod(bytes).map_err(cuda_error)?;
        Ok(CudaBuffer::from_allocation(allocation, reservation))
    }

    pub fn download_tensor_bytes(&mut self, buffer: &CudaBuffer) -> Result<Vec<u8>> {
        let guard = buffer.0.data.lock().unwrap();
        let downloaded = self
            .stream
            .clone_dtoh(guard.as_ref().expect("live CUDA buffer"))
            .map_err(cuda_error);
        self.stream.synchronize().map_err(cuda_error)?;
        downloaded
    }

    pub fn memory_usage(&self) -> DeviceMemoryUsage {
        self.memory.usage()
    }

    pub fn memory_budget(&self) -> DeviceMemoryBudget {
        self.memory.clone()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}

impl<F> KvStreamDevice for CudaArDevice<F> {
    type Buffer = CudaBuffer;

    fn allocate_bf16(&mut self, bytes: usize) -> Result<Self::Buffer> {
        if bytes == 0 || bytes % 2 != 0 {
            return Err(invalid_geometry());
        }
        let reservation = self
            .memory
            .reserve(bytes)
            .map_err(|_| FishError::InvalidTransition("CUDA device memory budget exhausted"))?;
        let allocation = self.stream.alloc_zeros::<u8>(bytes).map_err(cuda_error)?;
        Ok(CudaBuffer::from_allocation(allocation, reservation))
    }

    fn copy_2d_async(
        &mut self,
        destination: &Self::Buffer,
        source: &Self::Buffer,
        geometry: KvWrite2d,
    ) -> Result<()> {
        if Arc::ptr_eq(&destination.0, &source.0) {
            return Err(FishError::InvalidShape(
                "CUDA KV source aliases destination",
            ));
        }
        validate_kv_copy_geometry(geometry, destination.bytes(), source.bytes())?;
        self.context.bind_to_thread().map_err(cuda_error)?;
        let mut dst_guard = destination.0.data.lock().unwrap();
        let src_guard = source.0.data.lock().unwrap();
        let (dst, _dst_record) = dst_guard
            .as_mut()
            .expect("live CUDA buffer")
            .device_ptr_mut(&self.stream);
        let (src, _src_record) = src_guard
            .as_ref()
            .expect("live CUDA buffer")
            .device_ptr(&self.stream);
        let copy = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: src,
            srcArray: std::ptr::null_mut(),
            srcPitch: geometry.source_pitch_bytes,
            dstXInBytes: geometry.destination_offset_bytes,
            dstY: 0,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst,
            dstArray: std::ptr::null_mut(),
            dstPitch: geometry.destination_pitch_bytes,
            WidthInBytes: geometry.row_bytes,
            Height: geometry.rows,
        };
        unsafe { sys::cuMemcpy2DAsync_v2(&copy, self.stream.cu_stream()) }
            .result()
            .map_err(cuda_error)
    }

    fn synchronize(&mut self) -> Result<()> {
        self.stream.synchronize().map_err(cuda_error)
    }
}

impl<F> ArStreamDevice<Arc<TrtBlueprint>> for CudaArDevice<F>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    fn allocate_tensor(&mut self, ty: &TensorType) -> Result<Self::Buffer> {
        let bytes = fish_speech_core::ar_stream_executor::tensor_bytes(ty)?;
        if bytes == 0 {
            return Err(FishError::InvalidShape("CUDA output allocation is empty"));
        }
        if let Some(pool) = &self.scratch_pool {
            if let Some(cached) = pool.take(bytes, &self.stream)? {
                return Ok(CudaBuffer::from_scratch(
                    cached.data,
                    cached.reservation,
                    Some(ScratchRecycle {
                        pool: pool.clone(),
                        stream: Arc::clone(&self.stream),
                    }),
                ));
            }
        }
        let reservation = match self.memory.reserve(bytes) {
            Ok(lease) => lease,
            Err(_) => {
                if let Some(pool) = &self.scratch_pool {
                    pool.clear()?;
                }
                self.memory.reserve(bytes).map_err(|_| {
                    FishError::InvalidTransition("CUDA device memory budget exhausted")
                })?
            }
        };
        let allocation = unsafe { self.stream.alloc::<u8>(bytes) }.map_err(cuda_error)?;
        Ok(CudaBuffer::from_scratch(
            allocation,
            reservation,
            self.scratch_pool.as_ref().map(|pool| ScratchRecycle {
                pool: pool.clone(),
                stream: Arc::clone(&self.stream),
            }),
        ))
    }

    fn download_tensor_blocking(&mut self, buffer: &CudaBuffer) -> Result<Vec<u8>> {
        self.download_tensor_bytes(buffer)
    }

    fn enqueue_network(
        &mut self,
        plan: &Arc<TrtBlueprint>,
        buffers: &BTreeMap<Binding, Self::Buffer>,
    ) -> Result<()> {
        self.context.bind_to_thread().map_err(cuda_error)?;
        (self.dispatch)(plan, buffers, &self.stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fish_speech_core::kv_device::Bf16KvLayout;

    #[test]
    fn copies_one_prefix_slice_without_treating_capacity_as_length() {
        let geometry = Bf16KvLayout {
            batch: 2,
            kv_heads: 3,
            capacity: 32,
            head_dim: 64,
        }
        .write_plan(17, 1)
        .unwrap();
        assert!(
            validate_kv_copy_geometry(
                geometry,
                geometry.total_cache_bytes,
                geometry.rows * geometry.source_pitch_bytes
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_out_of_bounds_and_malformed_copy_geometry() {
        let mut geometry = Bf16KvLayout {
            batch: 1,
            kv_heads: 2,
            capacity: 8,
            head_dim: 4,
        }
        .write_plan(5, 2)
        .unwrap();
        assert!(validate_kv_copy_geometry(geometry, geometry.total_cache_bytes, 8).is_err());
        geometry.destination_offset_bytes = usize::MAX;
        assert!(
            validate_kv_copy_geometry(
                geometry,
                geometry.total_cache_bytes,
                geometry.rows * geometry.source_pitch_bytes
            )
            .is_err()
        );
    }
}
