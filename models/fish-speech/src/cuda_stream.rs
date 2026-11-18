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

use crate::device_memory::{DeviceMemoryBudget, DeviceMemoryLease, DeviceMemoryUsage};

#[derive(Debug)]
struct CudaAllocation {
    data: Mutex<CudaSlice<u8>>,
    _reservation: DeviceMemoryLease,
}

#[derive(Debug, Clone)]
pub struct CudaBuffer(Arc<CudaAllocation>);

impl CudaBuffer {
    fn from_allocation(data: CudaSlice<u8>, reservation: DeviceMemoryLease) -> Self {
        Self(Arc::new(CudaAllocation {
            data: Mutex::new(data),
            _reservation: reservation,
        }))
    }
}

impl CudaBuffer {
    pub fn bytes(&self) -> usize {
        self.0.data.lock().unwrap().num_bytes()
    }

    pub fn address(&self, stream: &CudaStream) -> u64 {
        let guard = self.0.data.lock().unwrap();
        let (address, _record) = guard.device_ptr(stream);
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
        let stream = context.new_stream().map_err(cuda_error)?;
        Ok(Self {
            context,
            stream,
            dispatch,
            memory,
        })
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
        let downloaded = self.stream.clone_dtoh(&*guard).map_err(cuda_error);
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
        let (dst, _dst_record) = dst_guard.device_ptr_mut(&self.stream);
        let (src, _src_record) = src_guard.device_ptr(&self.stream);
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

impl<F> ArStreamDevice for CudaArDevice<F>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    fn allocate_tensor(&mut self, ty: &TensorType) -> Result<Self::Buffer> {
        let bytes = fish_speech_core::ar_stream_executor::tensor_bytes(ty)?;
        if bytes == 0 {
            return Err(FishError::InvalidShape("CUDA output allocation is empty"));
        }
        let reservation = self
            .memory
            .reserve(bytes)
            .map_err(|_| FishError::InvalidTransition("CUDA device memory budget exhausted"))?;
        let allocation = unsafe { self.stream.alloc::<u8>(bytes) }.map_err(cuda_error)?;
        Ok(CudaBuffer::from_allocation(allocation, reservation))
    }

    fn enqueue_network(
        &mut self,
        plan: &TrtBlueprint,
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
