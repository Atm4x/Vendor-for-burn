//! Explicit CUDA Runtime transport for the TensorRT-RTX F5 executor.
//! Host uploads use owned pinned staging and the same CUDA stream as TRT inference.

use std::{
    ffi::{OsStr, c_void},
    ptr,
};

use libloading::Library;

use super::{trtx::DeviceBuffer, trtx_session::CudaTransport};
use crate::{F5Error, Result};

type CudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type CudaFree = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type CudaFreeHost = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, *mut c_void) -> i32;
type CudaStreamCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
type CudaStreamDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaStreamSync = unsafe extern "C" fn(*mut c_void) -> i32;

pub struct CudaRuntimeIo {
    _library: Library,
    malloc: CudaMalloc,
    free: CudaFree,
    host_alloc: CudaHostAlloc,
    host_free: CudaFreeHost,
    memcpy_async: CudaMemcpyAsync,
    stream_destroy: CudaStreamDestroy,
    stream_sync: CudaStreamSync,
    stream: *mut c_void,
    pending_uploads: Vec<*mut c_void>,
}

impl CudaRuntimeIo {
    /// The supplied path must name the exact CUDA Runtime shared library used for this session.
    pub fn open(path: impl AsRef<OsStr>) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .map_err(|e| F5Error::ModelExecution(format!("CUDA runtime library: {e}")))?;
        unsafe {
            let symbol = |name: &'static [u8]| -> Result<*const ()> {
                Ok(*library
                    .get::<*const ()>(name)
                    .map_err(|e| F5Error::ModelExecution(format!("CUDA runtime symbol: {e}")))?)
            };
            let malloc: CudaMalloc = std::mem::transmute(symbol(b"cudaMalloc\0")?);
            let free: CudaFree = std::mem::transmute(symbol(b"cudaFree\0")?);
            let host_alloc: CudaHostAlloc = std::mem::transmute(symbol(b"cudaHostAlloc\0")?);
            let host_free: CudaFreeHost = std::mem::transmute(symbol(b"cudaFreeHost\0")?);
            let memcpy_async: CudaMemcpyAsync = std::mem::transmute(symbol(b"cudaMemcpyAsync\0")?);
            let create: CudaStreamCreate =
                std::mem::transmute(symbol(b"cudaStreamCreateWithFlags\0")?);
            let stream_destroy: CudaStreamDestroy =
                std::mem::transmute(symbol(b"cudaStreamDestroy\0")?);
            let stream_sync: CudaStreamSync =
                std::mem::transmute(symbol(b"cudaStreamSynchronize\0")?);
            let mut stream = ptr::null_mut();
            check_cuda(create(&mut stream, 1), "cudaStreamCreateWithFlags")?;
            if stream.is_null() {
                return Err(F5Error::ModelExecution(
                    "CUDA returned a null stream".into(),
                ));
            }
            Ok(Self {
                _library: library,
                malloc,
                free,
                host_alloc,
                host_free,
                memcpy_async,
                stream_destroy,
                stream_sync,
                stream,
                pending_uploads: Vec::new(),
            })
        }
    }

    pub fn load_tensorrt_rtx(path: impl AsRef<std::path::Path>) -> Result<()> {
        trtx::dynamically_load_tensorrt(Some(path.as_ref().to_string_lossy().into_owned()))
            .map_err(|e| F5Error::ModelExecution(format!("TensorRT-RTX library: {e}")))
    }

    fn clear_completed_uploads(&mut self) {
        for memory in self.pending_uploads.drain(..) {
            unsafe {
                (self.host_free)(memory);
            }
        }
    }
}

fn check_cuda(code: i32, operation: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(F5Error::ModelExecution(format!(
            "{operation}: CUDA error {code}"
        )))
    }
}

impl CudaTransport for CudaRuntimeIo {
    fn allocate(&mut self, bytes: usize) -> Result<DeviceBuffer> {
        if bytes == 0 {
            return Err(F5Error::InvalidShape("Cannot allocate zero CUDA bytes"));
        }
        let mut pointer = ptr::null_mut();
        unsafe {
            check_cuda((self.malloc)(&mut pointer, bytes), "cudaMalloc")?;
        }
        if pointer.is_null() {
            return Err(F5Error::ModelExecution(
                "CUDA allocation returned null".into(),
            ));
        }
        Ok(DeviceBuffer {
            address: pointer as usize,
            bytes,
        })
    }

    fn free(&mut self, buffer: DeviceBuffer) -> Result<()> {
        if buffer.address == 0 {
            return Err(F5Error::InvalidShape("Cannot free null CUDA buffer"));
        }
        unsafe { check_cuda((self.free)(buffer.address as *mut c_void), "cudaFree") }
    }

    fn upload(&mut self, buffer: DeviceBuffer, contents: &[u8]) -> Result<()> {
        buffer.validate(contents.len())?;
        if contents.is_empty() {
            return Ok(());
        }
        let mut pinned = ptr::null_mut();
        unsafe {
            check_cuda(
                (self.host_alloc)(&mut pinned, contents.len(), 0),
                "cudaHostAlloc",
            )?;
            if pinned.is_null() {
                return Err(F5Error::ModelExecution(
                    "Null CUDA pinned host buffer".into(),
                ));
            }
            ptr::copy_nonoverlapping(contents.as_ptr(), pinned.cast::<u8>(), contents.len());
            self.pending_uploads.push(pinned);
            check_cuda(
                (self.memcpy_async)(
                    buffer.address as *mut c_void,
                    pinned,
                    contents.len(),
                    1,
                    self.stream,
                ),
                "cudaMemcpyAsync H2D",
            )
        }
    }

    fn download(&mut self, buffer: DeviceBuffer, contents: &mut [u8]) -> Result<()> {
        buffer.validate(contents.len())?;
        if contents.is_empty() {
            return Ok(());
        }
        unsafe {
            check_cuda(
                (self.memcpy_async)(
                    contents.as_mut_ptr().cast::<c_void>(),
                    buffer.address as *const c_void,
                    contents.len(),
                    2,
                    self.stream,
                ),
                "cudaMemcpyAsync D2H",
            )?;
        }
        self.synchronize()
    }

    fn synchronize(&mut self) -> Result<()> {
        let result =
            unsafe { check_cuda((self.stream_sync)(self.stream), "cudaStreamSynchronize") };
        if result.is_ok() {
            self.clear_completed_uploads();
        }
        result
    }

    fn stream(&self) -> *mut c_void {
        self.stream
    }
}

impl Drop for CudaRuntimeIo {
    fn drop(&mut self) {
        if self.synchronize().is_ok() {
            unsafe {
                (self.stream_destroy)(self.stream);
            }
        }
    }
}
