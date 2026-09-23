use crate::context::{OpenClEngine, OpenClError};
use crate::device::{DeviceKind, OpenClDeviceInfo};
use cl3::device::{
    CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
    CL_DEVICE_SVM_FINE_GRAIN_SYSTEM,
};
use cl3::memory::{
    svm_alloc, svm_free, CL_MEM_READ_WRITE as CL3_MEM_READ_WRITE, CL_MEM_SVM_FINE_GRAIN_BUFFER,
};
use libc::c_void;
use opencl3::memory::{
    Buffer, CL_MAP_READ, CL_MAP_WRITE, CL_MEM_ALLOC_HOST_PTR, CL_MEM_READ_WRITE,
};
use opencl3::svm::SvmVec;
use opencl3::types::{cl_context, cl_mem, cl_uchar, cl_uint, CL_BLOCKING, CL_NON_BLOCKING};
use std::alloc::{self, Layout};
use std::ptr;
use tracing::{debug, info};

/// How host↔device weight traffic should move for the selected accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferPath {
    /// Discrete GPU: pinned host pages + async DMA into device buffers.
    PinnedDma,
    /// APU / unified memory: Shared Virtual Memory zero-copy.
    SvmZeroCopy,
    /// CPU-only or OpenCL-on-CPU: plain host RAM (no device upload).
    HostRam,
}

/// Choose the Phase-1 streaming transfer strategy for a discovered device.
pub fn select_transfer_path(info: &OpenClDeviceInfo) -> TransferPath {
    match info.device_kind {
        // Unified-memory devices (APU/superchip) are first-class zero-copy.
        DeviceKind::Apu if info.supports_svm => TransferPath::SvmZeroCopy,
        DeviceKind::IntegratedGpu if info.supports_svm => TransferPath::SvmZeroCopy,
        DeviceKind::DiscreteGpu => TransferPath::PinnedDma,
        DeviceKind::CpuOpenCl => TransferPath::HostRam,
        // Accelerator/NPU via OpenCL: zero-copy when SVM is available.
        _ if info.supports_svm => TransferPath::SvmZeroCopy,
        _ => TransferPath::PinnedDma,
    }
}

/// A GPU-side device buffer that receives streamed layer weights.
pub struct DeviceLayerBuffer {
    pub cl_buffer: Buffer<cl_uchar>,
    pub size_bytes: usize,
}

impl DeviceLayerBuffer {
    /// Allocate a GPU device buffer of `size_bytes`.
    pub fn new(engine: &OpenClEngine, size_bytes: usize) -> Result<Self, OpenClError> {
        let cl_buffer = unsafe {
            Buffer::<cl_uchar>::create(
                &engine.context,
                CL_MEM_READ_WRITE,
                size_bytes,
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("Device buffer alloc failed: {e}")))?
        };
        debug!("Allocated device buffer: {} KB", size_bytes / 1024);
        Ok(Self {
            cl_buffer,
            size_bytes,
        })
    }
}

/// Host-side pinned memory (`CL_MEM_ALLOC_HOST_PTR`) for fast DMA to discrete GPUs.
pub struct PinnedHostBuffer {
    pub cl_buffer: Buffer<cl_uchar>,
    pub host_ptr: *mut u8,
    pub size_bytes: usize,
}

// SAFETY: Explicit ownership; callers must not alias concurrent mutable access.
unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    pub fn new(engine: &OpenClEngine, size_bytes: usize) -> Result<Self, OpenClError> {
        let cl_buffer = unsafe {
            Buffer::<cl_uchar>::create(
                &engine.context,
                CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                size_bytes,
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("Pinned buffer alloc failed: {e}")))?
        };

        let mut mapped: cl_mem = ptr::null_mut();
        unsafe {
            engine
                .queue
                .enqueue_map_buffer(
                    &cl_buffer,
                    CL_BLOCKING,
                    CL_MAP_WRITE | CL_MAP_READ,
                    0,
                    size_bytes,
                    &mut mapped,
                    &[],
                )
                .map_err(|e| OpenClError::ClError(format!("Buffer map failed: {e}")))?;
        }
        let host_ptr = mapped as *mut u8;

        info!("Allocated pinned host buffer: {} KB", size_bytes / 1024);
        Ok(Self {
            cl_buffer,
            host_ptr,
            size_bytes,
        })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.size_bytes) }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.host_ptr, self.size_bytes) }
    }

    /// Enqueue async DMA: pinned host → device buffer.
    pub fn enqueue_upload_to(
        &self,
        engine: &OpenClEngine,
        device_buf: &mut DeviceLayerBuffer,
    ) -> Result<opencl3::event::Event, OpenClError> {
        let bytes = self.as_slice();
        unsafe {
            engine
                .queue
                .enqueue_write_buffer(
                    &mut device_buf.cl_buffer,
                    CL_NON_BLOCKING,
                    0,
                    bytes,
                    &[],
                )
                .map_err(|e| OpenClError::ClError(format!("Enqueue upload failed: {e}")))
        }
    }
}

/// SVM-backed layer buffer for APU zero-copy streaming.
///
/// Host I/O writes directly into the mapped SVM region; the GPU kernel can
/// consume the same pointer without an explicit PCIe copy on unified memory.
pub struct SvmLayerBuffer<'a> {
    pub svm: SvmVec<'a, cl_uchar>,
    pub size_bytes: usize,
    fine_grained: bool,
}

impl<'a> SvmLayerBuffer<'a> {
    pub fn new(engine: &'a OpenClEngine, size_bytes: usize) -> Result<Self, OpenClError> {
        let caps = engine.context.get_svm_mem_capability();
        if caps == 0 {
            return Err(OpenClError::ClError(
                "Device reports no SVM capability".into(),
            ));
        }

        let mut svm = SvmVec::<cl_uchar>::allocate(&engine.context, size_bytes)
            .map_err(|e| OpenClError::ClError(format!("SVM alloc failed: {e}")))?;
        let fine_grained = svm.is_fine_grained();

        if !fine_grained {
            unsafe {
                engine
                    .queue
                    .enqueue_svm_map(
                        CL_BLOCKING,
                        CL_MAP_WRITE | CL_MAP_READ,
                        &mut svm[..],
                        &[],
                    )
                    .map_err(|e| OpenClError::ClError(format!("SVM map failed: {e}")))?;
            }
        }

        info!(
            "Allocated SVM layer buffer: {} KB (fine_grained={})",
            size_bytes / 1024,
            fine_grained
        );
        Ok(Self {
            svm,
            size_bytes,
            fine_grained,
        })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.svm[..]
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.svm[..]
    }

    /// For coarse-grain SVM: unmap before kernel use, map again for host writes.
    pub fn prepare_for_device(&mut self, engine: &OpenClEngine) -> Result<(), OpenClError> {
        if self.fine_grained {
            return Ok(());
        }
        unsafe {
            let ev = engine
                .queue
                .enqueue_svm_unmap(&self.svm[..], &[])
                .map_err(|e| OpenClError::ClError(format!("SVM unmap failed: {e}")))?;
            ev.wait()
                .map_err(|e| OpenClError::ClError(format!("SVM unmap wait failed: {e}")))?;
        }
        Ok(())
    }

    pub fn prepare_for_host(&mut self, engine: &OpenClEngine) -> Result<(), OpenClError> {
        if self.fine_grained {
            return Ok(());
        }
        unsafe {
            engine
                .queue
                .enqueue_svm_map(
                    CL_BLOCKING,
                    CL_MAP_WRITE | CL_MAP_READ,
                    &mut self.svm[..],
                    &[],
                )
                .map_err(|e| OpenClError::ClError(format!("SVM remap failed: {e}")))?;
        }
        Ok(())
    }
}

/// Owned SVM allocation (no engine lifetime) for generate / long-lived scratch.
///
/// Expert guidance: prefer **coarse-grained** `clSVMAlloc` on the APU context so host
/// I/O maps the region, then unmaps for device/SVM kernel access (zero-copy on APU).
pub struct OwnedSvmBuffer {
    ptr: *mut u8,
    pub size_bytes: usize,
    context: cl_context,
    fine_grain_system: bool,
    /// Coarse-grain buffer (needs explicit map/unmap).
    pub coarse: bool,
    mapped: bool,
}

// SAFETY: exclusive ownership; callers must not alias concurrent mutable access.
unsafe impl Send for OwnedSvmBuffer {}
unsafe impl Sync for OwnedSvmBuffer {}

impl OwnedSvmBuffer {
    pub fn new(engine: &OpenClEngine, size_bytes: usize) -> Result<Self, OpenClError> {
        let caps = engine.context.get_svm_mem_capability();
        if caps
            & (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
                | CL_DEVICE_SVM_FINE_GRAIN_BUFFER
                | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
            == 0
        {
            return Err(OpenClError::ClError(
                "Device reports no SVM capability".into(),
            ));
        }

        let fine_grain_system = caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM != 0;
        let has_coarse = caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER != 0;
        let fine_grain_buffer = caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0;

        let (ptr, coarse, mapped) = if fine_grain_system {
            let layout = Layout::from_size_align(size_bytes, 64)
                .map_err(|e| OpenClError::ClError(format!("SVM layout: {e}")))?;
            let p = unsafe { alloc::alloc(layout) };
            if p.is_null() {
                alloc::handle_alloc_error(layout);
            }
            (p, false, true)
        } else if has_coarse {
            // Prefer coarse-grain (expert: APU first-class + map/unmap protocol).
            let raw = unsafe {
                svm_alloc(
                    engine.context.get(),
                    CL3_MEM_READ_WRITE,
                    size_bytes,
                    64u32 as cl_uint,
                )
                .map_err(|e| OpenClError::ClError(format!("clSVMAlloc coarse failed: {e}")))?
            };
            let p = raw as *mut u8;
            let slice = unsafe { std::slice::from_raw_parts_mut(p, size_bytes) };
            unsafe {
                engine
                    .queue
                    .enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE | CL_MAP_READ, slice, &[])
                    .map_err(|e| OpenClError::ClError(format!("SVM map failed: {e}")))?;
            }
            (p, true, true)
        } else if fine_grain_buffer {
            let raw = unsafe {
                svm_alloc(
                    engine.context.get(),
                    CL_MEM_SVM_FINE_GRAIN_BUFFER | CL3_MEM_READ_WRITE,
                    size_bytes,
                    64u32 as cl_uint,
                )
                .map_err(|e| OpenClError::ClError(format!("clSVMAlloc fine failed: {e}")))?
            };
            (raw as *mut u8, false, true)
        } else {
            return Err(OpenClError::ClError("No usable SVM mode".into()));
        };

        info!(
            "Allocated owned SVM buffer: {} KB (coarse={}, system={})",
            size_bytes / 1024,
            coarse,
            fine_grain_system
        );
        // Retain the raw `cl_context`: this buffer can outlive the engine that
        // created it (e.g. `GenerationSession`'s fields drop before its `orch`
        // mutex). Without an explicit retain, `Drop` could call `clSVMFree` /
        // `clReleaseContext` on a destroyed context.
        unsafe {
            cl3::context::retain_context(engine.context.get())
                .map_err(|e| OpenClError::ClError(format!("retain context: {e}")))?;
        }
        Ok(Self {
            ptr,
            size_bytes,
            context: engine.context.get(),
            fine_grain_system,
            coarse,
            mapped,
        })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(
            self.mapped || !self.coarse,
            "OwnedSvmBuffer::as_mut_slice on a coarse SVM region that is not host-mapped"
        );
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_bytes) }
    }

    pub fn as_slice(&self) -> &[u8] {
        assert!(
            self.mapped || !self.coarse,
            "OwnedSvmBuffer::as_slice on a coarse SVM region that is not host-mapped"
        );
        unsafe { std::slice::from_raw_parts(self.ptr, self.size_bytes) }
    }

    pub fn prepare_for_host(&mut self, engine: &OpenClEngine) -> Result<(), OpenClError> {
        if !self.coarse || self.mapped {
            return Ok(());
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_bytes) };
        unsafe {
            engine
                .queue
                .enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE | CL_MAP_READ, slice, &[])
                .map_err(|e| OpenClError::ClError(format!("SVM map(host): {e}")))?;
        }
        self.mapped = true;
        Ok(())
    }

    pub fn prepare_for_device(&mut self, engine: &OpenClEngine) -> Result<(), OpenClError> {
        if !self.coarse || !self.mapped {
            return Ok(());
        }
        let slice = unsafe { std::slice::from_raw_parts(self.ptr, self.size_bytes) };
        unsafe {
            let ev = engine
                .queue
                .enqueue_svm_unmap(slice, &[])
                .map_err(|e| OpenClError::ClError(format!("SVM unmap: {e}")))?;
            ev.wait()
                .map_err(|e| OpenClError::ClError(format!("SVM unmap wait: {e}")))?;
        }
        self.mapped = false;
        Ok(())
    }
}

impl Drop for OwnedSvmBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if self.fine_grain_system {
            let layout = Layout::from_size_align(self.size_bytes, 64)
                .unwrap_or_else(|_| Layout::from_size_align(1, 1).unwrap());
            unsafe { alloc::dealloc(self.ptr, layout) };
        } else {
            unsafe {
                let _ = svm_free(self.context, self.ptr as *mut c_void);
            }
        }
        // Balance the `retain_context` taken in `OwnedSvmBuffer::new`.
        unsafe {
            let _ = cl3::context::release_context(self.context);
        }
        self.ptr = ptr::null_mut();
    }
}

/// SVM ping-pong for scoped sessions that already borrow an [`OpenClEngine`].
pub struct SvmScratch<'a> {
    pub slots: [SvmLayerBuffer<'a>; 2],
}

impl<'a> SvmScratch<'a> {
    pub fn allocate(engine: &'a OpenClEngine, layer_bytes: usize) -> Result<Self, OpenClError> {
        Ok(Self {
            slots: [
                SvmLayerBuffer::new(engine, layer_bytes)?,
                SvmLayerBuffer::new(engine, layer_bytes)?,
            ],
        })
    }

    pub fn host_slot_mut(&mut self, idx: usize) -> &mut [u8] {
        self.slots[idx % 2].as_mut_slice()
    }

    pub fn ingest_layer(
        &mut self,
        engine: &OpenClEngine,
        idx: usize,
        src: &[u8],
    ) -> Result<(), OpenClError> {
        let slot = idx % 2;
        self.slots[slot].prepare_for_host(engine)?;
        let dst = self.slots[slot].as_mut_slice();
        let n = src.len().min(dst.len());
        dst[..n].copy_from_slice(&src[..n]);
        self.slots[slot].prepare_for_device(engine)?;
        Ok(())
    }
}
