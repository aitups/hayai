//! Expert pattern: **Base buffer + parametric offset**.
//!
//! 1. Host ping-pong = `clSVMAlloc` on APU context (coarse-grain preferred).
//! 2. APU: SVM map/unmap + `set_arg_svm` → zero-copy GEMV with byte offsets.
//! 3. Each dGPU: VRAM mirror sized to the full layer-pack; DMA only the FFN
//!    slices that device will bind (`clEnqueueWriteBufferRect`).
//! 4. Kernels receive `layer_pack_base + weight_off`.

use crate::context::{OpenClEngine, OpenClError};
use crate::device::DeviceKind;
use crate::memory::{
    DeviceLayerBuffer, OwnedSvmBuffer, PinnedHostBuffer, TransferPath,
};
use crate::pool::OpenClDevicePool;
use hayai_io::AlignedBuffer;
use hayai_model::LayerPackLayout;
use libc::{c_void, size_t};
use opencl3::memory::Buffer;
use opencl3::types::{cl_device_id, cl_uchar, CL_NON_BLOCKING};
use tracing::{info, warn};

/// Per-dGPU VRAM mirror of the host layer-pack (one DMA target per ping-pong slot).
pub struct DgpuMirror {
    pub device_id: cl_device_id,
    pub slots: [DeviceLayerBuffer; 2],
}

enum HostBase {
    /// APU-first SVM (expert recommended).
    Svm {
        slots: [OwnedSvmBuffer; 2],
        owner_id: cl_device_id,
    },
    /// No SVM in pool: pinned host + primary device mirrors live in [`StreamingScratch::dgpu_mirrors`].
    Pinned {
        pinned: [PinnedHostBuffer; 2],
    },
    Host {
        slots: [AlignedBuffer; 2],
    },
}

/// Dual-slot hetero scratch for generate (owned; no engine lifetime).
pub struct StreamingScratch {
    host: HostBase,
    /// Discrete-GPU VRAM copies of each ping-pong slot (full pack layout; FFN slices DMA'd).
    pub dgpu_mirrors: Vec<DgpuMirror>,
    pub path: TransferPath,
    /// Resident mode: one base allocation of `resident_layers × resident_stride`
    /// holding every layer pack; layers are addressed by index (never `% 2`) and
    /// never re-read from disk nor re-DMA'd per token.
    pub resident: bool,
    pub resident_stride: usize,
    pub resident_layers: usize,
    /// Macro-chunk mode (1 < block_k < resident_layers): ping-pong slots hold
    /// `block_k × resident_stride` and are loaded once per block per token.
    pub block_k: usize,
    /// Which block (1-based) is currently staged in each ping-pong slot
    /// (macro-chunk only). `0` = slot empty / stale.
    pub block_staged: [usize; 2],
}

/// How a pool device should bind weights for Base+Offset GEMV.
pub enum WeightBind<'a> {
    /// APU / SVM owner: pass SVM pointer via `set_arg_svm`.
    Svm { ptr: *const u8 },
    /// dGPU: bind this device's mirrored `cl_mem` (already DMA'd).
    Device { buf: &'a Buffer<cl_uchar> },
}

impl StreamingScratch {
    /// Allocate scratch for the whole OpenCL pool (expert Base+Offset pattern).
    ///
    /// `k_chunk` = layers per ping-pong slot (macro-chunk). `1` keeps the classic
    /// per-layer ping-pong. Non-resident macro-chunk sizes each slot
    /// `k_chunk × layer_bytes`.
    pub fn allocate_for_pool(
        pool: &OpenClDevicePool,
        layer_bytes: usize,
        k_chunk: usize,
    ) -> Result<(TransferPath, Self), OpenClError> {
        let k_chunk = k_chunk.max(1);
        let slot_bytes = layer_bytes.saturating_mul(k_chunk);
        if pool.is_empty() {
            return Ok((
                TransferPath::HostRam,
                Self {
                    host: HostBase::Host {
                        slots: [
                            AlignedBuffer::zeroed(slot_bytes),
                            AlignedBuffer::zeroed(slot_bytes),
                        ],
                    },
                    dgpu_mirrors: Vec::new(),
                    path: TransferPath::HostRam,
                    resident: false,
                    resident_stride: layer_bytes,
                    resident_layers: 0,
                    block_k: k_chunk,
                    block_staged: [0, 0],
                },
            ));
        }

        let svm_owner = pool.apu_svm().or_else(|| {
            pool.engines
                .iter()
                .find(|e| e.device_info.supports_svm)
        });

        let (path, host) = if let Some(apu) = svm_owner {
            match (
                OwnedSvmBuffer::new(apu, slot_bytes),
                OwnedSvmBuffer::new(apu, slot_bytes),
            ) {
                (Ok(a), Ok(b)) => {
                    info!(
                        "HeteroScratch host base: SVM on {} (coarse preferred)",
                        apu.device_info.device_name
                    );
                    (
                        TransferPath::SvmZeroCopy,
                        HostBase::Svm {
                            slots: [a, b],
                            owner_id: apu.device_info.device_id,
                        },
                    )
                }
                (Err(e), _) | (_, Err(e)) => {
                    warn!("SVM host base failed ({e}); pinned fallback on primary");
                    let prim = pool.primary();
                    (
                        TransferPath::PinnedDma,
                        HostBase::Pinned {
                            pinned: [
                                PinnedHostBuffer::new(prim, slot_bytes)?,
                                PinnedHostBuffer::new(prim, slot_bytes)?,
                            ],
                        },
                    )
                }
            }
        } else {
            let prim = pool.primary();
            info!(
                "HeteroScratch host base: pinned on {} (no SVM in pool)",
                prim.device_info.device_name
            );
            (
                TransferPath::PinnedDma,
                HostBase::Pinned {
                    pinned: [
                        PinnedHostBuffer::new(prim, slot_bytes)?,
                        PinnedHostBuffer::new(prim, slot_bytes)?,
                    ],
                },
            )
        };

        let mut dgpu_mirrors = Vec::new();
        for eng in pool.engines.iter().filter(|e| {
            e.device_info.device_kind == DeviceKind::DiscreteGpu
        }) {
            dgpu_mirrors.push(DgpuMirror {
                device_id: eng.device_info.device_id,
                slots: [
                    DeviceLayerBuffer::new(eng, slot_bytes)?,
                    DeviceLayerBuffer::new(eng, slot_bytes)?,
                ],
            });
            info!(
                "HeteroScratch dGPU mirror: {} ({} KiB × 2, FFN-slice DMA)",
                eng.device_info.device_name,
                slot_bytes / 1024
            );
        }

        // Pinned host has no SVM bind path — every FFN device needs a VRAM/device mirror.
        if dgpu_mirrors.is_empty() {
            if let HostBase::Pinned { .. } = &host {
                let prim = pool.primary();
                dgpu_mirrors.push(DgpuMirror {
                    device_id: prim.device_info.device_id,
                    slots: [
                        DeviceLayerBuffer::new(prim, slot_bytes)?,
                        DeviceLayerBuffer::new(prim, slot_bytes)?,
                    ],
                });
            }
        }

        Ok((
            path,
            Self {
                host,
                dgpu_mirrors,
                path,
                resident: false,
                resident_stride: layer_bytes,
                resident_layers: 0,
                block_k: k_chunk,
                block_staged: [0, 0],
            },
        ))
    }

    /// Allocate a **resident** scratch: every layer pack lives in accelerator
    /// memory (`resident_layers × resident_stride`) for the whole session.
    ///
    /// Layers are addressed by index (no ping-pong), I/O and DMA happen once at
    /// preload ([`Self::fill_layer`]) and are eliminated from the decode loop.
    pub fn allocate_resident_for_pool(
        pool: &OpenClDevicePool,
        layer_bytes: usize,
        n_layers: usize,
    ) -> Result<(TransferPath, Self), OpenClError> {
        let total = layer_bytes.saturating_mul(n_layers.max(1));
        if pool.is_empty() {
            return Ok((
                TransferPath::HostRam,
                Self {
                    host: HostBase::Host {
                        slots: [AlignedBuffer::zeroed(total), AlignedBuffer::zeroed(1)],
                    },
                    dgpu_mirrors: Vec::new(),
                    path: TransferPath::HostRam,
                    resident: true,
                    resident_stride: layer_bytes,
                    resident_layers: n_layers,
                    block_k: 1,
                    block_staged: [0, 0],
                },
            ));
        }

        let svm_owner = pool
            .apu_svm()
            .or_else(|| pool.engines.iter().find(|e| e.device_info.supports_svm));
        if let Some(apu) = svm_owner {
            match OwnedSvmBuffer::new(apu, total) {
                Ok(buf) => {
                    info!(
                        "ResidentScratch: SVM on {} ({} layers × {} KiB)",
                        apu.device_info.device_name,
                        n_layers,
                        layer_bytes / 1024
                    );
                    let mut mirrors = Self::resident_mirrors(pool, total)?;
                    // Pinned-host fallback not needed here (SVM covers primary FFN).
                    if mirrors.is_empty() {
                        let prim = pool.primary();
                        if prim.device_info.device_kind == DeviceKind::DiscreteGpu {
                            mirrors.push(DgpuMirror {
                                device_id: prim.device_info.device_id,
                                slots: [
                                    DeviceLayerBuffer::new(prim, total)?,
                                    DeviceLayerBuffer::new(prim, 1)?,
                                ],
                            });
                        }
                    }
                    return Ok((
                        TransferPath::SvmZeroCopy,
                        Self {
                            host: HostBase::Svm {
                                slots: [buf, OwnedSvmBuffer::new(apu, 1)?],
                                owner_id: apu.device_info.device_id,
                            },
                            dgpu_mirrors: mirrors,
                            path: TransferPath::SvmZeroCopy,
                            resident: true,
                            resident_stride: layer_bytes,
                            resident_layers: n_layers,
                            block_k: 1,
                            block_staged: [0, 0],
                        },
                    ));
                }
                Err(e) => {
                    warn!("ResidentScratch SVM alloc failed ({e}); pinned fallback");
                }
            }
        }

        let prim = pool.primary();
        let mut mirrors = Self::resident_mirrors(pool, total)?;
        if mirrors.is_empty() {
            mirrors.push(DgpuMirror {
                device_id: prim.device_info.device_id,
                slots: [
                    DeviceLayerBuffer::new(prim, total)?,
                    DeviceLayerBuffer::new(prim, 1)?,
                ],
            });
        }
        info!(
            "ResidentScratch: pinned host on {} ({} layers × {} KiB)",
            prim.device_info.device_name,
            n_layers,
            layer_bytes / 1024
        );
        Ok((
            TransferPath::PinnedDma,
            Self {
                host: HostBase::Pinned {
                    pinned: [
                        PinnedHostBuffer::new(prim, total)?,
                        PinnedHostBuffer::new(prim, 1)?,
                    ],
                },
                dgpu_mirrors: mirrors,
                path: TransferPath::PinnedDma,
                resident: true,
                resident_stride: layer_bytes,
                resident_layers: n_layers,
                block_k: 1,
                block_staged: [0, 0],
            },
        ))
    }

    /// One big VRAM mirror per discrete GPU for the resident base.
    fn resident_mirrors(
        pool: &OpenClDevicePool,
        total: usize,
    ) -> Result<Vec<DgpuMirror>, OpenClError> {
        let mut mirrors = Vec::new();
        for eng in pool.engines.iter().filter(|e| {
            e.device_info.device_kind == DeviceKind::DiscreteGpu
        }) {
            mirrors.push(DgpuMirror {
                device_id: eng.device_info.device_id,
                slots: [
                    DeviceLayerBuffer::new(eng, total)?,
                    DeviceLayerBuffer::new(eng, 1)?,
                ],
            });
        }
        Ok(mirrors)
    }

    /// Legacy single-engine allocate (bench-io).
    pub fn allocate(
        engine: Option<&OpenClEngine>,
        layer_bytes: usize,
    ) -> Result<(TransferPath, Self), OpenClError> {
        let Some(engine) = engine else {
            return Ok((
                TransferPath::HostRam,
                Self {
                    host: HostBase::Host {
                        slots: [
                            AlignedBuffer::zeroed(layer_bytes),
                            AlignedBuffer::zeroed(layer_bytes),
                        ],
                    },
                    dgpu_mirrors: Vec::new(),
                    path: TransferPath::HostRam,
                    resident: false,
                    resident_stride: layer_bytes,
                    resident_layers: 0,
                    block_k: 1,
                    block_staged: [0, 0],
                },
            ));
        };
        if engine.device_info.supports_svm {
            match (
                OwnedSvmBuffer::new(engine, layer_bytes),
                OwnedSvmBuffer::new(engine, layer_bytes),
            ) {
                (Ok(a), Ok(b)) => {
                    let mut mirrors = Vec::new();
                    if engine.device_info.device_kind == DeviceKind::DiscreteGpu {
                        mirrors.push(DgpuMirror {
                            device_id: engine.device_info.device_id,
                            slots: [
                                DeviceLayerBuffer::new(engine, layer_bytes)?,
                                DeviceLayerBuffer::new(engine, layer_bytes)?,
                            ],
                        });
                    }
                    return Ok((
                        TransferPath::SvmZeroCopy,
                        Self {
                            host: HostBase::Svm {
                                slots: [a, b],
                                owner_id: engine.device_info.device_id,
                            },
                            dgpu_mirrors: mirrors,
                            path: TransferPath::SvmZeroCopy,
                            resident: false,
                            resident_stride: layer_bytes,
                            resident_layers: 0,
                            block_k: 1,
                            block_staged: [0, 0],
                        },
                    ));
                }
                (Err(e), _) | (_, Err(e)) => {
                    warn!("single-device SVM failed ({e}); pinned");
                }
            }
        }
        let path = TransferPath::PinnedDma;
        let mut mirrors = Vec::new();
        if engine.device_info.device_kind == DeviceKind::DiscreteGpu {
            mirrors.push(DgpuMirror {
                device_id: engine.device_info.device_id,
                slots: [
                    DeviceLayerBuffer::new(engine, layer_bytes)?,
                    DeviceLayerBuffer::new(engine, layer_bytes)?,
                ],
            });
        }
        Ok((
            path,
            Self {
                host: HostBase::Pinned {
                    pinned: [
                        PinnedHostBuffer::new(engine, layer_bytes)?,
                        PinnedHostBuffer::new(engine, layer_bytes)?,
                    ],
                },
                dgpu_mirrors: mirrors,
                path,
                resident: false,
                resident_stride: layer_bytes,
                resident_layers: 0,
                block_k: 1,
                block_staged: [0, 0],
            },
        ))
    }

    pub fn host_slot_mut(&mut self, idx: usize) -> &mut [u8] {
        if self.resident {
            let s = self.resident_stride;
            let base = match &mut self.host {
                HostBase::Svm { slots, .. } => slots[0].as_mut_slice(),
                HostBase::Pinned { pinned } => pinned[0].as_mut_slice(),
                HostBase::Host { slots } => &mut slots[0][..],
            };
            let len = base.len();
            let lo = idx.saturating_mul(s).min(len);
            return &mut base[lo..(lo + s).min(len)];
        }
        if self.block_k > 1 {
            // Macro-chunk: layer `idx` lives at `(idx % block_k) × stride` inside
            // its block's ping-pong slot.
            let slot = self.slot_for(idx);
            let s = self.resident_stride;
            let base = match &mut self.host {
                HostBase::Svm { slots, .. } => slots[slot].as_mut_slice(),
                HostBase::Pinned { pinned } => pinned[slot].as_mut_slice(),
                HostBase::Host { slots } => &mut slots[slot][..],
            };
            let len = base.len();
            let off = (idx % self.block_k).saturating_mul(s).min(len);
            return &mut base[off..(off + s).min(len)];
        }
        let slot = idx % 2;
        match &mut self.host {
            HostBase::Svm { slots, .. } => slots[slot].as_mut_slice(),
            HostBase::Pinned { pinned } => pinned[slot].as_mut_slice(),
            HostBase::Host { slots } => &mut slots[slot],
        }
    }

    pub fn host_slot(&self, idx: usize) -> &[u8] {
        if self.resident {
            let s = self.resident_stride;
            let base = match &self.host {
                HostBase::Svm { slots, .. } => slots[0].as_slice(),
                HostBase::Pinned { pinned } => pinned[0].as_slice(),
                HostBase::Host { slots } => &slots[0][..],
            };
            let len = base.len();
            let lo = idx.saturating_mul(s).min(len);
            return &base[lo..(lo + s).min(len)];
        }
        if self.block_k > 1 {
            let slot = self.slot_for(idx);
            let s = self.resident_stride;
            let base = match &self.host {
                HostBase::Svm { slots, .. } => slots[slot].as_slice(),
                HostBase::Pinned { pinned } => pinned[slot].as_slice(),
                HostBase::Host { slots } => &slots[slot][..],
            };
            let len = base.len();
            let off = (idx % self.block_k).saturating_mul(s).min(len);
            return &base[off..(off + s).min(len)];
        }
        let slot = idx % 2;
        match &self.host {
            HostBase::Svm { slots, .. } => slots[slot].as_slice(),
            HostBase::Pinned { pinned } => pinned[slot].as_slice(),
            HostBase::Host { slots } => &slots[slot],
        }
    }

    /// Ping-pong slot that owns `layer` (streaming: `layer % 2`; macro-chunk:
    /// `(layer / block_k) % 2`; resident: always `0`).
    pub fn slot_for(&self, layer: usize) -> usize {
        if self.resident {
            0
        } else {
            (layer / self.block_k.max(1)) % 2
        }
    }

    /// Raw pointer to a whole macro-chunk block slot (for block prefetch threads).
    pub fn host_slot_block_ptr_mut(&mut self, block_slot: usize) -> (*mut u8, usize) {
        let slot = block_slot % 2;
        let slice = match &mut self.host {
            HostBase::Svm { slots, .. } => slots[slot].as_mut_slice(),
            HostBase::Pinned { pinned } => pinned[slot].as_mut_slice(),
            HostBase::Host { slots } => &mut slots[slot][..],
        };
        (slice.as_mut_ptr(), slice.len())
    }

    /// Mark a ping-pong slot as holding `block` (1-based index stored).
    pub fn mark_block_staged(&mut self, block_slot: usize, block: usize) {
        self.block_staged[block_slot % 2] = block + 1;
    }

    /// Byte capacity of one ping-pong slot (does not require SVM host-map).
    pub fn slot_capacity(&self) -> usize {
        if self.resident {
            return self.resident_stride;
        }
        if self.block_k > 1 {
            return self.resident_stride.saturating_mul(self.block_k);
        }
        match &self.host {
            HostBase::Svm { slots, .. } => slots[0].size_bytes,
            HostBase::Pinned { pinned } => pinned[0].size_bytes,
            HostBase::Host { slots } => slots[0].len(),
        }
    }

    fn slot_len(&self, idx: usize) -> usize {
        let _ = idx % 2;
        self.slot_capacity()
    }

    /// Map SVM (if any) so the host can write/read the layer pack (Attn CPU GEMV).
    pub fn prepare_host_write(
        &mut self,
        pool: &OpenClDevicePool,
        idx: usize,
    ) -> Result<(), OpenClError> {
        if self.resident {
            // idx = layer index; map the whole resident base once.
            if let HostBase::Svm { slots, owner_id } = &mut self.host {
                if let Some(apu) = pool
                    .engines
                    .iter()
                    .find(|e| e.device_info.device_id == *owner_id)
                {
                    slots[0].prepare_for_host(apu)?;
                }
            }
            return Ok(());
        }
        let slot = self.slot_for(idx);
        if let HostBase::Svm { slots, owner_id } = &mut self.host {
            if let Some(apu) = pool
                .engines
                .iter()
                .find(|e| e.device_info.device_id == *owner_id)
            {
                slots[slot].prepare_for_host(apu)?;
            }
        }
        Ok(())
    }

    /// Alias: ensure CPU can read weight views in this slot (coarse SVM map).
    pub fn ensure_host_readable(
        &mut self,
        pool: &OpenClDevicePool,
        idx: usize,
    ) -> Result<(), OpenClError> {
        self.prepare_host_write(pool, idx)
    }

    /// Raw host-slot pointer for direct I/O prefetch (caller maps first).
    /// Safe to use concurrently with the other ping-pong slot.
    pub fn host_slot_ptr_mut(&mut self, idx: usize) -> (*mut u8, usize) {
        let slot = idx % 2;
        let slice = self.host_slot_mut(slot);
        (slice.as_mut_ptr(), slice.len())
    }

    /// DMA FFN slices from the **mapped** host slot (no heap snapshot). Keeps SVM mapped
    /// so Attn CPU GEMV can overlap with dGPU WriteBufferRect.
    pub fn dma_ffn_keep_mapped(
        &mut self,
        pool: &OpenClDevicePool,
        idx: usize,
        layout: &LayerPackLayout,
    ) -> Result<(), OpenClError> {
        if self.resident {
            // Every layer was DMA'd into the dGPU mirrors once at preload (`fill_layer`).
            return Ok(());
        }
        let slot = self.slot_for(idx);
        let n = layout.total.min(self.slot_len(slot));
        let ffn_base = layout.gate_off.min(n);
        self.dma_ffn_slices_mapped(pool, slot, layout, ffn_base, n)
    }

    /// Unmap SVM so the device can consume the slot (call after Attn, before FFN enqueue).
    pub fn unmap_host_for_device(
        &mut self,
        pool: &OpenClDevicePool,
        idx: usize,
    ) -> Result<(), OpenClError> {
        if self.resident {
            if let HostBase::Svm { slots, owner_id } = &mut self.host {
                if let Some(apu) = pool
                    .engines
                    .iter()
                    .find(|e| e.device_info.device_id == *owner_id)
                {
                    slots[0].prepare_for_device(apu)?;
                }
            }
            return Ok(());
        }
        let slot = self.slot_for(idx);
        if let HostBase::Svm { slots, owner_id } = &mut self.host {
            if let Some(apu) = pool
                .engines
                .iter()
                .find(|e| e.device_info.device_id == *owner_id)
            {
                slots[slot].prepare_for_device(apu)?;
            }
        }
        Ok(())
    }

    /// After host write: DMA FFN regions then unmap SVM (legacy combined path).
    pub fn commit_host_and_dma_ffn(
        &mut self,
        pool: &OpenClDevicePool,
        idx: usize,
        layout: &LayerPackLayout,
    ) -> Result<(), OpenClError> {
        self.dma_ffn_keep_mapped(pool, idx, layout)?;
        self.unmap_host_for_device(pool, idx)
    }

    /// Copy `src` into the host slot then DMA FFN slices (compat / bench path).
    pub fn ingest_layer_pool(
        &mut self,
        pool: &OpenClDevicePool,
        idx: usize,
        src: &[u8],
        layout: &LayerPackLayout,
    ) -> Result<(), OpenClError> {
        let slot = idx % 2;
        let n = src.len().min(self.slot_len(slot)).min(layout.total);
        self.prepare_host_write(pool, slot)?;
        self.host_slot_mut(slot)[..n].copy_from_slice(&src[..n]);
        self.commit_host_and_dma_ffn(pool, slot, layout)
    }

    /// Compatibility: full-pack ingest without layout (bench-io) — DMA entire blob.
    pub fn ingest_layer(
        &mut self,
        engine: Option<&OpenClEngine>,
        idx: usize,
        src: &[u8],
    ) -> Result<(), OpenClError> {
        let slot = idx % 2;
        let n = src.len().min(self.slot_len(slot));
        if let (HostBase::Svm { slots, .. }, Some(eng)) = (&mut self.host, engine) {
            slots[slot].prepare_for_host(eng)?;
            slots[slot].as_mut_slice()[..n].copy_from_slice(&src[..n]);
            slots[slot].prepare_for_device(eng)?;
        } else {
            self.host_slot_mut(slot)[..n].copy_from_slice(&src[..n]);
        }
        let host_bytes = self.host_slot(slot)[..n].to_vec();
        if let Some(eng) = engine {
            for mirror in &mut self.dgpu_mirrors {
                if mirror.device_id == eng.device_info.device_id {
                    unsafe {
                        eng.queue
                            .enqueue_write_buffer(
                                &mut mirror.slots[slot].cl_buffer,
                                CL_NON_BLOCKING,
                                0,
                                &host_bytes,
                                &[],
                            )
                            .map_err(|e| {
                                OpenClError::ClError(format!("dGPU DMA WriteBuffer: {e}"))
                            })?;
                    }
                }
            }
        }
        Ok(())
    }

    /// DMA FFN role slices directly from the mapped host slot (zero heap copy).
    fn dma_ffn_slices_mapped(
        &mut self,
        pool: &OpenClDevicePool,
        slot: usize,
        layout: &LayerPackLayout,
        ffn_base: usize,
        n: usize,
    ) -> Result<(), OpenClError> {
        // Collect (device_id, dest_off, host_off, len) while we can borrow host immutably,
        // then issue WriteBufferRect with raw pointers into the mapped slot.
        let host_ptr = self.host_slot(slot).as_ptr();
        let mut jobs: Vec<(cl_device_id, usize, usize, usize)> = Vec::new();
        for mirror in &self.dgpu_mirrors {
            for role in 0..3usize {
                if pool.for_role(role).device_info.device_id != mirror.device_id {
                    continue;
                }
                let (off, len) = layout.ffn_region(role);
                if len == 0 || off < ffn_base || off + len > n {
                    continue;
                }
                jobs.push((mirror.device_id, off, off, len));
            }
        }
        for (dev_id, dest_off, host_off, len) in jobs {
            let Some(eng) = pool
                .engines
                .iter()
                .find(|e| e.device_info.device_id == dev_id)
            else {
                continue;
            };
            let Some(mirror) = self
                .dgpu_mirrors
                .iter_mut()
                .find(|m| m.device_id == dev_id)
            else {
                continue;
            };
            let host = unsafe { std::slice::from_raw_parts(host_ptr.add(host_off), len) };
            dma_write_rect(eng, &mut mirror.slots[slot].cl_buffer, dest_off, host)?;
        }
        let _ = ffn_base;
        Ok(())
    }

    /// Resolve Base+Offset bind for a pool device. Errors if the device has no
    /// SVM ownership and no VRAM mirror (HostUpload removed from product path).
    pub fn weight_bind(&self, eng: &OpenClEngine, slot: usize) -> Result<WeightBind<'_>, OpenClError> {
        if self.resident {
            // Resident: single base allocation; `slot` is the layer index.
            if let HostBase::Svm { slots, owner_id } = &self.host {
                if eng.device_info.device_id == *owner_id {
                    return Ok(WeightBind::Svm {
                        ptr: slots[0].as_ptr(),
                    });
                }
            }
            if let Some(m) = self
                .dgpu_mirrors
                .iter()
                .find(|m| m.device_id == eng.device_info.device_id)
            {
                return Ok(WeightBind::Device {
                    buf: &m.slots[0].cl_buffer,
                });
            }
            return Err(OpenClError::ClError(format!(
                "no resident WeightBind for '{}' (need SVM owner or dGPU mirror)",
                eng.device_info.device_name
            )));
        }
        let slot = if self.resident { 0 } else { self.slot_for(slot) };
        if let HostBase::Svm { slots, owner_id } = &self.host {
            if eng.device_info.device_id == *owner_id {
                return Ok(WeightBind::Svm {
                    ptr: slots[slot].as_ptr(),
                });
            }
        }
        if let Some(m) = self
            .dgpu_mirrors
            .iter()
            .find(|m| m.device_id == eng.device_info.device_id)
        {
            return Ok(WeightBind::Device {
                buf: &m.slots[slot].cl_buffer,
            });
        }
        Err(OpenClError::ClError(format!(
            "no WeightBind for '{}' (need SVM owner or dGPU mirror; HostUpload disabled)",
            eng.device_info.device_name
        )))
    }

    /// Byte offset of a tensor inside a layer pack given the layer index.
    ///
    /// Resident: `layer × stride + tensor_off`. Macro-chunk: the layer sits at
    /// `(layer % block_k) × stride` inside its block slot. Streaming: `tensor_off`
    /// (the layer pack already sits at the base of its ping-pong slot).
    pub fn weight_offset(&self, layer: usize, tensor_off: usize) -> usize {
        if self.resident {
            layer.saturating_mul(self.resident_stride).saturating_add(tensor_off)
        } else if self.block_k > 1 {
            (layer % self.block_k)
                .saturating_mul(self.resident_stride)
                .saturating_add(tensor_off)
        } else {
            tensor_off
        }
    }

    /// Write one layer pack into the resident base and DMA it into every dGPU
    /// mirror once (one-time preload cost; zero I/O/DMA during decode).
    pub fn fill_layer(
        &mut self,
        pool: &OpenClDevicePool,
        layer: usize,
        blob: &[u8],
    ) -> Result<(), OpenClError> {
        debug_assert!(self.resident, "fill_layer requires resident scratch");
        self.prepare_host_write(pool, layer)?;
        {
            let dst = self.host_slot_mut(layer);
            let n = blob.len().min(dst.len());
            dst[..n].copy_from_slice(&blob[..n]);
        }
        self.dma_layer_to_mirrors(pool, layer, blob.len())
    }

    /// DMA `total` bytes of the already-written layer `layer` (host resident base)
    /// into every dGPU mirror. One-time cost; the decode loop never re-DMAs.
    pub fn dma_layer_to_mirrors(
        &mut self,
        pool: &OpenClDevicePool,
        layer: usize,
        total: usize,
    ) -> Result<(), OpenClError> {
        if self.dgpu_mirrors.is_empty() || total == 0 {
            return Ok(());
        }
        let (host_ptr, _total) = self.host_base_ptr_len();
        let stride = self.resident_stride;
        let base_off = layer.saturating_mul(stride);
        for mirror in self.dgpu_mirrors.iter_mut() {
            let Some(eng) = pool
                .engines
                .iter()
                .find(|e| e.device_info.device_id == mirror.device_id)
            else {
                continue;
            };
            // SAFETY: host_ptr points into the resident base owned by `self`, which
            // outlives this call; the base stays host-visible while mapped above.
            let src = unsafe { std::slice::from_raw_parts(host_ptr.add(base_off), total) };
            dma_write_rect(eng, &mut mirror.slots[0].cl_buffer, base_off, src)?;
        }
        Ok(())
    }

    /// Raw pointer to the resident host base plus its total length.
    fn host_base_ptr_len(&self) -> (*const u8, usize) {
        let n = self.resident_layers.saturating_mul(self.resident_stride);
        match &self.host {
            HostBase::Svm { slots, .. } => (slots[0].as_ptr(), n),
            HostBase::Pinned { pinned } => (pinned[0].host_ptr, n),
            HostBase::Host { slots } => (slots[0].as_ptr(), n),
        }
    }

    /// Legacy helper used by older call sites.
    pub fn device_slot(&self, idx: usize) -> Option<&DeviceLayerBuffer> {
        self.dgpu_mirrors
            .first()
            .map(|m| &m.slots[idx % 2])
    }
}

fn dma_write_rect(
    eng: &OpenClEngine,
    dest: &mut Buffer<cl_uchar>,
    dest_off: usize,
    host: &[u8],
) -> Result<(), OpenClError> {
    if host.is_empty() {
        return Ok(());
    }
    let buffer_origin: [size_t; 3] = [dest_off as size_t, 0, 0];
    let host_origin: [size_t; 3] = [0, 0, 0];
    let region: [size_t; 3] = [host.len() as size_t, 1, 1];
    unsafe {
        eng.queue
            .enqueue_write_buffer_rect(
                dest,
                CL_NON_BLOCKING,
                buffer_origin.as_ptr(),
                host_origin.as_ptr(),
                region.as_ptr(),
                0,
                0,
                0,
                0,
                host.as_ptr() as *mut c_void,
                &[],
            )
            .map_err(|e| OpenClError::ClError(format!("dGPU WriteBufferRect: {e}")))?;
    }
    Ok(())
}
