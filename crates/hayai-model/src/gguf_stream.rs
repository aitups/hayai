//! GGUF catalog with **deterministic** tensor I/O via [`hayai_io::WeightIo`].
//!
//! PRD path: weights are not left to OS `mmap` paging during generate.
//! Header/metadata lives in RAM; payloads are fetched with `io_uring` (Linux)
//! or blocking file reads (Windows) into caller / ping-pong buffers.

use crate::gguf::{parse_header_bytes, tensor_nbytes};
use crate::gguf_types::{GgufError, MetadataValue, TensorInfo};
use crate::quant::QuantMatrix;
use hayai_io::{open_weight_io, IoBackend, WeightIo};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::info;

/// On-disk GGUF with explicit reads (no mmap of weight payloads).
pub struct GgufCatalog {
    pub path: PathBuf,
    io: Box<dyn WeightIo>,
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_index: HashMap<String, usize>,
    pub data_offset: u64,
    pub header_bytes: usize,
}

impl GgufCatalog {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let path = path.as_ref().to_path_buf();
        let io = open_weight_io(&path).map_err(GgufError::from)?;
        Self::open_with_io(path, io)
    }

    fn open_with_io(path: PathBuf, mut io: Box<dyn WeightIo>) -> Result<Self, GgufError> {
        let header = Self::parse_header_from_start(&mut *io)?;
        let backend = io.backend();
        info!(
            "Opened GGUF catalog v{} ({}): {} tensors, header={} KiB, data@{}",
            header.version,
            backend.as_str(),
            header.tensors.len(),
            header.header_bytes / 1024,
            header.data_offset
        );
        Ok(Self {
            path,
            io,
            version: header.version,
            alignment: header.alignment,
            metadata: header.metadata,
            tensors: header.tensors,
            tensor_index: header.tensor_index,
            data_offset: header.data_offset,
            header_bytes: header.header_bytes,
        })
    }

    /// Read increasing prefixes from offset 0 until the GGUF header parses.
    fn parse_header_from_start(io: &mut dyn WeightIo) -> Result<crate::gguf::GgufHeader, GgufError> {
        let mut size = 256usize;
        loop {
            let mut buf = vec![0u8; size];
            match io.read_at(0, &mut buf) {
                Ok(()) => match parse_header_bytes(&buf) {
                    Ok(h) => return Ok(h),
                    Err(GgufError::Truncated(_)) if size < 64 << 20 => {
                        size = (size.saturating_mul(2)).min(64 << 20);
                        continue;
                    }
                    Err(e) => return Err(e),
                },
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Exact read of `size` failed — file may be shorter; try half.
                    if size <= 64 {
                        return Err(GgufError::Truncated("EOF before GGUF header complete"));
                    }
                    size /= 2;
                    continue;
                }
                Err(e) => return Err(GgufError::from(e)),
            }
        }
    }

    pub fn io_backend(&self) -> IoBackend {
        self.io.backend()
    }

    /// Independent handle with the same parsed header (no re-parse).
    pub fn fork_reader(&self) -> Result<Self, GgufError> {
        let io = open_weight_io(&self.path).map_err(GgufError::from)?;
        Ok(Self {
            path: self.path.clone(),
            io,
            version: self.version,
            alignment: self.alignment,
            metadata: self.metadata.clone(),
            tensors: self.tensors.clone(),
            tensor_index: self.tensor_index.clone(),
            data_offset: self.data_offset,
            header_bytes: self.header_bytes,
        })
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    pub fn meta_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        self.metadata.get(key).and_then(|v| v.as_bool_array())
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        let idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))?;
        Ok(&self.tensors[*idx])
    }

    pub fn tensor_abs_offset(&self, info: &TensorInfo) -> u64 {
        self.data_offset + info.offset
    }

    /// Deterministic full-tensor read into `dst`.
    pub fn read_tensor_into(
        &mut self,
        name: &str,
        dst: &mut [u8],
    ) -> Result<(usize, TensorInfo), GgufError> {
        let info = self.tensor(name)?.clone();
        let nbytes = tensor_nbytes(&info)?;
        if dst.len() < nbytes {
            return Err(GgufError::Msg(format!(
                "buffer too small for {name}: need {nbytes}, have {}",
                dst.len()
            )));
        }
        let abs = self.tensor_abs_offset(&info);
        self.io
            .read_at(abs, &mut dst[..nbytes])
            .map_err(GgufError::from)?;
        Ok((nbytes, info))
    }

    /// Load tensor into an owned [`QuantMatrix`] (temporary residency for one op).
    pub fn load_quant_matrix(&mut self, name: &str) -> Result<QuantMatrix, GgufError> {
        let info = self.tensor(name)?.clone();
        let nbytes = tensor_nbytes(&info)?;
        let mut buf = vec![0u8; nbytes];
        self.read_tensor_into(name, &mut buf)?;
        Ok(QuantMatrix::owned(
            info.name.clone(),
            info.ncols(),
            info.nrows(),
            info.ggml_type,
            buf,
        ))
    }

    /// Read one embedding row by seek (no full embd matrix in RAM).
    pub fn read_embed_row(
        &mut self,
        embd_name: &str,
        token: u32,
        hidden: usize,
        dst: &mut [f32],
    ) -> Result<(), GgufError> {
        if dst.len() != hidden {
            return Err(GgufError::Truncated("embed dst"));
        }
        let info = self.tensor(embd_name)?.clone();
        if info.ncols() != hidden {
            return Err(GgufError::Msg("embed ncols != hidden".into()));
        }
        let row = token as usize;
        if row >= info.nrows() {
            return Err(GgufError::Truncated("embed token OOB"));
        }
        let nbytes = tensor_nbytes(&info)?;
        let row_bytes = nbytes / info.nrows().max(1);
        let abs = self.tensor_abs_offset(&info) + (row * row_bytes) as u64;
        let mut buf = vec![0u8; row_bytes];
        self.io.read_at(abs, &mut buf).map_err(GgufError::from)?;
        let view = QuantMatrix::owned(
            format!("{embd_name}#row"),
            hidden,
            1,
            info.ggml_type,
            buf,
        );
        view.embed_row(0, dst)
    }

    pub fn dequant_f32(&mut self, name: &str) -> Result<Vec<f32>, GgufError> {
        let info = self.tensor(name)?.clone();
        let nbytes = tensor_nbytes(&info)?;
        let mut buf = vec![0u8; nbytes];
        let abs = self.tensor_abs_offset(&info);
        self.io.read_at(abs, &mut buf).map_err(GgufError::from)?;
        crate::gguf::dequantize(&info, &buf)
    }

    /// Load all weight matrices for one transformer block (deterministic reads).
    pub fn load_layer_pack(&mut self, layer: usize) -> Result<LayerWeightPack, GgufError> {
        let attn_gate = match self.tensor(&format!("blk.{layer}.attn_gate.weight")) {
            Ok(_) => Some(self.load_quant_matrix(&format!("blk.{layer}.attn_gate.weight"))?),
            Err(_) => None,
        };
        Ok(LayerWeightPack {
            wq: self.load_quant_matrix(&format!("blk.{layer}.attn_q.weight"))?,
            wk: self.load_quant_matrix(&format!("blk.{layer}.attn_k.weight"))?,
            wv: self.load_quant_matrix(&format!("blk.{layer}.attn_v.weight"))?,
            wo: self.load_quant_matrix(&format!("blk.{layer}.attn_output.weight"))?,
            gate: self.load_quant_matrix(&format!("blk.{layer}.ffn_gate.weight"))?,
            up: self.load_quant_matrix(&format!("blk.{layer}.ffn_up.weight"))?,
            down: self.load_quant_matrix(&format!("blk.{layer}.ffn_down.weight"))?,
            attn_gate,
        })
    }

    /// PRD §3.1: read the layer pack **directly** into `dst` (SVM/Pinned host slot).
    ///
    /// Returns `QuantMatrix::view` into `dst` (no heap copy). The pack must not
    /// outlive `dst` / the scratch slot that owns those bytes.
    pub fn load_layer_pack_into(
        &mut self,
        layer: usize,
        dst: &mut [u8],
    ) -> Result<(LayerWeightPack, LayerPackLayout), GgufError> {
        let names = [
            format!("blk.{layer}.attn_q.weight"),
            format!("blk.{layer}.attn_k.weight"),
            format!("blk.{layer}.attn_v.weight"),
            format!("blk.{layer}.attn_output.weight"),
            format!("blk.{layer}.ffn_gate.weight"),
            format!("blk.{layer}.ffn_up.weight"),
            format!("blk.{layer}.ffn_down.weight"),
        ];
        let mut offs = [0usize; 7];
        let mut lens = [0usize; 7];
        let mut dims = [(0usize, 0usize); 7];
        let mut types = [crate::gguf_types::GgmlType::F32; 7];
        let mut off = 0usize;
        for (i, name) in names.iter().enumerate() {
            let info = self.tensor(name)?.clone();
            let nbytes = tensor_nbytes(&info)?;
            if off + nbytes > dst.len() {
                return Err(GgufError::Msg(format!(
                    "layer {layer} pack {nbytes}B at {off} exceeds scratch {}",
                    dst.len()
                )));
            }
            let abs = self.tensor_abs_offset(&info);
            self.io
                .read_at(abs, &mut dst[off..off + nbytes])
                .map_err(GgufError::from)?;
            offs[i] = off;
            lens[i] = nbytes;
            dims[i] = (info.ncols(), info.nrows());
            types[i] = info.ggml_type;
            off += nbytes;
        }
        let gate_name = format!("blk.{layer}.attn_gate.weight");
        let (attn_gate_off, attn_gate_len, attn_gate_dim, attn_gate_ty) =
            if let Ok(info) = self.tensor(&gate_name).cloned() {
                let nbytes = tensor_nbytes(&info)?;
                if off + nbytes > dst.len() {
                    return Err(GgufError::Msg(format!(
                        "layer {layer} attn_gate exceeds scratch {}",
                        dst.len()
                    )));
                }
                let abs = self.tensor_abs_offset(&info);
                self.io
                    .read_at(abs, &mut dst[off..off + nbytes])
                    .map_err(GgufError::from)?;
                let ag_off = off;
                off += nbytes;
                (
                    ag_off,
                    nbytes,
                    (info.ncols(), info.nrows()),
                    info.ggml_type,
                )
            } else {
                (0, 0, (0, 0), crate::gguf_types::GgmlType::F32)
            };
        let layout = LayerPackLayout {
            wq_off: offs[0],
            wk_off: offs[1],
            wv_off: offs[2],
            wo_off: offs[3],
            gate_off: offs[4],
            up_off: offs[5],
            down_off: offs[6],
            attn_gate_off,
            wq_len: lens[0],
            wk_len: lens[1],
            wv_len: lens[2],
            wo_len: lens[3],
            gate_len: lens[4],
            up_len: lens[5],
            down_len: lens[6],
            attn_gate_len,
            total: off,
        };
        let mut pack = LayerWeightPack::views_from_base(dst, &layout, &names, &dims, &types);
        if attn_gate_len > 0 {
            pack.attn_gate = Some(QuantMatrix::view(
                gate_name,
                attn_gate_dim.0,
                attn_gate_dim.1,
                attn_gate_ty,
                &dst[attn_gate_off..attn_gate_off + attn_gate_len],
            ));
        }
        Ok((pack, layout))
    }

    /// Build layer-pack views into `base` **without any disk read** (resident mode).
    ///
    /// The pack's tensors must already live in `base[..layout.total]` (resident
    /// device memory). Unlike [`Self::load_layer_pack_into`] this never touches the
    /// weight payload — it only re-derives offsets/dims/types from the in-RAM index.
    pub fn layer_pack_views_from_base(
        &self,
        layer: usize,
        base: &[u8],
    ) -> Result<(LayerWeightPack, LayerPackLayout), GgufError> {
        let names = [
            format!("blk.{layer}.attn_q.weight"),
            format!("blk.{layer}.attn_k.weight"),
            format!("blk.{layer}.attn_v.weight"),
            format!("blk.{layer}.attn_output.weight"),
            format!("blk.{layer}.ffn_gate.weight"),
            format!("blk.{layer}.ffn_up.weight"),
            format!("blk.{layer}.ffn_down.weight"),
        ];
        let mut offs = [0usize; 7];
        let mut lens = [0usize; 7];
        let mut dims = [(0usize, 0usize); 7];
        let mut types = [crate::gguf_types::GgmlType::F32; 7];
        let mut off = 0usize;
        for (i, name) in names.iter().enumerate() {
            let info = self.tensor(name)?.clone();
            let nbytes = tensor_nbytes(&info)?;
            if off + nbytes > base.len() {
                return Err(GgufError::Msg(format!(
                    "layer {layer} pack {nbytes}B at {off} exceeds resident base {}",
                    base.len()
                )));
            }
            offs[i] = off;
            lens[i] = nbytes;
            dims[i] = (info.ncols(), info.nrows());
            types[i] = info.ggml_type;
            off += nbytes;
        }
        let gate_name = format!("blk.{layer}.attn_gate.weight");
        let (attn_gate_off, attn_gate_len, attn_gate_dim, attn_gate_ty) =
            if let Ok(info) = self.tensor(&gate_name).cloned() {
                let nbytes = tensor_nbytes(&info)?;
                if off + nbytes > base.len() {
                    return Err(GgufError::Msg(format!(
                        "layer {layer} attn_gate exceeds resident base {}",
                        base.len()
                    )));
                }
                let ag_off = off;
                off += nbytes;
                (ag_off, nbytes, (info.ncols(), info.nrows()), info.ggml_type)
            } else {
                (0, 0, (0, 0), crate::gguf_types::GgmlType::F32)
            };
        let layout = LayerPackLayout {
            wq_off: offs[0],
            wk_off: offs[1],
            wv_off: offs[2],
            wo_off: offs[3],
            gate_off: offs[4],
            up_off: offs[5],
            down_off: offs[6],
            attn_gate_off,
            wq_len: lens[0],
            wk_len: lens[1],
            wv_len: lens[2],
            wo_len: lens[3],
            gate_len: lens[4],
            up_len: lens[5],
            down_len: lens[6],
            attn_gate_len,
            total: off,
        };
        let mut pack = LayerWeightPack::views_from_base(base, &layout, &names, &dims, &types);
        if attn_gate_len > 0 {
            pack.attn_gate = Some(QuantMatrix::view(
                gate_name,
                attn_gate_dim.0,
                attn_gate_dim.1,
                attn_gate_ty,
                &base[attn_gate_off..attn_gate_off + attn_gate_len],
            ));
        }
        Ok((pack, layout))
    }

    /// Byte size of one layer pack (for StreamingScratch allocation).
    pub fn layer_pack_nbytes(&self, layer: usize) -> Result<usize, GgufError> {
        let names = [
            format!("blk.{layer}.attn_q.weight"),
            format!("blk.{layer}.attn_k.weight"),
            format!("blk.{layer}.attn_v.weight"),
            format!("blk.{layer}.attn_output.weight"),
            format!("blk.{layer}.ffn_gate.weight"),
            format!("blk.{layer}.ffn_up.weight"),
            format!("blk.{layer}.ffn_down.weight"),
        ];
        let mut total = 0usize;
        for n in &names {
            total += tensor_nbytes(self.tensor(n)?)?;
        }
        if let Ok(info) = self.tensor(&format!("blk.{layer}.attn_gate.weight")) {
            total += tensor_nbytes(info)?;
        }
        Ok(total)
    }

    /// Max layer pack size across all blocks.
    pub fn max_layer_pack_nbytes(&self) -> Result<usize, GgufError> {
        let n_meta = self
            .meta_u32("llama.block_count")
            .or_else(|| {
                let arch = self.meta_str("general.architecture").unwrap_or("llama");
                self.meta_u32(&format!("{arch}.block_count"))
            })
            .unwrap_or(0) as usize;
        let n_physical = self
            .tensors
            .iter()
            .filter_map(|t| {
                t.name
                    .strip_prefix("blk.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        // Prefer physical `blk.N` when meta encodes recurrence (HRM H×L×cycles).
        let layers = if n_physical > 0 && (n_meta == 0 || n_physical < n_meta) {
            n_physical
        } else if n_meta > 0 {
            n_meta
        } else {
            n_physical
        };
        let mut max = 0usize;
        for i in 0..layers {
            // Skip blocks that are not LLaMA-shaped packs (e.g. Qwen hybrid SSM / NextN).
            match self.layer_pack_nbytes(i) {
                Ok(n) => max = max.max(n),
                Err(_) => continue,
            }
        }
        if max == 0 {
            return Err(GgufError::Msg(
                "no LLaMA-shaped layer packs found (attn_q/k/v/o + ffn_*)".into(),
            ));
        }
        Ok(max)
    }
}

/// One decoder layer's packed weights (resident only while that layer is active / prefetched).
#[derive(Clone)]
pub struct LayerWeightPack {
    pub wq: QuantMatrix,
    pub wk: QuantMatrix,
    pub wv: QuantMatrix,
    pub wo: QuantMatrix,
    pub gate: QuantMatrix,
    pub up: QuantMatrix,
    pub down: QuantMatrix,
    /// Optional sigmoid attention gate (HRM / Qwen3-Next style). `None` for plain LLaMA.
    pub attn_gate: Option<QuantMatrix>,
}

/// Byte offsets + lengths of each matrix inside the contiguous layer-pack blob.
#[derive(Debug, Clone, Copy)]
pub struct LayerPackLayout {
    pub wq_off: usize,
    pub wk_off: usize,
    pub wv_off: usize,
    pub wo_off: usize,
    pub gate_off: usize,
    pub up_off: usize,
    pub down_off: usize,
    pub attn_gate_off: usize,
    pub wq_len: usize,
    pub wk_len: usize,
    pub wv_len: usize,
    pub wo_len: usize,
    pub gate_len: usize,
    pub up_len: usize,
    pub down_len: usize,
    pub attn_gate_len: usize,
    pub total: usize,
}

impl LayerPackLayout {
    /// FFN tensor regions for OpenCL role `0=gate, 1=up, 2=down`.
    pub fn ffn_region(&self, role: usize) -> (usize, usize) {
        match role % 3 {
            0 => (self.gate_off, self.gate_len),
            1 => (self.up_off, self.up_len),
            _ => (self.down_off, self.down_len),
        }
    }
}

impl LayerWeightPack {
    /// Bind seven LLaMA-style matrices as views into a contiguous pack blob.
    pub fn views_from_base(
        base: &[u8],
        layout: &LayerPackLayout,
        names: &[String; 7],
        dims: &[(usize, usize); 7],
        types: &[crate::gguf_types::GgmlType; 7],
    ) -> Self {
        let slice = |off: usize, len: usize| &base[off..off + len];
        Self {
            wq: QuantMatrix::view(
                names[0].clone(),
                dims[0].0,
                dims[0].1,
                types[0],
                slice(layout.wq_off, layout.wq_len),
            ),
            wk: QuantMatrix::view(
                names[1].clone(),
                dims[1].0,
                dims[1].1,
                types[1],
                slice(layout.wk_off, layout.wk_len),
            ),
            wv: QuantMatrix::view(
                names[2].clone(),
                dims[2].0,
                dims[2].1,
                types[2],
                slice(layout.wv_off, layout.wv_len),
            ),
            wo: QuantMatrix::view(
                names[3].clone(),
                dims[3].0,
                dims[3].1,
                types[3],
                slice(layout.wo_off, layout.wo_len),
            ),
            gate: QuantMatrix::view(
                names[4].clone(),
                dims[4].0,
                dims[4].1,
                types[4],
                slice(layout.gate_off, layout.gate_len),
            ),
            up: QuantMatrix::view(
                names[5].clone(),
                dims[5].0,
                dims[5].1,
                types[5],
                slice(layout.up_off, layout.up_len),
            ),
            down: QuantMatrix::view(
                names[6].clone(),
                dims[6].0,
                dims[6].1,
                types[6],
                slice(layout.down_off, layout.down_len),
            ),
            attn_gate: None,
        }
    }

    /// Re-bind an existing pack's metadata as views into a new base (after scratch ingest).
    pub fn rebind_views(&self, base: &[u8], layout: &LayerPackLayout) -> Self {
        let names = [
            self.wq.name.clone(),
            self.wk.name.clone(),
            self.wv.name.clone(),
            self.wo.name.clone(),
            self.gate.name.clone(),
            self.up.name.clone(),
            self.down.name.clone(),
        ];
        let dims = [
            (self.wq.ncols, self.wq.nrows),
            (self.wk.ncols, self.wk.nrows),
            (self.wv.ncols, self.wv.nrows),
            (self.wo.ncols, self.wo.nrows),
            (self.gate.ncols, self.gate.nrows),
            (self.up.ncols, self.up.nrows),
            (self.down.ncols, self.down.nrows),
        ];
        let types = [
            self.wq.ggml_type,
            self.wk.ggml_type,
            self.wv.ggml_type,
            self.wo.ggml_type,
            self.gate.ggml_type,
            self.up.ggml_type,
            self.down.ggml_type,
        ];
        let mut pack = Self::views_from_base(base, layout, &names, &dims, &types);
        if layout.attn_gate_len > 0 {
            let name = self
                .attn_gate
                .as_ref()
                .map(|m| m.name.clone())
                .unwrap_or_else(|| "attn_gate".into());
            let (nc, nr, ty) = self
                .attn_gate
                .as_ref()
                .map(|m| (m.ncols, m.nrows, m.ggml_type))
                .unwrap_or((0, 0, crate::gguf_types::GgmlType::F32));
            pack.attn_gate = Some(QuantMatrix::view(
                name,
                nc,
                nr,
                ty,
                &base[layout.attn_gate_off..layout.attn_gate_off + layout.attn_gate_len],
            ));
        }
        pack
    }

    pub fn nbytes(&self) -> usize {
        self.wq.nbytes()
            + self.wk.nbytes()
            + self.wv.nbytes()
            + self.wo.nbytes()
            + self.gate.nbytes()
            + self.up.nbytes()
            + self.down.nbytes()
            + self.attn_gate.as_ref().map(|m| m.nbytes()).unwrap_or(0)
    }

    pub fn layout(&self) -> LayerPackLayout {
        let wq_len = self.wq.nbytes();
        let wk_len = self.wk.nbytes();
        let wv_len = self.wv.nbytes();
        let wo_len = self.wo.nbytes();
        let gate_len = self.gate.nbytes();
        let up_len = self.up.nbytes();
        let down_len = self.down.nbytes();
        let attn_gate_len = self.attn_gate.as_ref().map(|m| m.nbytes()).unwrap_or(0);
        let mut off = 0usize;
        let wq_off = off;
        off += wq_len;
        let wk_off = off;
        off += wk_len;
        let wv_off = off;
        off += wv_len;
        let wo_off = off;
        off += wo_len;
        let gate_off = off;
        off += gate_len;
        let up_off = off;
        off += up_len;
        let down_off = off;
        off += down_len;
        let attn_gate_off = off;
        off += attn_gate_len;
        LayerPackLayout {
            wq_off,
            wk_off,
            wv_off,
            wo_off,
            gate_off,
            up_off,
            down_off,
            attn_gate_off,
            wq_len,
            wk_len,
            wv_len,
            wo_len,
            gate_len,
            up_len,
            down_len,
            attn_gate_len,
            total: off,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::write_minimal_gguf;
    use crate::gguf_types::MetadataValue;
    use std::env::temp_dir;

    #[test]
    fn catalog_reads_without_mmap_payload() {
        let path = temp_dir().join("hayai_catalog_stream.gguf");
        write_minimal_gguf(
            &path,
            &[("general.architecture", MetadataValue::String("llama".into()))],
            &[("token_embd.weight", vec![4, 2], vec![1.0f32; 8])],
        )
        .unwrap();
        let mut cat = GgufCatalog::open(&path).unwrap();
        let m = cat.load_quant_matrix("token_embd.weight").unwrap();
        assert_eq!(m.ncols, 4);
        assert!(!m.is_mapped());
        let _ = std::fs::remove_file(&path);
    }
}
