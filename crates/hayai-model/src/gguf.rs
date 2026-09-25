use crate::gguf_types::{GgmlType, GgufError, MetadataValue, TensorInfo};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tracing::info;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" LE

/// Parsed GGUF header + tensor directory (no weight payloads).
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub alignment: u64,
    pub metadata: Arc<HashMap<String, MetadataValue>>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_index: HashMap<String, usize>,
    pub data_offset: u64,
    /// Number of header bytes consumed from the start of the file.
    pub header_bytes: usize,
}

/// Parse GGUF header from an in-memory prefix of the file (must include full tensor dir).
pub fn parse_header_bytes(data: &[u8]) -> Result<GgufHeader, GgufError> {
    let mut r = Reader::new(data);
    let magic = r.u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }
    let version = r.u32()?;
    // v1 shares the v2/v3 on-disk layout (offsets are u64); v4 is accepted
    // forward-compatibly. Anything else fails loudly.
    if !(1..=4).contains(&version) {
        return Err(GgufError::BadVersion(version));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    // Each directory entry needs at least a few bytes; a hostile GGUF could
    // otherwise request `Vec::with_capacity(usize::MAX)` and abort the process.
    // Bound both counts by the number of bytes left in the header buffer.
    let remaining = (data.len().saturating_sub(r.pos)) as u128;
    if kv_count as u128 > remaining / 8 + 1 {
        return Err(GgufError::Truncated("metadata count exceeds file size"));
    }
    if tensor_count as u128 > remaining / 8 + 1 {
        return Err(GgufError::Truncated("tensor count exceeds file size"));
    }
    let tensor_count = tensor_count as usize;
    let kv_count = kv_count as usize;

    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = r.string()?;
        let value = r.metadata_value()?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut tensor_index = HashMap::with_capacity(tensor_count);
    for i in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        if n_dims > 4 {
            return Err(GgufError::Msg(format!("tensor {name} has {n_dims} dims")));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let d = r.u64()?;
            if d == 0 {
                return Err(GgufError::Msg(format!("tensor {name} has a zero dimension")));
            }
            dims.push(d);
        }
        let type_id = r.u32()?;
        let offset = r.u64()?;
        let ggml_type = GgmlType::from_u32(type_id);
        let info = TensorInfo {
            name,
            dims,
            ggml_type,
            type_id,
            offset,
        };
        // Validate the element product and byte size now so no later code can
        // overflow while computing offsets/allocations from this directory.
        tensor_nbytes(&info)?;
        if tensor_index.insert(info.name.clone(), i).is_some() {
            return Err(GgufError::Msg(format!("duplicate tensor name {}", info.name)));
        }
        tensors.push(info);
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::Msg(format!(
            "invalid general.alignment {alignment} (must be a non-zero power of two)"
        )));
    }
    let data_offset = align_up_checked(r.pos as u64, alignment)?;

    // Reject a directory whose absolute offsets/ranges would wrap `u64`. This
    // makes `tensor_abs_offset` (which returns a plain `u64`) overflow-free for
    // every tensor in the catalog.
    for t in &tensors {
        let nbytes = tensor_nbytes(t)?;
        let abs = data_offset
            .checked_add(t.offset)
            .ok_or_else(|| GgufError::Msg(format!("tensor {} absolute offset overflow", t.name)))?;
        abs.checked_add(nbytes as u64)
            .ok_or_else(|| GgufError::Msg(format!("tensor {} byte range overflow", t.name)))?;
    }

    Ok(GgufHeader {
        version,
        alignment,
        metadata: Arc::new(metadata),
        tensors,
        tensor_index,
        data_offset,
        header_bytes: r.pos,
    })
}

pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub metadata: Arc<HashMap<String, MetadataValue>>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_index: HashMap<String, usize>,
    /// Absolute file offset where tensor_data begins.
    pub data_offset: u64,
    /// DEV/TEST ONLY: full-file mmap. Production generate must use [`crate::gguf_stream::GgufCatalog`].
    mmap: Mmap,
}

impl GgufFile {
    /// Dev/test helper: maps the whole file. Prefer [`crate::gguf_stream::GgufCatalog`] for generate.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    pub fn parse(mmap: Mmap) -> Result<Self, GgufError> {
        let header = parse_header_bytes(&mmap)?;
        info!(
            "Opened GGUF v{} (mmap/dev): {} tensors, {} metadata keys, data@{}",
            header.version,
            header.tensors.len(),
            header.metadata.len(),
            header.data_offset
        );

        Ok(Self {
            version: header.version,
            alignment: header.alignment,
            metadata: header.metadata,
            tensors: header.tensors,
            tensor_index: header.tensor_index,
            data_offset: header.data_offset,
            mmap,
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

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        let idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))?;
        Ok(&self.tensors[*idx])
    }

    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8], GgufError> {
        let (start, nbytes) = self.tensor_range(info)?;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| GgufError::Truncated("tensor range overflow"))?;
        Ok(&self.mmap[start..end])
    }

    /// Absolute byte range of a tensor inside the mmap (for zero-copy views).
    pub fn tensor_range(&self, info: &TensorInfo) -> Result<(usize, usize), GgufError> {
        let start_u64 = self
            .data_offset
            .checked_add(info.offset)
            .ok_or_else(|| GgufError::Truncated("tensor absolute offset overflow"))?;
        let start = usize::try_from(start_u64)
            .map_err(|_| GgufError::Truncated("tensor offset too large"))?;
        let nbytes = tensor_nbytes(info)?;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| GgufError::Truncated("tensor range overflow"))?;
        if end > self.mmap.len() {
            return Err(GgufError::Truncated("tensor data out of bounds"));
        }
        Ok((start, nbytes))
    }

    pub fn mmap_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Dequantize tensor to dense FP32 in ggml layout (ne0 contiguous).
    pub fn dequant_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let info = self.tensor(name)?;
        let bytes = self.tensor_bytes(info)?;
        dequantize(info, bytes)
    }
}

fn align_up_checked(v: u64, align: u64) -> Result<u64, GgufError> {
    if align == 0 {
        return Err(GgufError::Msg("alignment is zero".into()));
    }
    let rem = v % align;
    if rem == 0 {
        Ok(v)
    } else {
        v.checked_add(align - rem)
            .ok_or_else(|| GgufError::Msg("alignment overflow".into()))
    }
}

pub fn tensor_nbytes(info: &TensorInfo) -> Result<usize, GgufError> {
    let n_u64 = info
        .n_elements_checked()
        .ok_or_else(|| GgufError::Msg(format!("tensor {} element count overflow", info.name)))?;
    let n = usize::try_from(n_u64).map_err(|_| {
        GgufError::Msg(format!("tensor {} element count too large for this platform", info.name))
    })?;
    // Every branch uses checked arithmetic: a malformed tensor directory must
    // surface as an error, never as a wrapped size that later under-allocates.
    let bytes = match info.ggml_type {
        GgmlType::F32 => n.checked_mul(4),
        GgmlType::F16 | GgmlType::BF16 => n.checked_mul(2),
        GgmlType::F64 | GgmlType::I64 => n.checked_mul(8),
        GgmlType::I8 => Some(n),
        GgmlType::I16 => n.checked_mul(2),
        GgmlType::I32 => n.checked_mul(4),
        GgmlType::Q4_0 => n.div_ceil(32).checked_mul(18),
        GgmlType::Q4_1 => n.div_ceil(32).checked_mul(20),
        GgmlType::Q5_0 => n.div_ceil(32).checked_mul(22),
        GgmlType::Q5_1 => n.div_ceil(32).checked_mul(24),
        GgmlType::Q8_0 => n.div_ceil(32).checked_mul(34),
        GgmlType::Q8_1 => n.div_ceil(32).checked_mul(36),
        GgmlType::IQ4_NL => n.div_ceil(32).checked_mul(18),
        GgmlType::Q2_K => n.div_ceil(256).checked_mul(84),
        GgmlType::Q3_K => n.div_ceil(256).checked_mul(110),
        GgmlType::Q4_K => n.div_ceil(256).checked_mul(144),
        GgmlType::Q5_K => n.div_ceil(256).checked_mul(176),
        GgmlType::Q6_K => n.div_ceil(256).checked_mul(210),
        GgmlType::Q8_K => n.div_ceil(256).checked_mul(292),
        GgmlType::IQ2_XXS => n.div_ceil(256).checked_mul(66),
        GgmlType::IQ2_XS => n.div_ceil(256).checked_mul(74),
        GgmlType::IQ2_S => n.div_ceil(256).checked_mul(82),
        GgmlType::IQ3_XXS => n.div_ceil(256).checked_mul(98),
        GgmlType::IQ3_S => n.div_ceil(256).checked_mul(110),
        GgmlType::IQ1_S => n.div_ceil(256).checked_mul(50),
        GgmlType::IQ1_M => n.div_ceil(256).checked_mul(56),
        GgmlType::IQ4_XS => n.div_ceil(256).checked_mul(136),
        GgmlType::TQ1_0 => n.div_ceil(256).checked_mul(54),
        GgmlType::TQ2_0 => n.div_ceil(256).checked_mul(66),
        other => {
            return Err(GgufError::UnsupportedType(other, info.name.clone()));
        }
    };
    bytes.ok_or_else(|| GgufError::Msg(format!("tensor {} byte size overflow", info.name)))
}

pub fn dequantize(info: &TensorInfo, bytes: &[u8]) -> Result<Vec<f32>, GgufError> {
    let n = info
        .n_elements_checked()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| GgufError::Msg(format!("tensor {} element count overflow", info.name)))?;
    match info.ggml_type {
        GgmlType::F32 => {
            let need = n
                .checked_mul(4)
                .ok_or_else(|| GgufError::Msg("f32 tensor size overflow".into()))?;
            if bytes.len() < need {
                return Err(GgufError::Truncated("f32 tensor"));
            }
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                out[i] = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
            Ok(out)
        }
        GgmlType::F16 => {
            let need = n
                .checked_mul(2)
                .ok_or_else(|| GgufError::Msg("f16 tensor size overflow".into()))?;
            if bytes.len() < need {
                return Err(GgufError::Truncated("f16 tensor"));
            }
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                let h = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out[i] = f16_to_f32(h);
            }
            Ok(out)
        }
        GgmlType::Q4_0 => dequant_q4_0(bytes, n),
        GgmlType::Q4_1 => dequant_q4_1(bytes, n),
        GgmlType::Q2_K => crate::q2k::dequant_q2_k(bytes, n),
        GgmlType::Q3_K => crate::q3k::dequant_q3_k(bytes, n),
        GgmlType::Q4_K => crate::q4k::dequant_q4_k(bytes, n),
        GgmlType::Q5_0 => crate::q5::dequant_q5_0(bytes, n),
        GgmlType::Q5_1 => crate::q5::dequant_q5_1(bytes, n),
        GgmlType::Q5_K => crate::q5k::dequant_q5_k(bytes, n),
        GgmlType::Q6_K => crate::q6k::dequant_q6_k(bytes, n),
        GgmlType::IQ4_NL => crate::iq4::dequant_iq4_nl(bytes, n),
        GgmlType::IQ4_XS => crate::iq4::dequant_iq4_xs(bytes, n),
        GgmlType::IQ3_XXS => crate::iq3::dequant_iq3_xxs(bytes, n),
        GgmlType::IQ3_S => crate::iq3::dequant_iq3_s(bytes, n),
        GgmlType::IQ2_XXS => crate::iq2::dequant_iq2_xxs(bytes, n),
        GgmlType::IQ2_XS => crate::iq2::dequant_iq2_xs(bytes, n),
        GgmlType::IQ2_S => crate::iq2::dequant_iq2_s(bytes, n),
        GgmlType::Q8_0 => dequant_q8_0(bytes, n),
        GgmlType::Q8_1 => dequant_q8_1(bytes, n),
        GgmlType::Q8_K => dequant_q8_k(bytes, n),
        GgmlType::BF16 => dequant_bf16(bytes, n),
        GgmlType::F64 => dequant_f64(bytes, n),
        GgmlType::I8 => dequant_i8(bytes, n),
        GgmlType::I16 => dequant_i16(bytes, n),
        GgmlType::I32 => dequant_i32(bytes, n),
        GgmlType::I64 => dequant_i64(bytes, n),
        other => Err(GgufError::UnsupportedType(other, info.name.clone())),
    }
}

fn dequant_q4_0(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let blocks = (n + 31) / 32;
    if bytes.len() < blocks * 18 {
        return Err(GgufError::Truncated("q4_0 tensor"));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..blocks {
        let base = b * 18;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        let qs = &bytes[base + 2..base + 18];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i8 - 8;
            let x1 = (qs[j] >> 4) as i8 - 8;
            let i0 = b * 32 + j;
            let i1 = b * 32 + j + 16;
            if i0 < n {
                out[i0] = x0 as f32 * d;
            }
            if i1 < n {
                out[i1] = x1 as f32 * d;
            }
        }
    }
    Ok(out)
}

fn dequant_q4_1(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let blocks = (n + 31) / 32;
    if bytes.len() < blocks * 20 {
        return Err(GgufError::Truncated("q4_1 tensor"));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..blocks {
        let base = b * 20;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([bytes[base + 2], bytes[base + 3]]));
        let qs = &bytes[base + 4..base + 20];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as f32;
            let x1 = (qs[j] >> 4) as f32;
            let i0 = b * 32 + j;
            let i1 = b * 32 + j + 16;
            if i0 < n {
                out[i0] = x0 * d + m;
            }
            if i1 < n {
                out[i1] = x1 * d + m;
            }
        }
    }
    Ok(out)
}

fn dequant_q8_0(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let blocks = (n + 31) / 32;
    if bytes.len() < blocks * 34 {
        return Err(GgufError::Truncated("q8_0 tensor"));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..blocks {
        let base = b * 34;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        for j in 0..32 {
            let idx = b * 32 + j;
            if idx < n {
                out[idx] = (bytes[base + 2 + j] as i8) as f32 * d;
            }
        }
    }
    Ok(out)
}

/// Q8_1: `{half d; half s; int8 qs[32]}` (36 bytes / 32 elements). `s` is the
/// precomputed sum used by ggml's dot kernels; dequant only needs `d`.
fn dequant_q8_1(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let blocks = (n + 31) / 32;
    if bytes.len() < blocks * 36 {
        return Err(GgufError::Truncated("q8_1 tensor"));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..blocks {
        let base = b * 36;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        for j in 0..32 {
            let idx = b * 32 + j;
            if idx < n {
                out[idx] = (bytes[base + 4 + j] as i8) as f32 * d;
            }
        }
    }
    Ok(out)
}

/// Q8_K: `{float d; int8 qs[256]; int16 bsums[16]}` (292 bytes / 256 elements).
fn dequant_q8_k(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let blocks = n.div_ceil(256);
    if bytes.len() < blocks * 292 {
        return Err(GgufError::Truncated("q8_k tensor"));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..blocks {
        let base = b * 292;
        let d = f32::from_le_bytes([bytes[base], bytes[base + 1], bytes[base + 2], bytes[base + 3]]);
        for j in 0..256 {
            let idx = b * 256 + j;
            if idx < n {
                out[idx] = (bytes[base + 4 + j] as i8) as f32 * d;
            }
        }
    }
    Ok(out)
}

/// BF16 (bfloat16): high 16 bits of an f32. `float32 = bits << 16`.
fn dequant_bf16(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if bytes.len() < n * 2 {
        return Err(GgufError::Truncated("bf16 tensor"));
    }
    Ok((0..n)
        .map(|i| {
            let bits = (u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32) << 16;
            f32::from_bits(bits)
        })
        .collect())
}

fn dequant_f64(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if bytes.len() < n * 8 {
        return Err(GgufError::Truncated("f64 tensor"));
    }
    Ok((0..n)
        .map(|i| {
            let o = i * 8;
            f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap()) as f32
        })
        .collect())
}

fn dequant_i8(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if bytes.len() < n {
        return Err(GgufError::Truncated("i8 tensor"));
    }
    Ok(bytes[..n].iter().map(|&b| b as i8 as f32).collect())
}

fn dequant_i16(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if bytes.len() < n * 2 {
        return Err(GgufError::Truncated("i16 tensor"));
    }
    Ok((0..n)
        .map(|i| i16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()) as f32)
        .collect())
}

fn dequant_i32(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if bytes.len() < n * 4 {
        return Err(GgufError::Truncated("i32 tensor"));
    }
    Ok((0..n)
        .map(|i| i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as f32)
        .collect())
}

fn dequant_i64(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if bytes.len() < n * 8 {
        return Err(GgufError::Truncated("i64 tensor"));
    }
    Ok((0..n)
        .map(|i| i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()) as f32)
        .collect())
}

/// IEEE-754 binary16 → f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal
            let mut m = mant;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn need(&self, n: usize) -> Result<(), GgufError> {
        match self.pos.checked_add(n) {
            Some(end) if end <= self.data.len() => Ok(()),
            _ => Err(GgufError::Truncated("reader")),
        }
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        self.need(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        self.need(2)?;
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.u8()? as i8)
    }

    fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.u16()? as i16)
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.u32()? as i32)
    }

    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.u64()? as i64)
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.u8()? != 0)
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()? as usize;
        self.need(len)?;
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + len]).into_owned();
        self.pos += len;
        Ok(s)
    }

    fn metadata_value(&mut self) -> Result<MetadataValue, GgufError> {
        let ty = self.u32()?;
        self.read_value(ty, 0)
    }

    fn read_value(&mut self, ty: u32, depth: usize) -> Result<MetadataValue, GgufError> {
        const MAX_METADATA_DEPTH: usize = 32;
        if depth > MAX_METADATA_DEPTH {
            return Err(GgufError::Truncated("metadata array nesting too deep"));
        }
        Ok(match ty {
            0 => MetadataValue::U8(self.u8()?),
            1 => MetadataValue::I8(self.i8()?),
            2 => MetadataValue::U16(self.u16()?),
            3 => MetadataValue::I16(self.i16()?),
            4 => MetadataValue::U32(self.u32()?),
            5 => MetadataValue::I32(self.i32()?),
            6 => MetadataValue::F32(self.f32()?),
            7 => MetadataValue::Bool(self.bool()?),
            8 => MetadataValue::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let len = self.u64()?;
                // Each element occupies at least one byte; a length beyond the
                // remaining buffer is malformed and would otherwise let a small
                // file request an enormous allocation.
                let remaining = self.data.len().saturating_sub(self.pos);
                if len > remaining as u64 {
                    return Err(GgufError::Truncated("metadata array length"));
                }
                let len = len as usize;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.read_value(elem_ty, depth + 1)?);
                }
                MetadataValue::Array(items)
            }
            10 => MetadataValue::U64(self.u64()?),
            11 => MetadataValue::I64(self.i64()?),
            12 => MetadataValue::F64(self.f64()?),
            other => {
                return Err(GgufError::Msg(format!("unknown metadata type {other}")));
            }
        })
    }
}

/// Write a minimal GGUF (F32 tensors + string metadata) for unit tests.
pub fn write_minimal_gguf(
    path: &Path,
    metadata: &[(&str, MetadataValue)],
    tensors: &[(&str, Vec<u64>, Vec<f32>)],
) -> Result<(), GgufError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (k, v) in metadata {
        write_string(&mut buf, k);
        write_metadata_value(&mut buf, v);
    }

    let mut offsets = Vec::new();
    let mut data = Vec::new();
    let align = 32u64;
    for (_name, _dims, values) in tensors {
        while (data.len() as u64) % align != 0 {
            data.push(0);
        }
        offsets.push(data.len() as u64);
        for x in values {
            data.extend_from_slice(&x.to_le_bytes());
        }
    }

    for (i, (name, dims, _)) in tensors.iter().enumerate() {
        write_string(&mut buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims.iter() {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&offsets[i].to_le_bytes());
    }

    while (buf.len() as u64) % align != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&data);
    std::fs::write(path, buf)?;
    Ok(())
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_metadata_value(buf: &mut Vec<u8>, v: &MetadataValue) {
    match v {
        MetadataValue::U8(x) => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.push(*x);
        }
        MetadataValue::I8(x) => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.push(*x as u8);
        }
        MetadataValue::U16(x) => {
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::I16(x) => {
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::U32(x) => {
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::I32(x) => {
            buf.extend_from_slice(&5u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::F32(x) => {
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::Bool(x) => {
            buf.extend_from_slice(&7u32.to_le_bytes());
            buf.push(*x as u8);
        }
        MetadataValue::String(s) => {
            buf.extend_from_slice(&8u32.to_le_bytes());
            write_string(buf, s);
        }
        MetadataValue::U64(x) => {
            buf.extend_from_slice(&10u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::I64(x) => {
            buf.extend_from_slice(&11u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::F64(x) => {
            buf.extend_from_slice(&12u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::Array(items) => {
            buf.extend_from_slice(&9u32.to_le_bytes());
            let elem_ty = match items.first() {
                Some(MetadataValue::String(_)) => 8u32,
                Some(MetadataValue::F32(_)) => 6,
                Some(MetadataValue::U32(_)) => 4,
                Some(MetadataValue::I32(_)) => 5,
                Some(MetadataValue::U16(_)) => 2,
                Some(MetadataValue::Bool(_)) => 7,
                Some(MetadataValue::U64(_)) => 10,
                _ => 8,
            };
            buf.extend_from_slice(&elem_ty.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                match it {
                    MetadataValue::String(s) => write_string(buf, s),
                    MetadataValue::F32(x) => buf.extend_from_slice(&x.to_le_bytes()),
                    MetadataValue::U32(x) => buf.extend_from_slice(&x.to_le_bytes()),
                    MetadataValue::I32(x) => buf.extend_from_slice(&x.to_le_bytes()),
                    MetadataValue::U16(x) => buf.extend_from_slice(&x.to_le_bytes()),
                    MetadataValue::Bool(x) => buf.push(*x as u8),
                    MetadataValue::U64(x) => buf.extend_from_slice(&x.to_le_bytes()),
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn roundtrip_minimal_gguf() {
        let path = temp_dir().join("hayai_test_minimal.gguf");
        write_minimal_gguf(
            &path,
            &[
                ("general.architecture", MetadataValue::String("llama".into())),
                ("llama.embedding_length", MetadataValue::U32(8)),
            ],
            &[("token_embd.weight", vec![8, 4], vec![1.0f32; 32])],
        )
        .unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.meta_str("general.architecture"), Some("llama"));
        assert_eq!(gguf.meta_u32("llama.embedding_length"), Some(8));
        let t = gguf.dequant_f32("token_embd.weight").unwrap();
        assert_eq!(t.len(), 32);
        assert!((t[0] - 1.0).abs() < 1e-6);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn f16_conversion_smoke() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-3); // 1.0
        assert!((f16_to_f32(0x0000)).abs() < 1e-6);
    }

    #[test]
    fn zero_alignment_is_rejected() {
        let path = temp_dir().join("hayai_test_zero_align.gguf");
        write_minimal_gguf(
            &path,
            &[("general.alignment", MetadataValue::U32(0))],
            &[("t.weight", vec![4], vec![0.0f32; 4])],
        )
        .unwrap();
        match GgufFile::open(&path) {
            Ok(_) => panic!("zero general.alignment must be rejected"),
            Err(e) => assert!(format!("{e}").contains("alignment"), "err={e}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn duplicate_tensor_names_are_rejected() {
        let path = temp_dir().join("hayai_test_dup_tensor.gguf");
        write_minimal_gguf(
            &path,
            &[],
            &[
                ("t.weight", vec![4], vec![0.0f32; 4]),
                ("t.weight", vec![4], vec![0.0f32; 4]),
            ],
        )
        .unwrap();
        match GgufFile::open(&path) {
            Ok(_) => panic!("duplicate tensor names must be rejected"),
            Err(e) => assert!(format!("{e}").contains("duplicate"), "err={e}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn tensor_nbytes_overflow_is_rejected() {
        let info = TensorInfo {
            name: "huge".into(),
            dims: vec![u64::MAX, 2],
            ggml_type: GgmlType::F32,
            type_id: 0,
            offset: 0,
        };
        assert!(tensor_nbytes(&info).is_err());
    }
}
