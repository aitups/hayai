use crate::gguf_types::{GgmlType, GgufError, MetadataValue, TensorInfo};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use tracing::info;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" LE

/// Parsed GGUF header + tensor directory (no weight payloads).
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, MetadataValue>,
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
    if version != 2 && version != 3 {
        return Err(GgufError::BadVersion(version));
    }
    let tensor_count = r.u64()? as usize;
    let kv_count = r.u64()? as usize;

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
            dims.push(r.u64()?);
        }
        let type_id = r.u32()?;
        let offset = r.u64()?;
        let ggml_type = GgmlType::from_u32(type_id);
        tensor_index.insert(name.clone(), i);
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            type_id,
            offset,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32);
    let data_offset = align_up(r.pos as u64, alignment);

    Ok(GgufHeader {
        version,
        alignment,
        metadata,
        tensors,
        tensor_index,
        data_offset,
        header_bytes: r.pos,
    })
}

pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, MetadataValue>,
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
        let start = (self.data_offset + info.offset) as usize;
        let nbytes = tensor_nbytes(info)?;
        let end = start + nbytes;
        if end > self.mmap.len() {
            return Err(GgufError::Truncated("tensor data out of bounds"));
        }
        Ok(&self.mmap[start..end])
    }

    /// Absolute byte range of a tensor inside the mmap (for zero-copy views).
    pub fn tensor_range(&self, info: &TensorInfo) -> Result<(usize, usize), GgufError> {
        let start = (self.data_offset + info.offset) as usize;
        let nbytes = tensor_nbytes(info)?;
        let end = start + nbytes;
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

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) / align * align
}

pub fn tensor_nbytes(info: &TensorInfo) -> Result<usize, GgufError> {
    let n = info.n_elements() as usize;
    Ok(match info.ggml_type {
        GgmlType::F32 => n * 4,
        GgmlType::F16 | GgmlType::BF16 => n * 2,
        GgmlType::F64 => n * 8,
        GgmlType::I8 => n,
        GgmlType::I16 => n * 2,
        GgmlType::I32 => n * 4,
        GgmlType::I64 => n * 8,
        GgmlType::Q4_0 => {
            let blocks = (n + 31) / 32;
            blocks * 18
        }
        GgmlType::Q4_1 => {
            let blocks = (n + 31) / 32;
            blocks * 20
        }
        GgmlType::Q2_K => {
            let blocks = (n + 255) / 256;
            blocks * 84
        }
        GgmlType::Q3_K => {
            let blocks = (n + 255) / 256;
            blocks * 110
        }
        GgmlType::Q4_K => {
            let blocks = (n + 255) / 256;
            blocks * 144
        }
        GgmlType::Q5_0 => {
            let blocks = (n + 31) / 32;
            blocks * 22
        }
        GgmlType::Q5_1 => {
            let blocks = (n + 31) / 32;
            blocks * 24
        }
        GgmlType::Q5_K => {
            let blocks = (n + 255) / 256;
            blocks * 176
        }
        GgmlType::Q6_K => {
            let blocks = (n + 255) / 256;
            blocks * 210
        }
        GgmlType::Q8_0 => {
            let blocks = (n + 31) / 32;
            blocks * 34
        }
        GgmlType::IQ4_NL => {
            let blocks = (n + 31) / 32;
            blocks * 18
        }
        GgmlType::IQ2_XXS => ((n + 255) / 256) * 66,
        GgmlType::IQ2_XS => ((n + 255) / 256) * 74,
        GgmlType::IQ2_S => ((n + 255) / 256) * 82,
        GgmlType::IQ3_XXS => ((n + 255) / 256) * 98,
        GgmlType::IQ3_S => ((n + 255) / 256) * 110,
        GgmlType::IQ1_S => ((n + 255) / 256) * 50,
        GgmlType::IQ1_M => ((n + 255) / 256) * 56,
        GgmlType::IQ4_XS => ((n + 255) / 256) * 136,
        GgmlType::TQ1_0 => ((n + 255) / 256) * 54,
        GgmlType::TQ2_0 => ((n + 255) / 256) * 66,
        GgmlType::Q8_K => ((n + 255) / 256) * 256,
        other => {
            return Err(GgufError::UnsupportedType(other, info.name.clone()));
        }
    })
}

pub fn dequantize(info: &TensorInfo, bytes: &[u8]) -> Result<Vec<f32>, GgufError> {
    let n = info.n_elements() as usize;
    match info.ggml_type {
        GgmlType::F32 => {
            if bytes.len() < n * 4 {
                return Err(GgufError::Truncated("f32 tensor"));
            }
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                out[i] = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
            Ok(out)
        }
        GgmlType::F16 => {
            if bytes.len() < n * 2 {
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
        if self.pos + n > self.data.len() {
            Err(GgufError::Truncated("reader"))
        } else {
            Ok(())
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
        self.read_value(ty)
    }

    fn read_value(&mut self, ty: u32) -> Result<MetadataValue, GgufError> {
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
                let len = self.u64()? as usize;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.read_value(elem_ty)?);
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
        MetadataValue::U32(x) => {
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::F32(x) => {
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        MetadataValue::String(s) => {
            buf.extend_from_slice(&8u32.to_le_bytes());
            write_string(buf, s);
        }
        MetadataValue::Array(items) => {
            buf.extend_from_slice(&9u32.to_le_bytes());
            // assume string array
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                if let MetadataValue::String(s) = it {
                    write_string(buf, s);
                }
            }
        }
        _ => {
            buf.extend_from_slice(&8u32.to_le_bytes());
            write_string(buf, "");
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
}
