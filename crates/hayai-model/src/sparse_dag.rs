//! FFN disperso (DAG irregular) — GGUF de `saor`.
//!
//! Implementa las secciones §3.A4 / §3.D de `pr_soporte_gguf_disperso_v3.md`:
//! carga del bloque disperso desde el catálogo (metadatos `saor.*` + tensores
//! `ffn_dag_*`) y conversión del bit-tensor de adyacencia + pesos activos a CSR,
//! con referencias CPU (SpMM CSR y denso enmascarado) para validación.

use crate::gguf::GgufFile;
use crate::{GgufCatalog, GgufError, MetadataValue};

/// Clave de metadato: dimensión de entrada (`saor.d_in`, UINT64).
pub const META_D_IN: &str = "saor.d_in";
/// Clave de metadato: dimensión de salida (`saor.d_out`, UINT64).
pub const META_D_OUT: &str = "saor.d_out";
/// Clave de metadato: umbral de esparsidad (`saor.tau`, FLOAT32).
pub const META_TAU: &str = "saor.tau";
/// Clave de metadato: genoma CPPN aplanado (`saor.genome`, ARRAY(F32)).
pub const META_GENOME: &str = "saor.genome";
/// Clave de metadato: marca de bloque disperso (`saor.sparse`, BOOL).
pub const META_SPARSE: &str = "saor.sparse";

/// Tensor: bit-tensor de adyacencia (`ffn_dag_adjacency`, I8 bytes, LSB-first).
pub const TENSOR_ADJACENCY: &str = "ffn_dag_adjacency";
/// Tensor: pesos activos del DAG (`ffn_dag_weights`, F32, orden i-mayor).
pub const TENSOR_WEIGHTS: &str = "ffn_dag_weights";

/// Bloque FFN disperso tal y como viene en el GGUF de `saor`.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseDagBlock {
    /// Dimensión de entrada del bloque.
    pub d_in: usize,
    /// Dimensión de salida del bloque.
    pub d_out: usize,
    /// Umbral de esparsidad dinámico.
    pub tau: f32,
    /// Genoma CPPN aplanado (referencia para regenerar/re-evolucionar).
    pub genome: Vec<f32>,
    /// Bit-tensor `ffn_dag_adjacency` (LSB-first, `conn = i*d_out + j`).
    pub adjacency: Vec<u8>,
    /// Pesos activos del DAG (orden de escaneo `(i, j)`, solo conexiones vivas).
    pub weights: Vec<f32>,
}

impl SparseDagBlock {
    /// Conexiones activas (popcount sobre el bit-tensor).
    pub fn active_connections(&self) -> usize {
        self.adjacency.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Esparcidad del candidato = `1 - nnz / (d_in * d_out)`.
    pub fn sparsity(&self) -> f32 {
        let total = (self.d_in * self.d_out).max(1);
        1.0 - self.active_connections() as f32 / total as f32
    }
}

/// Carga el bloque disperso de un `GgufCatalog` (metadatos `saor.*` + tensores
/// `ffn_dag_adjacency` / `ffn_dag_weights`). Devuelve `Ok(None)` si el GGUF no
/// está marcado como bloque disperso de `saor` (`saor.sparse` ausente/false).
///
/// Mitiga el bug de type-ID del productor (R1 del doc): la adyacencia se lee
/// como `dims[0]` bytes crudos (`read_raw_at`) en vez de confiar en
/// `tensor_nbytes`, que devolvería un tamaño erróneo si el `ggml_type` se
/// emite como 16 (`IQ2_XXS`) en lugar de 24 (`I8`).
pub fn load_sparse_dag(cat: &mut GgufCatalog) -> Result<Option<SparseDagBlock>, GgufError> {
    let meta = &cat.metadata;
    let is_sparse = matches!(meta.get(META_SPARSE), Some(MetadataValue::Bool(true)));
    if !is_sparse {
        return Ok(None);
    }
    let d_in = meta
        .get(META_D_IN)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| GgufError::MissingKey(META_D_IN.to_string()))? as usize;
    let d_out = meta
        .get(META_D_OUT)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| GgufError::MissingKey(META_D_OUT.to_string()))? as usize;
    let tau = meta
        .get(META_TAU)
        .and_then(|v| v.as_f32())
        .ok_or_else(|| GgufError::MissingKey(META_TAU.to_string()))?;
    let genome = meta
        .get(META_GENOME)
        .and_then(|v| v.as_f32_array())
        .unwrap_or_default();

    let adj_info = cat.tensor(TENSOR_ADJACENCY)?.clone();
    let w_info = cat.tensor(TENSOR_WEIGHTS)?.clone();

    // R1: la adyacencia es un bit-tensor de bytes; leer `dims[0]` bytes crudos.
    let adj_len = adj_info.dims.first().copied().unwrap_or(0) as usize;
    if adj_len as u64 > cat.file_size() {
        return Err(GgufError::Msg(
            "sparse DAG adjacency size exceeds the file size".into(),
        ));
    }
    let mut adjacency = vec![0u8; adj_len];
    let abs = cat.tensor_abs_offset(&adj_info);
    cat.read_raw_at(abs, &mut adjacency)?;

    // Pesos activos en F32 (tipo correcto → `read_tensor_into` es seguro).
    let w_nbytes = crate::tensor_nbytes(&w_info)?;
    if w_nbytes % 4 != 0 {
        return Err(GgufError::Msg(
            "sparse DAG weights payload is not a multiple of 4 bytes".into(),
        ));
    }
    let mut w_buf = vec![0u8; w_nbytes];
    cat.read_tensor_into(TENSOR_WEIGHTS, &mut w_buf)?;
    let mut weights = vec![0.0f32; w_buf.len() / 4];
    for (i, chunk) in w_buf.chunks_exact(4).enumerate() {
        weights[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    validate_sparse_block(&adjacency, &weights, d_in, d_out)?;

    Ok(Some(SparseDagBlock {
        d_in,
        d_out,
        tau,
        genome,
        adjacency,
        weights,
    }))
}

/// Validate that a sparse block's bit-tensor and weight vector agree with the
/// declared `d_in × d_out` shape. Shared by the loaders so `sparse_dag_to_csr`
/// can rely on consistent input.
fn validate_sparse_block(
    adjacency: &[u8],
    weights: &[f32],
    d_in: usize,
    d_out: usize,
) -> Result<(), GgufError> {
    let total = d_in
        .checked_mul(d_out)
        .ok_or_else(|| GgufError::Msg("sparse DAG d_in*d_out overflow".into()))?;
    if total > adjacency.len().saturating_mul(8) {
        return Err(GgufError::Msg(format!(
            "sparse DAG adjacency has {} bytes, too small for {d_in}x{d_out}",
            adjacency.len()
        )));
    }
    let active: usize = adjacency.iter().map(|b| b.count_ones() as usize).sum();
    if active != weights.len() {
        return Err(GgufError::Msg(format!(
            "sparse DAG weight count {} != {active} active connections",
            weights.len()
        )));
    }
    Ok(())
}

/// Carga un bloque disperso **embebido** en un GGUF completo (formato D16 de
/// `saor`): para la base `blk.0.ffn_gate` lee los tensores
/// `blk.0.ffn_gate.ffn_dag_adjacency` / `ffn_dag_weights` + los metadatos
/// `saor.blk.0.ffn_gate.{d_in,d_out,tau}`. Devuelve `Ok(None)` si el tensor
/// base no está marcado como disperso (ausencia del tensor de adyacencia).
pub fn load_embedded_block(
    gguf: &GgufFile,
    base: &str,
) -> Result<Option<SparseDagBlock>, GgufError> {
    let adj_name = format!("{base}.{TENSOR_ADJACENCY}");
    let w_name = format!("{base}.{TENSOR_WEIGHTS}");
    if gguf.tensor(&adj_name).is_err() {
        return Ok(None);
    }
    let adj_info = gguf.tensor(&adj_name)?;
    let w_info = gguf.tensor(&w_name)?;
    let adj_len = adj_info.dims.first().copied().unwrap_or(0) as usize;
    // The producer may misreport the adjacency type (R1), so `tensor_nbytes` can
    // be smaller than `dims[0]`. Read the raw `dims[0]` bytes from the mmap and
    // bounds-check the range instead of slicing the reported tensor extent.
    let (adj_start, _) = gguf.tensor_range(adj_info)?;
    let adj_end = adj_start
        .checked_add(adj_len)
        .ok_or_else(|| GgufError::Msg("sparse adjacency range overflow".into()))?;
    let adjacency = gguf
        .mmap_bytes()
        .get(adj_start..adj_end)
        .ok_or_else(|| GgufError::Truncated("sparse adjacency out of bounds"))?
        .to_vec();

    let w_bytes = gguf.tensor_bytes(w_info)?;
    let mut weights = vec![0.0f32; w_bytes.len() / 4];
    for (i, chunk) in w_bytes.chunks_exact(4).enumerate() {
        weights[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }

    let m = |k: &str| gguf.meta_u32(&format!("saor.{base}.{k}"));
    let d_in = m("d_in").unwrap_or(0) as usize;
    let d_out = m("d_out").unwrap_or(0) as usize;
    validate_sparse_block(&adjacency, &weights, d_in, d_out)?;
    Ok(Some(SparseDagBlock {
        d_in,
        d_out,
        tau: gguf.meta_f32(&format!("saor.{base}.tau")).unwrap_or(0.0),
        genome: Vec::new(),
        adjacency,
        weights,
    }))
}

/// Convierte `(adjacency, weights)` a CSR. `conn = i*d_out + j` es el índice de
/// conexión en el bit-tensor (LSB-first); `weights` está en orden i-mayor (solo
/// conexiones vivas). Filas del CSR = salidas `j`, columnas = entradas `i`.
///
/// Espejo de `saor_domain::topology::Topology::to_csr`.
///
/// # Panics
/// Panics if the inputs are inconsistent (adjacency too small or weight count
/// mismatched). Use [`try_sparse_dag_to_csr`] for untrusted input.
pub fn sparse_dag_to_csr(
    adjacency: &[u8],
    weights: &[f32],
    d_in: usize,
    d_out: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    try_sparse_dag_to_csr(adjacency, weights, d_in, d_out)
        .expect("sparse_dag_to_csr: inconsistent sparse block (use try_sparse_dag_to_csr)")
}

/// Fallible CSR conversion: validates the bit-tensor length and the weight count
/// against `d_in × d_out` before indexing anything, so a malformed GGUF yields a
/// clean error instead of an out-of-bounds panic.
pub fn try_sparse_dag_to_csr(
    adjacency: &[u8],
    weights: &[f32],
    d_in: usize,
    d_out: usize,
) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>), GgufError> {
    let total = d_in
        .checked_mul(d_out)
        .ok_or_else(|| GgufError::Msg("sparse DAG d_in*d_out overflow".into()))?;
    validate_sparse_block(adjacency, weights, d_in, d_out)?;
    // Peso por conexión `conn`, para iterar en orden j-mayor sin desordenar valores.
    let mut weight_by_conn = vec![0.0f32; total];
    let mut w_idx = 0usize;
    for i in 0..d_in {
        for j in 0..d_out {
            let conn = i * d_out + j;
            if (adjacency[conn / 8] & (1 << (conn % 8))) != 0 {
                weight_by_conn[conn] = weights[w_idx];
                w_idx += 1;
            }
        }
    }
    let mut row_ptr = vec![0i32; d_out + 1];
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    for j in 0..d_out {
        for i in 0..d_in {
            let conn = i * d_out + j;
            if (adjacency[conn / 8] & (1 << (conn % 8))) != 0 {
                col_idx.push(i as i32);
                vals.push(weight_by_conn[conn]);
            }
        }
        row_ptr[j + 1] = col_idx.len() as i32;
    }
    Ok((row_ptr, col_idx, vals))
}

/// SpMM CSR de referencia (CPU): `Y[b][j] = sum_k X[b][col[k]] * val[k]`.
pub fn spmm_csr_cpu(
    x: &[f32],
    row_ptr: &[i32],
    col_idx: &[i32],
    vals: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let batch = x.len().checked_div(d_in).unwrap_or(0);
    let mut y = vec![0.0f32; batch * d_out];
    for b in 0..batch {
        for j in 0..d_out {
            let mut acc = 0.0f32;
            for k in row_ptr[j] as usize..row_ptr[j + 1] as usize {
                acc += x[b * d_in + col_idx[k] as usize] * vals[k];
            }
            y[b * d_out + j] = acc;
        }
    }
    y
}

/// SpMM denso-enmascarado de referencia: materializa la matriz `d_out × d_in`
/// (ceros donde no hay conexión) **solo** en CPU y para validación — nunca en el
/// runtime de producción (que consume CSR, sin densificar).
pub fn spmm_dense_masked(
    x: &[f32],
    adjacency: &[u8],
    weights: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let batch = x.len().checked_div(d_in).unwrap_or(0);
    let total = d_in * d_out;
    let mut dense = vec![0.0f32; d_out * d_in];
    let mut w_idx = 0usize;
    for i in 0..d_in {
        for j in 0..d_out {
            let conn = i * d_out + j;
            if conn < total && (adjacency[conn / 8] & (1 << (conn % 8))) != 0 {
                dense[j * d_in + i] = weights[w_idx];
                w_idx += 1;
            }
        }
    }
    let mut y = vec![0.0f32; batch * d_out];
    for b in 0..batch {
        for j in 0..d_out {
            let mut acc = 0.0f32;
            for i in 0..d_in {
                acc += x[b * d_in + i] * dense[j * d_in + i];
            }
            y[b * d_out + j] = acc;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::write_minimal_gguf;
    use std::path::Path;

    fn sample_block() -> SparseDagBlock {
        // d_in=8, d_out=4 -> 32 conexiones; 6 activas (espejo de saor).
        SparseDagBlock {
            d_in: 8,
            d_out: 4,
            tau: 0.42,
            genome: (0..512).map(|i| i as f32 * 0.001).collect(),
            adjacency: vec![0b0101_0101u8, 0b0000_0011u8, 0u8, 0u8],
            weights: vec![1.0, -2.0, 3.5, 0.25, -0.5, 7.0],
        }
    }

    // ── Test-only writer: espejo del productor `saor-streamer` (GGUF v3). ─────
    fn wstr(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    fn align(buf: &mut Vec<u8>, a: u64) {
        while !(buf.len() as u64).is_multiple_of(a) {
            buf.push(0);
        }
    }

    fn write_saor_sparse_gguf(path: &Path, block: &SparseDagBlock) -> Result<(), String> {
        const ALIGN: u64 = 32;
        // ── tensor data section ──
        let mut data = Vec::new();
        align(&mut data, ALIGN);
        let adj_off = data.len() as u64;
        data.extend_from_slice(&block.adjacency);
        align(&mut data, ALIGN);
        let w_off = data.len() as u64;
        for w in &block.weights {
            data.extend_from_slice(&w.to_le_bytes());
        }

        // ── header ──
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&5u64.to_le_bytes()); // kv_count

        // Metadatos `saor.*` (UINT64=10, FLOAT32=6, BOOL=7, ARRAY=9).
        wstr(&mut buf, META_D_IN);
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&(block.d_in as u64).to_le_bytes());
        wstr(&mut buf, META_D_OUT);
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&(block.d_out as u64).to_le_bytes());
        wstr(&mut buf, META_TAU);
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&block.tau.to_le_bytes());
        wstr(&mut buf, META_SPARSE);
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(1u8);
        wstr(&mut buf, META_GENOME);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes()); // elemento FLOAT32
        buf.extend_from_slice(&(block.genome.len() as u64).to_le_bytes());
        for g in &block.genome {
            buf.extend_from_slice(&g.to_le_bytes());
        }

        // Directorio de tensores: `ffn_dag_adjacency` (GGML I8=24) + `ffn_dag_weights` (F32=0).
        wstr(&mut buf, TENSOR_ADJACENCY);
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&(block.adjacency.len() as u64).to_le_bytes());
        buf.extend_from_slice(&24u32.to_le_bytes()); // GGML_TYPE_I8
        buf.extend_from_slice(&adj_off.to_le_bytes());
        wstr(&mut buf, TENSOR_WEIGHTS);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(block.weights.len() as u64).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
        buf.extend_from_slice(&w_off.to_le_bytes());

        align(&mut buf, ALIGN);
        buf.extend_from_slice(&data);
        std::fs::write(path, buf).map_err(|e| e.to_string())
    }

    #[test]
    fn roundtrip_loads_sparse_block() {
        let path = std::env::temp_dir().join("hayai_saor_sparse_roundtrip.gguf");
        let block = sample_block();
        write_saor_sparse_gguf(&path, &block).expect("write");
        let mut cat = GgufCatalog::open(&path).expect("open");
        let loaded = load_sparse_dag(&mut cat).expect("load").expect("is sparse");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded, block);
        assert_eq!(loaded.active_connections(), 6);
    }

    #[test]
    fn non_sparse_gguf_returns_none() {
        let path = std::env::temp_dir().join("hayai_non_sparse.gguf");
        write_minimal_gguf(
            &path,
            &[("general.architecture", crate::MetadataValue::String("llama".into()))],
            &[("token_embd.weight", vec![4, 2], vec![1.0f32; 8])],
        )
        .unwrap();
        let mut cat = GgufCatalog::open(&path).unwrap();
        let r = load_sparse_dag(&mut cat).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(r.is_none());
    }

    #[test]
    fn try_to_csr_rejects_inconsistent_input() {
        // Bit-tensor too small for d_in*d_out.
        assert!(try_sparse_dag_to_csr(&[0u8], &[], 4, 4).is_err());
        // Active bit count != weight count.
        let adj = [0b0000_0001u8];
        assert!(try_sparse_dag_to_csr(&adj, &[1.0, 2.0], 4, 2).is_err());
        assert!(try_sparse_dag_to_csr(&adj, &[1.0], 4, 2).is_ok());
    }

    #[test]
    fn bit_tensor_lsb_first() {
        let bits = [0b1001_0110u8, 0b0000_1111u8];
        let unpack = |idx: usize| (bits[idx / 8] >> (idx % 8)) & 0x01 == 1;
        assert!(!unpack(0));
        assert!(unpack(1));
        assert!(unpack(2));
        assert!(!unpack(3));
        assert!(unpack(7));
        assert!(unpack(8));
        assert!(unpack(11));
        assert!(!unpack(12));
    }

    #[test]
    fn to_csr_preserves_weights_and_columns() {
        let block = sample_block();
        let (row_ptr, col_idx, vals) =
            sparse_dag_to_csr(&block.adjacency, &block.weights, block.d_in, block.d_out);
        // CSR esperado para el bloque de ejemplo (filas = salidas j).
        // activas: conn 0(i0,j0) 2(i0,j2) 4(i1,j0) 6(i1,j2) 8(i2,j0) 9(i2,j1)
        assert_eq!(row_ptr, vec![0, 3, 4, 6, 6]);
        assert_eq!(col_idx, vec![0, 1, 2, 2, 0, 1]);
        assert_eq!(vals, vec![1.0, 3.5, -0.5, 7.0, -2.0, 0.25]);
    }

    #[test]
    fn csr_matches_dense_masked() {
        let block = sample_block();
        let x: Vec<f32> = (0..block.d_in * 2).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let expected =
            spmm_dense_masked(&x, &block.adjacency, &block.weights, block.d_in, block.d_out);
        let (rp, ci, vv) =
            sparse_dag_to_csr(&block.adjacency, &block.weights, block.d_in, block.d_out);
        let got = spmm_csr_cpu(&x, &rp, &ci, &vv, block.d_in, block.d_out);
        assert_eq!(expected.len(), got.len());
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-6, "CSR vs dense mismatch: {a} != {b}");
        }
    }

    /// Test-only writer del formato **embebido** (D16): tensores por bloque
    /// `blk.0.ffn_gate.ffn_dag_*` + metadatos `saor.blk.0.ffn_gate.*`.
    fn write_embedded_gguf(path: &Path, base: &str, block: &SparseDagBlock) -> Result<(), String> {
        const ALIGN: u64 = 32;
        let adj_name = format!("{base}.{TENSOR_ADJACENCY}");
        let w_name = format!("{base}.{TENSOR_WEIGHTS}");

        let mut data = Vec::new();
        align(&mut data, ALIGN);
        let adj_off = data.len() as u64;
        data.extend_from_slice(&block.adjacency);
        align(&mut data, ALIGN);
        let w_off = data.len() as u64;
        for w in &block.weights {
            data.extend_from_slice(&w.to_le_bytes());
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&5u64.to_le_bytes()); // kv_count

        // Metadatos por bloque `saor.<base>.*`.
        for (k, vtype, value) in [
            (format!("saor.{base}.d_in"), 10u32, (block.d_in as u64).to_le_bytes().to_vec()),
            (format!("saor.{base}.d_out"), 10u32, (block.d_out as u64).to_le_bytes().to_vec()),
            (format!("saor.{base}.tau"), 6u32, block.tau.to_le_bytes().to_vec()),
            (format!("saor.{base}.sparse"), 7u32, vec![1u8]),
            (format!("saor.{base}.genome"), 9u32, {
                let mut v = Vec::new();
                v.extend_from_slice(&6u32.to_le_bytes());
                v.extend_from_slice(&(block.genome.len() as u64).to_le_bytes());
                for g in &block.genome {
                    v.extend_from_slice(&g.to_le_bytes());
                }
                v
            }),
        ] {
            wstr(&mut buf, &k);
            buf.extend_from_slice(&vtype.to_le_bytes());
            buf.extend_from_slice(&value);
        }

        wstr(&mut buf, &adj_name);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(block.adjacency.len() as u64).to_le_bytes());
        buf.extend_from_slice(&24u32.to_le_bytes()); // GGML I8
        buf.extend_from_slice(&adj_off.to_le_bytes());
        wstr(&mut buf, &w_name);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(block.weights.len() as u64).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&w_off.to_le_bytes());

        align(&mut buf, ALIGN);
        buf.extend_from_slice(&data);
        std::fs::write(path, buf).map_err(|e| e.to_string())
    }

    #[test]
    fn load_embedded_block_roundtrip() {
        let path = std::env::temp_dir().join("hayai_embed_block.gguf");
        let block = sample_block();
        write_embedded_gguf(&path, "blk.0.ffn_gate", &block).expect("write");
        let gguf = GgufFile::open(&path).expect("open");

        let loaded = load_embedded_block(&gguf, "blk.0.ffn_gate")
            .expect("load")
            .expect("is sparse");
        assert_eq!(loaded.d_in, block.d_in);
        assert_eq!(loaded.d_out, block.d_out);
        assert_eq!(loaded.tau, block.tau);
        assert_eq!(loaded.adjacency, block.adjacency);
        assert_eq!(loaded.weights, block.weights);

        // Un tensor no marcado como disperso devuelve None.
        let none = load_embedded_block(&gguf, "blk.1.ffn_up").expect("load");
        assert!(none.is_none());

        let _ = std::fs::remove_file(&path);
    }
}


