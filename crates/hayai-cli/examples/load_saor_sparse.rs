//! Validación Fase 7: carga un GGUF disperso de `saor` con hayai y verifica el
//! bloque (metadatos + bit-tensor + pesos) y el SpMM (CSR vs denso).
//!
//! Uso: cargo run --release --example load_saor_sparse -- <archivo.gguf>

use hayai_model::{
    load_sparse_dag, sparse_dag_to_csr, spmm_csr_cpu, spmm_dense_masked, GgufCatalog,
};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("uso: load_saor_sparse <archivo.gguf>");
    let mut cat = GgufCatalog::open(PathBuf::from(&path))?;
    let block = load_sparse_dag(&mut cat)?.expect("el GGUF debe ser un bloque disperso saor");

    println!("== Bloque disperso (hayai load_sparse_dag) ==");
    println!("d_in={} d_out={} tau={:.4}", block.d_in, block.d_out, block.tau);
    println!(
        "active={} sparsity={:.4} adj_bytes={} weights={} genome_len={}",
        block.active_connections(),
        block.sparsity(),
        block.adjacency.len(),
        block.weights.len(),
        block.genome.len(),
    );
    assert_eq!(block.adjacency.len() as u64, (block.d_in * block.d_out).div_ceil(8) as u64);

    // SpMM: CSR vs denso-enmascarado (referencia CPU) deben coincidir.
    let (row_ptr, col_idx, vals) =
        sparse_dag_to_csr(&block.adjacency, &block.weights, block.d_in, block.d_out);
    let batch = 16usize;
    let x: Vec<f32> = (0..batch * block.d_in)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let y_csr = spmm_csr_cpu(&x, &row_ptr, &col_idx, &vals, block.d_in, block.d_out);
    let y_dense = spmm_dense_masked(&x, &block.adjacency, &block.weights, block.d_in, block.d_out);
    assert_eq!(y_csr.len(), y_dense.len());
    let max_err = y_csr
        .iter()
        .zip(y_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("spmm_csr vs spmm_dense: max_err={max_err:e}");
    assert!(max_err < 1e-6, "CSR y denso deben coincidir");

    println!("== OK: bloque saor cargado y SpMM validado ==");
    Ok(())
}
