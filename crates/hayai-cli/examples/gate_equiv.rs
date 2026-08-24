//! Diagnóstico Fase 3: compara el GEMV cuantizado del gate original (Q5_0) con
//! el SpMM CSR del bloque denso embebido para la misma entrada.
//!
//!   cargo run --release --example gate_equiv -- <model.gguf> <embedded.gguf> <n>

use std::sync::Arc;

use hayai_model::{load_embedded_block, GgufFile, LlamaWeights, spmm_csr_cpu};

fn main() -> anyhow::Result<()> {
    let model = std::env::args().nth(1).expect("model gguf");
    let embedded = std::env::args().nth(2).expect("embedded gguf");
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let gguf = Arc::new(GgufFile::open(&model)?);
    let weights = LlamaWeights::load_arc(gguf)?;
    let gate = &weights.layers[0].gate;
    let block = load_embedded_block(
        &GgufFile::open(&embedded)?,
        "blk.0.ffn_gate",
    )?
    .expect("bloque disperso");

    let (row_ptr, col_idx, vals) =
        hayai_model::sparse_dag_to_csr(&block.adjacency, &block.weights, block.d_in, block.d_out);

    let mut rng = 0x12345678u64;
    let mut x = vec![0.0f32; gate.ncols];
    for v in x.iter_mut() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((rng >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
    let mut y_q = vec![0.0f32; gate.nrows];
    gate.gemv(&x, &mut y_q)?;
    let y_csr = spmm_csr_cpu(&x, &row_ptr, &col_idx, &vals, block.d_in, block.d_out);

    // Tercera vía: dequant_f32 del tensor original en el layout GGUF.
    let w_deq = weights.gguf.dequant_f32("blk.0.ffn_gate.weight")?;
    let mut y_deq = vec![0.0f32; gate.nrows];
    for j in 0..gate.nrows {
        let mut s = 0.0f32;
        for i in 0..gate.ncols {
            s += w_deq[j * gate.ncols + i] * x[i];
        }
        y_deq[j] = s;
    }

    let maxdiff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    println!(
        "gate ncols={} nrows={} | CSR nnz={}\n  |gemv - csr| = {:.6}\n  |gemv - deq| = {:.6}\n  |deq - csr|  = {:.6}",
        gate.ncols,
        gate.nrows,
        row_ptr.last().copied().unwrap_or(0),
        maxdiff(&y_q, &y_csr),
        maxdiff(&y_q, &y_deq),
        maxdiff(&y_deq, &y_csr),
    );
    Ok(())
}
