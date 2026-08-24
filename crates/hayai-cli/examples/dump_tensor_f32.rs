//! Vuelca un tensor del GGUF a F32 plano (fila-mayor `[d_out, d_in]` en la
//! convención `saor`), para la orquestación Python (Fase 5/7: profesor real).
//!
//! Uso: cargo run --release --example dump_tensor_f32 -- <gguf> <tensor> <out.bin>

use hayai_model::GgufFile;
use std::fs;

fn main() -> anyhow::Result<()> {
    let gguf = std::env::args().nth(1).expect("uso: dump_tensor_f32 <gguf> <tensor> <out.bin>");
    let name = std::env::args().nth(2).expect("tensor");
    let out = std::env::args().nth(3).expect("out.bin");

    let g = GgufFile::open(&gguf)?;
    let info = g.tensor(&name)?;
    let f32v = g.dequant_f32(&name)?;
    let dims: Vec<u64> = info.dims.clone();
    assert_eq!(f32v.len(), info.n_elements() as usize);

    let mut buf = Vec::with_capacity(f32v.len() * 4);
    for x in &f32v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    let nbytes = buf.len();
    fs::write(&out, buf)?;
    println!(
        "tensor={name} dims={dims:?} elems={} bytes={} -> {out}",
        f32v.len(),
        nbytes
    );
    Ok(())
}
