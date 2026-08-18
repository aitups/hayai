fn main() {
    let mut cat = hayai_model::GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    let m = cat.load_quant_matrix("blk.0.ffn_gate.weight").unwrap();
    println!("gate type={:?} {}x{}", m.ggml_type, m.nrows, m.ncols);
    let x: Vec<f32> = (0..m.ncols).map(|i| ((i*19+5)%97) as f32 / 97.0 - 0.5).collect();
    let mut y = vec![0.0f32; m.nrows];
    m.gemv(&x, &mut y).unwrap();
    // compare first 32 rows via dequant of whole? use second gemv path - just print stats
    let mut ms = 0.0f32; for &v in &y { ms += v*v; }
    println!("y_rms={:.6} y0..4={:?}", (ms/y.len() as f32).sqrt(), &y[..4]);
}
