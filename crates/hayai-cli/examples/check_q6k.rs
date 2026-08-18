fn main() {
    use hayai_model::GgufCatalog;
    let mut cat = GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    let emb = cat.load_quant_matrix("token_embd.weight").unwrap();
    let qkv = cat.load_quant_matrix("blk.0.attn_qkv.weight").unwrap();
    println!("emb type={:?} {}x{}", emb.ggml_type, emb.nrows, emb.ncols);
    println!("qkv type={:?} {}x{}", qkv.ggml_type, qkv.nrows, qkv.ncols);

    // random-ish input
    let x: Vec<f32> = (0..emb.ncols).map(|i| ((i * 17 + 3) % 100) as f32 / 100.0 - 0.5).collect();
    let mut y_gemv = vec![0.0f32; 8]; // only first 8 rows for speed via manual
    // full gemv too heavy for emb (248k rows) — check first 64 rows manually
    let rows = 64usize;
    let mut y_ref = vec![0.0f32; rows];
    let mut y_fast = vec![0.0f32; emb.nrows];
    emb.gemv(&x, &mut y_fast).unwrap();
    for r in 0..rows {
        let mut row = vec![0.0f32; emb.ncols];
        emb.extract_row(r, &mut row).unwrap();
        y_ref[r] = row.iter().zip(x.iter()).map(|(a,b)| a*b).sum();
    }
    let mut max_diff = 0.0f32;
    for r in 0..rows {
        max_diff = max_diff.max((y_ref[r] - y_fast[r]).abs());
    }
    println!("emb gemv vs extract first {rows} max_diff={max_diff}");

    let x2: Vec<f32> = (0..qkv.ncols).map(|i| ((i * 13 + 7) % 100) as f32 / 100.0 - 0.5).collect();
    let mut yq = vec![0.0f32; qkv.nrows];
    qkv.gemv(&x2, &mut yq).unwrap();
    let mut max_diff2 = 0.0f32;
    for r in 0..64 {
        let mut row = vec![0.0f32; qkv.ncols];
        qkv.extract_row(r, &mut row).unwrap();
        let s: f32 = row.iter().zip(x2.iter()).map(|(a,b)| a*b).sum();
        max_diff2 = max_diff2.max((s - yq[r]).abs());
    }
    println!("qkv gemv vs extract first 64 max_diff={max_diff2}");
    let _ = y_gemv;
}
