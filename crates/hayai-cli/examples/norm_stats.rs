fn main() {
    let mut cat = hayai_model::GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    for name in [
        "blk.3.attn_q_norm.weight",
        "blk.3.attn_k_norm.weight",
        "blk.3.attn_norm.weight",
        "blk.3.post_attention_norm.weight",
        "output_norm.weight",
        "blk.0.ssm_norm.weight",
    ] {
        let v = cat.dequant_f32(name).unwrap();
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        println!("{name}: len={} min={min:.4} max={max:.4} mean={mean:.4} first4={:?}", v.len(), &v[..4.min(v.len())]);
    }
}
