fn main() {
    let mut cat = hayai_model::GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    for name in [
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.post_attention_norm.weight",
    ] {
        let v = cat.dequant_f32(name).unwrap();
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        println!("{name}: len={} min={min:.6} max={max:.6} mean={mean:.6} first8={:?}", v.len(), &v[..8.min(v.len())]);
    }
}
