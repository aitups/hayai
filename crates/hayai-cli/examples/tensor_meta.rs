fn main() {
    let cat = hayai_model::GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    println!("data_offset={}", cat.data_offset);
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
        "token_embd.weight",
    ] {
        let t = cat.tensor(name).unwrap();
        println!("{name}: type={:?} dims={:?} offset={}", t.ggml_type, t.dims, t.offset);
    }
}
