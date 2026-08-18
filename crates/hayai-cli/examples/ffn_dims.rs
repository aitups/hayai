fn main() {
    let cat = hayai_model::GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    let cfg = {
        // use public API if any
        let arch = cat.meta_str("general.architecture").unwrap();
        println!("arch={arch}");
        for k in ["qwen35.feed_forward_length", "qwen35.embedding_length", "qwen35.block_count"] {
            println!("{k}={:?}", cat.meta_u32(k));
        }
        let gate = cat.tensor("blk.0.ffn_gate.weight").unwrap();
        println!("ffn_gate dims={:?} ncols={} nrows={}", gate.dims, gate.ncols(), gate.nrows());
    };
}
