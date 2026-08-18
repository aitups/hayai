fn main() {
    let cat = hayai_model::GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    println!("output.weight={}", cat.tensor("output.weight").is_ok());
    println!("token_embd={}", cat.tensor("token_embd.weight").is_ok());
    if let Ok(t) = cat.tensor("output.weight") {
        println!("output type={:?} dims={:?}", t.ggml_type, t.dims);
    }
    if let Ok(t) = cat.tensor("token_embd.weight") {
        println!("emb type={:?} dims={:?}", t.ggml_type, t.dims);
    }
}
