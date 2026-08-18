fn main() {
    let cat = hayai_model::GgufCatalog::open("models/google_gemma-4-E4B-it-Q4_K_M.gguf").unwrap();
    let tok = hayai_model::Tokenizer::from_catalog(&cat).unwrap();
    println!("spm={}", tok.spm);
    for p in [
        "The capital of France is",
        "2 + 2 =",
        "Hello, my name is",
    ] {
        let ids = tok.encode(p, true);
        println!("{p:?} => {:?}", ids);
    }
}
