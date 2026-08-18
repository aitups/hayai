fn main() {
    let mut cat = hayai_model::GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    let tok = hayai_model::Tokenizer::from_catalog(&cat).unwrap();
    let ids = tok.encode("The capital of France is", false);
    println!("prompt ids: {:?}", ids);
    for id in [279u32, 264, 9338, 310, 6511, 3750, 524, 303, 11751] {
        let s = tok.decode(&[id]);
        println!("id {id} => {:?}", s);
    }
    // also check if 11751 is in top elsewhere
}
