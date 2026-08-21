//! Validate a model's chat template: render a sample message, encode it, and
//! report whether the special markers are atomic (single IDs) — warning when the
//! template references special tokens missing from the vocab (GGUF dropped them).
use hayai_model::{GgufCatalog, Tokenizer};

fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/SmolLM2-135M-Instruct-Q4_K_M.gguf".into());
    let cat = GgufCatalog::open(&model).unwrap();
    let tok = Tokenizer::from_catalog(&cat).unwrap();
    println!("model: {model}");
    println!("has_jinja: {}", tok.chat_template.is_some());
    println!("bos={} eos={} spm={}", tok.bos_id, tok.eos_id, tok.spm);
    let mut warnings = Vec::new();
    match tok.render_chat_template(&[("user".into(), "Hi".into())], true, &mut warnings) {
        Some(r) => {
            println!("rendered: {r:?}");
            let ids = tok.encode(&r, false);
            println!("ids ({}): {:?}", ids.len(), ids);
            for w in &warnings {
                println!("WARNING missing special in vocab: {w}");
            }
            for m in ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|system|>", "<|turn|>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>"] {
                if tok.token_to_id.get(m).is_some() {
                    println!("marker {m}: ATOMIC id {}", tok.token_to_id[m]);
                } else {
                    println!("marker {m}: NOT in vocab");
                }
            }
        }
        None => {
            println!("no usable jinja template -> ChatML fallback");
            for w in &warnings {
                println!("  warning: {w}");
            }
        }
    }
}
