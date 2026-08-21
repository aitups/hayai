use crate::gguf::GgufFile;
use crate::gguf_types::GgufError;
use std::collections::HashMap;

/// GPT-2 / SentencePiece BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    pub tokens: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub merges: HashMap<(String, String), u32>,
    /// Control / special tokens matched atomically during encode (longest-first).
    pub special_tokens: Vec<(String, u32)>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub add_bos: bool,
    /// SentencePiece (Gemma/LLaMA): space marker is `▁` (U+2581). Else GPT-2 `Ġ`.
    pub spm: bool,
    /// GPT-2 byte-level BPE forward map (byte -> unicode char). Empty for SentencePiece.
    pub bytes_to_unicode: HashMap<u8, char>,
    /// GPT-2 byte-level BPE inverse map (unicode char -> byte). Empty for SentencePiece.
    pub unicode_to_byte: HashMap<char, u8>,
    /// Raw Jinja chat template stored in the GGUF (`tokenizer.chat_template`),
    /// if present. Used by the OpenAI-like API to render chat messages.
    pub chat_template: Option<String>,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        Self::from_metadata(&gguf.metadata)
    }

    pub fn from_catalog(cat: &crate::gguf_stream::GgufCatalog) -> Result<Self, GgufError> {
        Self::from_metadata(&cat.metadata)
    }

    pub fn from_metadata(
        metadata: &std::collections::HashMap<String, crate::gguf_types::MetadataValue>,
    ) -> Result<Self, GgufError> {
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .ok_or_else(|| GgufError::MissingKey("tokenizer.ggml.tokens".into()))?;

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let merges_raw = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array())
            .unwrap_or_default();

        let mut merges = HashMap::new();
        for (rank, m) in merges_raw.iter().enumerate() {
            if let Some((a, b)) = m.split_once(' ') {
                merges.insert((a.to_string(), b.to_string()), rank as u32);
            }
        }

        let bos_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);
        let eos_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);
        let add_bos = metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                crate::gguf_types::MetadataValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        let model = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| match v {
                crate::gguf_types::MetadataValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("");
        // Gemma4 / LLaMA SPM use ▁; GPT-2 BPE uses Ġ. Prefer model string, else vocab vote.
        let spm = {
            let m = model.to_ascii_lowercase();
            if m.contains("gemma") || m == "llama" || m.contains("spm") || m.contains("sentencepiece")
            {
                true
            } else if m.contains("gpt") {
                false
            } else {
                let mut n_spm = 0usize;
                let mut n_gpt = 0usize;
                for t in tokens.iter().take(8192) {
                    if t.starts_with('\u{2581}') {
                        n_spm += 1;
                    }
                    if t.starts_with('Ġ') {
                        n_gpt += 1;
                    }
                }
                n_spm > n_gpt
            }
        };

        let mut special_tokens: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                (t.starts_with("<|") && t.ends_with("|>"))
                    || t.starts_with("<0x")
                    || *t == "<|endoftext|>"
                    || *t == "<bos>"
                    || *t == "<eos>"
                    || *t == "<unk>"
                    || *t == "<pad>"
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        // Some converters store the added/special tokens in a separate list
        // (`tokenizer.ggml.special_tokens`). Merge any that are already in the
        // vocab but not matched by the heuristic above (e.g. `[INST]`, `<|user|>`).
        for s in metadata
            .get("tokenizer.ggml.special_tokens")
            .and_then(|v| v.as_string_array())
            .unwrap_or_default()
        {
            if let Some(&id) = token_to_id.get(&s) {
                if !special_tokens.iter().any(|(t, _)| t.as_str() == s.as_str()) {
                    special_tokens.push((s.clone(), id));
                }
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // GPT-2 byte-level BPE (byte_fallback) vocabularies (GPT-2, Qwen, SmolLM2, Llama-3)
        // store every raw byte as one token whose text is the `bytes_to_unicode` char.
        let (bytes_to_unicode, unicode_to_byte) = if spm {
            (HashMap::new(), HashMap::new())
        } else {
            build_byte_maps()
        };

        // Jinja chat template (transformers convention); also accept `general.chat_template`.
        let chat_template = metadata
            .get("tokenizer.chat_template")
            .and_then(|v| match v {
                crate::gguf_types::MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .or_else(|| {
                metadata
                    .get("general.chat_template")
                    .and_then(|v| match v {
                        crate::gguf_types::MetadataValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
            });

        Ok(Self {
            tokens,
            token_to_id,
            merges,
            special_tokens,
            bos_id,
            eos_id,
            add_bos,
            spm,
            bytes_to_unicode,
            unicode_to_byte,
            chat_template,
        })
    }

    /// Normalize a Jinja chat template for minijinja compatibility.
    ///
    /// Transformers templates commonly use dict `.get('key')` and string
    /// `.startswith(...)` / `.endswith(...)`, which minijinja (2.x) does not support
    /// as methods. We rewrite:
    ///   * literal `.get('x')` / `.get("x")` → `['x']` (index syntax), and
    ///   * `.startswith(` / `.endswith(` → `|startswith(` / `|endswith(` (filters).
pub fn normalize_jinja_template(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 16);
    let b = raw.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'.' && i + 5 < b.len() {
            let method = &raw[i + 1..i + 4];
            if method == "get" && b[i + 4] == b'(' {
                if let Some(&qc @ (b'\'' | b'"')) = b.get(i + 5) {
                    let mut k = i + 6;
                    while k < b.len() && b[k] != qc {
                        k += 1;
                    }
                    if k < b.len() && b.get(k + 1) == Some(&b')') {
                        // `.get('X')` -> `['X']` (keep the quotes).
                        out.push('[');
                        out.push_str(&raw[i + 5..=k]);
                        out.push(']');
                        i = k + 2;
                        continue;
                    }
                }
            } else if raw[i + 1..].starts_with("startswith(")
                || raw[i + 1..].starts_with("endswith(")
            {
                // `.startswith(` / `.endswith(` -> `|startswith(` / `|endswith(`.
                let method_len = if raw[i + 1..].starts_with("startswith(") {
                    11
                } else {
                    9
                };
                out.push('|');
                out.push_str(&raw[i + 1..i + 1 + method_len]);
                i += 1 + method_len;
                continue;
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Byte-level BPE encode (GPT-2 or SentencePiece). Special tokens are matched atomically.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_special && self.add_bos {
            ids.push(self.bos_id);
        }

        let mut rest = text;
        while !rest.is_empty() {
            if let Some((tok, id)) = self.match_special_at(rest) {
                ids.push(id);
                rest = &rest[tok.len()..];
                continue;
            }
            let mut cut = rest.len();
            for (i, _) in rest.char_indices().skip(1) {
                if self.match_special_at(&rest[i..]).is_some() {
                    cut = i;
                    break;
                }
            }
            let chunk = &rest[..cut];
            for word in split_words(chunk, self.spm) {
                ids.extend(self.bpe_encode_word(&word));
            }
            rest = &rest[cut..];
        }
        ids
    }

    fn match_special_at<'a>(&'a self, s: &'a str) -> Option<(&'a str, u32)> {
        for (tok, id) in &self.special_tokens {
            if s.starts_with(tok.as_str()) {
                return Some((tok.as_str(), *id));
            }
        }
        None
    }

    fn bpe_encode_word(&self, word: &str) -> Vec<u32> {
        if let Some(&id) = self.token_to_id.get(word) {
            return vec![id];
        }

        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if symbols.is_empty() {
            return Vec::new();
        }

        loop {
            let mut best: Option<(usize, u32)> = None; // (pair_index, rank)
            for i in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&key) {
                    if best.map(|(_, r)| rank < r).unwrap_or(true) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols[i] = merged;
            symbols.remove(i + 1);
        }

        let mut out = Vec::new();
        for s in symbols {
            if let Some(&id) = self.token_to_id.get(&s) {
                out.push(id);
            } else {
                // Byte fallback: byte-level BPE vocabs map raw bytes to the
                // `bytes_to_unicode` char tokens; also accept <0xNN> tokens if present.
                for b in s.as_bytes() {
                    let mut pushed = false;
                    if let Some(&ch) = self.bytes_to_unicode.get(b) {
                        if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                            out.push(id);
                            pushed = true;
                        }
                    }
                    if !pushed {
                        let tok = format!("<0x{b:02X}>");
                        if let Some(&id) = self.token_to_id.get(&tok) {
                            out.push(id);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(tok) = self.tokens.get(id as usize) {
                if tok == "<|endoftext|>"
                    || tok.starts_with("<|")
                    || (id == self.bos_id || id == self.eos_id)
                {
                    continue;
                }
                if let Some(hex) = tok.strip_prefix("<0x").and_then(|t| t.strip_suffix('>')) {
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        bytes.push(b);
                        continue;
                    }
                }
                // GPT-2 byte-level BPE: inverse bytes_to_unicode -> raw bytes;
                // SentencePiece: space marker is U+2581 and tokens are plain UTF-8.
                if self.spm {
                    bytes.extend_from_slice(tok.replace('\u{2581}', " ").as_bytes());
                } else {
                    for ch in tok.chars() {
                        match self.unicode_to_byte.get(&ch) {
                            Some(&b) => bytes.push(b),
                            None => {
                                let mut buf = [0u8; 4];
                                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// ChatML-style wrap used by SmolLM Instruct (and similar) GGUFs.
    pub fn apply_chat_template(&self, user: &str) -> String {
        format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    }

    /// Render a message list with the model's Jinja chat template (`tokenizer.chat_template`)
    /// when present. Returns `None` when there is no usable template (caller falls back to
    /// [`Self::apply_chat_template`]).
    ///
    /// `warnings` collects special-token markers (e.g. `<|user|>`) referenced by the rendered
    /// prompt that are **not** in the vocab — they would be BPE-split into subword tokens and
    /// degrade the prompt. This surfaces GGUF conversions that dropped added tokens.
    pub fn render_chat_template(
        &self,
        messages: &[(String, String)],
        add_generation_prompt: bool,
        warnings: &mut Vec<String>,
    ) -> Option<String> {
        let raw = self.chat_template.as_ref()?;
        let raw = Self::normalize_jinja_template(raw);
        let mut env = minijinja::Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        env.add_filter("startswith", |s: &str, prefix: &str| s.starts_with(prefix));
        env.add_filter("endswith", |s: &str, suffix: &str| s.ends_with(suffix));
        if let Err(e) = env.add_template_owned("chat", raw.clone()) {
            warnings.push(format!("chat template does not compile: {e}"));
            return None;
        }
        let tpl = match env.get_template("chat") {
            Ok(t) => t,
            Err(e) => {
                warnings.push(format!("chat template lookup failed: {e}"));
                return None;
            }
        };
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect();
        let bos = self
            .tokens
            .get(self.bos_id as usize)
            .cloned()
            .unwrap_or_default();
        let eos = self
            .tokens
            .get(self.eos_id as usize)
            .cloned()
            .unwrap_or_default();
        let ctx = minijinja::Value::from_serialize(&serde_json::json!({
            "messages": msgs,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": bos,
            "eos_token": eos,
        }));
        match tpl.render(ctx) {
            Ok(rendered) => {
                warnings.extend(self.unresolved_specials(&rendered));
                Some(rendered)
            }
            Err(e) => {
                warnings.push(format!("chat template render error: {e}"));
                None
            }
        }
    }

    /// Special-token markers (`<...|...>`) in `text` that are NOT representable as a
    /// single vocab token (they would be BPE-split into subword tokens, degrading
    /// the prompt). Checks `token_to_id` first — Gemma4 pairs like `<turn|>` are
    /// atomic even though they are not matched by the `<|...|>` heuristic.
    pub fn unresolved_specials(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let b = text.as_bytes();
        let mut i = 0usize;
        while i < b.len() {
            if b[i] == b'<' {
                // Read up to the next `>` (whitespace or 40 chars caps a "marker").
                let mut j = i + 1;
                while j < b.len() && j - i <= 40 {
                    let c = b[j];
                    if c == b'>' {
                        break;
                    }
                    if c.is_ascii_whitespace() {
                        j = b.len();
                        break;
                    }
                    j += 1;
                }
                if j < b.len() && b[j] == b'>' && j - i <= 32 && marker_like(&text[i..=j]) {
                    let marker = &text[i..=j];
                    if self.token_to_id.get(marker).is_none()
                        && !self
                            .special_tokens
                            .iter()
                            .any(|(t, _)| t.as_str() == marker)
                    {
                        out.push(marker.to_string());
                    }
                    i = j + 1;
                    continue;
                }
            }
            i += 1;
        }
        out
    }
}

/// GPT-2 `bytes_to_unicode` mapping (HuggingFace tokenizers). Every byte maps to a
/// printable char: printable ASCII/Latin-1 map to themselves, the rest to U+0100+.
/// Used by byte-level BPE vocabularies (GPT-2, Qwen, SmolLM2, Llama-3).
/// True when a `<...>` span looks like a special-token marker (contains `|`).
fn marker_like(s: &str) -> bool {
    s.len() >= 3 && s.contains('|')
}

fn build_byte_maps() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    let mut cs: Vec<char> = Vec::new();
    for b in b'!'..=b'~' { bs.push(b); cs.push(b as char); }
    for b in 0xa1..=0xac { bs.push(b); cs.push(b as char); }
    for b in 0xae..=0xff { bs.push(b); cs.push(b as char); }
    let mut n = 0u32;
    for b in 0..=255u32 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(char::from_u32(0x100 + n).expect("byte map range"));
            n += 1;
        }
    }
    let mut fwd = HashMap::with_capacity(256);
    let mut inv = HashMap::with_capacity(256);
    for (b, ch) in bs.into_iter().zip(cs.into_iter()) {
        fwd.insert(b, ch);
        inv.insert(ch, b);
    }
    (fwd, inv)
}

fn split_words(text: &str, spm: bool) -> Vec<String> {
    // Whitespace-aware segmentation:
    // - GPT-2: spaces→Ġ, newlines→Ċ, tabs→ċ
    // - SentencePiece (Gemma): spaces→▁ (U+2581); newlines stay as their own pieces if present
    let space_mark = if spm { "\u{2581}" } else { "Ġ" };
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if ch == ' ' || ch == '\t' {
                let piece = if ch == '\t' && !spm {
                    "ċ"
                } else if ch == '\t' && spm {
                    space_mark
                } else {
                    space_mark
                };
                cur.push_str(piece);
            } else if spm {
                // Prefer a literal newline token if the model has one; else emit ▁.
                out.push("\n".to_string());
            } else {
                out.push("Ċ".to_string());
            }
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() && cur != space_mark && cur != "ċ" {
        out.push(cur);
    }
    if out.is_empty() && !text.is_empty() {
        out.push(text.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{write_minimal_gguf, GgufFile};
    use crate::gguf_types::MetadataValue;
    use std::env::temp_dir;

    #[test]
    fn encode_decode_with_merges() {
        let path = temp_dir().join("hayai_tok_test.gguf");
        let tokens = vec![
            MetadataValue::String("<unk>".into()),
            MetadataValue::String("<s>".into()),
            MetadataValue::String("</s>".into()),
            MetadataValue::String("a".into()),
            MetadataValue::String("b".into()),
            MetadataValue::String("ab".into()),
            MetadataValue::String("Ġhi".into()),
        ];
        let merges = vec![MetadataValue::String("a b".into())];
        write_minimal_gguf(
            &path,
            &[
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                ("tokenizer.ggml.merges", MetadataValue::Array(merges)),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
            ],
            &[("dummy", vec![1], vec![0.0f32])],
        )
        .unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert!(!tok.spm);
        let ids = tok.bpe_encode_word("ab");
        assert_eq!(ids, vec![5]); // "ab"
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn byte_level_bpe_non_ascii_roundtrip() {
        let path = temp_dir().join("hayai_tok_bytes_test.gguf");
        // GPT-2 byte-level BPE vocab for "中" (UTF-8 bytes E4 B8 AD) plus ASCII.
        let (fwd, _inv) = build_byte_maps();
        let e4 = fwd[&0xe4u8].to_string();
        let b8 = fwd[&0xb8u8].to_string();
        let ad = fwd[&0xadu8].to_string();
        let tokens = vec![
            MetadataValue::String("<unk>".into()),
            MetadataValue::String("<s>".into()),
            MetadataValue::String("</s>".into()),
            MetadataValue::String(e4.clone()),
            MetadataValue::String(b8.clone()),
            MetadataValue::String(ad.clone()),
            MetadataValue::String("hi".into()),
        ];
        write_minimal_gguf(
            &path,
            &[
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                (
                    "tokenizer.ggml.merges",
                    MetadataValue::Array(vec![MetadataValue::String("a b".into())]),
                ),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
            ],
            &[("dummy", vec![1], vec![0.0f32])],
        )
        .unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert!(!tok.spm);
        // decode bytes E4 B8 AD -> "中"
        assert_eq!(tok.decode(&[3, 4, 5]), "中");
        // encode "中" back to the byte tokens (byte_fallback path)
        assert_eq!(tok.encode("中", false), vec![3, 4, 5]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn chat_template_renders_with_jinja_and_atomic_specials() {
        let path = temp_dir().join("hayai_tok_chat.gguf");
        let tokens = vec![
            MetadataValue::String("<unk>".into()),
            MetadataValue::String("<s>".into()),
            MetadataValue::String("</s>".into()),
            MetadataValue::String("<|im_start|>".into()),
            MetadataValue::String("<|im_end|>".into()),
            MetadataValue::String("hello".into()),
        ];
        write_minimal_gguf(
            &path,
            &[
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                ("tokenizer.ggml.merges", MetadataValue::Array(vec![])),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
                (
                    "tokenizer.chat_template",
                    MetadataValue::String(
                        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
                            .into(),
                    ),
                ),
            ],
            &[("dummy", vec![1], vec![0.0f32])],
        )
        .unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let mut warnings = Vec::new();
        let rendered = tok
            .render_chat_template(&[("user".into(), "hi".into())], true, &mut warnings)
            .unwrap();
        assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"));
        assert!(warnings.is_empty());
        // The special markers must encode atomically (ids 3 and 4).
        let ids = tok.encode(&rendered, false);
        assert!(ids.contains(&3) && ids.contains(&4));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unresolved_specials_reports_missing_markers() {
        // Template references <|user|> which is NOT in the vocab.
        let path = temp_dir().join("hayai_tok_chat_missing.gguf");
        let tokens = vec![
            MetadataValue::String("<unk>".into()),
            MetadataValue::String("<s>".into()),
            MetadataValue::String("</s>".into()),
            MetadataValue::String("<|endoftext|>".into()),
            MetadataValue::String("hi".into()),
        ];
        write_minimal_gguf(
            &path,
            &[
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                ("tokenizer.ggml.merges", MetadataValue::Array(vec![])),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(3)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(3)),
                (
                    "tokenizer.chat_template",
                    MetadataValue::String(
                        "<|endoftext|>{{ '<|user|>\\n' + messages[0]['content'] }}".into(),
                    ),
                ),
            ],
            &[("dummy", vec![1], vec![0.0f32])],
        )
        .unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let mut warnings = Vec::new();
        let rendered = tok
            .render_chat_template(&[("user".into(), "hi".into())], false, &mut warnings)
            .unwrap();
        assert!(rendered.contains("<|user|>"));
        assert!(warnings.iter().any(|w| w.contains("<|user|>")));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn special_tokens_metadata_is_merged() {
        // `[INST]` is in the vocab but not matched by the `<|...|>` heuristic; the
        // separate `tokenizer.ggml.special_tokens` list must add it.
        let path = temp_dir().join("hayai_tok_special_meta.gguf");
        let tokens = vec![
            MetadataValue::String("<unk>".into()),
            MetadataValue::String("<s>".into()),
            MetadataValue::String("</s>".into()),
            MetadataValue::String("[INST]".into()),
            MetadataValue::String("hi".into()),
        ];
        write_minimal_gguf(
            &path,
            &[
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                (
                    "tokenizer.ggml.special_tokens",
                    MetadataValue::Array(vec![MetadataValue::String("[INST]".into())]),
                ),
                ("tokenizer.ggml.merges", MetadataValue::Array(vec![])),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
            ],
            &[("dummy", vec![1], vec![0.0f32])],
        )
        .unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert!(tok.special_tokens.iter().any(|(t, _)| t == "[INST]"));
        assert_eq!(tok.encode("[INST] hi", false).first(), Some(&3));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn normalize_rewrites_startswith_and_get() {
        // Qwen-style template using `.startswith` and dict `.get(...)` must render.
        let path = temp_dir().join("hayai_tok_startswith.gguf");
        let tokens = vec![
            MetadataValue::String("<unk>".into()),
            MetadataValue::String("<s>".into()),
            MetadataValue::String("</s>".into()),
            MetadataValue::String("<|im_start|>".into()),
            MetadataValue::String("<|im_end|>".into()),
            MetadataValue::String("hi".into()),
        ];
        let tmpl = "{% if messages[0]['role'].startswith('u') %}ok{% endif %}{% for m in messages %}{{ m.get('role') }}:{{ m['content'] }}{% endfor %}";
        write_minimal_gguf(
            &path,
            &[
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                ("tokenizer.ggml.merges", MetadataValue::Array(vec![])),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
                (
                    "tokenizer.chat_template",
                    MetadataValue::String(tmpl.into()),
                ),
            ],
            &[("dummy", vec![1], vec![0.0f32])],
        )
        .unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let mut warnings = Vec::new();
        let rendered = tok
            .render_chat_template(&[("user".into(), "hi".into())], false, &mut warnings)
            .unwrap();
        assert!(rendered.starts_with("ok"), "rendered={rendered}");
        assert!(rendered.contains("user:hi"), "rendered={rendered}");
        let _ = std::fs::remove_file(path);
    }
}
