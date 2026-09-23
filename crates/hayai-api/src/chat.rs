//! Chat template rendering.
//!
//! Resolution chain (per model): CLI `--chat-template` override → GGUF metadata
//! `tokenizer.chat_template` (Jinja) → built-in ChatML fallback. All Jinja
//! rendering goes through [`Tokenizer::render_chat_template_override`] so the
//! CLI and the server share exactly one implementation (including the
//! `raise_exception` / `strftime_now` / `tools` shims).

use hayai_model::Tokenizer;

/// Default fallback: ChatML (SmolLM/Qwen-style).
const DEFAULT_CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/// A chat template for one model: an optional override plus the tokenizer's own.
pub struct ChatTemplate {
    override_template: Option<String>,
}

impl ChatTemplate {
    pub fn from_model(_tokenizer: &Tokenizer, override_template: Option<&str>) -> Self {
        Self {
            override_template: override_template.map(str::to_owned),
        }
    }

    /// Render the messages (plus optional generation prompt) to a prompt string.
    pub fn render(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
        tokenizer: &Tokenizer,
        tools: &serde_json::Value,
    ) -> String {
        let raw = self
            .override_template
            .clone()
            .or_else(|| tokenizer.chat_template.clone());
        if let Some(raw) = raw {
            let mut warnings = Vec::new();
            match tokenizer.render_chat_template_override_values(
                &raw,
                messages,
                add_generation_prompt,
                tools,
                &mut warnings,
            ) {
                Ok(rendered) => {
                    for w in warnings {
                        tracing::warn!("chat template: {w}");
                    }
                    return rendered;
                }
                Err(e) => {
                    tracing::warn!("{e}; using ChatML fallback");
                }
            }
        }
        Self::chatml_fallback(messages, add_generation_prompt)
    }

    fn chatml_fallback(messages: &[serde_json::Value], add_generation_prompt: bool) -> String {
        let mut s = String::new();
        for m in messages {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
            s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        if add_generation_prompt {
            s.push_str("<|im_start|>assistant\n");
        }
        s
    }

    /// Convenience: render without the generation prompt (used by tests).
    pub fn render_messages(
        &self,
        messages: &[serde_json::Value],
        tokenizer: &Tokenizer,
    ) -> String {
        self.render(messages, true, tokenizer, &serde_json::json!([]))
    }

    /// The built-in ChatML template (exposed so callers can detect the default).
    pub fn default_chatml() -> &'static str {
        DEFAULT_CHATML
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_with_template(tmpl: &str) -> Tokenizer {
        use hayai_model::MetadataValue;
        use std::collections::HashMap;
        let mut meta = HashMap::new();
        meta.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetadataValue::Array(
                ["<|im_start|>", "<|im_end|>", "hello", "world"]
                    .into_iter()
                    .map(|s| MetadataValue::String(s.to_string()))
                    .collect(),
            ),
        );
        meta.insert(
            "tokenizer.chat_template".to_string(),
            MetadataValue::String(tmpl.to_string()),
        );
        Tokenizer::from_metadata(&meta).unwrap()
    }

    #[test]
    fn chatml_default_fallback() {
        let tok = tokenizer_with_template("{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\\n{% endfor %}");
        let tpl = ChatTemplate::from_model(&tok, None);
        let out = tpl.render_messages(&[serde_json::json!({"role":"user","content":"hi"})], &tok);
        assert_eq!(out, "user:hi\\n");
    }

    #[test]
    fn override_template_wins() {
        let tok = tokenizer_with_template("IGNORED");
        let tpl = ChatTemplate::from_model(&tok, Some("{{ messages[0]['content'] }}"));
        let out = tpl.render_messages(&[serde_json::json!({"role":"user","content":"hi"})], &tok);
        assert_eq!(out, "hi");
    }

    #[test]
    fn raise_exception_falls_back_to_chatml() {
        let tok = tokenizer_with_template("{{ raise_exception('nope') }}");
        let tpl = ChatTemplate::from_model(&tok, None);
        let out = tpl.render_messages(&[serde_json::json!({"role":"user","content":"hi"})], &tok);
        assert!(out.contains("<|im_start|>user"));
    }

    #[test]
    fn tools_are_passed_to_template() {
        // Template advertises the available function names, as tool templates do.
        let tok = tokenizer_with_template(
            "{% for t in tools %}{{ t['function']['name'] }};{% endfor %}",
        );
        let tpl = ChatTemplate::from_model(&tok, None);
        let tools = serde_json::json!([
            {"type":"function","function":{"name":"get_weather","parameters":{}}}
        ]);
        let out = tpl.render(
            &[serde_json::json!({"role":"user","content":"hi"})],
            true,
            &tok,
            &tools,
        );
        assert_eq!(out, "get_weather;");
    }
}
