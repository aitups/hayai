//! Chat template rendering.
//!
//! Resolution chain (per model): CLI `--chat-template` override → GGUF metadata
//! `tokenizer.chat_template` (Jinja) → built-in ChatML fallback.

use hayai_model::Tokenizer;

/// Default fallback: ChatML (SmolLM/Qwen-style).
const DEFAULT_CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/// A compiled (or fallback) chat template for one model.
pub struct ChatTemplate {
    env: Option<minijinja::Environment<'static>>,
    default_chatml: bool,
}

impl ChatTemplate {
    pub fn from_model(tokenizer: &Tokenizer, override_template: Option<&str>) -> Self {
        let raw = override_template
            .map(str::to_owned)
            .or_else(|| tokenizer.chat_template.clone())
            .unwrap_or_else(|| DEFAULT_CHATML.to_string());

        if raw == DEFAULT_CHATML {
            return Self {
                env: None,
                default_chatml: true,
            };
        }

        let mut env = minijinja::Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        if env.add_template_owned("chat", raw.clone()).is_err() {
            // Invalid template: fall back to ChatML rather than failing requests.
            tracing::warn!("invalid chat template; falling back to ChatML");
            return Self {
                env: None,
                default_chatml: true,
            };
        }
        Self {
            env: Some(env),
            default_chatml: false,
        }
    }

    /// Render the messages (plus optional generation prompt) to a prompt string.
    pub fn render(
        &self,
        messages: &[(String, String)],
        add_generation_prompt: bool,
        tokenizer: &Tokenizer,
    ) -> String {
        match &self.env {
            Some(env) => {
                let tpl = env.get_template("chat").unwrap();
                let msgs: Vec<serde_json::Value> = messages
                    .iter()
                    .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
                    .collect();
                let bos = tokenizer
                    .tokens
                    .get(tokenizer.bos_id as usize)
                    .cloned()
                    .unwrap_or_default();
                let eos = tokenizer
                    .tokens
                    .get(tokenizer.eos_id as usize)
                    .cloned()
                    .unwrap_or_default();
                let ctx = minijinja::Value::from_serialize(&serde_json::json!({
                    "messages": msgs,
                    "add_generation_prompt": add_generation_prompt,
                    "bos_token": bos,
                    "eos_token": eos,
                }));
                tpl.render(ctx).unwrap_or_default()
            }
            None => {
                debug_assert!(self.default_chatml);
                let mut s = String::new();
                for (role, content) in messages {
                    s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                if add_generation_prompt {
                    s.push_str("<|im_start|>assistant\n");
                }
                s
            }
        }
    }

    /// Convenience: render without the generation prompt (used by tests).
    pub fn render_messages(&self, messages: &[(String, String)], tokenizer: &Tokenizer) -> String {
        self.render(messages, true, tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_with_template(tmpl: &str) -> Tokenizer {
        use std::collections::HashMap;
        use hayai_model::MetadataValue;
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
        let out = tpl.render_messages(
            &[("user".into(), "hi".into())],
            &tok,
        );
        assert_eq!(out, "user:hi\\n");
    }

    #[test]
    fn override_template_wins() {
        let tok = tokenizer_with_template("IGNORED");
        let tpl = ChatTemplate::from_model(&tok, Some("{{ messages[0]['content'] }}"));
        let out = tpl.render_messages(&[("user".into(), "hi".into())], &tok);
        assert_eq!(out, "hi");
    }
}
