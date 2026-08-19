//! Generation execution on top of [`hayai_core::GenerationSession`].

use crate::models::{ApiError, ChatMessage, sampler_from};
use crate::registry::ModelHandle;
use hayai_core::GenerationSession;
use std::sync::Arc;

/// Run one completion: prefill `prompt`, decode up to `max_tokens`, and push
/// each text delta through `on_delta` (streaming or no-op). Applies `stop`
/// sequences and trims the output at the first match.
///
/// Returns `(text, finish_reason, prompt_tokens, completion_tokens)`.
pub fn generate_text(
    handle: Arc<ModelHandle>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
    stop: &[String],
    mut on_delta: impl FnMut(&str) -> Result<(), ApiError>,
) -> Result<(String, String, usize, usize), ApiError> {
    let sampler = sampler_from(temperature, top_p);
    let mut session = GenerationSession::start(
        &handle.path,
        handle.sinks,
        handle.window,
        handle.memory_strategy,
        sampler,
        seed,
        handle.orch.clone(),
    )
    .map_err(|e| ApiError::internal(e.to_string()))?;

    let prompt_tokens = session
        .prefill(prompt)
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let mut decoded: String = String::new();
    let mut finish = "length";

    for _ in 0..max_tokens.max(1) {
        match session.next_token() {
            Ok(Some(_tok)) => {}
            Ok(None) => {
                finish = "stop";
                break;
            }
            Err(e) => return Err(ApiError::internal(e.to_string())),
        }
        let full = session.generated_text();
        // Stop-sequence check before emitting the delta (avoids emitting the stop).
        if let Some((_stop_str, stop_at)) = stop
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| full.find(s.as_str()).map(|i| (s.clone(), i)))
            .min_by_key(|(_, i)| *i)
        {
            let before_stop = &full[..stop_at];
            let delta = before_stop[decoded.len()..].to_string();
            if !delta.is_empty() {
                on_delta(&delta)?;
            }
            decoded = before_stop.to_string();
            finish = "stop";
            session.mark_finished();
            break;
        }
        let delta = full[decoded.len()..].to_string();
        decoded = full;
        if !delta.is_empty() {
            on_delta(&delta)?;
        }
    }

    let completion_tokens = session.completion_tokens();
    Ok((decoded, finish.to_string(), prompt_tokens, completion_tokens))
}

/// Render chat messages into a prompt using the model's chat template.
pub fn chat_prompt(handle: &ModelHandle, messages: &[ChatMessage]) -> String {
    let msgs: Vec<(String, String)> = messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    handle
        .chat_template
        .render(&msgs, true, &handle.tokenizer)
}

/// Parse chat messages for the legacy completions endpoint (single prompt).
pub fn legacy_prompt(handle: &ModelHandle, prompt: &str) -> String {
    let _ = handle;
    prompt.to_string()
}
