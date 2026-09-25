//! Generation execution on top of [`hayai_core::GenerationSession`].

use crate::models::{ApiError, ChatMessage, SamplingParams};
use crate::registry::ModelHandle;
use hayai_core::GenerationSession;
use std::sync::Arc;

/// Decode the accumulated ids byte-exactly, emit the new text delta, and detect
/// stop sequences. Returns `true` when a stop sequence matched.
fn emit_decoded(
    tokenizer: &hayai_model::Tokenizer,
    ids: &[u32],
    decoded: &mut String,
    stop: &[String],
    on_delta: &mut impl FnMut(&str) -> Result<(), ApiError>,
) -> Result<bool, ApiError> {
    // Byte-exact decode: emit only the complete UTF-8 prefix so a multi-byte
    // character split across tokens is held back instead of being rendered as
    // U+FFFD and later contradicted.
    let raw = tokenizer.decode_bytes(ids);
    let valid = match std::str::from_utf8(&raw) {
        Ok(s) => s,
        Err(e) => std::str::from_utf8(&raw[..e.valid_up_to()]).unwrap_or(""),
    };
    if valid.len() < decoded.len() {
        // Defensive: never slice below the already-emitted prefix.
        decoded.truncate(valid.len());
    }
    // Stop-sequence check before emitting the delta (avoids emitting the stop).
    if let Some((_stop_str, stop_at)) = stop
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| valid.find(s.as_str()).map(|i| (s.clone(), i)))
        .min_by_key(|(_, i)| *i)
    {
        if stop_at > decoded.len() {
            on_delta(&valid[decoded.len()..stop_at])?;
            decoded.push_str(&valid[decoded.len()..stop_at]);
        }
        return Ok(true);
    }
    if valid.len() > decoded.len() {
        let delta = &valid[decoded.len()..];
        on_delta(delta)?;
        decoded.push_str(delta);
    }
    Ok(false)
}

/// Run one completion: prefill `prompt`, decode up to `max_tokens`, and push
/// each text delta through `on_delta` (streaming or no-op). Applies `stop`
/// sequences and trims the output at the first match.
///
/// Returns `(text, finish_reason, prompt_tokens, completion_tokens)`.
pub fn generate_text(
    handle: Arc<ModelHandle>,
    prompt: &str,
    max_tokens: usize,
    params: SamplingParams,
    stop: &[String],
    on_delta: impl FnMut(&str) -> Result<(), ApiError>,
) -> Result<(String, String, usize, usize), ApiError> {
    if handle.encoder_decoder {
        return generate_text_t5(handle, prompt, max_tokens, params, stop, on_delta);
    }
    let sampler = params.sampler();
    let mut session = GenerationSession::start(
        &handle.path,
        handle.sinks,
        handle.window,
        handle.memory_strategy,
        sampler,
        params.seed,
        handle.orch.clone(),
    )
    .map_err(|e| ApiError::internal(e.to_string()))?;
    session.set_penalties(params.penalties);
    if let Some(grammar) = params.grammar {
        session.set_grammar(grammar);
    }

    let prompt_tokens = session
        .prefill(prompt)
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let mut decoded: String = String::new();
    let mut finish = "length";
    let mut on_delta = on_delta;

    for _ in 0..max_tokens.max(1) {
        match session.next_token() {
            Ok(Some(_tok)) => {}
            Ok(None) => {
                finish = "stop";
                break;
            }
            Err(e) => return Err(ApiError::internal(e.to_string())),
        }
        if emit_decoded(
            &handle.tokenizer,
            session.generated_ids(),
            &mut decoded,
            stop,
            &mut on_delta,
        )? {
            finish = "stop";
            session.mark_finished();
            break;
        }
    }

    let completion_tokens = session.completion_tokens();
    Ok((decoded, finish.to_string(), prompt_tokens, completion_tokens))
}

/// Encoder-decoder (T5/BART) completion via the token-by-token [`EncDecSession`].
fn generate_text_t5(
    handle: Arc<ModelHandle>,
    prompt: &str,
    max_tokens: usize,
    params: SamplingParams,
    stop: &[String],
    mut on_delta: impl FnMut(&str) -> Result<(), ApiError>,
) -> Result<(String, String, usize, usize), ApiError> {
    use hayai_core::encoder_decoder_infer::EncDecSession;
    let mut session = EncDecSession::start(&handle.path, params.sampler(), params.seed)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    session.set_penalties(params.penalties);
    let prompt_tokens = session
        .prefill(prompt)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let mut decoded = String::new();
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
        if emit_decoded(
            &handle.tokenizer,
            session.generated_ids(),
            &mut decoded,
            stop,
            &mut on_delta,
        )? {
            finish = "stop";
            session.mark_finished();
            break;
        }
    }
    let completion_tokens = session.completion_tokens();
    Ok((decoded, finish.to_string(), prompt_tokens, completion_tokens))
}

/// Render chat messages into a prompt using the model's chat template.
pub fn chat_prompt(handle: &ModelHandle, messages: &[ChatMessage], tools: &serde_json::Value) -> String {
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let mut o = serde_json::Map::new();
            o.insert("role".into(), serde_json::json!(m.role));
            o.insert("content".into(), serde_json::json!(m.content));
            if let Some(tc) = &m.tool_calls {
                o.insert("tool_calls".into(), tc.clone());
            }
            if let Some(id) = &m.tool_call_id {
                o.insert("tool_call_id".into(), serde_json::json!(id));
            }
            if let Some(n) = &m.name {
                o.insert("name".into(), serde_json::json!(n));
            }
            serde_json::Value::Object(o)
        })
        .collect();
    handle
        .chat_template
        .render(&msgs, true, &handle.tokenizer, tools)
}
