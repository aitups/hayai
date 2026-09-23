//! OpenAI-compatible wire types (serde).

use serde::{Deserialize, Serialize};

fn default_max_tokens() -> usize {
    128
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_top_k() -> usize {
    0
}
fn default_min_p() -> f32 {
    0.0
}
fn default_penalty() -> f32 {
    0.0
}
fn default_seed() -> u64 {
    42
}
fn default_stream() -> bool {
    false
}
fn default_n() -> usize {
    1
}

/// Hard cap on `max_tokens` per request (DoS guard). Requests above are rejected
/// with 400 instead of monopolizing the model's orchestrator and disk.
pub const MAX_TOKENS_LIMIT: usize = 32768;

/// `stop` accepts a single string or an array of strings.
fn deserialize_stop<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_json::Value::deserialize(d)?;
    match value {
        serde_json::Value::String(s) => Ok(vec![s]),
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| D::Error::custom("stop entries must be strings"))
            })
            .collect(),
        // OpenAI clients commonly send `"stop": null`.
        serde_json::Value::Null => Ok(Vec::new()),
        _ => Err(D::Error::custom("stop must be a string or array of strings")),
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// Assistant tool calls (OpenAI `tool_calls`), passed through to the Jinja template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// Tool-role message: id of the call being answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    #[serde(default = "default_penalty")]
    pub presence_penalty: f32,
    #[serde(default = "default_penalty")]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// GBNF grammar source for constrained decoding (Hayai extension).
    #[serde(default)]
    pub grammar: Option<String>,
    /// OpenAI `tools` array (function definitions), rendered by the chat template.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// OpenAI `tool_choice` (`auto`/`none`/`required`/object), passed to the template.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
    /// OpenAI `n`: only 1 is implemented.
    #[serde(default = "default_n")]
    pub n: usize,
    /// OpenAI `logprobs`: not implemented; any value is rejected.
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    #[serde(default = "default_penalty")]
    pub presence_penalty: f32,
    #[serde(default = "default_penalty")]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// GBNF grammar source for constrained decoding (Hayai extension).
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

/// Engine sampling parameters resolved from a request (temperature, top_p/top_k,
/// min_p, penalties, seed).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub penalties: hayai_model::Penalties,
    pub seed: u64,
    /// Optional GBNF grammar for constrained decoding.
    pub grammar: Option<hayai_model::Grammar>,
}

impl SamplingParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temperature: f32,
        top_p: f32,
        top_k: usize,
        min_p: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        logit_bias: Option<&std::collections::HashMap<String, f32>>,
        seed: u64,
    ) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            min_p,
            penalties: penalties_from(presence_penalty, frequency_penalty, logit_bias),
            seed,
            grammar: None,
        }
    }

    pub fn sampler(&self) -> hayai_model::SamplerConfig {
        sampler_from(self.temperature, self.top_p, self.top_k, self.min_p)
    }

    /// Parse and attach an optional GBNF grammar (empty/whitespace = none).
    pub fn apply_grammar(&mut self, src: Option<&str>) -> Result<(), String> {
        self.grammar = match src {
            Some(s) if !s.trim().is_empty() => {
                Some(hayai_model::Grammar::parse(s).map_err(|e| e.to_string())?)
            }
            _ => None,
        };
        Ok(())
    }
}

/// Reject request options that are accepted by the OpenAI schema but not
/// implemented, plus an unbounded `max_tokens`. Returns `Some(400)` on misuse.
pub fn validate_generation(
    max_tokens: usize,
    n: usize,
    logprobs: Option<&serde_json::Value>,
) -> Option<ApiError> {
    if n > 1 {
        return Some(ApiError::bad_request("`n` > 1 is not supported"));
    }
    if logprobs.is_some() {
        return Some(ApiError::bad_request("`logprobs` is not supported"));
    }
    if max_tokens > MAX_TOKENS_LIMIT {
        return Some(ApiError::bad_request(format!(
            "`max_tokens` {max_tokens} exceeds the limit of {MAX_TOKENS_LIMIT}"
        )));
    }
    None
}

/// Map temperature/top_p/top_k/min_p onto the engine sampler config
/// (greedy when temp≈0).
pub fn sampler_from(
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
) -> hayai_model::SamplerConfig {
    if temperature <= 0.0 {
        if top_k > 0 {
            // Still deterministic-ish: honor top_k in greedy path by filtering.
            return hayai_model::SamplerConfig::TopK {
                temperature: 1e-5,
                top_k,
            };
        }
        return hayai_model::SamplerConfig::Greedy;
    }
    if min_p > 0.0 {
        return hayai_model::SamplerConfig::MinP {
            temperature,
            top_p,
            min_p,
        };
    }
    if top_k > 0 && top_p < 1.0 {
        return hayai_model::SamplerConfig::TopKTopP {
            temperature,
            top_k,
            top_p,
        };
    }
    if top_k > 0 {
        return hayai_model::SamplerConfig::TopK {
            temperature,
            top_k,
        };
    }
    if top_p < 1.0 {
        return hayai_model::SamplerConfig::TopP {
            temperature,
            top_p,
        };
    }
    hayai_model::SamplerConfig::Temperature { temperature }
}

/// Build engine penalties from OpenAI-style request fields.
pub fn penalties_from(
    presence_penalty: f32,
    frequency_penalty: f32,
    logit_bias: Option<&std::collections::HashMap<String, f32>>,
) -> hayai_model::Penalties {
    let logit_bias = logit_bias
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, *v)))
                .collect()
        })
        .unwrap_or_default();
    hayai_model::Penalties {
        repetition: 1.0,
        presence: presence_penalty,
        frequency: frequency_penalty,
        logit_bias,
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Responses
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct TextChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

/// SSE chunk for chat completions.
#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// SSE chunk for legacy completions.
#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

impl ApiError {
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: msg.into(),
                ty: "invalid_request_error".into(),
                code: Some("model_not_found".into()),
            },
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: msg.into(),
                ty: "internal_error".into(),
                code: None,
            },
        }
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: msg.into(),
                ty: "invalid_request_error".into(),
                code: None,
            },
        }
    }
}

pub fn request_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:x}")
}

pub fn created_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Best-effort parse of model-emitted tool calls into OpenAI `tool_calls`.
///
/// Supports the common delimited formats (Qwen2.5 / Hermes / Llama-3 tool tags):
/// `<tool_call>{"name":..,"arguments":..}</tool_call>` (one or more), where
/// `arguments` is an object or a JSON string. Returns `None` when the text has no
/// tool-call block, so callers keep the plain assistant message.
pub fn parse_tool_calls(text: &str) -> Option<Vec<serde_json::Value>> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find(CLOSE) else { break };
        let body = after[..end].trim();
        if let Ok(mut v) = serde_json::from_str::<serde_json::Value>(body) {
            // Normalize to {"type":"function","function":{"name","arguments"}}.
            let name = v
                .get("name")
                .or_else(|| v.pointer("/function/name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let args = v
                .get_mut("arguments")
                .map(|a| a.take())
                .or_else(|| v.pointer("/function/arguments").cloned())
                .unwrap_or(serde_json::Value::Object(Default::default()));
            let args = match args {
                serde_json::Value::String(s) => {
                    serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s))
                }
                other => other,
            };
            out.push(serde_json::json!({
                "id": format!("call_{}", out.len()),
                "type": "function",
                "function": { "name": name, "arguments": args.to_string() }
            }));
        }
        rest = &after[end + CLOSE.len()..];
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_maps_top_k_and_min_p() {
        assert!(matches!(
            sampler_from(0.7, 0.9, 0, 0.05),
            hayai_model::SamplerConfig::MinP { .. }
        ));
        assert!(matches!(
            sampler_from(0.7, 1.0, 5, 0.0),
            hayai_model::SamplerConfig::TopK { .. }
        ));
        assert!(matches!(
            sampler_from(0.7, 0.9, 5, 0.0),
            hayai_model::SamplerConfig::TopKTopP { .. }
        ));
        assert!(matches!(
            sampler_from(0.0, 1.0, 0, 0.0),
            hayai_model::SamplerConfig::Greedy
        ));
    }

    #[test]
    fn request_defaults_deserialize() {
        let req: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"hi"}"#).unwrap();
        assert_eq!(req.top_k, 0);
        assert_eq!(req.min_p, 0.0);
        assert_eq!(req.presence_penalty, 0.0);
        assert!(req.logit_bias.is_none());
    }

    #[test]
    fn penalties_parse_logit_bias_keys() {
        let mut m = std::collections::HashMap::new();
        m.insert("42".to_string(), 5.0);
        m.insert("not-a-number".to_string(), 1.0);
        let p = penalties_from(0.5, 0.25, Some(&m));
        assert_eq!(p.logit_bias, vec![(42, 5.0)]);
        assert_eq!(p.presence, 0.5);
        assert_eq!(p.frequency, 0.25);
    }

    #[test]
    fn parses_qwen_style_tool_calls() {
        let text = "Sure.<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let calls = parse_tool_calls(text).expect("tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn parses_string_arguments_and_multiple_calls() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":\"{\\\"x\\\":1}\"}</tool_call>\
                    <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "a");
        let a: serde_json::Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a["x"], 1);
        assert_eq!(calls[1]["function"]["name"], "b");
    }

    #[test]
    fn no_tool_call_block_returns_none() {
        assert!(parse_tool_calls("just a normal answer").is_none());
        assert!(parse_tool_calls("<tool_call>not json</tool_call>").is_none());
    }

    #[test]
    fn tool_request_fields_deserialize() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"weather?"}],
                "tools":[{"type":"function","function":{"name":"get_weather","parameters":{}}}],
                "tool_choice":"auto"}"#,
        )
        .unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tool_choice.as_ref().unwrap(), "auto");
    }
}
