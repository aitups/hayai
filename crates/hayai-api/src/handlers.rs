//! axum handlers for the OpenAI-compatible endpoints.

use crate::engine;
use crate::models::{ApiError, ChatChunk, ChatCompletionResponse, ModelInfo, ModelList, Usage};
use crate::registry::{ModelHandle, ModelRegistry};
use crate::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Json};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

pub async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let data = state
        .registry
        .ids()
        .into_iter()
        .map(|id| ModelInfo {
            id,
            object: "model",
            created: 0,
            owned_by: "hayai",
        })
        .collect();
    Json(ModelList {
        object: "list",
        data,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// chat /completions
// ─────────────────────────────────────────────────────────────────────────────

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<crate::models::ChatCompletionRequest>,
) -> Response {
    if let Some(e) =
        crate::models::validate_generation(req.max_tokens, req.n, req.logprobs.as_ref())
    {
        return err_response(StatusCode::BAD_REQUEST, e);
    }
    let permit = match state.gen_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return err_response(
                StatusCode::SERVICE_UNAVAILABLE,
                ApiError::internal("server is shutting down"),
            )
        }
    };
    let handle = match state.registry.handle(&req.model) {
        Ok(h) => h,
        Err(e) => return err_response(StatusCode::NOT_FOUND, e),
    };

    if req.stream {
        return stream_chat_response(state.registry.clone(), handle, req, permit);
    }

    let rid = crate::models::request_id();
    let created = crate::models::created_now();
    let tools = req.tools.clone().unwrap_or_else(|| serde_json::json!([]));
    let prompt = engine::chat_prompt(&handle, &req.messages, &tools);
    let handle2 = handle.clone();
    let mut params = crate::models::SamplingParams::new(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.presence_penalty,
        req.frequency_penalty,
        req.logit_bias.as_ref(),
        req.seed,
    );
    if let Err(e) = params.apply_grammar(req.grammar.as_deref()) {
        return err_response(StatusCode::BAD_REQUEST, ApiError::bad_request(e));
    }
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        engine::generate_text(
            handle2,
            &prompt,
            req.max_tokens,
            params,
            &req.stop,
            |_| Ok(()),
        )
    })
    .await;

    match result {
        Ok(Ok((text, finish, prompt_tokens, completion_tokens))) => {
            // Best-effort: surface model-emitted tool calls as OpenAI `tool_calls`.
            let tool_calls = crate::models::parse_tool_calls(&text);
            let (content, finish) = if tool_calls.is_some() {
                (String::new(), "tool_calls".to_string())
            } else {
                (text, finish)
            };
            let body = ChatCompletionResponse {
                id: rid,
                object: "chat.completion",
                created,
                model: req.model,
                choices: vec![crate::models::ChatChoice {
                    index: 0,
                    message: crate::models::ChatMessage {
                        role: "assistant".into(),
                        content,
                        tool_calls: tool_calls.map(serde_json::Value::Array),
                        ..Default::default()
                    },
                    finish_reason: finish,
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            Json(body).into_response()
        }
        Ok(Err(e)) => err_response(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(_) => err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::internal("blocking task panicked"),
        ),
    }
}

/// SSE streaming for chat completions: one event per delta + `[DONE]`.
fn stream_chat_response(
    _registry: Arc<ModelRegistry>,
    handle: Arc<ModelHandle>,
    req: crate::models::ChatCompletionRequest,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    let tools = req.tools.clone().unwrap_or_else(|| serde_json::json!([]));
    let prompt = engine::chat_prompt(&handle, &req.messages, &tools);
    let handle2 = handle.clone();
    let model = req.model.clone();
    let stop = req.stop.clone();
    let rid = crate::models::request_id();
    let mut params = crate::models::SamplingParams::new(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.presence_penalty,
        req.frequency_penalty,
        req.logit_bias.as_ref(),
        req.seed,
    );
    if let Err(e) = params.apply_grammar(req.grammar.as_deref()) {
        return err_response(StatusCode::BAD_REQUEST, ApiError::bad_request(e));
    }
    tokio::spawn(async move {
        let _permit = permit;
        let _ = tokio::task::spawn_blocking(move || {
            stream_chat(tx, handle2, rid, model, &prompt, req.max_tokens, params, &stop)
        })
        .await;
    });
    let stream = ReceiverStream::new(rx);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

/// Streaming loop: emit one SSE chunk per delta, then a `[DONE]` event.
#[allow(clippy::too_many_arguments)]
fn stream_chat(
    tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    handle: Arc<ModelHandle>,
    rid: String,
    model: String,
    prompt: &str,
    max_tokens: usize,
    params: crate::models::SamplingParams,
    stop: &[String],
) {
    let model_name = model.clone();
    let rid_ = rid.clone();
    let send = |delta: &str| {
        let chunk = ChatChunk {
            id: rid.clone(),
            object: "chat.completion.chunk",
            created: crate::models::created_now(),
            model: model.clone(),
            choices: vec![crate::models::ChatChunkChoice {
                index: 0,
                delta: crate::models::ChatMessage {
                    role: "assistant".into(),
                    content: delta.to_string(),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let _ = tx.blocking_send(Ok(Event::default().data(
            serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".into()),
        )));
        Ok::<(), ApiError>(())
    };
    match engine::generate_text(handle, prompt, max_tokens, params, stop, send) {
        Ok((_text, finish, _p, _c)) => {
            let chunk = ChatChunk {
                id: rid_,
                object: "chat.completion.chunk",
                created: crate::models::created_now(),
                model: model_name,
                choices: vec![crate::models::ChatChunkChoice {
                    index: 0,
                    delta: crate::models::ChatMessage {
                        role: "assistant".into(),
                        content: String::new(),
                        ..Default::default()
                    },
                    finish_reason: Some(finish),
                }],
            };
            let _ = tx.blocking_send(Ok(Event::default().data(
                serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".into()),
            )));
        }
        Err(e) => {
            let _ = tx.blocking_send(Ok(Event::default().data(
                serde_json::to_string(&e).unwrap_or_else(|_| "{}".into()),
            )));
        }
    }
    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
}



// ─────────────────────────────────────────────────────────────────────────────
// /completions (legacy)
// ─────────────────────────────────────────────────────────────────────────────

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<crate::models::CompletionRequest>,
) -> Response {
    if let Some(e) =
        crate::models::validate_generation(req.max_tokens, req.n, req.logprobs.as_ref())
    {
        return err_response(StatusCode::BAD_REQUEST, e);
    }
    let permit = match state.gen_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return err_response(
                StatusCode::SERVICE_UNAVAILABLE,
                ApiError::internal("server is shutting down"),
            )
        }
    };
    let handle = match state.registry.handle(&req.model) {
        Ok(h) => h,
        Err(e) => return err_response(StatusCode::NOT_FOUND, e),
    };

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
        let prompt = req.prompt.clone();
        let handle2 = handle.clone();
        let model = req.model.clone();
        let stop = req.stop.clone();
        let rid = crate::models::request_id();
        let mut params = crate::models::SamplingParams::new(
            req.temperature,
            req.top_p,
            req.top_k,
            req.min_p,
            req.presence_penalty,
            req.frequency_penalty,
            req.logit_bias.as_ref(),
            req.seed,
        );
        if let Err(e) = params.apply_grammar(req.grammar.as_deref()) {
            return err_response(StatusCode::BAD_REQUEST, ApiError::bad_request(e));
        }
        tokio::spawn(async move {
            let _permit = permit;
            let _ = tokio::task::spawn_blocking(move || {
                stream_completion(
                    tx, handle2, rid, model, &prompt, req.max_tokens, params, &stop,
                )
            })
            .await;
        });
        let stream = ReceiverStream::new(rx);
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }

    let rid = crate::models::request_id();
    let created = crate::models::created_now();
    let prompt = req.prompt.clone();
    let handle2 = handle.clone();
    let mut params = crate::models::SamplingParams::new(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.presence_penalty,
        req.frequency_penalty,
        req.logit_bias.as_ref(),
        req.seed,
    );
    if let Err(e) = params.apply_grammar(req.grammar.as_deref()) {
        return err_response(StatusCode::BAD_REQUEST, ApiError::bad_request(e));
    }
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        engine::generate_text(handle2, &prompt, req.max_tokens, params, &req.stop, |_| Ok(()))
    })
    .await;

    match result {
        Ok(Ok((text, finish, prompt_tokens, completion_tokens))) => {
            let body = crate::models::CompletionResponse {
                id: rid,
                object: "text_completion",
                created,
                model: req.model,
                choices: vec![crate::models::TextChoice {
                    text,
                    index: 0,
                    finish_reason: finish,
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            Json(body).into_response()
        }
        Ok(Err(e)) => err_response(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(_) => err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::internal("blocking task panicked"),
        ),
    }
}



#[allow(clippy::too_many_arguments)]
fn stream_completion(
    tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    handle: Arc<ModelHandle>,
    rid: String,
    model: String,
    prompt: &str,
    max_tokens: usize,
    params: crate::models::SamplingParams,
    stop: &[String],
) {
    let model_name = model.clone();
    let rid_ = rid.clone();
    let send = |delta: &str| {
        let chunk = crate::models::CompletionChunk {
            id: rid.clone(),
            object: "text_completion",
            created: crate::models::created_now(),
            model: model.clone(),
            choices: vec![crate::models::CompletionChunkChoice {
                text: delta.to_string(),
                index: 0,
                finish_reason: None,
            }],
        };
        let _ = tx.blocking_send(Ok(Event::default().data(
            serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".into()),
        )));
        Ok::<(), ApiError>(())
    };
    match engine::generate_text(handle, prompt, max_tokens, params, stop, send) {
        Ok((_text, finish, _p, _c)) => {
            let chunk = crate::models::CompletionChunk {
                id: rid_,
                object: "text_completion",
                created: crate::models::created_now(),
                model: model_name,
                choices: vec![crate::models::CompletionChunkChoice {
                    text: String::new(),
                    index: 0,
                    finish_reason: Some(finish),
                }],
            };
            let _ = tx.blocking_send(Ok(Event::default().data(
                serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".into()),
            )));
        }
        Err(e) => {
            let _ = tx.blocking_send(Ok(Event::default().data(
                serde_json::to_string(&e).unwrap_or_else(|_| "{}".into()),
            )));
        }
    }
    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
}

fn err_response(status: StatusCode, e: ApiError) -> Response {
    (status, Json(e)).into_response()
}

