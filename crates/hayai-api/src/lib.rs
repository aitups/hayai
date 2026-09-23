//! OpenAI-compatible HTTP API for the Hayai engine.

pub mod chat;
pub mod engine;
pub mod handlers;
pub mod hf;
pub mod models;
pub mod registry;

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Router;
use models::ApiError;
use registry::ModelRegistry;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Shared application state passed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    /// Bounds concurrent generations so a request flood cannot exhaust RAM/disk.
    pub gen_permits: Arc<Semaphore>,
    /// When set, `/v1/*` requires `Authorization: Bearer <key>`.
    pub api_key: Option<Arc<String>>,
}

/// Reject `/v1/*` requests without the configured bearer token.
async fn require_api_key(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(key) = state.api_key.as_deref() {
        let authorized = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|t| constant_time_eq(t.as_bytes(), key.as_bytes()))
            .unwrap_or(false);
        if !authorized {
            return (
                StatusCode::UNAUTHORIZED,
                axum::Json(ApiError::bad_request("missing or invalid API key")),
            )
                .into_response();
        }
    }
    next.run(req).await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Build the axum router with the OpenAI-compatible endpoints.
pub fn router(registry: Arc<ModelRegistry>, max_concurrency: usize, api_key: Option<String>) -> Router {
    let state = AppState {
        registry,
        gen_permits: Arc::new(Semaphore::new(max_concurrency.max(1))),
        api_key: api_key.map(Arc::new),
    };
    let v1 = Router::new()
        .route("/v1/models", axum::routing::get(handlers::list_models))
        .route("/v1/completions", axum::routing::post(handlers::completions))
        .route(
            "/v1/chat/completions",
            axum::routing::post(handlers::chat_completions),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ));
    Router::new()
        .route("/healthz", axum::routing::get(handlers::healthz))
        .merge(v1)
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .with_state(state)
}
