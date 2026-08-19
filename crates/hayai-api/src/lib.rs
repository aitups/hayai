//! OpenAI-compatible HTTP API for the Hayai engine.

pub mod chat;
pub mod engine;
pub mod handlers;
pub mod hf;
pub mod models;
pub mod registry;

use axum::Router;
use registry::ModelRegistry;
use std::sync::Arc;

/// Shared application state passed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
}

/// Build the axum router with the OpenAI-compatible endpoints.
pub fn router(registry: Arc<ModelRegistry>) -> Router {
    Router::new()
        .route("/healthz", axum::routing::get(handlers::healthz))
        .route("/v1/models", axum::routing::get(handlers::list_models))
        .route("/v1/completions", axum::routing::post(handlers::completions))
        .route(
            "/v1/chat/completions",
            axum::routing::post(handlers::chat_completions),
        )
        .with_state(AppState { registry })
}
