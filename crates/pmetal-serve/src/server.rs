//! Server setup and configuration.

use crate::engine::InferenceEngine;
use crate::routes::{self, AppState, ServingMetrics};
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Port to listen on.
    pub port: u16,
    /// Host to bind to.
    pub host: String,
    /// Maximum concurrent requests.
    pub max_concurrent: usize,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "0.0.0.0".to_string(),
            max_concurrent: 16,
        }
    }
}

/// Build the axum router with all routes.
pub fn build_router(engine: InferenceEngine) -> Router {
    let state = Arc::new(AppState {
        engine,
        metrics: ServingMetrics::default(),
    });

    Router::new()
        .route("/health", axum::routing::get(routes::health))
        .route("/v1/models", axum::routing::get(routes::list_models))
        .route("/v1/metrics", axum::routing::get(routes::serving_metrics))
        .route(
            "/v1/chat/completions",
            axum::routing::post(routes::chat_completions),
        )
        .route("/v1/completions", axum::routing::post(routes::completions))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Start the server.
pub async fn run_server(engine: InferenceEngine, config: ServeConfig) -> anyhow::Result<()> {
    let router = build_router(engine);
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;

    tracing::info!("Starting PMetal inference server on {}", addr);
    tracing::info!("OpenAI-compatible API available at http://{}/v1", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
