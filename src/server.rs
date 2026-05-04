use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Sse, sse::Event},
    routing::post,
};
use futures::StreamExt;

use crate::adapter::AdapterError;
use crate::entry::{anthropic, openai};
use crate::ir;
use crate::router::Router as ModelRouter;
use tracing::{error, info};

pub struct AppState {
    pub router: ModelRouter,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat_handler))
        .route("/v1/messages", post(anthropic_messages_handler))
        .route("/health", axum::routing::get(health_handler))
        .route("/healthz", axum::routing::get(health_handler))
        .route("/readyz", axum::routing::get(ready_handler))
        .with_state(state)
}

// ── OpenAI-compatible endpoint ──────────────────────────────────────────────

async fn openai_chat_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(body): axum::Json<serde_json::Value>,
) -> Result<axum::response::Response, AppError> {
    let openai_req: openai::OpenAIChatRequest =
        serde_json::from_value(body).map_err(|e| AppError::bad_request(e.to_string()))?;
    let stream = openai_req.stream;
    let mut ir_req = openai::openai_to_ir(&openai_req);
    info!(entry = "openai", model = %ir_req.model, stream, "incoming request");
    let route = state.router.resolve(&ir_req.model).map_err(AppError::from_adapter)?;

    // Apply model rewrite so the backend sees the correct model name.
    let display_model = ir_req.model.clone();
    ir_req.model = route.model;
    info!(entry = "openai", display_model = %display_model, backend_model = %ir_req.model, provider = route.adapter.provider_name(), "resolved route");

    if stream {
        handle_openai_stream(route.adapter, ir_req, display_model).await
    } else {
        handle_openai_chat(route.adapter, ir_req, display_model).await
    }
}

async fn handle_openai_chat(
    adapter: Arc<dyn crate::adapter::Adapter>,
    ir_req: ir::ChatRequest,
    display_model: String,
) -> Result<axum::response::Response, AppError> {
    let mut ir_resp = adapter.chat(&ir_req).await.map_err(AppError::from_adapter)?;
    ir_resp.model = display_model;
    let openai_resp = openai::ir_to_openai_response(ir_resp);
    let body = serde_json::to_string(&openai_resp).unwrap_or_default();
    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response())
}

async fn handle_openai_stream(
    adapter: Arc<dyn crate::adapter::Adapter>,
    ir_req: ir::ChatRequest,
    model: String,
) -> Result<axum::response::Response, AppError> {
    let backend_stream = adapter
        .chat_stream(&ir_req)
        .await
        .map_err(AppError::from_adapter)?;

    let message_id = format!("chatcmpl-{}", uuid_v4());

    let sse_stream = backend_stream.filter_map(move |result| {
        let mid = message_id.clone();
        let m = model.clone();
        async move {
            match result {
                Ok(stream_event) => {
                    let line = openai::ir_to_openai_sse(stream_event, &mid, &m);
                    line.map(|data| {
                        Ok::<_, Infallible>(Event::default().data(openai_sse_data(&data)))
                    })
                }
                Err(e) => Some(Ok(Event::default().data(format!(
                    "data: {{\"error\": \"{}\"}}\n",
                    e
                )))),
            }
        }
    });

    Ok(Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response())
}

fn openai_sse_data(line: &str) -> String {
    line.trim()
        .strip_prefix("data: ")
        .unwrap_or(line.trim())
        .to_string()
}

// ── Anthropic-compatible endpoint ───────────────────────────────────────────

async fn anthropic_messages_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(body): axum::Json<serde_json::Value>,
) -> Result<axum::response::Response, AppError> {
    let anthro_req: anthropic::AnthropicMessageRequest =
        serde_json::from_value(body).map_err(|e| AppError::bad_request(e.to_string()))?;
    let stream = anthro_req.stream;
    let mut ir_req = anthropic::anthropic_to_ir(&anthro_req);
    info!(entry = "anthropic", model = %ir_req.model, stream, "incoming request");
    let route = state.router.resolve(&ir_req.model).map_err(AppError::from_adapter)?;

    // Apply model rewrite.
    let display_model = ir_req.model.clone();
    ir_req.model = route.model;
    info!(entry = "anthropic", display_model = %display_model, backend_model = %ir_req.model, provider = route.adapter.provider_name(), "resolved route");

    if stream {
        handle_anthropic_stream(route.adapter, ir_req, display_model).await
    } else {
        handle_anthropic_chat(route.adapter, ir_req, display_model).await
    }
}

async fn handle_anthropic_chat(
    adapter: Arc<dyn crate::adapter::Adapter>,
    ir_req: ir::ChatRequest,
    display_model: String,
) -> Result<axum::response::Response, AppError> {
    let mut ir_resp = adapter.chat(&ir_req).await.map_err(AppError::from_adapter)?;
    ir_resp.model = display_model;
    let anthro_resp = anthropic::ir_to_anthropic_response(ir_resp);
    let body = serde_json::to_string(&anthro_resp).unwrap_or_default();
    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response())
}

async fn handle_anthropic_stream(
    adapter: Arc<dyn crate::adapter::Adapter>,
    ir_req: ir::ChatRequest,
    _display_model: String,
) -> Result<axum::response::Response, AppError> {
    let backend_stream = adapter
        .chat_stream(&ir_req)
        .await
        .map_err(AppError::from_adapter)?;

    let sse_stream = backend_stream.filter_map(|result| async move {
        match result {
            Ok(stream_event) => {
                let sse = anthropic::ir_to_anthropic_sse(stream_event);
                sse.map(|(event_type, data)| {
                    Ok::<_, Infallible>(Event::default().event(event_type).data(data))
                })
            }
            Err(e) => Some(Ok(Event::default().data(format!(
                "{{\"error\": \"{}\"}}",
                e
            )))),
        }
    });

    Ok(Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response())
}

// ── Health ──────────────────────────────────────────────────────────────────

async fn health_handler() -> &'static str {
    "ok"
}

async fn ready_handler() -> &'static str {
    "ready"
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:032x}", ts)
}

// ── Error handling ──────────────────────────────────────────────────────────

struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg,
        }
    }

    fn from_adapter(err: AdapterError) -> Self {
        let status = match &err {
            AdapterError::BackendError(_) => StatusCode::BAD_GATEWAY,
            AdapterError::TranslationError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AdapterError::StreamError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AdapterError::UnsupportedFeature { .. } => StatusCode::NOT_IMPLEMENTED,
        };
        Self {
            status,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        error!(status = %self.status, error = %self.message, "request failed");
        let body = serde_json::json!({"error": {"message": self.message}});
        (self.status, axum::Json(body)).into_response()
    }
}
