//! HTTP API server, wire-compatible with the (now-shutdown) remove.bg API.
//!
//! Endpoint: `POST /v1.0/removebg`
//!
//! Accepted fields (multipart/form-data), matching remove.bg exactly:
//! - `image_file`  — the image to process (binary)
//! - `image_url`   — alternative: fetch the image from a URL
//! - `size`        — `auto` (default) | `preview` | `full` | `regular`
//! - `format`      — `auto` (default) | `png` | `jpg` | `zip`
//! - `bg_color`    — `#rrggbb` hex colour for a solid background (omit = transparent)
//!
//! Remove.bg-specific headers (`X-Api-Key`, `Accept`) are accepted and
//! ignored, so existing client code keeps working after changing only the URL.
//!
//! The same server also serves the web demo at `/` from `static/index.html`.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};

use crate::engine::{Engine, OutputFormat, OutputSize, RemoveBgOptions};

/// Run the HTTP server until cancelled.
pub async fn serve(engine: Engine, addr: &str, no_web_demo: bool) -> Result<()> {
    let socket_addr: SocketAddr = addr
        .parse()
        .with_context(|| format!("invalid bind address `{addr}` (expected host:port)"))?;

    let state: Arc<AppState> = Arc::new(AppState { engine });

    let router = axum::Router::new()
        .route("/v1.0/removebg", post(removebg))
        .route("/health", get(health))
        .layer(CorsLayer::very_permissive())
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state);

    let router = if no_web_demo {
        router.fallback(not_found)
    } else {
        router.fallback_service(ServeDir::new("static"))
    };

    let listener = tokio::net::TcpListener::bind(socket_addr)
        .await
        .with_context(|| format!("failed to bind {socket_addr}"))?;
    tracing::info!("OpenBG API server listening on http://{socket_addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

struct AppState {
    engine: Engine,
}

#[derive(serde::Serialize)]
struct ErrorBody {
    errors: Vec<ErrorItem>,
}

#[derive(serde::Serialize)]
struct ErrorItem {
    title: &'static str,
    detail: String,
    code: &'static str,
}

fn json_error(code: &'static str, detail: impl Into<String>) -> Response {
    let body = ErrorBody {
        errors: vec![ErrorItem {
            title: "Request failed",
            detail: detail.into(),
            code,
        }],
    };
    (StatusCode::BAD_REQUEST, Json(body)).into_response()
}

/// `POST /v1.0/removebg` — remove.bg-compatible endpoint.
async fn removebg(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    // remove.bg accepted API keys; we don't enforce one, but accept the header.
    let _api_key = headers.get("x-api-key");

    let mut image_file: Option<Bytes> = None;
    let mut image_url: Option<String> = None;
    let mut size: Option<String> = None;
    let mut format: Option<String> = None;
    let mut bg_color: Option<String> = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return json_error("bad_request", format!("multipart parse error: {e}")),
        };
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "image_file" => match field.bytes().await {
                Ok(b) => image_file = Some(b),
                Err(e) => return json_error("bad_request", format!("image_file: {e}")),
            },
            "image_url" => {
                if let Ok(text) = field.text().await {
                    image_url = Some(text);
                }
            }
            "size" => {
                if let Ok(text) = field.text().await {
                    size = Some(text);
                }
            }
            "format" => {
                if let Ok(text) = field.text().await {
                    format = Some(text);
                }
            }
            "bg_color" => {
                if let Ok(text) = field.text().await {
                    bg_color = Some(text);
                }
            }
            _ => {
                // Tolerate and ignore unknown fields (remove.bg had several optional ones).
                let _ = field.bytes().await;
            }
        }
    }

    let opts = match build_opts(size.as_deref(), format.as_deref(), bg_color.as_deref()) {
        Ok(o) => o,
        Err(e) => return json_error("bad_request", e),
    };

    let bytes = if let Some(b) = image_file {
        b
    } else if let Some(url) = image_url {
        match fetch_url(&url).await {
            Ok(b) => b,
            Err(e) => return json_error("bad_request", format!("image_url fetch failed: {e}")),
        }
    } else {
        return json_error(
            "missing_image",
            "either `image_file` or `image_url` is required",
        );
    };

    let result = match state.engine.remove_bg_from_bytes(&bytes, opts).await {
        Ok(r) => r,
        Err(e) => return json_error("processing_failed", format!("{e:#}")),
    };

    let mime = match result.format {
        OutputFormat::Png | OutputFormat::Auto => "image/png",
        OutputFormat::Jpeg => "image/jpeg",
    };

    (
        [(header::CONTENT_TYPE, mime)],
        axum::body::Body::from(result.image_bytes),
    )
        .into_response()
}

async fn fetch_url(url: &str) -> Result<Bytes> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        bail!("HTTP {}", resp.status());
    }
    Ok(resp.bytes().await?)
}

fn build_opts(
    size: Option<&str>,
    format: Option<&str>,
    bg_color: Option<&str>,
) -> Result<RemoveBgOptions, String> {
    let format = match format.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("auto") => OutputFormat::Auto,
        Some("png") => OutputFormat::Png,
        Some("jpg") | Some("jpeg") => OutputFormat::Jpeg,
        Some("zip") => OutputFormat::Auto, // zip not supported; fall back to PNG
        Some(other) => return Err(format!("unknown format `{other}`")),
    };
    let size = match size.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("auto") => OutputSize::Auto,
        Some("preview") => OutputSize::Preview,
        Some("full") => OutputSize::Full,
        Some("regular") => OutputSize::Regular,
        Some(other) => return Err(format!("unknown size `{other}`")),
    };
    let bg_color = match bg_color.map(str::trim) {
        None => None,
        Some(s) => Some(parse_hex_color(s).map_err(|e: anyhow::Error| e.to_string())?),
    };
    Ok(RemoveBgOptions {
        format,
        size,
        bg_color,
    })
}

fn parse_hex_color(s: &str) -> Result<[u8; 3]> {
    let s = s.trim().trim_start_matches('#');
    if s.len() != 6 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("expected `#rrggbb`, got `{s}`");
    }
    let mut bytes = [0u8; 3];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
    }
    Ok(bytes)
}

async fn health() -> impl IntoResponse {
    "ok"
}

async fn not_found() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, "not found")
}
