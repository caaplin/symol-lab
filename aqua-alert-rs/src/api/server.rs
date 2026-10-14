//! The HTTP service: the thing a co-operative actually runs.
//!
//! 1:1 port of `src/aqua_alert/api/server.py` (FastAPI -> axum). Endpoints:
//!
//! GET  /health                     liveness
//! GET  /                           the farmer alert screen (deliverable #3), embedded
//! GET  /catalog                    the public-data catalogue (deliverable #2), JSON
//! GET  /catalog/{entry_id}         one catalogue entry (404 with {"detail": …})
//! GET  /pond/{pond_id}/status      last verdict for a pond (cold-starts from the sample)
//! POST /pond/{pond_id}/reading     farmer reading -> verdict + alerts
//! POST /pond/{pond_id}/photo       multipart image -> disease triage
//! GET  /pond/{pond_id}/forecast    next 3 nights of DO-crash risk from Open-Meteo
//! GET  /escalation-matrix          who gets told what
//! POST /simulate/crash-night       demo: play the RED scenario end-to-end
//!
//! Response bodies match the FastAPI service field-for-field (serde == pydantic
//! `model_dump(mode="json")`), so the farmer screen and any existing client work
//! unchanged against either backend.

use crate::alerts::dispatcher::{self, AlertDispatcher};
use crate::disease::classifier;
use crate::pipeline::{self, FarmDay, Forecast};
use crate::schemas::PondReading;
use crate::sources::{registry, weather};
use crate::util::pyround;
use crate::{anomaly, risk, VERSION, SERVICE};

use axum::extract::{Multipart, Path, Query};
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{Duration, Local};
use once_cell::sync::Lazy;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// The farmer-facing alert screen (deliverable #3), embedded in the binary.
pub const FARMER_APP_HTML: &str = include_str!("../../assets/farmer_app.html");

/// In-memory store: a real deployment swaps this for SQLite/Postgres. Small co-ops
/// with 20-200 ponds genuinely do not need more than SQLite.
static STATE: Lazy<RwLock<HashMap<String, Value>>> = Lazy::new(|| RwLock::new(HashMap::new()));
/// tokio Mutex so the dedupe window can be held across the async weather fetch.
static DISPATCHER: Lazy<tokio::sync::Mutex<AlertDispatcher>> =
    Lazy::new(|| tokio::sync::Mutex::new(AlertDispatcher::new()));

const FARM_COORDS: (f64, f64) = (12.9716, 77.5946); // Bengaluru - override per pond in production

fn species_for(pond_id: &str) -> &'static str {
    match pond_id {
        "KA-BLR-001" => "rohu",
        "KA-BLR-002" => "tilapia",
        "AP-WGL-007" => "shrimp",
        _ => "tilapia",
    }
}

fn to_value<T: serde::Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

/// The STATE[pond_id] object: {assessment, night_band, night_reason, alerts}.
fn state_value(day: &FarmDay) -> Value {
    json!({
        "assessment": to_value(&day.assessment),
        "night_band": day.night_band.map(|b| b.as_str()),
        "night_reason": day.night_reason.clone(),
        "alerts": day.alerts.iter().map(|a| to_value(a)).collect::<Vec<_>>(),
    })
}

fn detail(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (status, Json(json!({ "detail": msg.into() })))
}

#[derive(Debug, Deserialize)]
struct ChannelQuery {
    channel: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ForecastQuery {
    lat: Option<f64>,
    lon: Option<f64>,
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true, "service": SERVICE, "version": VERSION }))
}

async fn farmer_screen() -> Html<&'static str> {
    Html(FARMER_APP_HTML)
}

async fn catalog() -> Json<Value> {
    Json(json!({ "count": registry::CATALOG.len(), "entries": registry::to_json() }))
}

async fn catalog_entry(Path(entry_id): Path<String>) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match registry::by_id(&entry_id) {
        Some(e) => Ok(Json(to_value(e))),
        None => Err(detail(StatusCode::NOT_FOUND, format!("no catalogue entry {entry_id}"))),
    }
}

async fn post_reading(
    Path(pond_id): Path<String>,
    Query(q): Query<ChannelQuery>,
    Json(mut reading): Json<PondReading>,
) -> Json<Value> {
    reading.pond_id = pond_id.clone();
    let species = species_for(&pond_id);
    let channel = q.channel.unwrap_or_else(|| "console".to_string());

    let mut disp = DISPATCHER.lock().await;
    let day = pipeline::assess_pond(
        reading,
        species,
        Some(&mut *disp),
        &channel,
        Some(FARM_COORDS.0),
        Some(FARM_COORDS.1),
        true,
    )
    .await;
    let val = state_value(&day);
    drop(disp);

    STATE.write().unwrap().insert(pond_id, val.clone());
    Json(val)
}

async fn status(Path(pond_id): Path<String>) -> Json<Value> {
    if let Some(v) = STATE.read().unwrap().get(&pond_id) {
        return Json(v.clone());
    }
    // cold start: derive from the bundled public-dataset-calibrated sample
    let species = species_for(&pond_id);
    let day = pipeline::demo_day(&pond_id, species, None);
    let val = json!({
        "assessment": to_value(&day.assessment),
        "night_band": Value::Null,
        "night_reason": Value::Null,
        "alerts": [],
    });
    STATE.write().unwrap().insert(pond_id, val.clone());
    Json(val)
}

async fn photo(
    Path(pond_id): Path<String>,
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut data: Option<Vec<u8>> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        // the form field is named "file"; accept the first field carrying bytes
        if let Ok(bytes) = field.bytes().await {
            if !bytes.is_empty() {
                data = Some(bytes.to_vec());
                break;
            }
        }
    }
    let bytes = data.ok_or_else(|| detail(StatusCode::BAD_REQUEST, "no file uploaded"))?;
    let species = species_for(&pond_id);
    match classifier::predict_bytes(&bytes, species, None) {
        Ok(pred) => Ok(Json(json!({ "pond_id": pond_id, "prediction": to_value(&pred) }))),
        Err(e) => Err(detail(StatusCode::BAD_REQUEST, format!("image triage failed: {e}"))),
    }
}

async fn forecast(
    Path(pond_id): Path<String>,
    Query(q): Query<ForecastQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let lat = q.lat.unwrap_or(FARM_COORDS.0);
    let lon = q.lon.unwrap_or(FARM_COORDS.1);
    let today = Local::now().date_naive();
    let days = match weather::fetch_daily(lat, lon, today, today + Duration::days(3), 20.0).await {
        Ok(d) => d,
        Err(_) => return Err(detail(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error")),
    };
    let nights: Vec<Value> = days
        .iter()
        .map(|d| {
            let (band, reason) =
                risk::do_crash_forecast(d.t_max, d.t_min, d.wind_ms, d.precip_mm, d.cloud_pct, None);
            json!({
                "date": d.day.format("%Y-%m-%d").to_string(),
                "band": band.as_str(),
                "reason": reason,
                "virtual_water_temp_c": d.virtual_water_temp(),
                "rain_mm": d.precip_mm,
                "t_max": d.t_max,
                "t_min": d.t_min,
                "wind_ms": pyround(d.wind_ms, 1),
            })
        })
        .collect();
    Ok(Json(json!({
        "pond_id": pond_id,
        "coords": { "lat": lat, "lon": lon },
        "nights": nights,
        "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)",
    })))
}

async fn escalation_matrix() -> Json<Value> {
    let map: serde_json::Map<String, Value> = dispatcher::escalation_matrix()
        .into_iter()
        .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
        .collect();
    Json(Value::Object(map))
}

async fn simulate_crash() -> Json<Value> {
    // Injects a classic pre-monsoon crash night: hot, still, 05:30 DO = 1.6 mg/L.
    let reading = PondReading {
        pond_id: "KA-BLR-001".into(),
        water_temp_c: Some(33.4),
        do_mgl: Some(1.6),
        ph: Some(8.9),
        turbidity_ntu: Some(180.0),
        ammonia_total_mgl: Some(1.9),
        nitrite_mgl: Some(0.8),
        source: "simulated crash night".into(),
        ..Default::default()
    };
    let mut disp = DISPATCHER.lock().await;
    let day = pipeline::assess_internal(reading, "rohu", Some(&mut *disp), "console", Forecast::Disabled);
    Json(json!({
        "assessment": to_value(&day.assessment),
        "alerts": day.alerts.iter().map(|a| to_value(a)).collect::<Vec<_>>(),
    }))
}

/// Swagger UI shell. FastAPI serves this at `/docs` from a CDN bundle; the axum port
/// does the same, pointed at our hand-written `/openapi.json`.
const DOCS_HTML: &str = r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AquaAlert API</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"/>
</head><body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>window.onload=()=>{SwaggerUIBundle({url:"/openapi.json",dom_id:"#swagger-ui"});};</script>
</body></html>"#;

async fn docs() -> Html<&'static str> {
    Html(DOCS_HTML)
}

/// A hand-written OpenAPI 3.0 document. FastAPI introspects this from the pydantic
/// models; axum has no equivalent auto-generation, so the contract is described here.
async fn openapi_json() -> Json<Value> {
    Json(json!({
        "openapi": "3.0.3",
        "info": {
            "title": "AquaAlert",
            "version": VERSION,
            "description": "Sensorless-first early disease & water-quality alerts for small aquaculture farms.",
        },
        "servers": [{ "url": "/" }],
        "paths": {
            "/health": { "get": { "summary": "Liveness", "responses": { "200": { "description": "OK" } } } },
            "/": { "get": { "summary": "Farmer alert screen (HTML)", "responses": { "200": { "description": "HTML" } } } },
            "/catalog": {
                "get": {
                    "summary": "Public-data catalogue",
                    "parameters": [{ "name": "kind", "in": "query", "schema": { "type": "string" } }],
                    "responses": { "200": { "description": "count + entries" } },
                }
            },
            "/catalog/{entry_id}": {
                "get": {
                    "summary": "One catalogue entry",
                    "parameters": [{ "name": "entry_id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": { "200": { "description": "entry" }, "404": { "description": "not found" } },
                }
            },
            "/escalation-matrix": { "get": { "summary": "Who is told what, and how fast", "responses": { "200": { "description": "band -> rule" } } } },
            "/pond/{pond_id}/reading": {
                "post": {
                    "summary": "Post one morning reading -> live verdict + alerts",
                    "parameters": [
                        { "name": "pond_id", "in": "path", "required": true, "schema": { "type": "string" } },
                        { "name": "channel", "in": "query", "schema": { "type": "string" } },
                    ],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/PondReading" } } } },
                    "responses": { "200": { "description": "assessment + night outlook + alerts" } },
                }
            },
            "/pond/{pond_id}/photo": {
                "post": {
                    "summary": "Multipart photo -> disease triage",
                    "parameters": [{ "name": "pond_id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "requestBody": { "required": true, "content": { "multipart/form-data": { "schema": { "type": "object", "properties": { "file": { "type": "string", "format": "binary" } } } } } },
                    "responses": { "200": { "description": "prediction" }, "400": { "description": "no/undecodable file" } },
                }
            },
            "/pond/{pond_id}/status": {
                "get": {
                    "summary": "Last verdict for a pond (cold-starts from the sample)",
                    "parameters": [{ "name": "pond_id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": { "200": { "description": "state object" } },
                }
            },
            "/pond/{pond_id}/forecast": {
                "get": {
                    "summary": "Next nights of DO-crash risk (Open-Meteo)",
                    "parameters": [
                        { "name": "pond_id", "in": "path", "required": true, "schema": { "type": "string" } },
                        { "name": "lat", "in": "query", "schema": { "type": "number" } },
                        { "name": "lon", "in": "query", "schema": { "type": "number" } },
                    ],
                    "responses": { "200": { "description": "nights" }, "500": { "description": "weather unavailable" } },
                }
            },
            "/simulate/crash-night": { "post": { "summary": "Play the RED crash scenario end-to-end", "responses": { "200": { "description": "assessment + alerts" } } } },
        },
        "components": {
            "schemas": {
                "PondReading": {
                    "type": "object",
                    "properties": {
                        "ts": { "type": "string", "format": "date-time" },
                        "water_temp_c": { "type": "number", "nullable": true },
                        "do_mgl": { "type": "number", "nullable": true },
                        "ph": { "type": "number", "nullable": true },
                        "turbidity_ntu": { "type": "number", "nullable": true },
                        "ammonia_total_mgl": { "type": "number", "nullable": true },
                        "nitrite_mgl": { "type": "number", "nullable": true },
                        "tds_mgl": { "type": "number", "nullable": true },
                        "secchi_cm": { "type": "number", "nullable": true },
                        "salinity_ppt": { "type": "number", "nullable": true },
                        "source": { "type": "string" },
                        "note": { "type": "string", "nullable": true },
                    },
                }
            }
        }
    }))
}

/// The application router (exposed for tests and for embedding).
pub fn router() -> Router {
    Router::new()
        .route("/", get(farmer_screen))
        .route("/health", get(health))
        .route("/docs", get(docs))
        .route("/openapi.json", get(openapi_json))
        .route("/catalog", get(catalog))
        .route("/catalog/:entry_id", get(catalog_entry))
        .route("/pond/:pond_id/reading", post(post_reading))
        .route("/pond/:pond_id/status", get(status))
        .route("/pond/:pond_id/photo", post(photo))
        .route("/pond/:pond_id/forecast", get(forecast))
        .route("/escalation-matrix", get(escalation_matrix))
        .route("/simulate/crash-night", post(simulate_crash))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(TraceLayer::new_for_http())
}

/// Run the API + farmer alert screen.
pub async fn serve(host: &str, port: u16) -> anyhow::Result<()> {
    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("AquaAlert listening on http://{addr}  (farmer screen at /, catalogue at /catalog)");
    axum::serve(listener, router())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown signal received");
        })
        .await?;
    Ok(())
}

/// Clear the in-memory store and per-pond drift history (mirrors `reset_state()`).
pub fn reset_state() {
    STATE.write().unwrap().clear();
    anomaly::reset();
}
