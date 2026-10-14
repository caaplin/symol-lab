//! AquaAlert — sensorless-first early disease & water-quality alerts for small
//! aquaculture farms.
//!
//! A faithful, 1:1 Rust rewrite of the Python/uv backend (`src/aqua_alert/`), so
//! the whole service — risk engine, chemistry, drift detection, alert dispatcher,
//! public-data catalogue, weather client, photo-triage AI and HTTP API — compiles
//! to a **single self-contained binary**.
//!
//! Module map (mirrors the Python package):
//! * [`schemas`]    <- `schemas.py`      typed contracts (serde == pydantic JSON)
//! * [`thresholds`] <- `thresholds.py`   species envelopes + water chemistry
//! * [`risk`]       <- `risk.py`         severity -> band fusion; night DO-crash forecast
//! * [`anomaly`]    <- `anomaly.py`      per-pond robust MAD z-score drift
//! * [`pipeline`]   <- `pipeline.py`     weather + readings + photo -> FarmDay
//! * [`sources`]    <- `sources/`        registry, weather, synthetic, usgs_wqp
//! * [`alerts`]     <- `alerts/`         dispatcher (SMS/IVR), escalation matrix
//! * [`disease`]    <- `disease/`        classifier (ONNX model + heuristic fallback)
//! * [`api`]        <- `api/server.py`   axum HTTP service (feature = "server")

#![forbid(unsafe_code)]

pub mod anomaly;
pub mod csv;
pub mod pipeline;
pub mod risk;
pub mod schemas;
pub mod thresholds;
pub mod util;

pub mod alerts;
pub mod disease;
pub mod sources;

#[cfg(feature = "server")]
pub mod api;

/// Service version, reported by `/health` (matches the Python `app.version`).
pub const VERSION: &str = "0.1.0";

/// Service name, reported by `/health`.
pub const SERVICE: &str = "aqua-alert";
