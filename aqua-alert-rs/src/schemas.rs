//! Typed contracts shared by ingestion, the risk engine, alerts and the API.
//!
//! 1:1 port of `src/aqua_alert/schemas.py`. JSON field names and shapes match
//! pydantic's `model_dump(mode="json")` exactly, so the Rust service is a
//! drop-in replacement for the FastAPI one.

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};

/// Farmer-facing severity. Deliberately only 4 levels — a farmer acts on colour,
/// not numbers. Serialises as the upper-case string ("GREEN", …) like the Python
/// `str`-`Enum`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Band {
    /// safe, keep routine
    Green,
    /// watch, cheap preventive action
    Yellow,
    /// act today, real risk of loss
    Orange,
    /// emergency, act within hours
    Red,
}

impl Band {
    pub fn as_str(self) -> &'static str {
        match self {
            Band::Green => "GREEN",
            Band::Yellow => "YELLOW",
            Band::Orange => "ORANGE",
            Band::Red => "RED",
        }
    }

    pub fn parse(s: &str) -> Band {
        match s.trim().to_uppercase().as_str() {
            "YELLOW" => Band::Yellow,
            "ORANGE" => Band::Orange,
            "RED" => Band::Red,
            _ => Band::Green,
        }
    }
}

impl std::fmt::Display for Band {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Timeseries,
    Images,
    Thresholds,
    RemoteSensing,
    Weather,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::Timeseries => "timeseries",
            SourceKind::Images => "images",
            SourceKind::Thresholds => "thresholds",
            SourceKind::RemoteSensing => "remote_sensing",
            SourceKind::Weather => "weather",
        }
    }
}

fn default_ts() -> DateTime<Utc> {
    Utc::now()
}

fn default_pond_id() -> String {
    String::new()
}

fn default_source() -> String {
    "manual".to_string()
}

/// Parse an ISO-8601 timestamp, tolerating the naive (no offset) form the bundled
/// CSV and hand-written JSON use — those are treated as UTC, matching
/// `datetime.fromisoformat` in the Python code paths.
pub fn parse_ts(s: &str) -> DateTime<Utc> {
    let s = s.trim();
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return dt.with_timezone(&Utc);
    }
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ] {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
            return ndt.and_utc();
        }
    }
    Utc::now()
}

mod ts_utc {
    use super::parse_ts;
    use chrono::{DateTime, SecondsFormat, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(dt: &DateTime<Utc>, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // RFC3339 with automatic fractional-second precision + trailing "Z" —
        // exactly what pydantic emits for an aware UTC datetime.
        ser.serialize_str(&dt.to_rfc3339_opts(SecondsFormat::AutoSi, true))
    }

    pub fn deserialize<'de, D>(de: D) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(de)?;
        Ok(parse_ts(&s))
    }
}

/// One observation set for one pond at one time.
///
/// Every measurement field is `Option` on purpose: the sensorless path almost
/// never has all of them. The risk engine degrades gracefully and reports which
/// parameters are *missing*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PondReading {
    #[serde(default = "default_pond_id")]
    pub pond_id: String,
    #[serde(with = "ts_utc", default = "default_ts")]
    pub ts: DateTime<Utc>,
    #[serde(default)]
    pub water_temp_c: Option<f64>,
    /// dissolved oxygen
    #[serde(default)]
    pub do_mgl: Option<f64>,
    #[serde(default)]
    pub ph: Option<f64>,
    #[serde(default)]
    pub turbidity_ntu: Option<f64>,
    /// total ammonia nitrogen (TAN)
    #[serde(default)]
    pub ammonia_total_mgl: Option<f64>,
    #[serde(default)]
    pub nitrite_mgl: Option<f64>,
    #[serde(default)]
    pub tds_mgl: Option<f64>,
    #[serde(default)]
    pub secchi_cm: Option<f64>,
    #[serde(default)]
    pub salinity_ppt: Option<f64>,
    /// manual test kit | sensor | virtual | model
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default)]
    pub note: Option<String>,
}

impl Default for PondReading {
    fn default() -> Self {
        Self {
            pond_id: default_pond_id(),
            ts: default_ts(),
            water_temp_c: None,
            do_mgl: None,
            ph: None,
            turbidity_ntu: None,
            ammonia_total_mgl: None,
            nitrite_mgl: None,
            tds_mgl: None,
            secchi_cm: None,
            salinity_ppt: None,
            source: default_source(),
            note: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRisk {
    pub parameter: String,
    pub value: Option<f64>,
    pub unit: String,
    /// (ideal_low, ideal_high) — serialised as a 2-element array like the Python tuple
    pub ideal: (f64, f64),
    /// 0 safe -> 1 extreme
    pub severity: f64,
    /// "ok" | "watch" | "act" | "critical" | "missing"
    pub status: String,
    pub reason: String,
}

fn default_model_version() -> String {
    "rules-0.1".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub pond_id: String,
    pub species: String,
    #[serde(with = "ts_utc")]
    pub ts: DateTime<Utc>,
    pub band: Band,
    pub score: f64,
    pub headline_en: String,
    pub headline_local: String,
    pub actions: Vec<String>,
    pub drivers: Vec<ParameterRisk>,
    pub missing: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default = "default_model_version")]
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiseasePrediction {
    pub label: String,
    pub confidence: f64,
    /// [(label, score), …] — serialised as an array of 2-element arrays
    pub top_k: Vec<(String, f64)>,
    /// "onnx:<model>" | "heuristic-fallback"
    pub backend: String,
    /// "healthy" | "monitor" | "treat" | "escalate_to_expert"
    pub triage: String,
    pub advice_en: String,
    pub advice_local: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub pond_id: String,
    #[serde(with = "ts_utc")]
    pub created_at: DateTime<Utc>,
    pub band: Band,
    /// "sms" | "voice" | "whatsapp" | "app" | "console"
    pub channel: String,
    pub message: String,
    pub dedupe_key: String,
    #[serde(default)]
    pub delivered: bool,
    #[serde(default)]
    pub error: Option<String>,
}

/// One public dataset / API in the data catalogue (deliverable #2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    pub kind: SourceKind,
    pub url: String,
    #[serde(rename = "license")]
    pub license: String,
    pub what_it_provides: String,
    pub replaces_sensor: String,
    pub role_in_system: String,
    #[serde(default = "default_access_cost")]
    pub access_cost: String,
}

fn default_access_cost() -> String {
    "free".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_serialises_uppercase() {
        assert_eq!(serde_json::to_string(&Band::Red).unwrap(), "\"RED\"");
        assert_eq!(serde_json::to_string(&Band::Green).unwrap(), "\"GREEN\"");
        assert_eq!(Band::parse("orange"), Band::Orange);
    }

    #[test]
    fn naive_ts_is_read_as_utc() {
        let dt = parse_ts("2026-10-01T05:30:00");
        assert_eq!(dt.format("%d-%b %H:%M").to_string(), "01-Oct 05:30");
        let dt2 = parse_ts("2026-10-06T06:00");
        assert_eq!(dt2.format("%d-%b %H:%M").to_string(), "06-Oct 06:00");
    }

    #[test]
    fn reading_defaults_match_python() {
        let r: PondReading = serde_json::from_str("{}").unwrap();
        assert_eq!(r.pond_id, "");
        assert_eq!(r.source, "manual");
        assert!(r.do_mgl.is_none());
    }
}
