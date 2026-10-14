//! USGS/EPA Water Quality Portal client — proof that the pipeline ingests a real
//! government water-quality API, not only CSV files.
//!
//! 1:1 port of `src/aqua_alert/sources/usgs_wqp.py`. WQP is a cooperative service
//! of USGS + US EPA + 400 agencies, free, no API key, CSV/JSON/GeoJSON out.
//! Parameter codes we care about:
//!   00300 Dissolved oxygen (mg/L)      00400 pH
//!   00610 Ammonia, total (mg/L as N)   63676 Turbidity (NTU)
//!   32209 Chlorophyll a (ug/L)         00613 Nitrite (mg/L as N)

use crate::csv;
use crate::schemas::PondReading;
use anyhow::Result;
use chrono::{NaiveDate, Utc};
use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;

pub const WQP_RESULT_URL: &str = "https://www.waterqualitydata.us/data/Result/search";

pub fn param_codes() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("00300", "do_mgl");
    m.insert("00400", "ph");
    m.insert("00610", "ammonia_total_mgl");
    m.insert("00613", "nitrite_mgl");
    m.insert("63676", "turbidity_ntu");
    m.insert("32209", "chlorophyll_a_ugl");
    m.insert("00010", "water_temp_c");
    m
}

#[derive(Debug, Clone, Serialize)]
pub struct WqpRow {
    pub site: Option<String>,
    pub date: Option<String>,
    pub characteristic: String,
    pub value: Option<f64>,
    pub unit: Option<String>,
    pub param: String,
    pub source: String,
}

fn to_float(v: Option<&String>) -> Option<f64> {
    v.and_then(|s| s.trim().parse::<f64>().ok())
}

/// Download discrete water-quality results as CSV and normalise them.
pub async fn fetch_results(
    state_code: &str,
    start: Option<NaiveDate>,
    end: Option<NaiveDate>,
    params: &[&str],
    max_rows: usize,
    timeout: f64,
) -> Result<Vec<WqpRow>> {
    let mut q: Vec<(&str, String)> = vec![
        ("statecode", state_code.to_string()),
        ("mimeType", "csv".to_string()),
        ("sorted", "no".to_string()),
        ("count", "no".to_string()),
        ("pCode", params.join(";")),
    ];
    if let Some(s) = start {
        q.push(("startDateLo", s.format("%m-%d-%Y").to_string()));
    }
    if let Some(e) = end {
        q.push(("startDateHi", e.format("%m-%d-%Y").to_string()));
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs_f64(timeout))
        .build()?;
    let text = client.get(WQP_RESULT_URL).query(&q).send().await?.error_for_status()?.text().await?;

    let codes = param_codes();
    let records = csv::parse_records(&text);
    let mut rows: Vec<WqpRow> = Vec::new();
    for rec in records.into_iter().take(max_rows) {
        let get = |k: &str| -> Option<String> { rec.get(k).cloned().filter(|s| !s.is_empty()) };
        let characteristic = rec.get("CharacteristicName").cloned().unwrap_or_default();
        let pcode = rec.get("USGSPCode").cloned().unwrap_or_default();
        rows.push(WqpRow {
            site: get("MonitoringLocationIdentifier"),
            date: get("ActivityStartDate"),
            characteristic: characteristic.trim().to_string(),
            value: to_float(rec.get("ResultMeasureValue")),
            unit: get("ResultMeasure/MeasureUnitCode"),
            param: codes.get(pcode.as_str()).copied().unwrap_or("other").to_string(),
            source: "USGS/EPA WQP".to_string(),
        });
    }
    Ok(rows)
}

/// Pivot WQP rows for one site/date into a PondReading (public data as a virtual sensor).
pub fn to_pond_reading(rows: &[WqpRow], pond_id: &str) -> PondReading {
    let mut fields: HashMap<&str, f64> = HashMap::new();
    let mut site: Option<String> = None;
    for r in rows {
        let tracked = matches!(
            r.param.as_str(),
            "do_mgl" | "ph" | "ammonia_total_mgl" | "nitrite_mgl" | "turbidity_ntu" | "water_temp_c"
        );
        if tracked {
            if let Some(v) = r.value {
                fields.insert(r.param.as_str(), v);
                if site.is_none() {
                    site = r.site.clone();
                }
            }
        }
    }
    let g = |k: &str| fields.get(k).copied();
    PondReading {
        pond_id: pond_id.to_string(),
        ts: Utc::now(),
        water_temp_c: g("water_temp_c"),
        do_mgl: g("do_mgl"),
        ph: g("ph"),
        turbidity_ntu: g("turbidity_ntu"),
        ammonia_total_mgl: g("ammonia_total_mgl"),
        nitrite_mgl: g("nitrite_mgl"),
        source: format!("USGS WQP site {}", site.unwrap_or_else(|| "None".to_string())),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pivot_builds_a_virtual_sensor_reading() {
        let rows = vec![
            WqpRow {
                site: Some("USGS-05370500".into()),
                date: Some("2024-07-01".into()),
                characteristic: "Dissolved oxygen (DO)".into(),
                value: Some(6.7),
                unit: Some("mg/l".into()),
                param: "do_mgl".into(),
                source: "USGS/EPA WQP".into(),
            },
            WqpRow {
                site: Some("USGS-05370500".into()),
                date: Some("2024-07-01".into()),
                characteristic: "pH".into(),
                value: Some(7.9),
                unit: Some("std".into()),
                param: "ph".into(),
                source: "USGS/EPA WQP".into(),
            },
        ];
        let r = to_pond_reading(&rows, "VIRT-1");
        assert_eq!(r.do_mgl, Some(6.7));
        assert_eq!(r.ph, Some(7.9));
        assert!(r.ammonia_total_mgl.is_none());
        assert_eq!(r.source, "USGS WQP site USGS-05370500");
    }

    #[test]
    fn param_code_mapping() {
        let c = param_codes();
        assert_eq!(c["00300"], "do_mgl");
        assert_eq!(c["63676"], "turbidity_ntu");
    }
}
