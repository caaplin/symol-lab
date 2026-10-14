//! The glue: one function that turns "whatever we have" into "what the farmer should do".
//!
//! 1:1 port of `src/aqua_alert/pipeline.py`.
//!
//! Data flow (all optional inputs, none mandatory):
//!   weather API (Open-Meteo)  -> virtual water temp + night-DO-crash forecast -┐
//!   public pond datasets      -> baselines / thresholds                        ├-> risk::assess -> alerts
//!   farmer's own readings     -> anomaly::drift_scores (per-pond z-scores)     │      │
//!   phone photo               -> disease::classifier::predict -----------------┘      └-> screens/app
//!
//! The assessment core is synchronous; only the Open-Meteo fetch is async, so the
//! offline demo path needs no runtime.

use crate::alerts::dispatcher::AlertDispatcher;
use crate::schemas::{Alert, Band, DiseasePrediction, PondReading, RiskAssessment};
use crate::sources::weather::{self, DayWeather};
use crate::util::py_str_float;
use crate::{anomaly, risk};
use crate::sources::synthetic;
use chrono::{Duration, Local, Timelike};

#[derive(Debug, Clone)]
pub struct FarmDay {
    pub pond_id: String,
    pub reading: PondReading,
    pub assessment: RiskAssessment,
    pub night_band: Option<Band>,
    pub night_reason: Option<String>,
    pub disease: Option<DiseasePrediction>,
    pub alerts: Vec<Alert>,
}

/// Outcome of the (optional) weather fetch, resolved by the async caller.
pub enum Forecast {
    Disabled,
    Ok(Vec<DayWeather>),
    /// The Python `type(exc).__name__`, for the "unavailable" note.
    Err(String),
}

/// Synchronous core of `assess_pond`: drift -> fusion -> (optional) night forecast
/// -> dispatch. `forecast` carries the already-resolved weather attempt.
pub fn assess_internal(
    reading: PondReading,
    species: &str,
    dispatcher: Option<&mut AlertDispatcher>,
    channel: &str,
    forecast: Forecast,
) -> FarmDay {
    // 1. per-pond drift (cheap: no external data needed) — same parameter order as Python
    anomaly::update(&reading.pond_id, "do_mgl", reading.do_mgl);
    anomaly::update(&reading.pond_id, "ph", reading.ph);
    anomaly::update(&reading.pond_id, "water_temp_c", reading.water_temp_c);
    anomaly::update(&reading.pond_id, "ammonia_total_mgl", reading.ammonia_total_mgl);
    anomaly::update(&reading.pond_id, "turbidity_ntu", reading.turbidity_ntu);
    anomaly::update(&reading.pond_id, "nitrite_mgl", reading.nitrite_mgl);
    let drift = anomaly::drift_scores(&reading.pond_id);

    // 2. absolute + drift fusion
    let mut assessment = risk::assess(&reading, species, Some(&drift));
    if anomaly::severity_from_drift_default(&drift) > 0.5 && assessment.band == Band::Green {
        assessment.band = Band::Yellow;
        assessment.headline_en = "Water is off its normal pattern".to_string();
        assessment.headline_local = "ನೀರು ಸಾಮಾನ್ಯ ರೀತಿಯಲ್ಲಿಲ್ಲ".to_string();
        assessment.notes.push("Pond is drifting from its own 14-day baseline - re-test tomorrow".to_string());
    }

    // 3. night-time DO crash forecast from free weather (no DO probe required)
    let mut night_band: Option<Band> = None;
    let mut night_reason: Option<String> = None;
    match forecast {
        Forecast::Disabled => {}
        Forecast::Err(name) => {
            assessment.notes.push(format!("Weather forecast unavailable ({name})"));
        }
        Forecast::Ok(days) => {
            if let Some(tomorrow) = days.last() {
                let do_evening = if reading.ts.hour() >= 16 { reading.do_mgl } else { None };
                let (nb, nr) = risk::do_crash_forecast(
                    tomorrow.t_max,
                    tomorrow.t_min,
                    tomorrow.wind_ms,
                    tomorrow.precip_mm,
                    tomorrow.cloud_pct,
                    do_evening,
                );
                night_band = Some(nb);
                night_reason = Some(nr);
                let vwt = tomorrow.virtual_water_temp();
                assessment.notes.push(format!(
                    "Virtual water temp from Open-Meteo: {} C (air max {} C, min {} C)",
                    py_str_float(vwt),
                    py_str_float(tomorrow.t_max),
                    py_str_float(tomorrow.t_min)
                ));
                // Mirrors the Python string comparison `assessment.band.value < night_band.value`.
                if (nb == Band::Orange || nb == Band::Red) && assessment.band.as_str() < nb.as_str() {
                    assessment.band = nb;
                    assessment.actions.insert(
                        0,
                        "Run the aerator from 22:00 tonight until 07:00 \
                         (ರಾತ್ರಿ 10 ರಿಂದ ಬೆಳಿಗ್ಗೆ 7 ರವರೆಗೆ ಏರೇಟರ್ ಚಾಲು ಮಾಡಿ)"
                            .to_string(),
                    );
                }
            }
        }
    }

    let mut alerts: Vec<Alert> = Vec::new();
    if let Some(d) = dispatcher {
        alerts.push(d.dispatch(&assessment, channel, None));
    }

    FarmDay {
        pond_id: reading.pond_id.clone(),
        reading,
        assessment,
        night_band,
        night_reason,
        disease: None,
        alerts,
    }
}

/// Assess one pond reading, add a night-oxygen forecast if coordinates allow, and
/// push any alert through the dispatcher.
#[allow(clippy::too_many_arguments)]
pub async fn assess_pond(
    reading: PondReading,
    species: &str,
    dispatcher: Option<&mut AlertDispatcher>,
    channel: &str,
    lat: Option<f64>,
    lon: Option<f64>,
    include_forecast: bool,
) -> FarmDay {
    let forecast = if include_forecast && lat.is_some() && lon.is_some() {
        let today = Local::now().date_naive();
        match weather::fetch_daily(lat.unwrap(), lon.unwrap(), today - Duration::days(6), today, 20.0).await {
            Ok(days) => Forecast::Ok(days),
            Err(e) => Forecast::Err(weather::anyhow_error_name(&e).to_string()),
        }
    } else {
        Forecast::Disabled
    };
    assess_internal(reading, species, dispatcher, channel, forecast)
}

/// Runs the final *dawn* reading of the bundled 14-day sample — fully offline.
/// Dawn (06:00) is the reading that matters: DO is lowest and the farmer is at the pond.
pub fn demo_day(pond_id: &str, species: &str, dispatcher: Option<&mut AlertDispatcher>) -> FarmDay {
    use std::collections::HashMap;
    let all = synthetic::load_csv();
    let filtered: Vec<&HashMap<String, String>> = all
        .iter()
        .filter(|r| r.get("pond_id").map(|s| s.as_str()) == Some(pond_id))
        .collect();
    let rows: Vec<&HashMap<String, String>> =
        if filtered.is_empty() { all.iter().collect() } else { filtered };

    let last_row: &HashMap<String, String> = *rows.last().expect("sample log is non-empty");
    let final_day = &last_row["ts"][..10];
    let dawn: Vec<&HashMap<String, String>> = rows
        .iter()
        .filter(|r| &r["ts"][..10] == final_day && &r["ts"][11..13] == "06")
        .copied()
        .collect();
    let last: &HashMap<String, String> = dawn.first().copied().unwrap_or(last_row);

    let f = |k: &str| -> Option<f64> { last.get(k).and_then(|v| v.parse::<f64>().ok()) };
    let reading = PondReading {
        pond_id: last["pond_id"].clone(),
        ts: crate::schemas::parse_ts(&last["ts"]),
        water_temp_c: f("water_temp_c"),
        do_mgl: f("do_mgl"),
        ph: f("ph"),
        turbidity_ntu: f("turbidity_ntu"),
        ammonia_total_mgl: f("ammonia_total_mgl"),
        nitrite_mgl: f("nitrite_mgl"),
        source: "bundled sample (public-dataset-calibrated stand-in)".to_string(),
        ..Default::default()
    };

    assess_internal(reading, species, dispatcher, "console", Forecast::Disabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_demo_produces_a_verdict_and_an_alert() {
        let _g = crate::anomaly::lock_history();
        anomaly::reset();
        let mut disp = AlertDispatcher::new();
        let day = demo_day("KA-BLR-001", "tilapia", Some(&mut disp));
        // captured from the Python reference build
        assert_eq!(day.assessment.band, Band::Orange);
        assert_eq!(day.assessment.score, 0.2);
        assert_eq!(day.assessment.species, "tilapia");
        assert!(!day.assessment.actions.is_empty());
        assert!(!disp.sent.is_empty(), "a non-green verdict must reach the farmer");
        assert!(day.alerts[0].delivered);
        anomaly::reset();
    }

    #[test]
    fn green_pond_sends_nothing() {
        let _g = crate::anomaly::lock_history();
        anomaly::reset();
        let mut disp = AlertDispatcher::new();
        let r = PondReading {
            pond_id: "G1".into(),
            do_mgl: Some(7.5),
            ph: Some(7.5),
            water_temp_c: Some(28.0),
            ..Default::default()
        };
        let day = assess_internal(r, "tilapia", Some(&mut disp), "console", Forecast::Disabled);
        assert_eq!(day.assessment.band, Band::Green);
        assert!(!day.alerts[0].delivered);
        assert!(day.alerts[0].error.clone().unwrap().contains("suppressed"));
        anomaly::reset();
    }
}
