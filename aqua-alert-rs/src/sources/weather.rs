//! Open-Meteo client — the 'free sensor' that replaces thermometers, rain gauges
//! and the dissolved-oxygen probe's night-time job.
//!
//! 1:1 port of `src/aqua_alert/sources/weather.py`. Open-Meteo is CC BY 4.0, needs
//! no API key, and covers the whole globe with hourly ERA5 reanalysis back to 1940
//! plus a 16-day forecast.
//!
//! Sensorless trick: pond water temperature lags air temperature and, in a small,
//! shallow, still pond, tracks roughly the diurnal mean. We model it from the daily
//! mean air temperature plus a solar-radiation bump, cloud damping and wind cooling
//! — a 2-parameter model a farmer never has to touch.

use crate::util::pyround;
use anyhow::{anyhow, Result};
use chrono::{Local, NaiveDate};
use serde::Serialize;
use std::time::Duration;

pub const FORECAST_URL: &str = "https://api.open-meteo.com/v1/forecast";
pub const ARCHIVE_URL: &str = "https://archive-api.open-meteo.com/v1/archive";

const DAILY_VARS: &str = "temperature_2m_max,temperature_2m_min,precipitation_sum,\
                          wind_speed_10m_max,cloud_cover_mean,shortwave_radiation_sum";

#[derive(Debug, Clone, Serialize)]
pub struct DayWeather {
    pub day: NaiveDate,
    pub t_max: f64,
    pub t_min: f64,
    pub precip_mm: f64,
    pub wind_ms: f64,
    pub cloud_pct: f64,
    pub radiation_mj: f64,
    pub source: String,
}

impl DayWeather {
    /// Estimate pond water temperature (degC), calibrated against the tropical pond
    /// datasets in the catalogue (pond temp sits within ~1.5 C of the daily mean air).
    pub fn virtual_water_temp(&self) -> f64 {
        let mean_air = (self.t_max + self.t_min) / 2.0;
        let solar_bonus = (1.8f64).min(self.radiation_mj / 22.0); // shallow pond heats fast
        let cloud_damping = 1.0 - 0.006 * self.cloud_pct; // cloud cover reduces heating
        let wind_cooling = (1.2f64).min(0.18 * self.wind_ms); // evaporative cooling
        pyround(mean_air + solar_bonus * cloud_damping - wind_cooling, 1)
    }
}

/// Short, Python-`type(exc).__name__`-style label for the "forecast unavailable" note.
pub fn error_name(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "TimeoutException"
    } else if err.is_connect() {
        "ConnectError"
    } else if err.is_status() {
        "HTTPStatusError"
    } else if err.is_decode() {
        "ValueError"
    } else {
        "RequestError"
    }
}

/// Classify an `anyhow::Error` from a fetch into a Python-style exception name.
pub fn anyhow_error_name(err: &anyhow::Error) -> &'static str {
    match err.downcast_ref::<reqwest::Error>() {
        Some(re) => error_name(re),
        None => "Error",
    }
}

fn num(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn parse(payload: &serde_json::Value) -> Result<Vec<DayWeather>> {
    let d = payload.get("daily").ok_or_else(|| anyhow!("no 'daily' block in Open-Meteo response"))?;
    let times = d.get("time").and_then(|t| t.as_array()).ok_or_else(|| anyhow!("no daily.time"))?.clone();
    let arr = |key: &str| -> Vec<serde_json::Value> {
        d.get(key).and_then(|a| a.as_array()).cloned().unwrap_or_default()
    };
    let t_max = arr("temperature_2m_max");
    let t_min = arr("temperature_2m_min");
    let precip = arr("precipitation_sum");
    let wind = arr("wind_speed_10m_max");
    let cloud = arr("cloud_cover_mean");
    let rad = arr("shortwave_radiation_sum");

    let get = |arr: &[serde_json::Value], i: usize| -> serde_json::Value {
        arr.get(i).cloned().unwrap_or(serde_json::Value::Null)
    };

    let mut out = Vec::with_capacity(times.len());
    for (i, day) in times.iter().enumerate() {
        let day_str = day.as_str().unwrap_or("");
        let date = NaiveDate::parse_from_str(day_str, "%Y-%m-%d").unwrap_or_else(|_| Local::now().date_naive());
        out.push(DayWeather {
            day: date,
            t_max: num(&get(&t_max, i)),
            t_min: num(&get(&t_min, i)),
            precip_mm: num(&get(&precip, i)),
            wind_ms: num(&get(&wind, i)) / 3.6, // km/h -> m/s
            cloud_pct: num(&get(&cloud, i)),
            radiation_mj: num(&get(&rad, i)),
            source: "open-meteo".to_string(),
        });
    }
    Ok(out)
}

fn client(timeout: f64) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs_f64(timeout))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Free, no-key forecast/recent weather for a pond coordinate.
pub async fn fetch_daily(lat: f64, lon: f64, start: NaiveDate, end: NaiveDate, timeout: f64) -> Result<Vec<DayWeather>> {
    let today = Local::now().date_naive();
    // future ranges need the forecast endpoint, past ones the archive
    let url = if end >= today { FORECAST_URL } else { ARCHIVE_URL };
    let params = vec![
        ("latitude", lat.to_string()),
        ("longitude", lon.to_string()),
        ("daily", DAILY_VARS.to_string()),
        ("timezone", "auto".to_string()),
        ("start_date", start.format("%Y-%m-%d").to_string()),
        ("end_date", end.format("%Y-%m-%d").to_string()),
    ];
    let resp = client(timeout).get(url).query(&params).send().await?.error_for_status()?;
    let payload: serde_json::Value = resp.json().await?;
    parse(&payload)
}

/// Next N days — the input to the pre-dawn oxygen-crash warning.
pub async fn fetch_forecast(lat: f64, lon: f64, days: u32) -> Result<Vec<DayWeather>> {
    let params = vec![
        ("latitude", lat.to_string()),
        ("longitude", lon.to_string()),
        ("daily", DAILY_VARS.to_string()),
        ("timezone", "auto".to_string()),
        ("forecast_days", days.to_string()),
    ];
    let resp = client(20.0).get(FORECAST_URL).query(&params).send().await?.error_for_status()?;
    let payload: serde_json::Value = resp.json().await?;
    parse(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::py_str_float;

    fn day(t_max: f64, t_min: f64, precip: f64, wind: f64, cloud: f64, rad: f64) -> DayWeather {
        DayWeather {
            day: NaiveDate::from_ymd_opt(2026, 10, 7).unwrap(),
            t_max,
            t_min,
            precip_mm: precip,
            wind_ms: wind,
            cloud_pct: cloud,
            radiation_mj: rad,
            source: "open-meteo".into(),
        }
    }

    #[test]
    fn virtual_water_temp_blends_air_solar_cloud_wind() {
        // mean air 30, some sun, light cloud, light wind -> a couple of degrees above mean
        let d = day(35.0, 25.0, 0.0, 1.0, 20.0, 22.0);
        let vwt = d.virtual_water_temp();
        assert!(vwt > 29.0 && vwt < 33.0);
        // one decimal, like Python round(..., 1)
        assert_eq!(vwt, pyround(vwt, 1));
    }

    #[test]
    fn parse_maps_daily_arrays_and_converts_wind() {
        let payload = serde_json::json!({
            "daily": {
                "time": ["2026-10-07", "2026-10-08"],
                "temperature_2m_max": [34.0, null],
                "temperature_2m_min": [26.0, 24.0],
                "precipitation_sum": [0.0, 3.5],
                "wind_speed_10m_max": [3.6, null],   // 3.6 km/h -> 1.0 m/s
                "cloud_cover_mean": [10.0, 80.0],
                "shortwave_radiation_sum": [20.0, 12.0]
            }
        });
        let days = parse(&payload).unwrap();
        assert_eq!(days.len(), 2);
        assert_eq!(days[0].day, NaiveDate::from_ymd_opt(2026, 10, 7).unwrap());
        assert!((days[0].wind_ms - 1.0).abs() < 1e-9);
        assert_eq!(days[1].t_max, 0.0); // null -> 0.0
        assert_eq!(days[1].precip_mm, 3.5);
    }

    #[test]
    fn py_str_float_formats_temps_like_python() {
        assert_eq!(py_str_float(34.0), "34.0");
        assert_eq!(py_str_float(27.5), "27.5");
    }
}
