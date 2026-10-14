//! End-to-end HTTP contract test: boots the real axum server on an ephemeral port
//! and checks the responses against values captured from the Python/FastAPI build.
//!
//! Mirrors tests/test_pipeline_and_alerts.py::test_api_smoke and
//! test_api_coerces_json_datetime_and_keeps_old_readings.

#![cfg(feature = "server")]

use serde_json::{json, Value};

async fn spawn_server() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, aqua_alert::api::server::router()).await.unwrap();
    });
    // the socket is already listening (backlog), but give the task a beat to poll
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    format!("http://{addr}")
}

#[tokio::test]
async fn api_end_to_end_matches_python() {
    aqua_alert::api::server::reset_state();
    let base = spawn_server().await;
    let c = reqwest::Client::new();

    // --- /health ---
    let h: Value = c.get(format!("{base}/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(h["ok"], json!(true));
    assert_eq!(h["service"], json!("aqua-alert"));
    assert_eq!(h["version"], json!("0.1.0"));

    // --- GET / serves the farmer screen (deliverable #3), embedded ---
    let screen = c.get(format!("{base}/")).send().await.unwrap().text().await.unwrap();
    assert!(screen.to_lowercase().contains("<html") || screen.to_lowercase().contains("<!doctype"));

    // --- /catalog (deliverable #2) ---
    let cat: Value = c.get(format!("{base}/catalog")).send().await.unwrap().json().await.unwrap();
    assert_eq!(cat["count"], json!(20));
    assert!(cat["entries"].as_array().unwrap().len() >= 12);
    assert_eq!(cat["entries"][0]["id"], json!("kaggle-aquaponics-nigeria"));
    assert_eq!(cat["entries"][0]["access_cost"], json!("free"));

    // --- /catalog/{id} and its 404 ---
    let one: Value = c.get(format!("{base}/catalog/open-meteo")).send().await.unwrap().json().await.unwrap();
    assert_eq!(one["kind"], json!("weather"));
    let missing = c.get(format!("{base}/catalog/nope")).send().await.unwrap();
    assert_eq!(missing.status(), 404);
    let missing_body: Value = missing.json().await.unwrap();
    assert_eq!(missing_body["detail"], json!("no catalogue entry nope"));

    // --- /escalation-matrix ---
    let m: Value = c.get(format!("{base}/escalation-matrix")).send().await.unwrap().json().await.unwrap();
    for band in ["GREEN", "YELLOW", "ORANGE", "RED"] {
        assert!(m.get(band).is_some(), "matrix missing {band}");
    }
    assert!(m["ORANGE"].as_str().unwrap().to_lowercase().contains("voice"));

    // --- POST /pond/{id}/reading -> RED (robust whether or not Open-Meteo is reachable) ---
    let body = json!({"do_mgl": 1.7, "ph": 8.9, "water_temp_c": 33.0, "ammonia_total_mgl": 1.8, "nitrite_mgl": 0.7});
    let r: Value = c
        .post(format!("{base}/pond/KA-BLR-001/reading"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["assessment"]["band"], json!("RED"));
    assert_eq!(r["assessment"]["species"], json!("indian_major_carp")); // KA-BLR-001 -> rohu
    assert!(!r["assessment"]["headline_local"].as_str().unwrap().is_empty());
    // the STATE object shape
    for key in ["assessment", "night_band", "night_reason", "alerts"] {
        assert!(r.get(key).is_some(), "reading response missing {key}");
    }

    // --- GET /pond/{id}/status returns what we just stored ---
    let st: Value = c.get(format!("{base}/pond/KA-BLR-001/status")).send().await.unwrap().json().await.unwrap();
    assert_eq!(st["assessment"]["band"], json!("RED"));

    // --- POST /simulate/crash-night: exact parity with the captured Python JSON ---
    let s: Value = c.post(format!("{base}/simulate/crash-night")).send().await.unwrap().json().await.unwrap();
    let a = &s["assessment"];
    assert_eq!(a["band"], json!("RED"));
    assert_eq!(a["score"], json!(0.634));
    assert_eq!(a["species"], json!("indian_major_carp"));
    assert_eq!(a["model_version"], json!("rules-0.1"));
    // full field set matches pydantic model_dump(mode="json")
    for key in [
        "pond_id", "species", "ts", "band", "score", "headline_en", "headline_local",
        "actions", "drivers", "missing", "notes", "model_version",
    ] {
        assert!(a.get(key).is_some(), "assessment missing {key}");
    }
    // NH3-N derived and severity-sorted first
    let d0 = &a["drivers"][0];
    assert_eq!(d0["parameter"], json!("nh3_mgl"));
    assert_eq!(d0["status"], json!("critical"));
    assert_eq!(d0["severity"], json!(1.0));
    assert_eq!(d0["reason"], json!("derived from TAN + pH + temperature"));
    assert_eq!(d0["ideal"], json!([0.0, 0.02]));
    let order: Vec<&str> = a["drivers"].as_array().unwrap().iter().map(|d| d["parameter"].as_str().unwrap()).collect();
    assert_eq!(order, vec!["nh3_mgl", "do_mgl", "ph", "turbidity_ntu", "nitrite_mgl", "water_temp_c"]);

    // --- datetime coercion + per-pond species registry ---
    let body2 = json!({"ts": "2026-10-01T05:30:00", "do_mgl": 3.1, "ph": 8.1, "water_temp_c": 30.0});
    let r2 = c.post(format!("{base}/pond/AP-WGL-007/reading")).json(&body2).send().await.unwrap();
    assert_eq!(r2.status(), 200);
    let st2: Value = c.get(format!("{base}/pond/AP-WGL-007/status")).send().await.unwrap().json().await.unwrap();
    assert_eq!(st2["assessment"]["species"], json!("pacific_white_shrimp")); // AP-WGL-007 -> shrimp
    // the naive timestamp was accepted and re-emitted as RFC3339 UTC
    assert!(st2["assessment"]["ts"].as_str().unwrap().starts_with("2026-10-01T05:30:00"));

    aqua_alert::api::server::reset_state();
}
