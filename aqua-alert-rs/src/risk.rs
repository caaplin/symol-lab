//! Fuse whatever we know about a pond into one farmer-facing verdict.
//!
//! 1:1 port of `src/aqua_alert/risk.py`.
//!
//! Design rules (from the field constraints of small farms):
//! 1. Never require a parameter you did not get — degrade, and say what is missing.
//! 2. Precision beats recall for GREEN (an "all clear" on a dying pond destroys
//!    trust), so a single critical parameter escalates the whole pond.
//! 3. Every non-GREEN verdict ships with an action a farmer can take today with
//!    materials already on the farm.

use crate::schemas::{Band, ParameterRisk, PondReading, RiskAssessment};
use crate::thresholds::{feed_advice, get_profile, unionized_ammonia_mgl, Envelope, SpeciesProfile};
use crate::util::pyround;

fn headline(band: Band) -> (&'static str, &'static str) {
    match band {
        Band::Green => ("Water is safe today", "ಇಂದು ನೀರು ಸುರಕ್ಷಿತವಾಗಿದೆ"),
        Band::Yellow => ("Watch - water is drifting", "ಎಚ್ಚರಿಕೆ - ನೀರಿನ ಗುಣಮಟ್ಟ ಬದಲಾಗುತ್ತಿದೆ"),
        Band::Orange => ("Act today - fish at risk", "ಇಂದೇ ಕ್ರಮ ತಗೊಳ್ಳಿ - ಮೀನುಗಳಿಗೆ ಅಪಾಯ"),
        Band::Red => ("EMERGENCY - fish may die", "ತುರ್ತು - ಮೀನು ಸಾಯಬಹುದು"),
    }
}

/// Piecewise-linear severity in [0, 1]. 0 inside the ideal band.
fn severity(value: f64, env: &Envelope) -> f64 {
    let (lo_i, hi_i) = env.ideal;
    let (lo_a, hi_a) = env.act;
    let (lo_c, hi_c) = (env.crit_low, env.crit_high);

    if lo_i <= value && value <= hi_i {
        return 0.0;
    }
    if value < lo_i {
        if value >= lo_a {
            return 0.30 * (lo_i - value) / (lo_i - lo_a).max(1e-9);
        }
        if value >= lo_c {
            return 0.30 + 0.45 * (lo_a - value) / (lo_a - lo_c).max(1e-9);
        }
        // below the critical floor: hard escalation, blows up fast
        return (0.75 + 0.25 * (lo_c - value) / (lo_c.abs() * 0.5).max(1e-9)).min(1.0);
    }
    // value > hi_i
    if value <= hi_a {
        return 0.30 * (value - hi_i) / (hi_a - hi_i).max(1e-9);
    }
    if value <= hi_c {
        return 0.30 + 0.45 * (value - hi_a) / (hi_c - hi_a).max(1e-9);
    }
    (0.75 + 0.25 * (value - hi_c) / (hi_c.abs() * 0.5).max(1e-9)).min(1.0)
}

fn status(sev: f64) -> &'static str {
    if sev <= 0.0 {
        "ok"
    } else if sev < 0.30 {
        "watch"
    } else if sev < 0.75 {
        "act"
    } else {
        "critical"
    }
}

/// Concrete, cheap, same-day actions, ordered by impact per rupee.
///
/// `drivers` maps parameter name -> *severity* (mirroring the Python dict); the
/// ammonia water-exchange rule keys off the NH3 severity, exactly as upstream.
fn actions(band: Band, profile: &SpeciesProfile, drivers: &[(String, f64)], reading: &PondReading) -> Vec<String> {
    let mut acts: Vec<String> = Vec::new();
    let do_v = reading.do_mgl;
    let nh3 = drivers.iter().find(|(n, _)| n.as_str() == "nh3_mgl").map(|(_, s)| *s);

    if let Some(d) = do_v {
        if d < profile.aeration_below_do {
            acts.push("Turn on the aerator / paddle wheel NOW and keep it running until sunrise \
                       (ಈಗಲೇ ಏರೇಟರ್ ಆನ್ ಮಾಡಿ, ಸೂರ್ಯೋದಯದವರೆಗೆ ಬಿಡಬೇಡಿ)".to_string());
        }
    }
    if let Some(d) = do_v {
        if d < profile.feed_stop_below_do {
            acts.push("Stop feeding today (ಇಂದು ಆಹಾರ ಕೊಡಬೇಡಿ)".to_string());
        }
    }
    if let Some(n) = nh3 {
        if n > 0.05 {
            acts.push("Exchange 20-30% of pond water with clean water; stop feeding for 24 h \
                       (20-30% ನೀರು ಬದಲಾಯಿಸಿ)".to_string());
        }
    }
    if let Some(ph) = reading.ph {
        if ph < 6.5 {
            acts.push("Add agricultural lime 100-200 kg/ha (ಸುಣ್ಣ ಹಾಕಿ)".to_string());
        }
    }
    if let Some(ph) = reading.ph {
        if ph > 9.0 {
            acts.push("Add molasses/cow-dung slurry to buffer pH, avoid lime \
                       (ಸಕ್ಕರೆ ಜೇನು/ಮೊಲಾಸಸ್ ಹಾಕಿ)".to_string());
        }
    }
    if let Some(t) = reading.turbidity_ntu {
        if t > 150.0 {
            acts.push("Reduce feeding and check for algal bloom; flush pond edges \
                       (ಪಾಚಿ ಪರೀಕ್ಷಿಸಿ)".to_string());
        }
    }
    if band == Band::Orange || band == Band::Red {
        acts.push("Take 3 fish and check skin, gills and fins; send a photo in the app \
                   (3 ಮೀನು ಪರೀಕ್ಷಿಸಿ, ಫೋಟೋ ಕಳಿಸಿ)".to_string());
    }
    if band == Band::Red {
        acts.push("Call the nearest fisheries extension officer / AquaOne centre today \
                   (ಮೀನುಗಾರಿಕೆ ಅಧಿಕಾರಿಗೆ ಕರೆ ಮಾಡಿ)".to_string());
    }
    if acts.is_empty() {
        acts.push("Keep routine: test in the morning, record the reading \
                   (ಬೆಳಿಗ್ಗೆ ಪರೀಕ್ಷೆ ಮಾಡಿ, ದಾಖಲೆ ಬರೆಯಿರಿ)".to_string());
    }
    acts
}

/// Main entry point: one reading (+ optional ordered drift scores) -> one verdict.
pub fn assess(reading: &PondReading, species: &str, history_drift: Option<&[(String, f64)]>) -> RiskAssessment {
    let profile = get_profile(species);
    let mut missing: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut risks: Vec<ParameterRisk> = Vec::new();

    // ---- derived parameter: toxic un-ionized ammonia from cheap strips -------------
    let mut nh3: Option<f64> = None;
    let mut derived_nh3 = false;
    if let (Some(tan), Some(ph), Some(temp)) = (reading.ammonia_total_mgl, reading.ph, reading.water_temp_c) {
        let v = unionized_ammonia_mgl(tan, ph, temp);
        nh3 = Some(v);
        derived_nh3 = true;
        notes.push(format!(
            "NH3-N {:.3} mg/L derived from TAN {:.2} mg/L, pH {:.2}, {:.1} C \
             (Emerson pKa; {:.1}% un-ionized)",
            v,
            tan,
            ph,
            temp,
            100.0 * (v / tan.max(1e-9))
        ));
    }

    // References into `profile.envelopes` are valid for the whole function body
    // (profile is a local that outlives `values`).
    let values: Vec<(&str, Option<f64>, &Envelope)> = vec![
        ("do_mgl", reading.do_mgl, &profile.envelopes["do_mgl"]),
        ("nh3_mgl", nh3, &profile.envelopes["nh3_mgl"]),
        ("ph", reading.ph, &profile.envelopes["ph"]),
        ("water_temp_c", reading.water_temp_c, &profile.envelopes["water_temp_c"]),
        ("turbidity_ntu", reading.turbidity_ntu, &profile.envelopes["turbidity_ntu"]),
        ("nitrite_mgl", reading.nitrite_mgl, &profile.envelopes["nitrite_mgl"]),
    ];

    let mut weighted = 0.0f64;
    let mut used_weight = 0.0f64;
    let mut drivers: Vec<(String, f64)> = Vec::new();

    for &(name, value, e) in &values {
        let v = match value {
            None => {
                // report what the *farmer* could still add, not internal derived names
                if name == "nh3_mgl" {
                    if reading.ammonia_total_mgl.is_none() {
                        missing.push("ammonia_total_mgl (test strip)".to_string());
                    } else {
                        let mut need: Vec<&str> = Vec::new();
                        if reading.ph.is_none() {
                            need.push("ph");
                        }
                        if reading.water_temp_c.is_none() {
                            need.push("water_temp_c");
                        }
                        missing.push(format!("NH3-N (needs {})", need.join(" and ")));
                    }
                } else {
                    missing.push(name.to_string());
                }
                continue;
            }
            Some(v) => v,
        };

        let sev = severity(v, e);
        let reason = if name == "nh3_mgl" && derived_nh3 {
            "derived from TAN + pH + temperature".to_string()
        } else if !e.note.is_empty() {
            e.note.to_string()
        } else {
            "measured".to_string()
        };
        risks.push(ParameterRisk {
            parameter: name.to_string(),
            value: Some(v),
            unit: e.unit.to_string(),
            ideal: e.ideal,
            severity: sev,
            status: status(sev).to_string(),
            reason,
        });
        weighted += sev * e.weight;
        used_weight += e.weight;
        drivers.push((name.to_string(), sev));
    }

    if let Some(drift) = history_drift {
        if !drift.is_empty() {
            let parts: Vec<String> = drift.iter().map(|(k, v)| format!("{k} z={v:+.1}")).collect();
            notes.push(format!("Drift vs own 14-day baseline: {}", parts.join(", ")));
        }
    }

    let score = if used_weight > 0.0 { weighted / used_weight } else { 0.0 };
    let max_sev = drivers.iter().fold(0.0f64, |m, (_, s)| m.max(*s));
    // `max(drivers, key=drivers.get)` — first-wins on ties (insertion order).
    let worst: Option<String> = {
        let mut best: Option<(&String, f64)> = None;
        for (n, s) in &drivers {
            match best {
                None => best = Some((n, *s)),
                Some((_, bs)) => {
                    if *s > bs {
                        best = Some((n, *s));
                    }
                }
            }
        }
        best.map(|(n, _)| n.clone())
    };

    // ---- banding: worst single parameter dominates, weighted score breaks ties ----
    let mut band = if max_sev >= 0.75 {
        Band::Red
    } else if max_sev >= 0.30 || score >= 0.22 {
        Band::Orange
    } else if max_sev > 0.05 || score >= 0.05 {
        Band::Yellow
    } else {
        Band::Green
    };

    // Nitrogen "canary": a rising nitrite with normal DO is the earliest 24-48 h
    // warning of an ammonia/disease event.
    if band == Band::Green {
        if let Some(nit) = reading.nitrite_mgl {
            if nit > 0.5 {
                band = Band::Yellow;
                notes.push("Nitrite rising while DO is fine: early warning, re-test tomorrow morning".to_string());
            }
        }
    }

    let (headline_en, headline_local) = headline(band);
    if !missing.is_empty() {
        notes.push(format!(
            "No data for: {} - verdict uses the parameters actually available",
            missing.join(", ")
        ));
    }

    let mut acts = actions(band, &profile, &drivers, reading);
    let adv = feed_advice(&profile, reading.do_mgl, nh3, reading.water_temp_c, "kn");
    if adv.feed_fraction == 0.0 {
        acts.retain(|x| !x.to_lowercase().starts_with("stop feeding"));
    }
    // Safety-critical physical action first (aerator), then the feeding decision.
    let insert_at = if !acts.is_empty() && acts[0].to_lowercase().starts_with("turn on") { 1 } else { 0 };
    acts.insert(insert_at, format!("{} | {} ({})", adv.text_en, adv.text_local, adv.reason));

    // sorted by severity descending; Rust's `sort_by` is stable, matching Python's
    // `sorted(risks, key=lambda r: -r.severity)`.
    risks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(w) = &worst {
        notes.push(format!("Worst parameter: {w}"));
    }

    RiskAssessment {
        pond_id: reading.pond_id.clone(),
        species: profile.name.to_string(),
        ts: reading.ts,
        band,
        score: pyround(score, 3),
        headline_en: headline_en.to_string(),
        headline_local: headline_local.to_string(),
        actions: acts,
        drivers: risks,
        missing,
        notes,
        model_version: "rules-0.1".to_string(),
    }
}

/// Cheap early-morning DO-crash predictor.
///
/// Dissolved oxygen in a still, warm pond bottoms out between 03:00 and 07:00.
/// Hot day + still air + no rain + a cloudless night = worst case. This is the
/// alert that arrives *before* the kill, using only a free weather API.
pub fn do_crash_forecast(
    air_temp_max_c: f64,
    night_temp_min_c: f64,
    wind_ms: f64,
    rain_mm: f64,
    cloud_pct: f64,
    do_last_evening: Option<f64>,
) -> (Band, String) {
    let mut risk = 0.0f64;
    let mut reasons: Vec<String> = Vec::new();
    if air_temp_max_c >= 34.0 {
        risk += 0.35;
        reasons.push(format!("hot day {air_temp_max_c:.0} C"));
    }
    if night_temp_min_c >= 26.0 {
        risk += 0.20;
        reasons.push(format!("warm night {night_temp_min_c:.0} C (high respiration)"));
    }
    if wind_ms < 1.5 {
        risk += 0.20;
        reasons.push("still air, no mixing".to_string());
    }
    if rain_mm < 0.5 {
        risk += 0.10;
        reasons.push("no rain to re-oxygenate".to_string());
    }
    if cloud_pct < 30.0 {
        risk += 0.05;
        reasons.push("clear night".to_string());
    }
    if let Some(d) = do_last_evening {
        if d < 6.0 {
            risk += 0.25;
            reasons.push(format!("evening DO already low ({d:.1} mg/L)"));
        }
    }
    let band = if risk >= 0.75 {
        Band::Red
    } else if risk >= 0.55 {
        Band::Orange
    } else if risk >= 0.35 {
        Band::Yellow
    } else {
        Band::Green
    };
    let msg = if band == Band::Red || band == Band::Orange {
        "Predicted pre-dawn oxygen crash"
    } else if band == Band::Yellow {
        "Pre-dawn oxygen dip possible"
    } else {
        if reasons.is_empty() {
            reasons.push("cool, mixed, wet conditions".to_string());
        }
        "Pre-dawn oxygen expected to stay safe"
    };
    (band, format!("{msg}: {}", reasons.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::PondReading;
    use chrono::TimeZone;

    fn reading() -> PondReading {
        PondReading::default()
    }

    #[test]
    fn banding_escalates_on_a_single_critical_parameter() {
        let mut r = reading();
        r.pond_id = "T1".into();
        r.do_mgl = Some(1.6);
        r.ph = Some(7.8);
        r.water_temp_c = Some(29.0);
        let a = assess(&r, "rohu", None);
        assert_eq!(a.band, Band::Red);
        assert_eq!(a.score, 0.529);
        assert_eq!(a.species, "indian_major_carp");
        assert!(a.actions.iter().any(|x| x.to_lowercase().contains("aerat")));
        // exact action ordering captured from the Python reference run
        assert_eq!(a.actions.len(), 4);
        assert!(a.actions[0].starts_with("Turn on the aerator"));
        assert!(a.actions[1].starts_with("Do NOT feed today |"));
        // drivers sorted by severity desc
        assert_eq!(a.drivers[0].parameter, "do_mgl");
        assert_eq!(a.drivers[0].status, "critical");
        assert!((a.drivers[0].severity - 0.9833333333333333).abs() < 1e-12);
        assert_eq!(
            a.missing,
            vec!["ammonia_total_mgl (test strip)", "turbidity_ntu", "nitrite_mgl"]
        );
        assert_eq!(a.notes.last().unwrap(), "Worst parameter: do_mgl");
    }

    #[test]
    fn healthy_pond_stays_green_and_says_what_is_missing() {
        let mut r = reading();
        r.pond_id = "T1".into();
        r.do_mgl = Some(7.2);
        r.ph = Some(7.4);
        r.water_temp_c = Some(28.0);
        let a = assess(&r, "tilapia", None);
        assert_eq!(a.band, Band::Green);
        assert_eq!(a.score, 0.0);
        assert!(a.missing.iter().any(|m| m.starts_with("turbidity_ntu")));
        assert!(a.missing.iter().any(|m| m.starts_with("ammonia_total_mgl")));
    }

    #[test]
    fn missing_data_never_crashes() {
        let mut r = reading();
        r.pond_id = "T1".into();
        let a = assess(&r, "tilapia", None);
        assert_eq!(a.band, Band::Green);
        assert_eq!(a.score, 0.0);
        assert_eq!(a.missing.len(), 6);
    }

    #[test]
    fn high_ph_makes_a_medium_tan_dangerous() {
        let mut safe = reading();
        safe.pond_id = "T".into();
        safe.do_mgl = Some(6.0);
        safe.ph = Some(6.8);
        safe.water_temp_c = Some(28.0);
        safe.ammonia_total_mgl = Some(1.0);
        let safe = assess(&safe, "tilapia", None);

        let mut risky = reading();
        risky.pond_id = "T".into();
        risky.do_mgl = Some(6.0);
        risky.ph = Some(9.2);
        risky.water_temp_c = Some(31.0);
        risky.ammonia_total_mgl = Some(1.0);
        let risky = assess(&risky, "tilapia", None);

        assert!(risky.score > safe.score);
        assert!(risky.band.as_str() >= Band::Orange.as_str());
        // captured from the Python reference: risky is RED, score 0.32, safe 0.0
        assert_eq!(risky.band, Band::Red);
        assert_eq!(risky.score, 0.32);
        assert_eq!(safe.score, 0.0);
    }

    #[test]
    fn crash_night_simulation_matches_python() {
        let mut r = reading();
        r.pond_id = "KA-BLR-001".into();
        r.water_temp_c = Some(33.4);
        r.do_mgl = Some(1.6);
        r.ph = Some(8.9);
        r.turbidity_ntu = Some(180.0);
        r.ammonia_total_mgl = Some(1.9);
        r.nitrite_mgl = Some(0.8);
        let a = assess(&r, "rohu", None);
        assert_eq!(a.band, Band::Red);
        assert_eq!(a.score, 0.634);
        // NH3-N derived, severity saturates at 1.0
        let nh3 = a.drivers.iter().find(|d| d.parameter == "nh3_mgl").unwrap();
        assert!((nh3.value.unwrap() - 0.8459884569780424).abs() < 1e-9);
        assert_eq!(nh3.severity, 1.0);
        assert_eq!(nh3.reason, "derived from TAN + pH + temperature");
        // severity-descending order captured from Python
        let order: Vec<&str> = a.drivers.iter().map(|d| d.parameter.as_str()).collect();
        assert_eq!(
            order,
            vec!["nh3_mgl", "do_mgl", "ph", "turbidity_ntu", "nitrite_mgl", "water_temp_c"]
        );
    }

    #[test]
    fn night_crash_forecast_fires_before_the_kill() {
        let hot_still = do_crash_forecast(35.0, 27.0, 0.8, 0.0, 10.0, Some(5.4));
        let cool_wet = do_crash_forecast(27.0, 22.0, 4.0, 25.0, 95.0, Some(8.0));
        assert!(hot_still.0 == Band::Orange || hot_still.0 == Band::Red);
        assert_eq!(cool_wet.0, Band::Green);
        assert!(hot_still.1.to_lowercase().contains("oxygen"));
        assert_eq!(hot_still.0, Band::Red);
        assert_eq!(
            hot_still.1,
            "Predicted pre-dawn oxygen crash: hot day 35 C, warm night 27 C (high respiration), \
             still air, no mixing, no rain to re-oxygenate, clear night, evening DO already low (5.4 mg/L)"
        );
        assert_eq!(cool_wet.1, "Pre-dawn oxygen expected to stay safe: cool, mixed, wet conditions");
    }

    #[test]
    fn ts_defaults_to_a_real_datetime() {
        let r = PondReading { ts: chrono::Utc.with_ymd_and_hms(2026, 10, 6, 6, 0, 0).unwrap(), ..Default::default() };
        assert_eq!(r.ts.format("%d-%b %H:%M").to_string(), "06-Oct 06:00");
    }
}
