//! Drift detection: catch the slow failure that absolute thresholds miss.
//!
//! 1:1 port of `src/aqua_alert/anomaly.py`. A pond can sit at DO = 5.5 mg/L for a
//! week and still be dying — because *this farmer's* pond normally sits at 8 mg/L.
//! Absolute FAO thresholds are the floor; the farm's own history is the ceiling.
//! We keep a rolling baseline per pond using a robust (MAD-based) z-score, so a
//! single bad entry or a stuck sensor does not trigger an alert storm.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

/// pond_id -> ordered list of (parameter, rolling series). The order preserves the
/// sequence parameters were first seen in, matching Python's dict iteration order
/// (which the drift note string depends on).
type History = HashMap<String, Vec<(String, Vec<f64>)>>;

static HISTORY: Lazy<Mutex<History>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Serialises the unit tests that mutate the process-global `HISTORY`, so one
/// test's `reset()` cannot wipe another's in-progress baseline under `cargo test`'s
/// default multi-threaded runner. Poison-tolerant (a panicking test must not cascade).
#[cfg(test)]
pub static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg(test)]
pub fn lock_history() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// 14 days x 4 readings/day
pub const WINDOW: usize = 56;
/// convert MAD -> sigma for a normal distribution
pub const MAD_SCALE: f64 = 1.4826;

/// Median-absolute-deviation z-score. Returns 0.0 when the baseline is too thin.
pub fn robust_z(value: f64, baseline: &[f64]) -> f64 {
    if baseline.len() < 6 {
        return 0.0;
    }
    let mut sorted = baseline.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[sorted.len() / 2];

    let mut devs: Vec<f64> = baseline.iter().map(|x| (x - med).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = devs[devs.len() / 2];

    if mad < 1e-9 {
        // flat baseline: any move is meaningful, but cap it
        return if (value - med).abs() < 1e-9 {
            0.0
        } else {
            3.0f64.copysign(value - med)
        };
    }
    (value - med) / (MAD_SCALE * mad)
}

pub fn update(pond_id: &str, parameter: &str, value: Option<f64>) {
    let v = match value {
        Some(v) => v,
        None => return,
    };
    let mut hist = HISTORY.lock().unwrap();
    let list = hist.entry(pond_id.to_string()).or_default();
    // `position` (not `iter_mut().find`) keeps the lookup borrow from overlapping the
    // `list.push` in the None arm, and avoids binding `p` as `&mut String`.
    match list.iter().position(|item| item.0 == parameter) {
        Some(i) => {
            let buf = &mut list[i].1;
            buf.push(v);
            if buf.len() > WINDOW {
                buf.remove(0);
            }
        }
        None => list.push((parameter.to_string(), vec![v])),
    }
}

/// z-scores of the *latest* value against the pond's own baseline, in insertion order.
pub fn drift_scores(pond_id: &str) -> Vec<(String, f64)> {
    let hist = HISTORY.lock().unwrap();
    let mut out: Vec<(String, f64)> = Vec::new();
    if let Some(list) = hist.get(pond_id) {
        for (param, buf) in list {
            if buf.len() > 6 {
                let last = buf.len() - 1;
                let z = robust_z(buf[last], &buf[..last]);
                out.push((param.clone(), z));
            }
        }
    }
    out
}

/// Map the largest |z| to a 0-1 severity used by the fusion layer.
pub fn severity_from_drift(z: &[(String, f64)], threshold: f64) -> f64 {
    if z.is_empty() {
        return 0.0;
    }
    let m = z.iter().fold(0.0f64, |acc, (_, v)| acc.max(v.abs()));
    if m <= threshold {
        return 0.0;
    }
    (1.0f64).min((m - threshold) / (2.0 * threshold))
}

/// Convenience wrapper using the default threshold (2.5), matching the Python default.
pub fn severity_from_drift_default(z: &[(String, f64)]) -> f64 {
    severity_from_drift(z, 2.5)
}

pub fn reset() {
    HISTORY.lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_flags_a_pond_that_is_fine_but_not_itself() {
        let _g = lock_history();
        reset();
        for _ in 0..12 {
            update("DRIFT", "do_mgl", Some(7.8));
        }
        let z = drift_scores("DRIFT");
        // flat baseline -> z is 0 (or the param is present with ~0)
        assert!(z.is_empty() || z.iter().find(|(k, _)| k == "do_mgl").unwrap().1.abs() < 0.1);

        update("DRIFT", "do_mgl", Some(4.6));
        let z2 = drift_scores("DRIFT");
        let do_z = z2.iter().find(|(k, _)| k == "do_mgl").unwrap().1;
        assert!(do_z.abs() >= 3.0);
        assert!((do_z - (-3.0)).abs() < 1e-12); // capped, negative
        assert!(severity_from_drift_default(&z2) > 0.0);
        assert!((severity_from_drift_default(&z2) - 0.1).abs() < 1e-12);
        reset();
    }

    #[test]
    fn robust_z_needs_six_points() {
        assert_eq!(robust_z(10.0, &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn window_caps_history() {
        let _g = lock_history();
        reset();
        for i in 0..70 {
            update("CAP", "ph", Some(7.0 + (i as f64) * 0.001));
        }
        let hist = HISTORY.lock().unwrap();
        let buf = &hist["CAP"][0].1;
        assert_eq!(buf.len(), WINDOW);
        reset();
    }
}
