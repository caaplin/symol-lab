//! THE ALERT TOOL (deliverable #3).
//!
//! 1:1 port of `src/aqua_alert/alerts/dispatcher.py`.
//!
//! A farmer with a ₹0 data plan and a ₹1,000 feature phone must still get the
//! warning, so the alert layer is channel-first (SMS / Voice-IVR / WhatsApp / app).
//!
//! Anti-alert-fatigue rules baked in:
//! * one message per (pond, band-change, 6 h) -> dedupe_key;
//! * RED bypasses dedupe but is rate-limited to once per hour;
//! * GREEN is silenced (no SMS credits spent on good news);
//! * every message ends with a single action.

use crate::schemas::{Alert, Band, RiskAssessment};
use chrono::{DateTime, Duration, Utc};
use sha1::{Digest, Sha1};
use std::collections::HashMap;

/// GSM-7 one-part SMS budget.
pub const SMS_LIMIT: usize = 160;

pub fn band_prefix(band: Band) -> &'static str {
    match band {
        Band::Green => "SAFE",
        Band::Yellow => "WATCH",
        Band::Orange => "ALERT",
        Band::Red => "EMERGENCY",
    }
}

/// Pluggable dispatch. `console` prints; `sms`/`whatsapp` would POST to a provider
/// (MSG91 / Gupshup / Twilio in India) when AQUA_SMS_API_KEY is set — the provider
/// adapter is scaffolded exactly as in the Python demo build.
#[derive(Debug, Default)]
pub struct AlertDispatcher {
    pub sent: Vec<Alert>,
    last: HashMap<String, DateTime<Utc>>,
}

impl AlertDispatcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stable 12-hex-char key for (pond, band).
    ///
    /// NOTE on parity: the Python `f"{pond_id}:{band}"` interpolates a `str`-`Enum`,
    /// whose formatted form is *Python-version dependent* (3.11 yields
    /// "KA-BLR-001:Band.ORANGE"; 3.10 and `StrEnum` yield "KA-BLR-001:ORANGE"). The
    /// dedupe_key is an opaque internal token, so this port hashes the stable band
    /// **value** ("…:ORANGE"). Dedup *behaviour* is identical; only the raw hex of
    /// `Alert.id` differs from a Python-3.11 run. See README "Behavioural notes".
    pub fn dedupe_key(&self, pond_id: &str, band: Band) -> String {
        let mut hasher = Sha1::new();
        hasher.update(format!("{pond_id}:{}", band.as_str()).as_bytes());
        let digest = hasher.finalize();
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        hex[..12].to_string()
    }

    fn suppressed(&self, key: &str, band: Band, now: DateTime<Utc>) -> bool {
        match self.last.get(key) {
            None => false,
            Some(last) => {
                let window = if band == Band::Red {
                    Duration::hours(1)
                } else {
                    Duration::hours(6)
                };
                (now - *last) < window
            }
        }
    }

    /// Compose the farmer-facing message for a channel.
    pub fn compose(&self, assessment: &RiskAssessment, channel: &str) -> String {
        let prefix = band_prefix(assessment.band);
        let pond = &assessment.pond_id;
        let action = assessment
            .actions
            .first()
            .cloned()
            .unwrap_or_else(|| "Re-test in the morning.".to_string());
        // strip the "(kannada)" parenthetical and any "| local" tail for the SMS budget
        let short_action = action
            .split('|')
            .next()
            .unwrap_or("")
            .split('(')
            .next()
            .unwrap_or("")
            .trim()
            .trim_end_matches('.')
            .to_string();

        if channel == "voice" {
            return format!(
                "{prefix}. Pond {pond}. {}. {short_action}. {}.",
                assessment.headline_en, assessment.headline_local
            );
        }
        let msg = format!(
            "AQUA{prefix} Pond {pond} {} score {:.2}. {short_action}.",
            assessment.ts.format("%d-%b %H:%M"),
            assessment.score
        );
        // char-safe truncation to the one-part SMS budget
        msg.chars().take(SMS_LIMIT).collect()
    }

    /// Dispatch one assessment. Mirrors the Python control flow exactly: GREEN and
    /// suppressed cases return early (and do NOT update the dedupe window), so only
    /// a genuinely composed alert advances `_last` and is recorded in `sent`.
    pub fn dispatch(&mut self, assessment: &RiskAssessment, channel: &str, to: Option<&str>) -> Alert {
        let now = Utc::now();
        let key = self.dedupe_key(&assessment.pond_id, assessment.band);

        if assessment.band == Band::Green {
            return Alert {
                id: key.clone(),
                pond_id: assessment.pond_id.clone(),
                created_at: now,
                band: assessment.band,
                channel: channel.to_string(),
                message: String::new(),
                dedupe_key: key,
                delivered: false,
                error: Some("GREEN suppressed: do not spend SMS credits on good news".to_string()),
            };
        }

        if self.suppressed(&key, assessment.band, now) {
            return Alert {
                id: key.clone(),
                pond_id: assessment.pond_id.clone(),
                created_at: now,
                band: assessment.band,
                channel: channel.to_string(),
                message: String::new(),
                dedupe_key: key,
                delivered: false,
                error: Some("suppressed by anti-fatigue window".to_string()),
            };
        }

        let message = self.compose(assessment, channel);
        let mut alert = Alert {
            id: key.clone(),
            pond_id: assessment.pond_id.clone(),
            created_at: now,
            band: assessment.band,
            channel: channel.to_string(),
            message: message.clone(),
            dedupe_key: key.clone(),
            delivered: false,
            error: None,
        };

        let api_key = std::env::var("AQUA_SMS_API_KEY").ok().filter(|s| !s.is_empty());
        if channel == "console" || api_key.is_none() {
            println!("[{} -> {}] {}", channel.to_uppercase(), to.unwrap_or("farmer"), message);
            alert.delivered = true;
        } else {
            alert.error = Some("provider adapter not configured in this demo build".to_string());
        }

        self.last.insert(key, now);
        self.sent.push(alert.clone());
        alert
    }
}

/// Who gets told what, and how fast — the operational half of the alert tool.
/// Returned in the Python dict order (GREEN, YELLOW, ORANGE, RED).
pub fn escalation_matrix() -> Vec<(&'static str, &'static str)> {
    vec![
        ("GREEN", "Farmer app only, silent. Weekly digest SMS on Sunday."),
        ("YELLOW", "Farmer SMS (1 part) + app card. Co-operative dashboard updated."),
        (
            "ORANGE",
            "Farmer SMS + voice call in Kannada + WhatsApp photo request; co-operative officer notified in dashboard.",
        ),
        (
            "RED",
            "Farmer SMS + automated voice call + WhatsApp; district fisheries extension officer paged; 6-hourly re-check until band drops.",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline;
    use crate::risk;
    use crate::schemas::PondReading;

    fn assess(pond: &str, do_mgl: f64, ph: f64, temp: f64) -> RiskAssessment {
        let r = PondReading {
            pond_id: pond.into(),
            do_mgl: Some(do_mgl),
            ph: Some(ph),
            water_temp_c: Some(temp),
            ..Default::default()
        };
        risk::assess(&r, "tilapia", None)
    }

    #[test]
    fn sms_fits_one_gsm_part() {
        let _g = crate::anomaly::lock_history();
        crate::anomaly::reset();
        let disp = AlertDispatcher::new();
        let day = pipeline::demo_day("KA-BLR-001", "tilapia", None);
        let msg = disp.compose(&day.assessment, "sms");
        assert!(msg.chars().count() <= SMS_LIMIT);
        assert!(msg.contains("AQUA"));
        // exact string captured from the Python reference build
        assert_eq!(
            msg,
            "AQUAALERT Pond KA-BLR-001 06-Oct 06:00 score 0.20. \
             Turn on the aerator / paddle wheel NOW and keep it running until sunrise."
        );
        crate::anomaly::reset();
    }

    #[test]
    fn voice_channel_includes_local_headline() {
        let _g = crate::anomaly::lock_history();
        crate::anomaly::reset();
        let disp = AlertDispatcher::new();
        let day = pipeline::demo_day("KA-BLR-001", "tilapia", None);
        let v = disp.compose(&day.assessment, "voice");
        assert!(v.starts_with("ALERT. Pond KA-BLR-001. Act today - fish at risk."));
        assert!(v.contains("ಇಂದೇ ಕ್ರಮ ತಗೊಳ್ಳಿ"));
        crate::anomaly::reset();
    }

    #[test]
    fn green_is_suppressed_and_does_not_advance_the_window() {
        let mut disp = AlertDispatcher::new();
        let a = assess("G1", 7.5, 7.5, 28.0);
        assert_eq!(a.band, Band::Green);
        let alert = disp.dispatch(&a, "console", None);
        assert!(!alert.delivered);
        assert!(alert.error.unwrap().contains("suppressed"));
        assert!(disp.sent.is_empty(), "GREEN must not be recorded as sent");
    }

    #[test]
    fn dedupe_stops_sms_fatigue() {
        let mut disp = AlertDispatcher::new();
        let a = assess("D1", 2.0, 7.6, 29.0); // tilapia DO 2.0 -> critical -> RED
        assert_eq!(a.band, Band::Red);
        let first = disp.dispatch(&a, "console", None);
        let second = disp.dispatch(&a, "console", None);
        assert!(first.delivered);
        assert!(!second.delivered);
        assert!(second.error.unwrap().contains("anti-fatigue"));
        assert_eq!(disp.sent.len(), 1);
    }

    #[test]
    fn escalation_matrix_covers_every_band() {
        let m = escalation_matrix();
        let keys: Vec<&str> = m.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec!["GREEN", "YELLOW", "ORANGE", "RED"]);
        let orange = m.iter().find(|(k, _)| *k == "ORANGE").unwrap().1;
        assert!(orange.contains("IVR") || orange.to_lowercase().contains("voice"));
    }
}
