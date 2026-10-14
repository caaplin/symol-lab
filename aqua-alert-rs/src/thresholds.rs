//! Species envelopes + the water-chemistry maths that turns cheap inputs into
//! parameters small farmers cannot afford to measure.
//!
//! 1:1 port of `src/aqua_alert/thresholds.py`.
//!
//! Why this module matters
//! -----------------------
//! A small farm cannot buy a dissolved-oxygen probe (INR 8k-25k), an ammonia
//! photometer (INR 15k+) or a turbidity meter. But it *can* buy, or already owns:
//! a 30-rupee thermometer, a 150-rupee pH strip, a rain gauge / the sky, and a
//! mobile phone with a camera. Everything else here is derived, not measured.
//!
//! References
//! ----------
//! [FAO-DO]   FAO Training Series, "Improving pond water quality", x6709e ch.2.5.
//! [FAO-NH3]  FAO, "Site selection for aquaculture: chemical features of water",
//!            AC175E18 ch.18 (Emerson et al. 1975 / EIFAC 1986 pKa tables).
//! [EXT]      eXtension/SRAC "Water Quality in Aquaculture" species table.
//! [TIL]      El-Sherif & El-Feky 2008; Nehemia et al. 2012 (PMC9640449).
//! [APHA]     Standard Methods / Benson-Krause DO saturation equation.

use crate::util::{py_str_float, pyround};
use once_cell::sync::Lazy;
use std::collections::HashMap;

// --------------------------------------------------------------------------- //
// Derived-parameter chemistry
// --------------------------------------------------------------------------- //

/// Fraction of total ammonia (TAN) present as toxic NH3.
///
/// Emerson et al. (1975) pKa model, as tabulated by EIFAC 1986 and reproduced by
/// FAO [FAO-NH3]:  pKa = 0.09018 + 2729.92 / (T + 273.15);  f = 1 / (1 + 10^(pKa - pH)).
pub fn unionized_ammonia_fraction(ph: f64, temp_c: f64) -> f64 {
    let pka = 0.09018 + 2729.92 / (temp_c + 273.15);
    1.0 / (1.0 + 10f64.powf(pka - ph))
}

/// Toxic NH3-N in mg/L from a cheap total-ammonia test strip + pH strip + thermometer.
pub fn unionized_ammonia_mgl(tan_mgl: f64, ph: f64, temp_c: f64) -> f64 {
    tan_mgl * unionized_ammonia_fraction(ph, temp_c)
}

/// DO at 100% air saturation (mg/L). Benson-Krause / APHA [APHA].
pub fn do_saturation_mgl(temp_c: f64, salinity_ppt: f64) -> f64 {
    let tk = temp_c + 273.15;
    let ln_c = -139.34411
        + 1.575701e5 / tk
        - 6.642308e7 / tk.powi(2)
        + 1.243800e10 / tk.powi(3)
        - 8.621949e11 / tk.powi(4);
    let c = ln_c.exp();
    if salinity_ppt <= 0.0 {
        // freshwater pond: no correction
        return c;
    }
    // Chen & Millero salinity correction, linearised (coastal shrimp ponds only)
    c - 0.0841 * salinity_ppt + 0.000255 * salinity_ppt * temp_c
}

/// Freshwater convenience wrapper (`salinity_ppt = 0.0`).
pub fn do_saturation_freshwater(temp_c: f64) -> f64 {
    do_saturation_mgl(temp_c, 0.0)
}

// --------------------------------------------------------------------------- //
// Species envelopes
// --------------------------------------------------------------------------- //

/// Safe operating window for one parameter.
///
/// * `ideal`       — no action, no cost.
/// * `act`         — sub-optimal, cheap preventive action + a YELLOW warning.
/// * `crit_low/high` — physiology breaking down -> ORANGE/RED, must act today.
/// * `weight`      — contribution to the fused risk score (weights sum to 1.0).
#[derive(Debug, Clone)]
pub struct Envelope {
    pub ideal: (f64, f64),
    pub act: (f64, f64),
    pub crit_low: f64,
    pub crit_high: f64,
    pub weight: f64,
    pub unit: &'static str,
    pub note: &'static str,
}

impl Envelope {
    #[allow(clippy::too_many_arguments)]
    const fn new(
        ideal: (f64, f64),
        act: (f64, f64),
        crit_low: f64,
        crit_high: f64,
        weight: f64,
        unit: &'static str,
        note: &'static str,
    ) -> Self {
        Self { ideal, act, crit_low, crit_high, weight, unit, note }
    }
}

#[derive(Debug, Clone)]
pub struct SpeciesProfile {
    pub name: &'static str,
    pub local_names: HashMap<&'static str, &'static str>,
    pub envelopes: HashMap<&'static str, Envelope>,
    pub source: &'static str,
    /// below this, advise feeding stop
    pub feed_stop_below_do: f64,
    /// below this, advise aerator / agitation
    pub aeration_below_do: f64,
}

fn names(en: &'static str, kn: &'static str, hi: &'static str) -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("en", en);
    m.insert("kn", kn);
    m.insert("hi", hi);
    m
}

fn envelopes(list: Vec<(&'static str, Envelope)>) -> HashMap<&'static str, Envelope> {
    list.into_iter().collect()
}

fn tilapia() -> SpeciesProfile {
    SpeciesProfile {
        name: "tilapia",
        local_names: names("Tilapia", "ತಿಲಾಪಿಯಾ", "तिलापिया"),
        envelopes: envelopes(vec![
            // FAO: tilapia hard minimum 2 mg/L; growth <3.5; optimum 5-8 [FAO-DO][TIL]
            ("do_mgl", Envelope::new((5.0, 12.0), (3.0, 14.0), 2.0, 18.0, 0.35, "mg/L",
                "Feeding stops <3 mg/L; mortality risk <1 mg/L [TIL]")),
            // Toxic NH3-N (not TAN). >0.2 mg/L prolonged exposure starts mortality [TIL]
            ("nh3_mgl", Envelope::new((0.0, 0.02), (0.02, 0.05), 0.0, 0.20, 0.20, "mg/L NH3-N",
                "NH3-N >0.2 mg/L causes kills; appetite loss at 0.08 [TIL]")),
            ("ph", Envelope::new((6.5, 8.5), (6.0, 9.0), 5.0, 9.5, 0.15, "pH", "")),
            ("water_temp_c", Envelope::new((26.0, 32.0), (22.0, 35.0), 15.0, 38.0, 0.15, "degC", "")),
            ("turbidity_ntu", Envelope::new((0.0, 80.0), (0.0, 150.0), 0.0, 300.0, 0.10, "NTU", "")),
            ("nitrite_mgl", Envelope::new((0.0, 0.5), (0.0, 1.0), 0.0, 5.0, 0.05, "mg/L NO2-N", "")),
        ]),
        source: "[FAO-DO], [TIL], [EXT]",
        feed_stop_below_do: 3.0,
        aeration_below_do: 4.0,
    }
}

fn rohu() -> SpeciesProfile {
    // rohu / catla / mrigal polyculture - the dominant Indian pond system
    SpeciesProfile {
        name: "indian_major_carp",
        local_names: names("Rohu-Catla (IMC)", "ರೋಹು-ಕಟ್ಲಾ", "रोहू-कतला"),
        envelopes: envelopes(vec![
            // FAO: common carp minimum 3 mg/L, preferred >=5 [FAO-DO]
            ("do_mgl", Envelope::new((5.0, 12.0), (3.5, 14.0), 3.0, 18.0, 0.35, "mg/L", "")),
            ("nh3_mgl", Envelope::new((0.0, 0.02), (0.02, 0.05), 0.0, 0.10, 0.20, "mg/L NH3-N",
                "Carp tolerate total ammonia 3-4 mg/L at pH<8 [FAO-NH3]")),
            ("ph", Envelope::new((6.5, 8.5), (6.0, 9.0), 5.5, 9.5, 0.15, "pH", "")),
            ("water_temp_c", Envelope::new((24.0, 32.0), (20.0, 35.0), 12.0, 38.0, 0.15, "degC", "")),
            ("turbidity_ntu", Envelope::new((0.0, 100.0), (0.0, 200.0), 0.0, 350.0, 0.10, "NTU", "")),
            ("nitrite_mgl", Envelope::new((0.0, 0.5), (0.0, 1.0), 0.0, 5.0, 0.05, "mg/L NO2-N", "")),
        ]),
        source: "[FAO-DO], [FAO-NH3], [EXT]",
        feed_stop_below_do: 3.5,
        aeration_below_do: 4.5,
    }
}

fn pangasius() -> SpeciesProfile {
    SpeciesProfile {
        name: "pangasius_catfish",
        local_names: names("Pangasius / Catfish", "ಪಂಗಾಸಿಯಸ್", "पंगासियस"),
        envelopes: envelopes(vec![
            ("do_mgl", Envelope::new((4.0, 12.0), (2.5, 16.0), 1.0, 20.0, 0.35, "mg/L",
                "Air-breathers: survive very low DO but grow poorly <2.5 mg/L")),
            ("nh3_mgl", Envelope::new((0.0, 0.02), (0.02, 0.06), 0.0, 0.15, 0.20, "mg/L NH3-N", "")),
            ("ph", Envelope::new((6.5, 8.5), (6.0, 9.0), 5.0, 9.5, 0.15, "pH", "")),
            ("water_temp_c", Envelope::new((26.0, 32.0), (23.0, 36.0), 18.0, 39.0, 0.15, "degC", "")),
            ("turbidity_ntu", Envelope::new((0.0, 100.0), (0.0, 200.0), 0.0, 350.0, 0.10, "NTU", "")),
            ("nitrite_mgl", Envelope::new((0.0, 0.5), (0.0, 1.0), 0.0, 5.0, 0.05, "mg/L NO2-N", "")),
        ]),
        source: "[FAO-DO], [FAO-NH3]",
        feed_stop_below_do: 2.5,
        aeration_below_do: 3.0,
    }
}

fn prawn() -> SpeciesProfile {
    SpeciesProfile {
        name: "pacific_white_shrimp",
        local_names: names("Whiteleg shrimp", "ಸೀಗಡಿ", "झींगा"),
        envelopes: envelopes(vec![
            ("do_mgl", Envelope::new((5.0, 12.0), (3.5, 14.0), 2.5, 16.0, 0.35, "mg/L", "")),
            ("nh3_mgl", Envelope::new((0.0, 0.01), (0.01, 0.03), 0.0, 0.10, 0.20, "mg/L NH3-N",
                "Shrimp are the most ammonia-sensitive animal in the pond")),
            ("ph", Envelope::new((7.2, 8.5), (7.0, 9.0), 6.5, 9.5, 0.15, "pH", "")),
            ("water_temp_c", Envelope::new((26.0, 31.0), (23.0, 34.0), 18.0, 36.0, 0.15, "degC", "")),
            ("turbidity_ntu", Envelope::new((0.0, 60.0), (0.0, 120.0), 0.0, 250.0, 0.10, "NTU", "")),
            ("nitrite_mgl", Envelope::new((0.0, 0.3), (0.0, 0.8), 0.0, 3.0, 0.05, "mg/L NO2-N", "")),
        ]),
        source: "[FAO-DO], [EXT]",
        feed_stop_below_do: 3.5,
        aeration_below_do: 4.5,
    }
}

/// The species registry, keyed by the short names the API/CLI accept.
pub static PROFILES: Lazy<HashMap<&'static str, SpeciesProfile>> = Lazy::new(|| {
    let mut m: HashMap<&'static str, SpeciesProfile> = HashMap::new();
    m.insert("tilapia", tilapia());
    let r = rohu();
    m.insert("rohu", r.clone());
    m.insert("carp", r);
    m.insert("pangasius", pangasius());
    m.insert("shrimp", prawn());
    m
});

/// Look up a profile by (case-insensitive, trimmed) species key; falls back to
/// tilapia exactly like the Python `PROFILES.get(key, _TILAPIA)`.
pub fn get_profile(species: &str) -> SpeciesProfile {
    let key = species.trim().to_lowercase();
    match PROFILES.get(key.as_str()) {
        Some(p) => p.clone(),
        None => tilapia(),
    }
}

// --------------------------------------------------------------------------- //
// Feeding & aeration decision aid (this is what saves money day-to-day)
// --------------------------------------------------------------------------- //

#[derive(Debug, Clone)]
pub struct FeedAdvice {
    /// fraction of normal ration to give today
    pub feed_fraction: f64,
    pub text_en: String,
    pub text_local: String,
    pub reason: String,
}

/// Feeding into low DO is the classic small-farm money burner: uneaten feed
/// decays, ammonia rises, oxygen falls further [TIL].
pub fn feed_advice(
    profile: &SpeciesProfile,
    do_mgl: Option<f64>,
    nh3_mgl: Option<f64>,
    temp_c: Option<f64>,
    _local_lang: &str,
) -> FeedAdvice {
    let mut reasons: Vec<String> = Vec::new();
    let mut frac = 1.0f64;

    let do_ideal_lo = profile.envelopes["do_mgl"].ideal.0;
    let nh3_act_hi = profile.envelopes["nh3_mgl"].act.1;
    let temp_ideal_hi = profile.envelopes["water_temp_c"].ideal.1;

    if let Some(do_v) = do_mgl {
        if do_v < profile.feed_stop_below_do {
            frac = 0.0;
            reasons.push(format!(
                "DO {:.1} < {} mg/L",
                do_v,
                py_str_float(profile.feed_stop_below_do)
            ));
        } else if do_v < do_ideal_lo {
            frac = 0.5;
            reasons.push(format!("DO {:.1} mg/L below optimum", do_v));
        }
    }
    if let Some(nh3) = nh3_mgl {
        if nh3 > nh3_act_hi {
            frac = frac.min(0.5);
            reasons.push(format!("NH3-N {:.3} mg/L high", nh3));
        }
    }
    if let Some(t) = temp_c {
        if t > temp_ideal_hi {
            frac = frac.min(0.7);
            reasons.push(format!("water {:.0} C hot", t));
        }
    }

    let reason = if reasons.is_empty() {
        "all monitored parameters in range".to_string()
    } else {
        reasons.join("; ")
    };

    let (en, local) = match pyround(frac, 1) {
        f if f == 0.0 => ("Do NOT feed today", "ಇಂದು ಮೀನುಗಳಿಗೆ ಆಹಾರ ಕೊಡಬೇಡಿ"),
        f if f == 0.5 => ("Feed only half the normal ration", "ಅರ್ಧ ಪ್ರಮಾಣದಷ್ಟೇ ಆಹಾರ ಕೊಡಿ"),
        f if f == 0.7 => ("Feed about 70% of the normal ration", "ಸಾಮಾನ್ಯದ 70% ಆಹಾರ ಕೊಡಿ"),
        _ => ("Feed the normal ration", "ಸಾಮಾನ್ಯ ಪ್ರಮಾಣದಲ್ಲಿ ಆಹಾರ ಕೊಡಿ"),
    };

    FeedAdvice {
        feed_fraction: frac,
        text_en: en.to_string(),
        text_local: local.to_string(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn unionized_fraction_matches_eifac_tables() {
        // FAO AC175E18 reproduces the Emerson/EIFAC %-un-ionized-NH3 tables.
        assert!(approx(unionized_ammonia_fraction(8.5, 20.0), 0.111, 0.004));
        assert!(approx(unionized_ammonia_fraction(7.0, 20.0), 0.004, 0.002));
        // doctest points from the Python module
        assert_eq!(pyround(unionized_ammonia_fraction(8.5, 30.0) * 100.0, 1), 20.2);
        assert_eq!(pyround(unionized_ammonia_fraction(8.5, 20.0) * 100.0, 1), 11.1);
        assert_eq!(pyround(unionized_ammonia_fraction(7.0, 20.0) * 100.0, 1), 0.4);
        // warmer + more alkaline = dramatically more toxic ammonia at the same TAN
        assert!(unionized_ammonia_fraction(8.5, 30.0) > unionized_ammonia_fraction(8.5, 20.0));
    }

    #[test]
    fn ammonia_conversion_is_the_free_sensor() {
        assert!(unionized_ammonia_mgl(1.0, 9.0, 30.0) > 0.2);
        assert!(unionized_ammonia_mgl(1.0, 6.5, 30.0) < 0.02);
        assert!(approx(unionized_ammonia_mgl(1.0, 9.0, 30.0), 0.4453264567, 1e-9));
    }

    #[test]
    fn do_saturation_is_physical() {
        assert!(do_saturation_freshwater(25.0) > 8.0 && do_saturation_freshwater(25.0) < 8.6);
        assert!(do_saturation_freshwater(0.0) > 14.0 && do_saturation_freshwater(0.0) < 14.8);
        assert!(do_saturation_freshwater(10.0) > do_saturation_freshwater(30.0));
        assert!(approx(do_saturation_freshwater(25.0), 8.2634566978, 1e-6));
    }

    #[test]
    fn feed_advice_matches_python() {
        let p = get_profile("tilapia");
        let a = feed_advice(&p, Some(2.5), Some(0.0), Some(29.0), "kn");
        assert_eq!(a.feed_fraction, 0.0);
        assert_eq!(a.text_en, "Do NOT feed today");
        assert_eq!(a.reason, "DO 2.5 < 3.0 mg/L");

        let b = feed_advice(&p, Some(3.4), Some(0.0), Some(29.0), "kn");
        assert_eq!(b.feed_fraction, 0.5);
        assert_eq!(b.reason, "DO 3.4 mg/L below optimum");

        let c = feed_advice(&p, Some(6.0), Some(0.0), Some(34.0), "kn");
        assert_eq!(c.feed_fraction, 0.7);
        assert_eq!(c.reason, "water 34 C hot");

        let d = feed_advice(&p, None, None, None, "kn");
        assert_eq!(d.feed_fraction, 1.0);
        assert_eq!(d.reason, "all monitored parameters in range");
    }

    #[test]
    fn profile_lookup_and_fallback() {
        assert_eq!(get_profile("rohu").name, "indian_major_carp");
        assert_eq!(get_profile("CARP").name, "indian_major_carp");
        assert_eq!(get_profile("shrimp").name, "pacific_white_shrimp");
        assert_eq!(get_profile("pangasius").feed_stop_below_do, 2.5);
        // unknown species falls back to tilapia
        assert_eq!(get_profile("salmon").name, "tilapia");
    }
}
