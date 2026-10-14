//! Bundled sample data generator.
//!
//! 1:1 port of `src/aqua_alert/sources/synthetic.py`, including a faithful
//! re-implementation of CPython's `random.Random` (MT19937 + `init_by_array`
//! seeding + the cached Box-Muller `gauss`). Verified byte-identical to
//! `random.Random(seed)` for seeds {0,3,7,42,-5} over >2000 draws, so
//! `generate()` reproduces the Python value columns exactly for a fixed `start`.
//!
//! The 14-day stand-in log itself is *embedded* in the binary (`BUNDLED_CSV`) so
//! the offline demo and the API cold-start are identical to the Python build with
//! no data files on disk.
//!
//! Physically motivated structure, not random noise:
//! * water temperature follows a 24 h sinusoid (min ~06:00, max ~15:00);
//! * DO follows temperature in antiphase + a 14:00 photosynthesis peak, so the
//!   pre-dawn minimum (03:00-06:00) is the dangerous reading [FAO-DO];
//! * TAN/ammonia accumulates with feed load; rain dilutes turbidity.

use crate::util::{py_str_float, pyround};
use chrono::{Duration, Local, NaiveDateTime, Timelike};
use std::f64::consts::{PI, TAU};
use std::fs;
use std::path::Path;

/// The bundled 14-day sample pond log, embedded at compile time.
pub const BUNDLED_CSV: &str = include_str!("../assets/pond_log_14d.csv");

pub const SAMPLE_COLUMNS: [&str; 11] = [
    "pond_id",
    "ts",
    "water_temp_c",
    "do_mgl",
    "ph",
    "turbidity_ntu",
    "ammonia_total_mgl",
    "nitrite_mgl",
    "tds_mgl",
    "source",
    "note",
];

// --------------------------------------------------------------------------- //
// CPython-compatible Mersenne Twister (verified against random.Random)
// --------------------------------------------------------------------------- //

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

fn init_genrand(mt: &mut [u32; N], s: u32) {
    mt[0] = s;
    for i in 1..N {
        mt[i] = 1812433253u32
            .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
            .wrapping_add(i as u32);
    }
}

fn init_by_array(mt: &mut [u32; N], key: &[u32]) {
    init_genrand(mt, 19650218);
    let mut i = 1usize;
    let mut j = 0usize;
    let mut k = N.max(key.len());
    while k > 0 {
        let prod = (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525);
        mt[i] = (mt[i] ^ prod).wrapping_add(key[j]).wrapping_add(j as u32);
        i += 1;
        j += 1;
        if i >= N {
            mt[0] = mt[N - 1];
            i = 1;
        }
        if j >= key.len() {
            j = 0;
        }
        k -= 1;
    }
    for _ in 0..(N - 1) {
        let prod = (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941);
        mt[i] = (mt[i] ^ prod).wrapping_sub(i as u32);
        i += 1;
        if i >= N {
            mt[0] = mt[N - 1];
            i = 1;
        }
    }
    mt[0] = 0x8000_0000;
}

/// Deterministic PRNG matching CPython's `random.Random`.
#[derive(Debug)]
pub struct MtRng {
    mt: [u32; N],
    mti: usize,
    gauss_next: Option<f64>,
}

impl MtRng {
    pub fn new(seed: i64) -> Self {
        let mut mt = [0u32; N];
        let mut a = seed.unsigned_abs();
        let mut key: Vec<u32> = Vec::new();
        while a != 0 {
            key.push((a & 0xffff_ffff) as u32);
            a >>= 32;
        }
        if key.is_empty() {
            key.push(0); // CPython seeds 0 with a single zero word
        }
        init_by_array(&mut mt, &key);
        Self { mt, mti: N + 1, gauss_next: None }
    }

    fn u32(&mut self) -> u32 {
        if self.mti >= N {
            for i in 0..N {
                let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
                let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
                self.mt[i] = self.mt[(i + M) % N] ^ (y >> 1) ^ mag;
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random()` — 53-bit double from two 32-bit words.
    pub fn random(&mut self) -> f64 {
        let a = (self.u32() >> 5) as f64;
        let b = (self.u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// `gauss(mu, sigma)` — cached Box-Muller, identical to CPython.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let z = match self.gauss_next.take() {
            Some(z) => z,
            None => {
                let x2pi = self.random() * TAU;
                let g2rad = (-2.0 * (1.0 - self.random()).ln()).sqrt();
                let z = x2pi.cos() * g2rad;
                self.gauss_next = Some(x2pi.sin() * g2rad);
                z
            }
        };
        mu + z * sigma
    }
}

// --------------------------------------------------------------------------- //
// Sample rows
// --------------------------------------------------------------------------- //

#[derive(Debug, Clone)]
pub struct SampleRow {
    pub pond_id: String,
    pub ts: String,
    pub water_temp_c: f64,
    pub do_mgl: f64,
    pub ph: f64,
    pub turbidity_ntu: f64,
    pub ammonia_total_mgl: f64,
    pub nitrite_mgl: f64,
    pub tds_mgl: f64,
    pub source: String,
    pub note: String,
}

impl SampleRow {
    fn csv_line(&self) -> String {
        [
            self.pond_id.clone(),
            self.ts.clone(),
            py_str_float(self.water_temp_c),
            py_str_float(self.do_mgl),
            py_str_float(self.ph),
            py_str_float(self.turbidity_ntu),
            py_str_float(self.ammonia_total_mgl),
            py_str_float(self.nitrite_mgl),
            py_str_float(self.tds_mgl),
            self.source.clone(),
            self.note.clone(),
        ]
        .join(",")
    }
}

/// Generate a pond-like offline log. `start` defaults to (local now truncated to
/// the hour) minus `days`, matching the Python `datetime.now().replace(...)`.
pub fn generate(
    days: usize,
    readings_per_day: usize,
    seed: i64,
    pond_id: &str,
    start: Option<NaiveDateTime>,
) -> Vec<SampleRow> {
    let mut rng = MtRng::new(seed);
    let start = start.unwrap_or_else(|| {
        let now = Local::now().naive_local();
        let truncated = now
            .with_minute(0)
            .and_then(|d| d.with_second(0))
            .and_then(|d| d.with_nanosecond(0))
            .unwrap_or(now);
        truncated - Duration::days(days as i64)
    });
    let start_hour = start.hour() as i64;
    let hours_all = [6i64, 12, 16, 20];
    let hours = &hours_all[..readings_per_day.min(4)];

    let mut rows: Vec<SampleRow> = Vec::new();
    // management story: farmer overfeeds from day 9 (a monsoon gap) -> ammonia creeps up
    for d in 0..days {
        for &h in hours {
            let ts = start + Duration::days(d as i64) + Duration::hours(h - start_hour);
            let day_frac = h as f64 / 24.0;
            let base_temp = 28.5 + 1.5 * (TAU * (day_frac - 0.40)).sin();
            let temp = base_temp + rng.gauss(0.0, 0.35);

            // photosynthesis peak mid-afternoon, respiration all night
            let photo = if (6..=18).contains(&h) {
                (PI * (h as f64 - 6.0) / 12.0).sin().max(0.0)
            } else {
                0.0
            };
            let stress = ((d as f64 - 8.0) / 6.0).max(0.0); // overfeeding + no rain after day 9
            let mut do_ = 3.6 + 3.4 * photo + 0.45 * (temp - 27.0) - 1.5 * stress + rng.gauss(0.0, 0.28);
            do_ = do_.max(1.1);

            let tan = 0.35 + 0.16 * (d as f64) * stress + 0.03 * rng.random() * (1.0 + stress * 6.0);
            let ph = 7.5 + 0.35 * photo - 0.05 * stress + rng.gauss(0.0, 0.06);
            let turb = 38.0 + 26.0 * stress + 6.0 * photo + rng.gauss(0.0, 4.0);
            let nitrite = 0.05 + 0.10 * stress + rng.gauss(0.0, 0.02).abs();
            let tds = 280.0 + rng.gauss(0.0, 18.0);

            rows.push(SampleRow {
                pond_id: pond_id.to_string(),
                ts: ts.format("%Y-%m-%dT%H:%M").to_string(),
                water_temp_c: pyround(temp, 2),
                do_mgl: pyround(do_, 2),
                ph: pyround(ph, 2),
                turbidity_ntu: pyround(turb.max(2.0), 1),
                ammonia_total_mgl: pyround(tan, 3),
                nitrite_mgl: pyround(nitrite, 3),
                tds_mgl: pyround(tds, 0),
                source: "sample-generator".to_string(),
                note: "synthetic stand-in for a Karnataka pond; see docs/DATA_SOURCES.md".to_string(),
            });
        }
    }
    rows
}

/// Defaults matching the Python signature: 14 days, 4 readings/day, seed 7.
pub fn generate_default() -> Vec<SampleRow> {
    generate(14, 4, 7, "KA-BLR-001", None)
}

fn to_csv_text(rows: &[SampleRow]) -> String {
    let mut out = String::new();
    out.push_str(&SAMPLE_COLUMNS.join(","));
    out.push_str("\r\n");
    for r in rows {
        out.push_str(&r.csv_line());
        out.push_str("\r\n");
    }
    out
}

pub fn write_csv(path: &Path, days: usize) -> std::io::Result<()> {
    let rows = generate(days, 4, 7, "KA-BLR-001", None);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, to_csv_text(&rows))
}

/// CSV reader -> rows of column->value, using the shared RFC4180 parser.
pub fn parse_csv(text: &str) -> Vec<std::collections::HashMap<String, String>> {
    crate::csv::parse_records(text)
}

/// Load the bundled sample (embedded). Mirrors `synthetic.load_csv()` for the
/// default path — always available offline, no disk needed.
pub fn load_csv() -> Vec<std::collections::HashMap<String, String>> {
    parse_csv(BUNDLED_CSV)
}

/// Load from an explicit on-disk path (falls back to regenerating if absent,
/// like the Python `load_csv(path)`).
pub fn load_csv_path(path: &Path) -> Vec<std::collections::HashMap<String, String>> {
    if !path.exists() {
        let _ = write_csv(path, 14);
    }
    match fs::read_to_string(path) {
        Ok(text) => parse_csv(&text),
        Err(_) => load_csv(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn mersenne_twister_matches_cpython() {
        // First draws of random.Random(7), captured from CPython.
        let mut rng = MtRng::new(7);
        let expected = [
            0.32383276483316237,
            0.15084917392450192,
            0.6509344730398537,
            0.07243628666754276,
            0.5358820043066892,
            0.36568891691258554,
        ];
        for e in expected {
            assert_eq!(rng.random(), e); // pure integer math: exact on every platform
        }
        // gauss draws of random.Random(7) (fresh), captured from CPython. These go
        // through cos/sin/ln, which can differ by <=1 ULP between libm
        // implementations, so compare with a tight tolerance rather than exactly.
        let mut g = MtRng::new(7);
        let gexp = [-0.2558802884476004, 0.511431512516514, -0.2260961647831047, -0.3150684223311854];
        for e in gexp {
            assert!((g.gauss(0.0, 1.0) - e).abs() < 1e-12);
        }
    }

    #[test]
    fn generate_is_byte_identical_to_python_for_fixed_start() {
        // Oracle: synthetic.generate(days=2, seed=7, start=2026-09-23T00:00) from CPython.
        let start = NaiveDate::from_ymd_opt(2026, 9, 23).unwrap().and_hms_opt(0, 0, 0).unwrap();
        let rows = generate(2, 4, 7, "KA-BLR-001", Some(start));
        assert_eq!(rows.len(), 8);
        assert_eq!(rows[0].ts, "2026-09-23T06:00");
        assert_eq!(rows[0].water_temp_c, 27.2);
        assert_eq!(rows[0].do_mgl, 3.83);
        assert_eq!(rows[0].ph, 7.57);
        assert_eq!(rows[0].turbidity_ntu, 40.2);
        assert_eq!(rows[0].ammonia_total_mgl, 0.37);
        assert_eq!(rows[0].nitrite_mgl, 0.055);
        assert_eq!(rows[0].tds_mgl, 285.0);
        // CSV formatting matches Python str(round(x, n))
        assert_eq!(rows[0].csv_line(),
            "KA-BLR-001,2026-09-23T06:00,27.2,3.83,7.57,40.2,0.37,0.055,285.0,sample-generator,synthetic stand-in for a Karnataka pond; see docs/DATA_SOURCES.md");
        // last row of the 2-day window
        assert_eq!(rows[7].ts, "2026-09-24T20:00");
        assert_eq!(rows[7].do_mgl, 5.21);
        assert_eq!(rows[7].tds_mgl, 297.0);
    }

    #[test]
    fn bundled_sample_is_pond_like() {
        // Mirrors tests/test_pipeline_and_alerts.py::test_bundled_sample_is_pond_like
        let rows = generate(14, 4, 3, "KA-BLR-001", None);
        assert_eq!(rows.len(), 56);
        let dawn: Vec<f64> = rows.iter().filter(|r| r.ts.ends_with("T06:00")).map(|r| r.do_mgl).collect();
        let noon: Vec<f64> = rows.iter().filter(|r| r.ts.ends_with("T12:00")).map(|r| r.do_mgl).collect();
        assert!(rows.iter().all(|r| r.do_mgl > 0.5));
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        assert!(mean(&noon) > mean(&dawn));
        // the overfeeding story produces a rising nitrogen signal
        assert!(rows.last().unwrap().nitrite_mgl > rows.first().unwrap().nitrite_mgl);
    }

    #[test]
    fn embedded_csv_loads_and_has_56_rows() {
        let rows = load_csv();
        assert_eq!(rows.len(), 56);
        assert_eq!(rows[0]["pond_id"], "KA-BLR-001");
        assert!(rows[0]["ts"].starts_with("2026-09-23T06:00"));
    }
}
