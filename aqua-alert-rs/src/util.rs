//! Tiny helpers that make the Rust output byte-identical to the Python original.

/// Replicates Python's built-in `round(x, ndigits)`.
///
/// CPython rounds the *exact* binary value of the double to `ndigits` decimals
/// (half-to-even). Rust's `{:.n$}` formatter rounds the same exact value, so
/// formatting-then-parsing reproduces `round()` for every value this system
/// produces (none of which land on an exact decimal tie).
pub fn pyround(x: f64, ndigits: u32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let s = format!("{:.*}", ndigits as usize, x);
    s.parse::<f64>().unwrap_or(x)
}

/// Replicates Python's `str(float)` for the "plain" values that appear in the
/// farmer-facing note strings (temperatures, thresholds): shortest round-trip
/// digits, but always keeping a decimal point (`35` -> `35.0`).
pub fn py_str_float(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    let s = format!("{x}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_matches_python() {
        assert_eq!(pyround(0.5294871794871795, 3), 0.529);
        assert_eq!(pyround(0.6341666666666667, 3), 0.634);
        assert_eq!(pyround(0.8459884569780424, 3), 0.846);
        assert_eq!(pyround(29.29123, 2), 29.29);
        assert_eq!(pyround(285.0, 0), 285.0);
        // feed fractions are exact 1-decimal values, never ties
        assert_eq!(pyround(0.7, 1), 0.7);
        assert_eq!(pyround(0.5, 1), 0.5);
    }

    #[test]
    fn str_float_matches_python() {
        assert_eq!(py_str_float(35.0), "35.0");
        assert_eq!(py_str_float(34.7), "34.7");
        assert_eq!(py_str_float(3.5), "3.5");
    }
}
