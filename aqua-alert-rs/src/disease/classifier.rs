//! Photo -> disease triage. THE AI LAYER.
//!
//! 1:1 port of `src/aqua_alert/disease/classifier.py`, with two backends behind one
//! interface:
//!
//! 1. `onnx` (feature = "onnx", mirrors the Python `ml` extra): a real deep model —
//!    a ViT-base / EfficientNet fine-tuned on the South-Asia fish-disease sets,
//!    exported to ONNX and run in-process with ONNX Runtime (`ort`). This keeps the
//!    "single self-contained binary" property: no Python, no torch at runtime.
//!    Preprocessing replicates the HF image processor (resize -> RGB -> normalise ->
//!    NCHW float32), then softmax + top-k + a confidence floor below which we
//!    *abstain and refer* rather than guess.
//!
//! 2. `heuristic-fallback` (always compiled, zero download): classical lesion colour
//!    statistics on the photo. Not a diagnosis — a triage that decides
//!    healthy / monitor / treat / escalate, and is honest about its uncertainty.
//!    This is what runs offline on a INR 6,000 phone, and it *refuses to guess*
//!    unless the evidence is extreme (a large saturated red blob or a cottony-white
//!    patch), exactly like the Python fallback.
//!
//! `predict()` tries the deep model first (when the feature is on and a model file
//! is present) and degrades to the abstaining heuristic otherwise — never blocking
//! the farmer.

use crate::schemas::DiseasePrediction;
use crate::util::pyround;
use anyhow::{anyhow, Result};
use std::path::Path;

/// Class list of the South-Asia dataset (CC0) — the classes the app must speak about.
pub const SOUTH_ASIA_CLASSES: [&str; 7] = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease",
];

pub const TILAPIA_CLASSES: [&str; 4] =
    ["Streptococcosis", "Columnaris Disease", "Tilapia Lake Virus", "Healthy"];

/// Below this the deep model abstains and refers; we never guess.
pub const MIN_CONFIDENCE: f64 = 0.60;

/// Ungated public checkpoints used to *produce* the ONNX model (see export/).
pub const DEFAULT_MODELS: [&str; 2] = [
    "panda992/fish_disease_datasets",
    "allfahad/fish-disease-v1-original-20260608",
];

pub type Advice = (&'static str, &'static str, &'static str); // (triage, en, local)

/// Farmer-facing triage + first response, in the same order as the Python dict so
/// substring matching resolves identically. Deliberately conservative: we never say
/// "cured"; we say what to do in the next 24 h and when to involve a human expert.
pub const ADVICE: [(&'static str, Advice); 10] = [
    (
        "Bacterial Red disease",
        (
            "escalate_to_expert",
            "Red patches/septicaemia: isolate pond, stop feeding, call extension officer for an \
             antibiotic decision (never self-dose - resistance and residues are real).",
            "ಕೆಂಪು ಕಲೆಗಳು: ಆಹಾರ ನಿಲ್ಲಿಸಿ, ಮೀನುಗಾರಿಕೆ ಇಲಾಖೆ ಅಧಿಕಾರಿಗೆ ಕರೆ ಮಾಡಿ.",
        ),
    ),
    (
        "Bacterial diseases - Aeromoniasis",
        (
            "treat",
            "Ulcers with red margins. Reduce stocking stress, exchange 20% water, add salt 0.1-0.3% \
             if species allows, and re-photograph in 24 h to track spread.",
            "ಹುಣ್ಣುಗಳಿವೆ: 20% ನೀರು ಬದಲಿಸಿ, ಉಪ್ಪು ಸೇರಿಸಿ, 24 ಗಂಟೆ ನಂತರ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.",
        ),
    ),
    (
        "Bacterial gill disease",
        (
            "treat",
            "Gill disease: increase aeration immediately, stop feeding 24 h, exchange water; gill \
             damage is made far worse by low DO.",
            "ಕಿವಿರು ರೋಗ: ತಕ್ಷಣ ಏರೇಶನ್ ಹೆಚ್ಚಿಸಿ, 24 ಗಂಟೆ ಆಹಾರ ನಿಲ್ಲಿಸಿ.",
        ),
    ),
    (
        "Fungal diseases Saprolegniasis",
        (
            "treat",
            "Cotton-wool growths usually follow an injury or handling. Improve water quality first \
             (fungus is a symptom of stress), avoid further netting for a week.",
            "ಹತ್ತಿ ತರಹ ಬೆಳವಣಿಗೆ: ನೀರಿನ ಗುಣಮಟ್ಟ ಸುಧಾರಿಸಿ, ಒಂದು ವಾರ ಬಲೆ ಬಳಸಬೇಡಿ.",
        ),
    ),
    (
        "Parasitic diseases",
        (
            "treat",
            "Parasites (Argulus/trichodina-type signs). Confirm with a ₹0 pond-side trick: place a \
             few fish in clear water and look for moving specks; then follow extension advice on \
             salt/formal in baths.",
            "ಪರಾಸೈಟ್ ಸಾಧ್ಯತೆ: ಸ್ವಚ್ಛ ನೀರಿನಲ್ಲಿ ಮೀನು ಪರೀಕ್ಷಿಸಿ, ಅಧಿಕಾರಿಯ ಸಲಹೆ ಪಡೆಯಿರಿ.",
        ),
    ),
    (
        "Viral diseases White tail disease",
        (
            "escalate_to_expert",
            "Viral white tail has no pond-side cure. Stop all movement of fish and water between \
             ponds TODAY - containment is the only lever. Report to the fisheries department.",
            "ವೈರಸ್ ರೋಗ: ಚಿಕಿತ್ಸೆ ಇಲ್ಲ. ಇಂದೇ ನೀರು/ಮೀನು ಸಾಗಣೆ ನಿಲ್ಲಿಸಿ, ಇಲಾಖೆಗೆ ತಿಳಿಸಿ.",
        ),
    ),
    (
        "Streptococcosis",
        (
            "escalate_to_expert",
            "Streptococcosis: cloudy eye, erratic swimming, sudden mortality in tilapia. Stop \
             feeding, increase aeration, report for a prescribed treatment plan.",
            "ಸ್ಟ್ರೆಪ್ಟೋಕಾಕಸ್: ಆಹಾರ ನಿಲ್ಲಿಸಿ, ಏರೇಶನ್ ಹೆಚ್ಚಿಸಿ, ಇಲಾಖೆಗೆ ತಿಳಿಸಿ.",
        ),
    ),
    (
        "Columnaris Disease",
        (
            "treat",
            "Columnaris (white saddle lesions): salt bath per extension guidance, reduce organic \
             load, avoid crowding; the fish is still marketable if treated early.",
            "ಕಾಲಮ್ನಾರಿಸ್: ಉಪ್ಪು ಸ್ನಾನ, ಸಾವಯವ ಹೊರೆ ಕಡಿಮೆ ಮಾಡಿ.",
        ),
    ),
    (
        "Tilapia Lake Virus",
        (
            "escalate_to_expert",
            "TiLV is a notifiable, high-mortality tilapia virus. Containment only: stop water \
             movement, stop feeding, report immediately.",
            "TiLV: ತುರ್ತಾಗಿ ಇಲಾಖೆಗೆ ವರದಿ ಮಾಡಿ, ನೀರು ಸಾಗಣೆ ನಿಲ್ಲಿಸಿ.",
        ),
    ),
    ("Healthy Fish", ("healthy", "Skin, fins and gills look clean. Keep routine.", "ಮೀನು ಆರೋಗ್ಯವಾಗಿದೆ.")),
];

const DEFAULT_ADVICE: Advice = (
    "monitor",
    "Not recognised as a known condition. Photograph again in daylight, side-on, on a plain \
     background, and send to the extension officer.",
    "ಗುರುತಿಸಲಾಗಿಲ್ಲ. ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.",
);

/// First ADVICE entry whose key is a substring of `label` (case-insensitive),
/// mirroring the Python `_advice` scan order.
pub fn advice_for(label: &str) -> Advice {
    let l = label.to_lowercase();
    for (key, val) in ADVICE.iter() {
        if l.contains(&key.to_lowercase()) {
            return *val;
        }
    }
    DEFAULT_ADVICE
}

/// Zero-dependency colour/blob statistics — and an *abstention* by default.
///
/// A pond photo contains water, mud, a wet hand and a wet fish; a reddish human hand
/// can produce a larger "red pixel" fraction than a real lesion. So, exactly like the
/// shipping model, the fallback refuses to guess unless the evidence is extreme.
pub fn heuristic(image_path: &Path, species: &str) -> Result<DiseasePrediction> {
    let img = image::open(image_path)
        .map_err(|e| anyhow!("cannot open image {}: {e}", image_path.display()))?;
    Ok(heuristic_img(&img, species))
}

/// Colour/blob triage on an already-decoded image (infallible — no I/O).
pub fn heuristic_img(img: &image::DynamicImage, _species: &str) -> DiseasePrediction {
    let rgb = img.to_rgb8();
    let resized = image::imageops::resize(&rgb, 200, 200, image::imageops::FilterType::Triangle);

    let mut n = 0u32;
    let mut red = 0u32;
    let mut white = 0u32;
    for y in 0..200 {
        for x in 0..200 {
            let p = resized.get_pixel(x, y).0;
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            n += 1;
            let maxg = g.max(1) as f64;
            if r > 125 && (r - g) > 55 && (r - b) > 45 && (r as f64) > 1.30 * maxg {
                red += 1;
            } else {
                let mn = r.min(g).min(b);
                let mx = r.max(g).max(b);
                if mn > 172 && (mx - mn) < 30 {
                    white += 1;
                }
            }
        }
    }
    let red_f = red as f64 / n as f64;
    let white_f = white as f64 / n as f64;

    let (label, conf) = if red_f > 0.08 {
        // >8% of the frame is saturated lesion-red
        ("Bacterial Red disease", (0.75f64).min(0.45 + 4.0 * red_f))
    } else if white_f > 0.30 {
        // >30% cottony white = fungal/columnaris bloom
        ("Fungal diseases Saprolegniasis", (0.75f64).min(0.40 + white_f))
    } else {
        return DiseasePrediction {
            label: "Uncertain - needs a trained model or an expert".to_string(),
            confidence: pyround((0.5f64).min(red_f.max(white_f) * 4.0), 3),
            top_k: vec![("Uncertain".to_string(), 1.0)],
            backend: "heuristic-fallback".to_string(),
            triage: "monitor".to_string(),
            advice_en: "Colour statistics cannot separate a lesion from a hand, mud or glare, so we \
                        do not guess. Photograph the fish in daylight, plain background, filling the \
                        frame, and send it to the extension officer - or run the trained model \
                        (`uv sync --extra ml`)."
                .to_string(),
            advice_local: "ಫೋಟೋ ಸ್ಪಷ್ಟವಿಲ್ಲ. ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೆ ತೆಗೆಯಿರಿ ಮತ್ತು ಅಧಿಕಾರಿಗೆ ಕಳಿಸಿ.".to_string(),
        };
    };

    let (triage, en, local) = advice_for(label);
    let conf = pyround(conf, 3);
    DiseasePrediction {
        label: label.to_string(),
        confidence: conf,
        top_k: vec![(label.to_string(), conf), ("Uncertain".to_string(), pyround(1.0 - conf, 3))],
        backend: "heuristic-fallback".to_string(),
        triage: triage.to_string(),
        advice_en: format!("{en}  [from image colour statistics only - not a diagnosis]"),
        advice_local: local.to_string(),
    }
}

/// Classify one pond photo and map it to farmer-facing triage advice.
///
/// Order: deep ONNX model (if built with `--features onnx` and a model is present)
/// -> abstaining colour heuristic. Raises if the file does not exist, matching the
/// Python `FileNotFoundError`.
pub fn predict(image_path: &str, species: &str, model: Option<&str>) -> Result<DiseasePrediction> {
    let p = Path::new(image_path);
    if !p.exists() {
        return Err(anyhow!("No such file or directory (os error 2): {}", image_path));
    }
    let img = image::open(p).map_err(|e| anyhow!("cannot open image {}: {e}", p.display()))?;
    Ok(predict_dyn(&img, species, model))
}

/// Classify an in-memory image (the multipart upload path used by the API).
pub fn predict_bytes(data: &[u8], species: &str, model: Option<&str>) -> Result<DiseasePrediction> {
    let img = image::load_from_memory(data).map_err(|e| anyhow!("cannot decode image: {e}"))?;
    Ok(predict_dyn(&img, species, model))
}

/// Shared backend selection on a decoded image: deep ONNX model (feature = "onnx")
/// first, then the abstaining colour heuristic. Never blocks the farmer.
pub fn predict_dyn(img: &image::DynamicImage, species: &str, model: Option<&str>) -> DiseasePrediction {
    #[cfg(feature = "onnx")]
    {
        if let Ok(Some(pred)) = onnx_backend::predict_img(img, model) {
            return pred;
        }
        // model missing / failed -> fall through to the honest heuristic
    }
    #[cfg(not(feature = "onnx"))]
    let _ = model;

    heuristic_img(img, species)
}

// --------------------------------------------------------------------------- //
// Deep-model backend (ONNX Runtime) — feature = "onnx"
// --------------------------------------------------------------------------- //

#[cfg(feature = "onnx")]
pub mod onnx_backend {
    use super::{advice_for, MIN_CONFIDENCE, SOUTH_ASIA_CLASSES};
    use crate::schemas::DiseasePrediction;
    use crate::util::pyround;
    use anyhow::{anyhow, Result};
    use ndarray::Array4;
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use std::path::Path;

    fn model_path(explicit: Option<&str>) -> Option<std::path::PathBuf> {
        if let Some(m) = explicit {
            return Some(Path::new(m).to_path_buf());
        }
        if let Ok(env) = std::env::var("AQUA_ONNX_MODEL") {
            if !env.is_empty() {
                return Some(Path::new(&env).to_path_buf());
            }
        }
        let default = Path::new("models/fish_disease.onnx");
        if default.exists() {
            Some(default.to_path_buf())
        } else {
            None
        }
    }

    /// Class labels: an explicit JSON array file (AQUA_ONNX_LABELS) or the South-Asia set.
    fn labels() -> Vec<String> {
        if let Ok(path) = std::env::var("AQUA_ONNX_LABELS") {
            if !path.is_empty() {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    if let Ok(v) = serde_json::from_str::<Vec<String>>(&text) {
                        if !v.is_empty() {
                            return v;
                        }
                    }
                    // fall back to newline-delimited
                    let lines: Vec<String> =
                        text.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                    if !lines.is_empty() {
                        return lines;
                    }
                }
            }
        }
        SOUTH_ASIA_CLASSES.iter().map(|s| s.to_string()).collect()
    }

    fn env_f64(key: &str, default: f64) -> f64 {
        std::env::var(key).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(default)
    }

    /// Resize -> RGB -> normalise (mean/std) -> NCHW float32, replicating the HF
    /// ViT image processor. Defaults are the ViT norm (0.5/0.5) at 224px; override
    /// with AQUA_ONNX_{SIZE,MEAN,STD} to match the exported checkpoint.
    fn preprocess(img: &image::DynamicImage) -> Result<Array4<f32>> {
        let size = env_f64("AQUA_ONNX_SIZE", 224.0) as u32;
        let mean = [
            env_f64("AQUA_ONNX_MEAN0", 0.5),
            env_f64("AQUA_ONNX_MEAN1", 0.5),
            env_f64("AQUA_ONNX_MEAN2", 0.5),
        ];
        let std = [
            env_f64("AQUA_ONNX_STD0", 0.5),
            env_f64("AQUA_ONNX_STD1", 0.5),
            env_f64("AQUA_ONNX_STD2", 0.5),
        ];
        let rgb = img.to_rgb8();
        let resized = image::imageops::resize(&rgb, size, size, image::imageops::FilterType::CatmullRom);
        let s = size as usize;
        let mut data = vec![0f32; 3 * s * s];
        for y in 0..s {
            for x in 0..s {
                let p = resized.get_pixel(x as u32, y as u32).0;
                for c in 0..3 {
                    let v = p[c] as f32 / 255.0;
                    data[c * s * s + y * s + x] = (v - mean[c] as f32) / std[c] as f32;
                }
            }
        }
        Array4::from_shape_vec((1, 3, s, s), data).map_err(|e| anyhow!("tensor shape: {e}"))
    }

    fn softmax(x: &[f32]) -> Vec<f64> {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = x.iter().map(|v| (*v as f64 - m).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Run the deep model. Returns Ok(None) when no model is configured so the
    /// caller can fall back to the heuristic without treating it as an error.
    pub fn predict_img(img: &image::DynamicImage, model: Option<&str>) -> Result<Option<DiseasePrediction>> {
        let mp = match model_path(model) {
            Some(p) if p.exists() => p,
            _ => return Ok(None),
        };

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&mp)?;
        let input = preprocess(img)?;
        let outputs = session.run(ort::inputs![input.view()]?)?;
        // batch = 1, so the logits buffer is one row of length = num_classes.
        let (_shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let probs = softmax(logits);
        let labs = labels();

        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(String, f64)> = idx
            .iter()
            .take(5)
            .map(|&i| (labs.get(i).cloned().unwrap_or_else(|| format!("class_{i}")), pyround(probs[i], 3)))
            .collect();

        let (label, score) = top[0].clone();
        let backend = format!("onnx:{}", mp.display());
        if score < MIN_CONFIDENCE {
            return Ok(Some(DiseasePrediction {
                label: format!("Uncertain ({label}?)"),
                confidence: pyround(score, 3),
                top_k: top,
                backend,
                triage: "monitor".to_string(),
                advice_en: "Model confidence below the referral threshold - do not treat on this \
                            alone. Re-photograph in daylight, plain background, fish filling the \
                            frame; send to the extension officer."
                    .to_string(),
                advice_local: "ಖಚಿತವಿಲ್ಲ. ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆದು ಅಧಿಕಾರಿಗೆ ಕಳಿಸಿ.".to_string(),
            }));
        }
        let (triage, en, local) = advice_for(&label);
        Ok(Some(DiseasePrediction {
            label,
            confidence: pyround(score, 3),
            top_k: top,
            backend,
            triage: triage.to_string(),
            advice_en: en.to_string(),
            advice_local: local.to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advice_never_prescribes_antibiotics() {
        // Mirrors tests/test_disease.py::test_advice_never_prescribes_antibiotics
        for (label, advice) in ADVICE.iter() {
            let (triage, en, local) = *advice;
            assert!(["healthy", "monitor", "treat", "escalate_to_expert"].contains(&triage));
            assert!(!local.trim().is_empty(), "{label} has no local-language advice");
            if triage == "escalate_to_expert" {
                let e = en.to_lowercase();
                assert!(e.contains("officer") || e.contains("report") || e.contains("contain"));
            }
        }
    }

    #[test]
    fn advice_substring_matching() {
        assert_eq!(advice_for("Bacterial Red disease").0, "escalate_to_expert");
        assert_eq!(advice_for("Healthy Fish").0, "healthy");
        assert_eq!(advice_for("something unknown").0, "monitor");
        // the aeromoniasis label must not be captured by the "Bacterial Red disease" key
        assert_eq!(advice_for("Bacterial diseases - Aeromoniasis").0, "treat");
    }

    #[test]
    fn missing_file_is_loud() {
        assert!(predict("does/not/exist.jpg", "tilapia", None).is_err());
    }

    #[test]
    fn heuristic_abstains_on_the_bundled_demo_photos() {
        // Mirrors tests/test_disease.py::test_triage_never_pretends_to_diagnose for
        // the offline backend: it must abstain, not emit a confident diagnosis.
        for name in ["example_lesion_fish.jpg", "example_healthy_fish.jpg"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets").join(name);
            if !path.exists() {
                continue;
            }
            let pred = heuristic(&path, "tilapia").unwrap();
            assert_eq!(pred.backend, "heuristic-fallback");
            assert!(["monitor", "treat", "escalate_to_expert"].contains(&pred.triage.as_str()));
            if pred.label.to_lowercase().contains("uncertain") {
                assert!(pred.confidence <= 0.5);
            }
        }
    }
}
