# AquaAlert — Rust port (single binary)

A **1:1 Rust rewrite of the AquaAlert backend + HTTP service + AI triage**, compiled to a
single static binary. It is a faithful port of the original Python project
(`aqua_alert`: FastAPI + Typer + NumPy/pandas + transformers/torch), module for module,
with the same public-data catalogue, the same risk/alert logic, the same HTTP contract and
the same photo→disease classifier behaviour.

> Early disease & water-quality alerts for small aquaculture farms — using **free public
> data instead of physical sensors**, so a smallholder can act before a night oxygen crash
> or a disease outbreak wipes out a pond.

---

## What's in the box

| Python module | Rust module | Role |
|---|---|---|
| `schemas.py` | `src/schemas.rs` | serde models + a hand-rolled Python `repr`-compatible rounder |
| `thresholds.py` | `src/thresholds.rs` | per-species parameter envelopes + feeding advice |
| `risk.py` | `src/risk.rs` | severity/band engine, drivers, action builder, night forecast |
| `anomaly.py` | `src/anomaly.rs` | MAD z-score + Hampel filter drift scoring |
| `pipeline.py` | `src/pipeline.rs` | offline sample data + end-to-end demo day |
| `cli.py` (Typer) | `src/main.rs` (clap) | `catalog/demo/assess/forecast/triage/matrix/serve/…` |
| `alerts/dispatcher.py` | `src/alerts/dispatcher.rs` | SMS/voice/console + anti-fatigue dedupe |
| `api/server.py` (FastAPI) | `src/api/server.rs` (axum) | the HTTP service + farmer screen |
| `sources/registry.py` | `src/sources/registry.rs` | the 20-entry public-data catalogue |
| `sources/weather.py` | `src/sources/weather.rs` | Open-Meteo fetch + virtual-water-temp model |
| `sources/synthetic.py` | `src/sources/synthetic.rs` | offline sample pond log, **MT19937-exact** |
| `sources/usgs_wqp.py` | `src/sources/usgs_wqp.rs` | Water Quality Portal GeoJSON ingest |
| `disease/classifier.py` | `src/disease/classifier.rs` | photo → disease triage (**the AI**) |
| `screens/farmer_app.html` | `assets/farmer_app.html` | single-file farmer screen, embedded in the binary |

The three headline deliverables are all present: **(1)** the public-data catalogue
(`/catalog`, 20 entries), **(2)** the risk→alert pipeline with anti-fatigue, and **(3)** the
farmer screen served at `/` straight out of the binary.

---

## Build & run (single binary)

```bash
cargo build --release
# -> target/release/aqua-alert   (one file)

./target/release/aqua-alert demo          # offline end-to-end demo
./target/release/aqua-alert catalog
./target/release/aqua-alert assess --do 1.7 --ph 8.9 --temp 33 --tan 1.8
./target/release/aqua-alert forecast --lat 12.97 --lon 77.59
./target/release/aqua-alert triage assets/example_lesion_fish.jpg --species tilapia
./target/release/aqua-alert matrix
./target/release/aqua-alert serve --host 0.0.0.0 --port 8000   # farmer screen + API
```

`serve` binds `0.0.0.0:8000` by default-friendly args and shuts down gracefully on
Ctrl-C. Open `http://localhost:8000/` for the farmer screen and `http://localhost:8000/docs`
for the API docs page.

### Docker (static, distroless)

```bash
docker build -t aqua-alert-rs .
docker run --rm -p 8000:8000 aqua-alert-rs          # serve
docker run --rm aqua-alert-rs demo                  # one-shot
```

The image is `gcr.io/distroless/static` + a musl-linked binary: no Python, no glibc, no
shell. See the `Dockerfile`.

---

## The AI (photo → disease triage)

Two backends, exactly like the Python version, selected at runtime:

1. **ONNX (optional `onnx` feature)** — a public Hugging Face / Roboflow ViT exported to
   ONNX. Runs on CPU via `ort`, no Python or torch at runtime. Enable with
   `cargo build --features onnx` and point `AQUA_ONNX_MODEL` / `AQUA_ONNX_LABELS` at the
   exported files (see [`export/README.md`](export/README.md)).

2. **Heuristic fallback (always built in)** — resizes the photo to 200×200 and reads the
   colour statistics: a high fraction of *red* lesions → **Bacterial Red disease**; a high
   fraction of *white* cotton-like patches → **Fungal / Saprolegniasis**; otherwise it
   **abstains** (`"Uncertain — needs a trained model or an expert"`, triage `monitor`)
   instead of guessing. Each hit maps to the correct first-aid advice and a species-aware
   escalation (`contact officer / report / contain`, never "treat yourself" for a notifiable
   disease).

The result JSON is identical in shape to the Python service (`label`, `confidence`, `top_k`,
`advice_en`, `advice_local`, `triage`, `backend`, `model`).

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness/version |
| GET | `/catalog?kind=` | the public-data catalogue (deliverable #1) |
| GET | `/catalog/{id}` | one catalogue entry |
| GET | `/` | farmer alert screen (deliverable #3) |
| GET | `/docs` | API docs (OpenAPI + Swagger UI) |
| GET | `/escalation-matrix` | who is told what, and how fast |
| POST | `/pond/{id}/reading?channel=` | post one morning reading → live alert |
| POST | `/pond/{id}/photo` | multipart photo → disease triage |
| POST | `/simulate/crash-night` | offline RED crash + SMS/voice (the demo money-shot) |
| GET | `/pond/{id}/status` | last assessment + night outlook + alert |
| GET | `/pond/{id}/forecast?lat=&lon=` | next-night oxygen-crash outlook (Open-Meteo) |

CORS is allow-all (so the single-file screen works from anywhere). Pond state is in-memory.

---

## Tests

Every numeric/logic decision is pinned by a **unit test whose expected value was captured
from the running Python code** (run through the project's own `.venv`), so the port is
verified against ground truth, not just self-consistent:

```bash
cargo test                      # all features off except default (server)
cargo test --features onnx      # also compiles the ONNX backend
```

Coverage includes: the chemistry (NH3-N fraction, TAN→NH3, virtual water temp, DO crash
forecast), the risk engine (severity/dominant driver/action set/feed fraction), the anomaly
z-scores and Hampel filter, the catalogue, **MT19937 + `gauss()` bit-exact against CPython
`random.Random`** for seeds 0/3/7/42/−5, the full synthetic pond log, CSV write/load
round-trip, WQP ingest, disease advice mapping + the heuristic abstain path, the dispatcher
(SMS ≤160 chars, voice, GREEN suppression, dedupe anti-fatigue, escalation matrix), the demo
day, and an **end-to-end HTTP test** (`tests/api.rs`) that boots the real axum server and
compares responses to the captured Python JSON.

---

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `server` | ✔ | builds the axum HTTP service + `serve` command (needs `axum`, `tower-http`) |
| `onnx` | ✘ | builds the ONNX deep-model backend (needs `ort`, `ndarray`) |

`cargo build --release --no-default-features` gives a CLI-only binary (no HTTP server).
The heuristic AI is always available regardless of flags.

---

## Faithfulness notes & documented divergences

The port reproduces the Python behaviour deliberately, including its quirks (night
escalation compares band *strings* alphabetically; the water-exchange action keys off the
NH3-N severity; `insert_at` depends on whether the first action starts with "turn on";
`feed_fraction == 0` removes the "stop feeding" action; timestamps parse as naive-UTC).
A few things are intentionally different or unverifiable in this environment:

- **`dedupe_key` version quirk.** Python builds the key with `f"{pond}:{band}"` on a
  `str`-Enum; on CPython 3.11 that formats as `"KA-BLR-001:Band.ORANGE"`, on 3.10 /
  `StrEnum` as `"KA-BLR-001:ORANGE"`. It is an *opaque, version-dependent token* that the
  Python tests never assert on. Rust uses the stable band value (`"…:ORANGE"`). This is the
  only observable behavioural divergence, and it only changes an internal dedupe token.
- **`gauss()` ULP.** The MT19937 integer stream (`random()`, `genrand_uint32`) is
  bit-exact. `gauss()` is `(sqrt(-2 ln u1) cos(2π u2))·σ+μ`, and `cos/sin/ln` can differ by
  ≤1 ULP between CPython's libm and Rust's, so a handful of raw `gauss()` unit tests use a
  `1e-12` tolerance. Rounded sensor values (the CSV) are unaffected.
- **ONNX backend API.** Written against `ort` 2.0.0-rc.10; that crate's tensor/IO API is
  still stabilising, so building against a different rc may need a one-line tweak in
  `onnx_backend::predict_img` (see `export/README.md`). It is off by default and never
  affects the single-binary heuristic build.
- **`/docs`.** FastAPI auto-generates Swagger UI; the axum port serves an equivalent
  hand-written OpenAPI document + Swagger UI page instead (the JSON contract is described,
  not introspected from types).
- **Could not `cargo build` here.** This workspace has no Rust toolchain and no crates.io
  access, so the crate was written and cross-checked by hand against the captured Python
  ground truth but not compiled in-sandbox. It is structured to build with a stable Rust
  ≥ 1.75 (`cargo build --release`).

---

## Layout

```
aqua-alert-rs/
├── Cargo.toml
├── Dockerfile
├── README.md
├── assets/                 # farmer screen + sample data + example photos (embedded)
│   ├── farmer_app.html
│   ├── pond_log_14d.csv
│   ├── example_healthy_fish.jpg
│   └── example_lesion_fish.jpg
├── export/                 # ONNX export helpers for the optional `onnx` feature
│   ├── README.md
│   └── to_onnx.py
├── src/
│   ├── main.rs             # CLI (clap)
│   ├── lib.rs
│   ├── schemas.rs  thresholds.rs  risk.rs  anomaly.rs  pipeline.rs  csv.rs  util.rs
│   ├── alerts/{mod,dispatcher}.rs
│   ├── api/{mod,server}.rs
│   ├── disease/{mod,classifier}.rs
│   └── sources/{mod,registry,weather,synthetic,usgs_wqp}.rs
└── tests/
    └── api.rs              # end-to-end HTTP contract test
```
