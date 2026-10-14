//! CLI — the demo path a judge/officer can run in one line.
//!
//! 1:1 port of `src/aqua_alert/cli.py` (Typer -> clap):
//!
//!     aqua-alert catalog
//!     aqua-alert demo
//!     aqua-alert assess --do 1.7 --ph 8.9 --temp 33 --tan 1.8
//!     aqua-alert forecast --lat 12.97 --lon 77.59
//!     aqua-alert triage photo.jpg
//!     aqua-alert matrix
//!     aqua-alert serve            # farmer screen + API at :8000
//!
//! Everything compiles into the same single binary as the HTTP service.

use aqua_alert::alerts::dispatcher::{self, AlertDispatcher};
use aqua_alert::disease::classifier;
use aqua_alert::pipeline::{self, Forecast};
use aqua_alert::schemas::{Band, PondReading, RiskAssessment};
use aqua_alert::sources::{registry, synthetic, weather};
use aqua_alert::risk;
use chrono::{Duration, Local};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

fn band_color(b: Band) -> &'static str {
    match b {
        Band::Green => "\x1b[32m",
        Band::Yellow => "\x1b[33m",
        Band::Orange => "\x1b[38;5;208m",
        Band::Red => "\x1b[31m",
    }
}

/// Approximate Python's `{:.3g}` for the driver-value column (cosmetic CLI output).
fn fmt_val(v: f64) -> String {
    let s = format!("{v:.3}");
    let t = s.trim_end_matches('0').trim_end_matches('.');
    if t.is_empty() || t == "-" {
        "0".to_string()
    } else {
        t.to_string()
    }
}

#[derive(Parser)]
#[command(
    name = "aqua-alert",
    version,
    about = "Early disease & water-quality alerts for small farms"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print the public-data catalogue (deliverable #2).
    Catalog {
        /// filter: timeseries|images|weather|remote_sensing|thresholds
        #[arg(long)]
        kind: Option<String>,
    },
    /// Offline end-to-end demo on the bundled sample pond log.
    Demo {
        #[arg(long, default_value_t = 7)]
        seed: i64,
        #[arg(long, default_value = "console")]
        channel: String,
    },
    /// Assess one morning reading typed in by the farmer.
    Assess {
        /// dissolved oxygen mg/L
        #[arg(long = "do")]
        do_mgl: f64,
        /// pH
        #[arg(long)]
        ph: f64,
        /// water temp C
        #[arg(long)]
        temp: f64,
        /// total ammonia mg/L (test strip)
        #[arg(long)]
        tan: Option<f64>,
        /// NTU (visual/disc estimate)
        #[arg(long)]
        turbidity: Option<f64>,
        #[arg(long, default_value = "rohu")]
        species: String,
        #[arg(long, default_value = "KA-BLR-001")]
        pond: String,
    },
    /// Next nights' oxygen-crash risk from free Open-Meteo data (no DO probe).
    Forecast {
        #[arg(long, default_value_t = 12.9716)]
        lat: f64,
        #[arg(long, default_value_t = 77.5946)]
        lon: f64,
        #[arg(long, default_value_t = 3)]
        days: u32,
    },
    /// Photo -> disease triage (uses the public datasets in the catalogue).
    Triage {
        image: PathBuf,
        #[arg(long, default_value = "rohu")]
        species: String,
        /// explicit ONNX model path (feature = "onnx"); else AQUA_ONNX_MODEL/models/fish_disease.onnx
        #[arg(long)]
        model: Option<String>,
    },
    /// (Re)generate the bundled offline sample pond log.
    SampleData {
        #[arg(long, default_value_t = 14)]
        days: usize,
        #[arg(long, default_value = "data/samples/pond_log_14d.csv")]
        out: PathBuf,
    },
    /// Who gets told what, and how fast.
    Matrix,
    /// Run the API + farmer alert screen.
    #[cfg(feature = "server")]
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// accepted for CLI parity; hot-reload is a dev-server concern, not the compiled binary
        #[arg(long)]
        reload: bool,
    },
    /// Replay an exported reading log (JSONL of PondReading) to see the alert stream.
    Replay {
        jsonl: PathBuf,
        #[arg(long, default_value = "rohu")]
        species: String,
        #[arg(long, default_value = "console")]
        channel: String,
    },
}

fn print_assessment(a: &RiskAssessment) {
    let c = band_color(a.band);
    println!("{c}{BOLD}{} - {}{RESET}", a.band.as_str(), a.headline_en);
    println!("{DIM}{} | score {:.2} | {}{RESET}", a.headline_local, a.score, a.species);
    println!();
    for (i, x) in a.actions.iter().enumerate() {
        println!("{}. {}", i + 1, x);
    }
    if !a.drivers.is_empty() {
        println!();
        println!("{DIM}{:<16}{:>14}  {:<9}{}{RESET}", "parameter", "value", "status", "severity");
        for d in &a.drivers {
            let val = match d.value {
                Some(v) => format!("{} {}", fmt_val(v), d.unit),
                None => format!("-- {}", d.unit),
            };
            println!("{:<16}{:>14}  {:<9}{:.2}", d.parameter, val, d.status, d.severity);
        }
    }
    if !a.missing.is_empty() {
        println!("{DIM}missing parameters: {}{RESET}", a.missing.join(", "));
    }
    for n in &a.notes {
        println!("{DIM}note: {n}{RESET}");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Catalog { kind } => {
            println!("{BOLD}Public data replacing physical sensors{RESET}");
            println!("{DIM}{:<28}{:<16}{:<44}{}{RESET}", "id", "kind", "replaces sensor", "license");
            let mut shown = 0;
            for e in registry::CATALOG.iter() {
                if let Some(k) = &kind {
                    if e.kind.as_str() != k.as_str() {
                        continue;
                    }
                }
                println!("{:<28}{:<16}{:<44}{}", e.id, e.kind.as_str(), e.replaces_sensor, e.license);
                shown += 1;
            }
            println!("{DIM}{shown} entries ({} total) - full detail: GET /catalog{RESET}", registry::CATALOG.len());
        }

        Commands::Demo { seed: _seed, channel } => {
            let mut disp = AlertDispatcher::new();
            let day = pipeline::demo_day("KA-BLR-001", "tilapia", Some(&mut disp));
            print_assessment(&day.assessment);
            if let Some(a) = day.alerts.first() {
                if a.delivered {
                    println!("\n{BOLD}\x1b[31m── SMS sent to farmer ──{RESET}");
                    println!("{}", a.message);
                }
            }
            let _ = channel;
        }

        Commands::Assess { do_mgl, ph, temp, tan, turbidity, species, pond } => {
            let r = PondReading {
                pond_id: pond,
                do_mgl: Some(do_mgl),
                ph: Some(ph),
                water_temp_c: Some(temp),
                ammonia_total_mgl: tan,
                turbidity_ntu: turbidity,
                source: "manual kit".into(),
                ..Default::default()
            };
            let a = risk::assess(&r, &species, None);
            print_assessment(&a);
        }

        Commands::Forecast { lat, lon, days } => {
            let today = Local::now().date_naive();
            let end = today + Duration::days(days as i64 - 1);
            let rows = match weather::fetch_daily(lat, lon, today, end, 20.0).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("\x1b[31mweather fetch failed:{RESET} {e}");
                    eprintln!("Offline? The rest of the system still works.");
                    std::process::exit(1);
                }
            };
            println!("{BOLD}Night oxygen outlook - {lat},{lon}{RESET}");
            println!("{DIM}{:<12}{:<8}{:<60}{:>10}{RESET}", "date", "band", "why", "vwt C");
            for d in &rows {
                let (band, reason) = risk::do_crash_forecast(d.t_max, d.t_min, d.wind_ms, d.precip_mm, d.cloud_pct, None);
                let c = band_color(band);
                println!(
                    "{:<12}{c}{}{RESET}  {DIM}rain {:.1} mm{RESET}",
                    d.day.format("%Y-%m-%d"),
                    band.as_str(),
                    d.precip_mm
                );
                println!("    {DIM}{reason}{RESET}");
                println!("    {DIM}virtual water temp {} C{RESET}", aqua_alert::util::py_str_float(d.virtual_water_temp()));
            }
            println!("{DIM}Weather data by Open-Meteo.com (CC BY 4.0){RESET}");
        }

        Commands::Triage { image, species, model } => {
            if !image.exists() {
                eprintln!("\x1b[31mno such file:{RESET} {}", image.display());
                std::process::exit(1);
            }
            let p = classifier::predict(image.to_string_lossy().as_ref(), &species, model.as_deref())?;
            println!("{BOLD}── Disease triage ──{RESET}");
            println!("{BOLD}{}{RESET}  confidence {:.2}  ({})", p.label, p.confidence, p.backend);
            println!("triage: {BOLD}{}{RESET}", p.triage);
            println!();
            println!("{}", p.advice_en);
            println!();
            println!("{}", p.advice_local);
            println!();
            let top: Vec<String> = p.top_k.iter().map(|(k, v)| format!("{k} {v:.2}")).collect();
            println!("{DIM}top-{}: {}{RESET}", top.len(), top.join(", "));
        }

        Commands::SampleData { days, out } => {
            synthetic::write_csv(&out, days)?;
            let n = synthetic::load_csv_path(&out).len();
            println!("wrote {BOLD}{}{RESET} ({n} rows)", out.display());
        }

        Commands::Matrix => {
            for (band, rule) in dispatcher::escalation_matrix() {
                println!("{BOLD}── {band} ──{RESET}");
                println!("{rule}");
                println!();
            }
        }

        #[cfg(feature = "server")]
        Commands::Serve { host, port, reload } => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
                )
                .init();
            if reload {
                eprintln!("{DIM}note: --reload is a dev-server concern; the compiled binary runs as-is.{RESET}");
            }
            aqua_alert::api::server::serve(&host, port).await?;
        }

        Commands::Replay { jsonl, species, channel } => {
            let text = std::fs::read_to_string(&jsonl)?;
            let mut disp = AlertDispatcher::new();
            let mut n = 0;
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let r: PondReading = serde_json::from_str(line)?;
                pipeline::assess_internal(r, &species, Some(&mut disp), &channel, Forecast::Disabled);
                n += 1;
            }
            println!("replayed {n} readings; {} alerts dispatched", disp.sent.len());
        }
    }
    Ok(())
}
