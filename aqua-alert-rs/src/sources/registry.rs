//! THE DATA CATALOGUE (deliverable #2).
//!
//! 1:1 port of `src/aqua_alert/sources/registry.py`. Every entry is a *public,
//! existing* dataset or API that substitutes for a physical sensor the small farm
//! cannot buy. `replaces_sensor` says which sensor it stands in for;
//! `role_in_system` says what our code does with it. Verified 2026-10-07.

use crate::schemas::{CatalogEntry, SourceKind};
use once_cell::sync::Lazy;

#[allow(clippy::too_many_arguments)]
fn entry(
    id: &str,
    name: &str,
    kind: SourceKind,
    url: &str,
    license: &str,
    what_it_provides: &str,
    replaces_sensor: &str,
    role_in_system: &str,
) -> CatalogEntry {
    CatalogEntry {
        id: id.to_string(),
        name: name.to_string(),
        kind,
        url: url.to_string(),
        license: license.to_string(),
        what_it_provides: what_it_provides.to_string(),
        replaces_sensor: replaces_sensor.to_string(),
        role_in_system: role_in_system.to_string(),
        access_cost: "free".to_string(),
    }
}

pub static CATALOG: Lazy<Vec<CatalogEntry>> = Lazy::new(|| {
    vec![
        // ------------- A. Water-quality time series (replaces pH/DO/ammonia probes) --
        entry(
            "kaggle-aquaponics-nigeria",
            "Sensor Based Aquaponics Fish Pond Datasets (U. Nigeria Nsukka)",
            SourceKind::Timeseries,
            "https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets",
            "CC BY 4.0 (see Kaggle page)",
            "5-second-interval temperature, turbidity, DO, pH, ammonia, nitrate from 12 catfish ponds (Jun-Oct 2021), with fish population/length/weight",
            "pH probe, DO probe, ammonia sensor, turbidity sensor",
            "Primary training + calibration set for the absolute thresholds and for the DO-crash forecaster; gives labelled 'normal pond' baselines.",
        ),
        entry(
            "kaggle-pondsdata-iot",
            "Pondsdata - water quality data for aquaculture (IoT, 74,759 records)",
            SourceKind::Timeseries,
            "https://www.kaggle.com/datasets/apgopi/pondsdata",
            "See Kaggle page",
            "pH, DO, temperature, turbidity, ammonia, nitrate, manganese, Feb 2022-Jan 2023",
            "Full sensor suite",
            "Seasonal drift baselines (monsoon vs summer response of the same pond).",
        ),
        entry(
            "mendeley-aquaculture-monteria",
            "Environmental Parameters in Aquaculture (Monteria, Colombia)",
            SourceKind::Timeseries,
            "https://data.mendeley.com/datasets/8s73jfvgr5/1",
            "CC BY 4.0",
            "Temperature, DO, pH, turbidity with date/hour from tropical ponds, 2024",
            "pH/DO/turbidity probes",
            "DOI-citable calibration set; tropical pond diurnal curves for the night-oxygen model.",
        ),
        entry(
            "usgs-wqp",
            "Water Quality Portal (USGS + US EPA + 400 agencies)",
            SourceKind::Timeseries,
            "https://www.waterqualitydata.us/data/Result/search",
            "Public domain (US federal data)",
            "REST/CSV discrete water-chemistry results (parameter codes 00300 DO, 00400 pH, 00610 ammonia, 63676 turbidity, 32209 chlorophyll-a)",
            "Lab bench for ammonia/nitrite/nitrate/chl-a",
            "Live ingestion of a real government API (code path exercised in src/aqua_alert/sources/usgs_wqp.py); also the source of the 'this is what your pond should look like' reference distributions.",
        ),
        entry(
            "cpcb-nwmp-opencity",
            "CPCB National Water Monitoring Programme - lakes/ponds/tanks/wetlands (2017-2022) via OpenCity",
            SourceKind::Timeseries,
            "https://data.opencity.in/dataset/water-quality-data-of-lakes-and-ponds-in-india",
            "Public Domain",
            "Indian inland water body data: pH, DO, BOD, nitrate, faecal coliform, temperature, per year CSVs/PDFs sourced from CPCB",
            "Regional baseline for Indian surface water (not just US/EU)",
            "Locality prior: what 'normal' looks like in Karnataka standing water, used to sanity-check an alert before it reaches the farmer.",
        ),
        entry(
            "data-gov-in-aqi-cpcb",
            "data.gov.in Open Government Data (CPCB) API",
            SourceKind::Timeseries,
            "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
            "Government Open Data License - India",
            "Free-tier JSON API (needs a free api-key) streaming CPCB environmental station feeds, incl. water-quality resources",
            "Not a sensor replacement - a free national data channel",
            "Optional Indian deployment channel; keeps the stack on Indian public infrastructure for a government-sponsored rollout.",
        ),
        // ------------- B. Remote sensing (replaces chlorophyll/turbidity probes) -------
        entry(
            "sentinel2-ndci",
            "Copernicus Sentinel-2 MSI (Level-2A) + NDCI algorithm",
            SourceKind::RemoteSensing,
            "https://dataspace.copernicus.eu/ (Copernicus Data Space Ecosystem)",
            "Copernicus open data (CC BY-SA 3.0 IGO / free, open)",
            "10 m x 5-day multi-spectral reflectance; NDCI = (B5-B4)/(B5+B4) is a validated chlorophyll-a proxy for small water bodies (adj. R^2 up to 0.84 in small reservoirs)",
            "Chlorophyll-a / algal-bloom probe (INR 40k+), and spot turbidity checks",
            "Weekly pond-scale 'greenness' score per farm polygon -> bloom-risk input; validated against WQP chlorophyll-a (R^2 > 0.5 nationwide).",
        ),
        entry(
            "nasa-pace-sst",
            "NASA PACE / MODIS-Aqua chlorophyll-a and SST products",
            SourceKind::RemoteSensing,
            "https://oceancolor.gsfc.nasa.gov/",
            "Public domain (NASA)",
            "Global chlorophyll-a and sea-surface temperature (MODIS Jul 2002-Feb 2022 as used in peer-reviewed aquaculture-impact studies)",
            "Regional water-temperature logger + bloom monitor",
            "Fallback for farms whose pond is below Sentinel-2's useful size, and long-run climate context for risk scoring.",
        ),
        // ------------- C. Weather / hydrology (replaces the oxygen & rain sensors) -----
        entry(
            "open-meteo",
            "Open-Meteo Historical Weather Archive + Forecast API (ERA5)",
            SourceKind::Weather,
            "https://api.open-meteo.com/v1/forecast",
            "CC BY 4.0; free for non-commercial use, no API key, ~10,000 calls/day",
            "Hourly/daily air temperature, humidity, wind, rainfall, cloud cover, solar radiation, soil moisture from 1940 to 16-day forecast, global",
            "Rain gauge, air thermometer, anemometer, solar radiometer",
            "Drives the pre-dawn DO-crash forecaster and the virtual water-temperature model (risk.do_crash_forecast, sources/weather.py). This is the single highest-leverage sensorless component.",
        ),
        entry(
            "noaa-gfs-ecmwf",
            "NOAA GFS / ECMWF IFS forecasts (via Open-Meteo multi-model layer)",
            SourceKind::Weather,
            "https://open-meteo.com/en/docs",
            "US Government public domain / ECMWF open data",
            "16-day ensemble-ish forecast horizons from 30+ national weather models",
            "Nothing physical - extends lead time",
            "Lets the app warn 'Thursday night is dangerous' 2-4 days ahead, which is when a farmer can still hire an aerator or pump.",
        ),
        // ------------- D. Disease / image data (replaces the microscope + vet) ---------
        entry(
            "kaggle-freshwater-south-asia",
            "Freshwater Fish Disease Aquaculture in South Asia (7 classes x 250 imgs)",
            SourceKind::Images,
            "https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia",
            "CC0: Public Domain",
            "Bacterial (Aeromoniasis, Gill disease, Red disease), Fungal (Saprolegniasis), Parasitic, Viral (White Tail Disease), Healthy - collected partly from a farm in Odisha, India; 2,444 images in v4",
            "Field vet visit / wet-lab examination",
            "Primary classifier training set for the photo triage screen (disease/classifier.py); CC0 license means it can be redistributed to cooperatives.",
        ),
        entry(
            "kaggle-nile-tilapia",
            "Enhancing Disease Detection in Nile Tilapia (NTD-1)",
            SourceKind::Images,
            "https://www.kaggle.com/datasets/engineeringubu/enhancing-disease-detection-in-nile-tilapia",
            "See Kaggle page (open access)",
            "1,043 annotated tilapia images: Streptococcosis (327), Columnaris (362), Tilapia Lake Virus (354); bounding boxes + lesion masks (LabelMe)",
            "Pathology lab for the three diseases that cause sudden tilapia kills",
            "Second training/specialisation set; lesion masks let the model highlight *where* the sore is and how big it is, so the farmer can see progress after treatment.",
        ),
        entry(
            "hf-panda992-vit",
            "Pretrained ViT-base fish-disease classifier (panda992/fish_disease_datasets)",
            SourceKind::Images,
            "https://huggingface.co/panda992/fish_disease_datasets",
            "Apache-2.0",
            "google/vit-base-patch16-224-in21k fine-tuned on the 7-class South-Asia fish-disease set (Bacterial Red disease, Aeromoniasis, Gill disease, Saprolegniasis, Parasitic, White Tail Disease, Healthy)",
            "Immediate disease triage with no training cost; the vet's first look",
            "Default checkpoint of the photo triage screen (disease/classifier.py). Verified on our demo photos: lesion -> 'Bacterial Red disease' 0.97; healthy fish -> 'Healthy Fish' 0.98.",
        ),
        entry(
            "hf-allfahad-eus",
            "Secondary checkpoint: EUS / gill-rot / lice classifier (allfahad/fish-disease-v1-original-20260608)",
            SourceKind::Images,
            "https://huggingface.co/allfahad/fish-disease-v1-original-20260608",
            "Not stated on the model card - verify before commercial use",
            "Classes incl. Epizootic Ulcerative Syndrome, Bacterial gill rot, Lice, Healthy Fish (Bangladesh pond conditions)",
            "Second opinion on the EUS-type ulcers common in South Asian ponds",
            "Fallback/secondary checkpoint in disease/classifier.py; useful where EUS rather than 'red disease' is the local vocabulary.",
        ),
        entry(
            "roboflow-fish-disease",
            "Roboflow 'Fish diseases detection and classification using YOLOv8' pond dataset",
            SourceKind::Images,
            "https://universe.roboflow.com/ (search: fish disease YOLOv8)",
            "See workspace license (CC BY 4.0 typical)",
            "Object-detection annotations for ulcer/EUS lesions in real pond photos",
            "Field observation by an extension officer",
            "Detection (not just classification) head: counts and localises lesions. Used to fine-tune a small YOLO-nano model for INR ~0 phone inference.",
        ),
        entry(
            "f4k",
            "Fish4Knowledge (F4K) underwater video/image corpus",
            SourceKind::Images,
            "https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/",
            "Academic, open for research use",
            "27,370 verified fish images from 5,824 video clips, 23 species, 9 fixed cameras at 3 angles, pre-split 16,430 train / 10,940 test",
            "Manual count + behaviour observation (feeding response, crowding, surface gasping)",
            "Behaviour model: fish crowding at the surface / listlessness is a disease and hypoxia signal that appears *before* lesions. Also the species-ID stage so the risk engine picks the right thresholds.",
        ),
        // ------------- E. Threshold / advisory knowledge (the 'expert in a box') -------
        entry(
            "fao-x6709e",
            "FAO Training Series - Improving Pond Water Quality (x6709e)",
            SourceKind::Thresholds,
            "https://www.fao.org/fishery/docs/CDrom/FAO_Training/FAO_Training/General/x6709e/x6709e02.htm",
            "FAO open access",
            "Species DO minima and % saturation rules: trout 5, common carp 3, tilapia 2 mg/L; saturation-vs-temperature tables",
            "Not data - the decision rules that turn a reading into a verdict",
            "Encoded in thresholds.py as the ORANGE/RED floors.",
        ),
        entry(
            "fao-ac175e18",
            "FAO - Site Selection for Aquaculture: chemical features of water (AC175E18)",
            SourceKind::Thresholds,
            "https://www.fao.org/4/ac175e/AC175E18.htm",
            "FAO open access",
            "Ammonia chemistry: NH3 is 300-400x more toxic than NH4+, with Emerson (1975)/EIFAC (1986) % un-ionized NH3 tables by pH and temperature",
            "Ammonia photometer - we convert a cheap TAN strip reading into toxic-NH3 using pH + temperature",
            "thresholds.unionized_ammonia_mgl(); the single biggest 'free sensor' win in this design.",
        ),
        entry(
            "extension-srac-table",
            "eXtension / SRAC 'Water Quality in Aquaculture' species table",
            SourceKind::Thresholds,
            "https://freshwater-aquaculture.extension.org/water-quality-in-aquaculture/",
            "US Cooperative Extension (public)",
            "Catfish/carp and tilapia windows: temp 65-80 F, DO 3-10 mg/L, pH 6-8, alkalinity 50-250, ammonia % 0-0.03, nitrite 0-0.6 mg/L",
            "Extension officer's memory",
            "Cross-check on the envelope values (kept in the same units for tests).",
        ),
        entry(
            "tilapia-review-pmc9640449",
            "Effect of environmental factors on Nile tilapia growth (review, PMC9640449)",
            SourceKind::Thresholds,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC9640449/",
            "Open access (CC BY)",
            "DO 5-7 mg/L optimum, growth penalty <3.5, mortality <1; prolonged NH3 0.2 mg/L starts deaths, appetite loss at 0.08 mg/L",
            "Aquaculture specialist consult",
            "Tilapia profile + the stop-feed rule.",
        ),
    ]
});

pub fn by_id(entry_id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|c| c.id == entry_id)
}

/// `model_dump(mode="json")` for every entry — the `/catalog` payload.
pub fn to_json() -> Vec<serde_json::Value> {
    CATALOG.iter().map(|c| serde_json::to_value(c).unwrap_or_default()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn catalogue_is_complete_and_cited() {
        assert!(CATALOG.len() >= 12);
        assert_eq!(CATALOG.len(), 20); // exact parity with the Python catalogue
        let ids: Vec<&str> = CATALOG.iter().map(|e| e.id.as_str()).collect();
        let unique: HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "catalogue ids must be unique");
        for e in CATALOG.iter() {
            assert!(e.url.starts_with("http"));
            assert!(!e.license.is_empty(), "{} is missing a license", e.id);
            assert!(!e.replaces_sensor.is_empty(), "{} does not say which sensor it replaces", e.id);
            assert!(!e.role_in_system.is_empty());
        }
    }

    #[test]
    fn catalogue_covers_every_deliverable_kind() {
        let kinds: HashSet<&str> = CATALOG.iter().map(|e| e.kind.as_str()).collect();
        for k in ["timeseries", "images", "weather", "remote_sensing", "thresholds"] {
            assert!(kinds.contains(k), "missing kind {k}");
        }
    }

    #[test]
    fn by_id_lookup() {
        assert_eq!(by_id("open-meteo").unwrap().kind, SourceKind::Weather);
        assert!(by_id("does-not-exist").is_none());
    }
}
