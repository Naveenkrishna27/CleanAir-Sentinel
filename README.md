# CleanAir Sentinel — Clean Air & Climate Resilience, Bengaluru
### Build with AI: Code for Communities (2nd Edition)

**Track 2 — BRICS Theme: Sustainability**

## Problem

Major BRICS cities monitor macro-level air quality but miss hyper-local and
cross-border pollution events — industrial emissions, agricultural burning,
trans-boundary smog. The lack of real-time, granular data blocks coordinated
climate action and threatens public health.

## What this is

A platform that fuses:
- **Citizen-sourced reports** — photos + location, submitted through the app,
  scored for haze/smog severity
- **Satellite imagery** — Sentinel-5P (NO2, aerosol index) via Google Earth Engine
- **Meteorological data** — wind, humidity, temperature via NASA POWER
- **A live city-scale AQI reading** — Open-Meteo / Copernicus CAMS model, for
  independent context alongside the hyper-local grid

...into hotspot detection, short-horizon AQ forecasting, and decision-ready
alerts enriched with nearby schools/hospitals and a recommended response —
plus a documented, versioned model-export design so predictive models (not
raw data) could be shared across BRICS nodes.

## Status

Scoped to **Bengaluru** for the working demo, with a live deployment.

| Component | Status |
|---|---|
| Data ingestion (Sentinel-5P + NASA POWER) | Live — 49-date timeseries, 20+ point grid |
| Citizen report intake (photo upload + geotagging) | Live — real submitted reports |
| Hotspot detection (robust scoring + anomaly detection) | Live |
| AQ forecasting (multi-model comparison + time-series CV) | Live — model trained and deployed |
| Live city AQI (Open-Meteo / CAMS) | Live |
| Alerts enriched with nearby schools/hospitals (OpenStreetMap) | Live |
| Dashboard (map, submit, forecast, alerts, adjustable alert threshold) | Live |
| Federated model-export manifest | Built (design artifact, see below) |

## Modeling approach

**Forecasting** — `forecasting.py` cross-validates Ridge, Random Forest,
Gradient Boosting, and (if installed) XGBoost/LightGBM under **chronological
time-series splits** — never shuffled, so validation never leaks future data
into the past. The best performer by mean MAE is auto-selected and refit on
all available data.

**Hotspot scoring** — `hotspot_detection.py` combines three signals per grid
cell: citizen photo severity (used directly, since it's already a 0–1 score
and too sparse for percentile scaling to behave sensibly), percentile-scaled
satellite NO2/aerosol readings (so one outlier reading doesn't distort every
other cell), and an Isolation Forest anomaly score, so a location ranks as a
hotspot for being genuinely unusual, not just for having the highest raw
average. The alert threshold is adjustable live in the dashboard.

**Decision support** — `impact_context.py` looks up nearby schools and
hospitals for each alert via OpenStreetMap's Overpass API, and derives a
rule-based (not ML — with this little data, a trained recommender would be
overfitting theater) recommended action: a public-health advisory when
aerosol/haze dominates, a traffic/emissions check when NO2 dominates, flagged
URGENT when a school or health facility is nearby.

**Federated model exchange** — rather than implementing real federated
learning (a multi-week distributed-systems project, out of scope for this
hackathon), `model_export.py` produces a versioned JSON manifest describing a
trained model's features, region, and evaluation metric, so another BRICS
node could evaluate and adopt it without either side sharing raw data. Full
design rationale in `federated_model_exchange.md`.

## Architecture

```
Citizen reports (app upload) ─┐
Satellite imagery ────────────┼──▶ Fusion & scoring ──┬──▶ Hotspot map ──▶ Alerts + nearby schools/hospitals ──▶ Recommended action
Met + AQ sensors ──────────────┘                       └──▶ AQ forecasting ──▶ Model manifest (BRICS exchange)

Live city AQI (independent check) ──▶ Dashboard KPI
```

## Setup

```bash
python -m venv venv
source venv/bin/activate    # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Set `EE_PROJECT_ID` in `.env` — must be a project registered under "Earth
Engine enabled Cloud Projects" at code.earthengine.google.com. Running the
app the first time opens a browser for Earth Engine OAuth. No key is needed
for NASA POWER, OpenStreetMap, or the AQI source.

## Running the pipeline

```bash
# 1. Pull real satellite + weather data, and save an offline sample snapshot
python data_ingestion.py

# 2. Train & select the best forecasting model, and write the model manifest
python forecasting.py

# 3. Launch the dashboard
streamlit run app.py
```

Submit citizen reports through the app's "Submit a report" tab, then run
`python save_citizen_sample.py` to snapshot them (metadata-stripped,
coordinates rounded) into the committed offline-fallback tier.

The dashboard always shows which data tier it's displaying (live / cached
sample / synthetic placeholder) for both satellite and citizen data, so it's
never ambiguous what's real during a demo.

## Project structure

```
CleanAir Sentinel/
├── data/
│   ├── sample/             # committed offline-fallback snapshot (real data)
│   ├── citizen_photos/     # uploaded report photos (gitignored)
│   └── *.csv, *.joblib     # live-generated data (committed: model + manifest; gitignored: raw CSVs)
├── app.py                  # Streamlit dashboard
├── data_ingestion.py       # satellite + met data, writes sample snapshot
├── hotspot_detection.py    # citizen + satellite fusion, anomaly scoring
├── forecasting.py          # model comparison + manifest export
├── model_export.py         # federated model manifest
├── impact_context.py       # nearby schools/hospitals + recommended action
├── aqi_fetch.py            # live city-scale AQI (Open-Meteo/CAMS)
├── save_citizen_sample.py  # sanitizes + snapshots citizen reports for commit
├── federated_model_exchange.md
└── requirements.txt
```

## Target corridor

**Bengaluru metro area** (bbox: 77.40–77.75°E, 12.85–13.15°N), centered on
(12.9716, 77.5946) — the same point used in the team's existing
[AI-Weather-Forecasting](https://github.com/Naveenkrishna27/AI-Weather-Forecasting)
project.

## Data sources & attribution

- Satellite: Copernicus Sentinel-5P, via Google Earth Engine
- Meteorology: NASA POWER
- Live AQI: Open-Meteo / Copernicus Atmosphere Monitoring Service (CC BY 4.0)
- Nearby institutions: © OpenStreetMap contributors