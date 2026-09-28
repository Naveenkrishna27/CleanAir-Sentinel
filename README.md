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
- **Citizen-sourced reports** — photos + location, submitted through the app
- **Satellite imagery** — Sentinel-5P (NO2, aerosol index) via Google Earth Engine
- **Meteorological data** — wind, humidity, temperature via NASA POWER

...into hotspot detection, short-horizon AQ forecasting, and alerts, plus a
documented, versioned model-export design so predictive models (not raw
data) could be shared across BRICS nodes.

## Scope for this hackathon (MVP)

Scoped to **Bengaluru** for the working demo. The federated/cross-border
layer is a documented model-export manifest design (see
`docs/federated_model_exchange.md`), not a live multi-node system — that's
the honest scaling story for the pitch, not something built live in 8 days.

| Component | Status |
|---|---|
| Data ingestion (satellite + met, via `data_ingestion.py`) | Built — **not yet run against live data** |
| Citizen report intake (Streamlit upload form) | Built |
| Hotspot detection (CV heuristic + robust scoring + anomaly detection) | Built |
| AQ forecasting (multi-model comparison + time-series CV) | Built — **not yet run against live data** |
| Dashboard (map, submit, forecast, alerts tabs) | Built |
| Sample-data offline fallback | Built — **needs a real snapshot saved once ingestion runs** |
| Federated model-export manifest | Built |
| Demo video / final submission | Not started |

## Modeling approach

**Forecasting** — `forecasting.py` cross-validates Ridge, Random Forest,
Gradient Boosting, and (if installed) XGBoost/LightGBM under **chronological
time-series splits** — never shuffled, so validation never leaks future
data into the past. The best performer by mean MAE is auto-selected and
refit on all available data. Cite the leaderboard in the pitch: "we
evaluated N models under time-series CV and selected X (MAE: Y)."

**Hotspot scoring** — `hotspot_detection.py` uses percentile-based robust
scaling instead of min-max (so one outlier reading doesn't distort every
other location's score), blended with an Isolation Forest anomaly score, so
a location ranks as a hotspot for being genuinely unusual, not just for
having the highest raw average.

**Federated model exchange** — rather than implementing real federated
learning (out of scope for 8 days), `model_export.py` produces a versioned
JSON manifest describing a trained model's features, region, and evaluation
metric, so another BRICS node could evaluate and adopt it without either
side sharing raw data. Full design rationale in
`docs/federated_model_exchange.md`.

## Architecture

```
Citizen reports (app upload) ─┐
Satellite imagery ────────────┼──▶ Fusion & AI layer ──┬──▶ Hotspot detection ──▶ Authority alerts
Met + AQ sensors ──────────────┘                        └──▶ AQ forecasting ────▶ Model manifest (BRICS exchange)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then set `EE_PROJECT_ID` in `.env` — **important**: this must be a project
that shows under "Earth Engine enabled Cloud Projects" at
code.earthengine.google.com (check the project switcher there), not just any
Cloud project. Running `earthengine authenticate` the first time opens a
browser for OAuth.

## Running the pipeline

```bash
# 1. Pull real satellite + weather data, and save an offline sample snapshot
python src/data_ingestion.py

# 2. Train & select the best forecasting model, and write the model manifest
python src/forecasting.py

# 3. Launch the dashboard
streamlit run src/app.py
```

The dashboard always shows which data tier it's displaying (live / cached
sample / synthetic placeholder), so it's never ambiguous what's real during
a demo.

## Project structure

```
clean-air-platform/
├── data/
│   ├── sample/             # committed offline-fallback snapshot (real data)
│   ├── citizen_photos/     # uploaded report photos (gitignored)
│   └── *.csv, *.joblib     # live-generated data (gitignored)
├── docs/
│   └── federated_model_exchange.md
├── src/
│   ├── data_ingestion.py   # satellite + met data, writes sample snapshot
│   ├── hotspot_detection.py
│   ├── forecasting.py      # model comparison + manifest export
│   ├── model_export.py     # federated model manifest
│   └── app.py               # Streamlit dashboard (map, submit, forecast, alerts)
└── requirements.txt
```

## Target corridor

**Bengaluru metro area** (bbox: 77.40–77.75°E, 12.85–13.15°N), centered on
(12.9716, 77.5946).

## Earth Engine project

Registered project: `gee-nk-projects` (confirmed working — verified via a
live Sentinel-5P query in the Earth Engine Code Editor).
