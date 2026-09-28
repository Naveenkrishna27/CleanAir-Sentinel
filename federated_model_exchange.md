# Federated Model Exchange — Design Note

**Why this exists:** the hackathon brief asks for a platform "designed for
interoperability so BRICS nations can share predictive models and
coordinate resources." Building real federated learning (distributed
training across nodes with secure aggregation) is a multi-week
distributed-systems project — not realistic in an 8-day hackathon window,
and pretending otherwise would be a worse pitch than being upfront about
scope.

What we built instead is the **artifact and contract** a real federated
system would need: a versioned, shareable model manifest (see
`src/model_export.py`) that describes a trained model well enough for
another node to evaluate and adopt it, without either side ever exchanging
raw data.

## Why data never needs to move

Each participating city/node:
1. Trains its own forecasting model locally on its own citizen reports,
   sensor readings, and satellite data (all of which may be sensitive,
   large, or governed by local data-residency rules).
2. Exports only the trained model weights + a manifest describing the
   model's features, target variable, training-data volume, and
   validation performance.
3. Publishes that manifest + weights file to a shared registry (could be
   as simple as a shared object store bucket, or a lightweight API).
4. Other nodes can fetch a manifest, check whether its feature schema and
   region are compatible with their own pipeline, and decide whether to
   adopt the model as-is, fine-tune it further on local data, or ignore it.

This is close to a simplified **model registry pattern**, not full
federated averaging — but it delivers the actual stated goal (share
predictive models without sharing resources/data) with a fraction of the
engineering risk.

## Manifest schema (v1.0)

```json
{
  "manifest_version": "1.0",
  "project": "CleanAir Sentinel",
  "region": "bengaluru",
  "generated_at": "2026-09-25T10:00:00+00:00",
  "model": {
    "type": "gradient_boosting",
    "target_variable": "no2_mol_m2",
    "feature_schema": ["no2_mol_m2_lag1", "no2_mol_m2_lag7", "dayofweek", "..."],
    "training_rows": 45
  },
  "evaluation": {
    "metric": "mean_absolute_error",
    "value": 0.0000123,
    "validation_method": "time_series_cross_validation"
  },
  "data_sources": [
    {"name": "Sentinel-5P NO2", "type": "satellite", "provider": "Copernicus / ESA"},
    {"name": "Surface meteorology", "type": "reanalysis", "provider": "NASA POWER"},
    {"name": "Citizen photo severity", "type": "crowdsourced", "provider": "CleanAir Sentinel app users"}
  ],
  "sharing_terms": {
    "raw_data_included": false,
    "pii_included": false,
    "note": "Only weights and this manifest are shared cross-node."
  },
  "weights_artifact": "aq_forecaster.joblib"
}
```

## What we'd build next, post-hackathon

- A lightweight registry API (list/fetch manifests across regions).
- Feature-schema compatibility checking, so a node doesn't blindly load a
  model trained on incompatible features.
- Optional differential-privacy noise on shared evaluation metrics, so
  even aggregate numbers can't be reverse-engineered to infer local
  conditions precisely.
- A real secure-aggregation federated-averaging path for teams that want
  to go beyond model-sharing into joint training — genuinely out of scope
  here, but the manifest schema above is designed to be extensible toward it.
