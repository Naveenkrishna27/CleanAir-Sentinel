"""
Federated model exchange — export manifest.

We deliberately did NOT implement live federated learning across nodes for
this hackathon (that's a multi-week distributed-systems project, not an
8-day one). Instead, this produces a concrete, documented artifact: a
versioned JSON manifest describing a trained model well enough that another
BRICS city/node could evaluate whether to pull and use it, WITHOUT either
side sharing raw citizen or sensor data.

See docs/federated_model_exchange.md for the full design rationale.
"""

import json
import os
from datetime import datetime, timezone


def export_model_manifest(
    model_name: str,
    feature_cols: list[str],
    mae: float,
    train_rows: int,
    region: str = "bengaluru",
    target: str = "no2_mol_m2",
    path: str = "data/model_manifest.json",
) -> dict:
    """
    Writes a shareable manifest describing the trained model — architecture,
    features, region, and evaluation metric — without embedding any raw
    citizen photos, sensor readings, or personally identifiable data.

    A real federated-exchange node would fetch this manifest (and the
    accompanying .joblib weights file) rather than raw training data.
    """
    manifest = {
        "manifest_version": "1.0",
        "project": "CleanAir Sentinel",
        "region": region,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "type": model_name,
            "target_variable": target,
            "feature_schema": feature_cols,
            "training_rows": train_rows,
        },
        "evaluation": {
            "metric": "mean_absolute_error",
            "value": round(float(mae), 6),
            "validation_method": "time_series_cross_validation",
        },
        "data_sources": [
            {"name": "Sentinel-5P NO2", "type": "satellite", "provider": "Copernicus / ESA"},
            {"name": "Sentinel-5P Aerosol Index", "type": "satellite", "provider": "Copernicus / ESA"},
            {"name": "Surface meteorology", "type": "reanalysis", "provider": "NASA POWER"},
            {"name": "Citizen photo severity", "type": "crowdsourced", "provider": "CleanAir Sentinel app users"},
        ],
        "sharing_terms": {
            "raw_data_included": False,
            "pii_included": False,
            "note": (
                "Only model weights and this manifest are intended for cross-node "
                "sharing. Raw citizen photos and exact sensor coordinates stay local "
                "to the originating node/region."
            ),
        },
        "weights_artifact": "aq_forecaster.joblib",
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote federated model manifest to {path}")
    return manifest


if __name__ == "__main__":
    # Example standalone run
    export_model_manifest(
        model_name="gradient_boosting",
        feature_cols=["no2_mol_m2_lag1", "no2_mol_m2_lag7", "dayofweek"],
        mae=0.0,
        train_rows=0,
    )
