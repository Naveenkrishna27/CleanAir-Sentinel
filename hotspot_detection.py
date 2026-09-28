"""
Hotspot detection: combines citizen photo severity scoring with satellite
AQ index to flag pollution hotspots on a grid.

- Percentile-based robust scaling, so one broken sensor reading or one
  extreme photo doesn't distort every other cell's score.
- An Isolation Forest anomaly score as a second signal, rewarding cells
  that are genuinely unusual relative to the whole dataset.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class PhotoReport:
    image_path: str
    lat: float
    lon: float
    timestamp: str


def score_photo_severity(image_path: str) -> float:
    """
    Returns a 0-1 haze/smog severity score using a dark-channel-prior style
    heuristic: hazy images have low contrast and washed-out saturation.
    A fast, explainable baseline — swap for a trained classifier only if
    you get labeled data with time to spare.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean() / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std() / 128.0

    haze_score = 1.0 - min(1.0, (saturation + contrast) / 2)
    return round(float(np.clip(haze_score, 0, 1)), 3)


def _robust_scale(series: pd.Series, low_pct: float = 5, high_pct: float = 95) -> pd.Series:
    """Scales to 0-1 using the 5th-95th percentile range instead of min-max,
    so one extreme reading doesn't compress every other value."""
    lo, hi = np.percentile(series, [low_pct, high_pct])
    if hi - lo == 0:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def flag_hotspots(
    photo_reports: pd.DataFrame,
    satellite_df: pd.DataFrame,
    grid_size_km: float = 5.0,
    anomaly_weight: float = 0.4,
) -> pd.DataFrame:
    """
    photo_reports: DataFrame with columns [lat, lon, severity_score, timestamp]
    satellite_df: DataFrame with columns [lat, lon, no2_mol_m2, aerosol_index]

    Bins both onto a shared grid, scores each cell with a robust weighted
    average plus an Isolation Forest anomaly score, and returns a ranked
    table with real lat/lon per cell for mapping.
    """
    def to_grid_cell(lat, lon, size_km):
        deg_size = size_km / 111.0
        return round(lat / deg_size), round(lon / deg_size)

    photo_reports = photo_reports.copy()
    photo_reports["cell"] = photo_reports.apply(
        lambda r: to_grid_cell(r["lat"], r["lon"], grid_size_km), axis=1
    )
    citizen_agg = photo_reports.groupby("cell").agg(
        citizen_severity=("severity_score", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    )

    satellite_df = satellite_df.copy()
    satellite_df["cell"] = satellite_df.apply(
        lambda r: to_grid_cell(r["lat"], r["lon"], grid_size_km), axis=1
    )
    sat_agg = satellite_df.groupby("cell").agg(
        no2_mol_m2=("no2_mol_m2", "mean"),
        aerosol_index=("aerosol_index", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    )

    combined = citizen_agg.join(sat_agg, how="outer", lsuffix="_photo", rsuffix="_sat")
    combined["lat"] = combined["lat_photo"].fillna(combined["lat_sat"])
    combined["lon"] = combined["lon_photo"].fillna(combined["lon_sat"])
    combined = combined.drop(columns=["lat_photo", "lon_photo", "lat_sat", "lon_sat"])
    combined[["citizen_severity", "no2_mol_m2", "aerosol_index"]] = combined[
        ["citizen_severity", "no2_mol_m2", "aerosol_index"]
    ].fillna(0)

    feature_cols = ["citizen_severity", "no2_mol_m2", "aerosol_index"]
    for col in feature_cols:
        combined[f"{col}_norm"] = _robust_scale(combined[col])

    combined["avg_score"] = combined[[f"{c}_norm" for c in feature_cols]].mean(axis=1)

    if len(combined) >= 5:
        iso = IsolationForest(contamination=0.2, random_state=42)
        raw_scores = -iso.fit(combined[feature_cols]).score_samples(combined[feature_cols])
        combined["anomaly_score"] = _robust_scale(pd.Series(raw_scores, index=combined.index))
    else:
        combined["anomaly_score"] = combined["avg_score"]

    combined["hotspot_score"] = (
        (1 - anomaly_weight) * combined["avg_score"] + anomaly_weight * combined["anomaly_score"]
    )

    return combined.sort_values("hotspot_score", ascending=False).reset_index()
