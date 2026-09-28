"""
Data ingestion for CleanAir Sentinel (Bengaluru).

Pulls two data sources for the target corridor:
1. Satellite pollution data (Sentinel-5P, via Google Earth Engine)
2. Meteorological data (wind, humidity, temperature) via NASA POWER

Also writes a copy of whatever it fetches into data/sample/, so the
Streamlit app always has a real (not synthetic) fallback dataset if live
data is unavailable on demo day.
"""

import os
import shutil

import ee
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

EE_PROJECT_ID = os.getenv("EE_PROJECT_ID", "")

# --- Config -----------------------------------------------------------------

# Bengaluru metro area, centered on (12.9716, 77.5946) — the same point used
# in the team's existing AI-Weather-Forecasting project.
TARGET_REGION = {
    "name": "bengaluru",
    "bbox": [77.40, 12.85, 77.75, 13.15],
}

DATA_DIR = "data"
SAMPLE_DIR = "data/sample"


# --- Satellite data (Sentinel-5P via Earth Engine) --------------------------

def init_earth_engine():
    """Run once per session. First run opens a browser for OAuth; needs
    EE_PROJECT_ID set in .env — must be a project that has actually
    completed Earth Engine registration (check at code.earthengine.google.com
    under 'Earth Engine enabled Cloud Projects', not just any Cloud project
    you own)."""
    if not EE_PROJECT_ID:
        raise RuntimeError("Set EE_PROJECT_ID in your .env file (your registered Earth Engine project ID)")
    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=EE_PROJECT_ID)
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize Earth Engine with project '{EE_PROJECT_ID}'. "
                "Confirm this project shows under 'Earth Engine enabled Cloud Projects' "
                "at code.earthengine.google.com, not just under 'All Cloud Projects'."
            ) from e


def fetch_sentinel5p_no2(start_date: str, end_date: str, region: dict = TARGET_REGION) -> pd.DataFrame:
    """
    Pull mean tropospheric NO2 column density for the target region
    over the given date range from Sentinel-5P.

    Fetches date + value together per image, rather than as two separate
    aggregate_array() calls — aggregate_array silently drops entries where
    the property is null (e.g. a date with no valid pixels over the bbox,
    from cloud cover or swath gaps), so pulling two arrays separately and
    zipping them by position breaks the moment that happens, since the
    arrays end up different lengths.

    Returns a DataFrame with columns: date, no2_mol_m2
    """
    bbox = ee.Geometry.Rectangle(region["bbox"])

    collection = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
        .select("tropospheric_NO2_column_number_density")
        .filterDate(start_date, end_date)
        .filterBounds(bbox)
    )

    def reduce_image(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=bbox, scale=1000, maxPixels=1e9
        )
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "no2_mol_m2": stats.get("tropospheric_NO2_column_number_density"),
        })

    features = ee.FeatureCollection(collection.map(reduce_image))
    records = [f["properties"] for f in features.getInfo()["features"]]
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["date", "no2_mol_m2"])
    return df.dropna(subset=["no2_mol_m2"]).sort_values("date").reset_index(drop=True)


def fetch_sentinel5p_aerosol(start_date: str, end_date: str, region: dict = TARGET_REGION) -> pd.DataFrame:
    """Pull mean UV aerosol index for the target region over the date range.
    Same date+value-together fix as fetch_sentinel5p_no2 above."""
    bbox = ee.Geometry.Rectangle(region["bbox"])

    collection = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI")
        .select("absorbing_aerosol_index")
        .filterDate(start_date, end_date)
        .filterBounds(bbox)
    )

    def reduce_image(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=bbox, scale=1000, maxPixels=1e9
        )
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "aerosol_index": stats.get("absorbing_aerosol_index"),
        })

    features = ee.FeatureCollection(collection.map(reduce_image))
    records = [f["properties"] for f in features.getInfo()["features"]]
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["date", "aerosol_index"])
    return df.dropna(subset=["aerosol_index"]).sort_values("date").reset_index(drop=True)


# --- Satellite data, sampled per grid point (for the hotspot map) -----------

def fetch_sentinel5p_grid(date_str: str, region: dict = TARGET_REGION, n_points_per_side: int = 5) -> pd.DataFrame:
    """
    Samples NO2 + aerosol index at a grid of points across the region for a
    single date, giving hotspot_detection.flag_hotspots() real per-location
    rows instead of one region-wide average.

    Returns columns: lat, lon, no2_mol_m2, aerosol_index
    """
    lon_min, lat_min, lon_max, lat_max = region["bbox"]
    lats = [lat_min + i * (lat_max - lat_min) / (n_points_per_side - 1) for i in range(n_points_per_side)]
    lons = [lon_min + i * (lon_max - lon_min) / (n_points_per_side - 1) for i in range(n_points_per_side)]

    next_day = (pd.to_datetime(date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    no2_img = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
        .select("tropospheric_NO2_column_number_density")
        .filterDate(date_str, next_day)
        .mean()
    )
    aer_img = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI")
        .select("absorbing_aerosol_index")
        .filterDate(date_str, next_day)
        .mean()
    )
    combined_img = no2_img.addBands(aer_img)

    points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {"lat": lat, "lon": lon})
        for lat in lats for lon in lons
    ])
    sampled = combined_img.reduceRegions(collection=points, reducer=ee.Reducer.first(), scale=1000).getInfo()

    records = []
    for feature in sampled["features"]:
        props = feature["properties"]
        records.append({
            "lat": props["lat"],
            "lon": props["lon"],
            "no2_mol_m2": props.get("tropospheric_NO2_column_number_density"),
            "aerosol_index": props.get("absorbing_aerosol_index"),
        })
    df = pd.DataFrame(records).dropna()
    if df.empty:
        raise RuntimeError(
            f"No Sentinel-5P grid data returned for {date_str}. Try an earlier date — "
            "Sentinel-5P has occasional coverage gaps and processing latency of a few days."
        )
    return df


def fetch_sentinel5p_grid_with_retry(region: dict = TARGET_REGION, n_points_per_side: int = 5, max_days_back: int = 10) -> pd.DataFrame:
    """
    Tries fetch_sentinel5p_grid() starting from a few days ago and walking
    backward, since Sentinel-5P has real coverage gaps for any single date
    over a small bbox (cloud cover, swath timing, processing latency).
    Returns the first date that has usable data.
    """
    start = pd.Timestamp.now() - pd.Timedelta(days=3)  # NASA/Copernicus processing lag
    for days_back in range(max_days_back):
        date_str = (start - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            df = fetch_sentinel5p_grid(date_str, region, n_points_per_side)
            print(f"Grid data found for {date_str} ({len(df)} points)")
            return df
        except RuntimeError:
            print(f"No grid data for {date_str}, trying an earlier date...")
            continue
    raise RuntimeError(f"No Sentinel-5P grid data found in the last {max_days_back} days — try a wider bbox or check the dataset status.")


# --- Meteorological data (NASA POWER — free, no API key, real history) ------

def fetch_weather_history(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull real historical wind/humidity/temperature for a point from NASA
    POWER's daily point API. No API key required. Note: NASA POWER data
    typically lags a few days behind the present, so don't request dates
    right up to today.

    Returns columns: date, wind_speed, humidity, temp
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "WS2M,RH2M,T2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"NASA POWER request failed: {e}") from e

    props = resp.json()["properties"]["parameter"]

    records = []
    for day_str in props["T2M"].keys():
        records.append({
            "date": f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:]}",
            "wind_speed": props["WS2M"].get(day_str),
            "humidity": props["RH2M"].get(day_str),
            "temp": props["T2M"].get(day_str),
        })

    df = pd.DataFrame(records)
    df[["wind_speed", "humidity", "temp"]] = df[["wind_speed", "humidity", "temp"]].replace(-999, pd.NA)
    return df


# --- Merge into one training-ready table -------------------------------------

def build_dataset(start_date: str, end_date: str, region: dict = TARGET_REGION) -> pd.DataFrame:
    """
    Combine NO2, aerosol, and weather data into one dataframe keyed by date.

    Sentinel-5P has real coverage gaps (cloud cover, swath timing) — it's
    common for no single date to have valid NO2 AND aerosol AND weather all
    at once. If we required all three to be non-null, a downstream blanket
    dropna() could wipe out every row. So: only the target (no2_mol_m2) is
    required to be present; the covariate columns (aerosol, weather) are
    interpolated across small gaps instead of causing row loss.
    """
    no2_df = fetch_sentinel5p_no2(start_date, end_date, region)
    aerosol_df = fetch_sentinel5p_aerosol(start_date, end_date, region)

    lon_min, lat_min, lon_max, lat_max = region["bbox"]
    center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    weather_df = fetch_weather_history(center_lat, center_lon, start_date, end_date)

    merged = no2_df.merge(aerosol_df, on="date", how="outer").merge(weather_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    covariate_cols = ["aerosol_index", "wind_speed", "humidity", "temp"]
    merged[covariate_cols] = merged[covariate_cols].interpolate(limit_direction="both")

    before = len(merged)
    merged = merged.dropna(subset=["no2_mol_m2"]).reset_index(drop=True)
    print(f"build_dataset: {before} merged dates -> {len(merged)} with valid NO2 (target)")

    return merged


def save_sample_snapshot():
    """
    Copies the current real data files into data/sample/, which IS committed
    to git (see .gitignore). This is the offline fallback the Streamlit app
    uses if live data is missing — run this once after a successful
    ingestion so demo day doesn't depend on APIs being reachable.
    """
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    copied = []
    for fname in ["satellite_timeseries.csv", "satellite_data.csv"]:
        src = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SAMPLE_DIR, fname))
            copied.append(fname)
    if copied:
        print(f"Saved sample snapshot: {copied} -> {SAMPLE_DIR}/")
    else:
        print("Nothing to snapshot yet — run ingestion first.")


if __name__ == "__main__":
    init_earth_engine()

    try:
        # Time series (region-wide average) — feeds forecasting.py
        # NASA POWER lags a few days behind today, so end date is kept safely
        # in the past. Adjust if you need a more recent window.
        ts_df = build_dataset("2026-08-01", "2026-09-18")
        os.makedirs(DATA_DIR, exist_ok=True)
        ts_df.to_csv(f"{DATA_DIR}/satellite_timeseries.csv", index=False)
        print(f"Saved {len(ts_df)} rows to {DATA_DIR}/satellite_timeseries.csv")

        # Grid snapshot (per-location, latest available date) — feeds the hotspot map
        grid_df = fetch_sentinel5p_grid_with_retry()
        grid_df.to_csv(f"{DATA_DIR}/satellite_data.csv", index=False)
        print(f"Saved {len(grid_df)} rows to {DATA_DIR}/satellite_data.csv")
    finally:
        # Always snapshot whatever succeeded, even if a later step failed —
        # so a coverage-gap error on the grid fetch doesn't cost you the
        # offline fallback for data that DID come through.
        save_sample_snapshot()