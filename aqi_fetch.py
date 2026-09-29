"""
Current air quality (US AQI) for Bengaluru, from the Open-Meteo Air Quality
API - free, no API key, no signup. Data: Copernicus Atmosphere Monitoring
Service (CAMS), served by Open-Meteo under CC BY 4.0.

IMPORTANT - this is a MODELLED value, not a ground-station measurement.
CAMS is an atmospheric model (roughly tens of km resolution outside Europe),
so treat it as a city-scale estimate. The dashboard labels it "modelled" for
that reason. It is an independent source from our own Sentinel-5P satellite
readings and citizen reports.

Why not a ground-station feed: the nearest WAQI stations for Bengaluru had
stopped reporting (the closest last reported in June), and WAQI's city-name
lookup once returned a station in Delhi. A live modelled number, honestly
labelled, is better than a stale or wrong-city measurement.
"""

import time
from datetime import datetime, timezone

import requests

BENGALURU_LAT, BENGALURU_LON = 12.9716, 77.5946
STALE_AFTER_HOURS = 3
IST_OFFSET_SECONDS = 19800  # Asia/Kolkata is a fixed UTC+05:30, no daylight saving

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_bengaluru_aqi() -> dict:
    """
    Returns the current modelled US AQI for Bengaluru, its category, the
    time the value is for, and whether it is stale. Never raises - on any
    failure the AQI is None so one bad API call can't break the dashboard.
    """
    empty = {"aqi": None, "category": None, "station": None, "updated_text": None, "stale": False}

    params = {
        "latitude": BENGALURU_LAT,
        "longitude": BENGALURU_LON,
        "current": "us_aqi,pm2_5",
        "timezone": "Asia/Kolkata",
    }
    try:
        resp = requests.get(URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        return {**empty, "error": f"Open-Meteo request failed: {e}"}

    current = payload.get("current") or {}
    try:
        aqi = int(round(float(current["us_aqi"])))
    except (KeyError, TypeError, ValueError):
        return {**empty, "error": "Open-Meteo returned no AQI value"}

    updated_text, age_hours = None, None
    stamp = current.get("time")  # e.g. "2026-09-28T16:00", local time (Asia/Kolkata)
    if stamp:
        try:
            local = datetime.strptime(stamp, "%Y-%m-%dT%H:%M")
            updated_text = local.strftime("%d %b, %H:%M")
            offset = payload.get("utc_offset_seconds", IST_OFFSET_SECONDS)
            epoch = local.replace(tzinfo=timezone.utc).timestamp() - offset
            age_hours = (time.time() - epoch) / 3600
        except ValueError:
            pass

    return {
        "aqi": aqi,
        "category": _aqi_category(aqi),
        "station": "CAMS atmospheric model",
        "updated_text": updated_text,
        "age_hours": age_hours,
        "stale": age_hours is not None and age_hours > STALE_AFTER_HOURS,
    }


def _aqi_category(aqi) -> str:
    """Standard US EPA AQI category bands."""
    if aqi is None or not isinstance(aqi, (int, float)):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy (sensitive groups)"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"