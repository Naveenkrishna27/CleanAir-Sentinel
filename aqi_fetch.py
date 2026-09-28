"""
Real ground-station AQI for Bengaluru, via the WAQI (World Air Quality
Index) API. Free token, issued instantly by email at
https://aqicn.org/data-platform/token/ — no approval wait.

Deliberately NOT derived from our own Sentinel-5P NO2 column density:
converting a satellite vertical column to ground-level AQI needs
boundary-layer-height assumptions that would make the number scientifically
shaky. This is a second, independent, real data source instead — actual
ground monitoring stations reporting the standard AQI.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

WAQI_TOKEN = os.getenv("WAQI_TOKEN", "")


def fetch_bengaluru_aqi() -> dict:
    """
    Returns the current AQI for Bengaluru from the nearest reporting
    ground station, plus its category label. Returns None values on any
    failure (missing token, network issue, no station data) rather than
    raising — an AQI card failing shouldn't break the rest of the
    dashboard.
    """
    if not WAQI_TOKEN:
        return {"aqi": None, "category": None, "station": None, "error": "WAQI_TOKEN not set in .env"}

    url = f"https://api.waqi.info/feed/bengaluru/"
    try:
        resp = requests.get(url, params={"token": WAQI_TOKEN}, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "ok":
            return {"aqi": None, "category": None, "station": None, "error": payload.get("data", "WAQI request failed")}

        data = payload["data"]
        aqi = data.get("aqi")
        return {
            "aqi": aqi,
            "category": _aqi_category(aqi),
            "station": data.get("city", {}).get("name"),
        }
    except Exception as e:
        return {"aqi": None, "category": None, "station": None, "error": str(e)}


def _aqi_category(aqi) -> str:
    """Standard US EPA-style AQI category bands."""
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
