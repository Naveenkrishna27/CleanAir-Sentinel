"""
Impact context for alerts: for each flagged hotspot, finds nearby schools
and hospitals/clinics via OpenStreetMap's Overpass API (free, no key), and
derives a simple rule-based recommended action.

This turns a bare hotspot_score into "who's affected and what should
happen" - the decision-support angle judges care about, not just a number.

Deliberately rule-based, not ML: with only a few dozen data points overall,
training a real recommendation model would be overfitting theater. A
transparent if/else is more honest AND more explainable to a judge who
asks "why did it recommend that."

Robustness: Overpass's main public instance (overpass-api.de) is shared
across many apps and can be slow or briefly unavailable, especially from a
shared hosting IP (e.g. Streamlit Cloud). A single failed request should
not look identical to "genuinely zero schools nearby" - so failures are
retried across TWO independent public Overpass mirrors before giving up,
with an explicit status the caller can act on. app.py caches successes
longer than failures so a transient outage self-heals quickly instead of
freezing a wrong "None" on screen for a long time.
"""

import time

import requests

# Two independent public mirrors - if the main instance is overloaded, the
# second (a different operator, different infrastructure) often still works.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def fetch_nearby_institutions(lat: float, lon: float, radius_m: int = 2000, retries: int = 1) -> dict:
    """
    Returns counts of schools and hospitals/clinics within radius_m meters
    of (lat, lon), using OpenStreetMap's Overpass API. No API key required.

    Returns {"schools": int, "hospitals": int, "ok": True} on success, or
    {"schools": None, "hospitals": None, "ok": False, "error": str} if every
    mirror failed - callers should treat ok=False as "unknown", not "zero",
    and should not cache it for long.
    """
    query = f"""
    [out:json][timeout:20];
    (
      node["amenity"="school"](around:{radius_m},{lat},{lon});
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"](around:{radius_m},{lat},{lon});
    );
    out body;
    """
    last_error = None
    for url in OVERPASS_URLS:
        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, data={"data": query}, timeout=25)
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
                schools = sum(1 for e in elements if e.get("tags", {}).get("amenity") == "school")
                hospitals = sum(1 for e in elements if e.get("tags", {}).get("amenity") in ("hospital", "clinic"))
                return {"schools": schools, "hospitals": hospitals, "ok": True}
            except Exception as e:
                last_error = str(e)
                if attempt < retries:
                    time.sleep(1.5)

    return {"schools": None, "hospitals": None, "ok": False, "error": last_error}


def recommend_action(hotspot_row, ctx: dict) -> str:
    """
    Rule-based recommendation combining the hotspot's dominant pollutant
    signal with what's nearby. Not ML - see module docstring for why.
    """
    no2 = hotspot_row.get("no2_mol_m2", 0) or 0
    aerosol = hotspot_row.get("aerosol_index", 0) or 0

    if aerosol >= no2:
        action = "Recommend public health advisory + reduce outdoor activity"
    else:
        action = "Recommend traffic diversion + industrial emissions check"

    if not ctx.get("ok"):
        return action + " (nearby-institution lookup unavailable - retry)"

    schools = ctx.get("schools") or 0
    hospitals = ctx.get("hospitals") or 0
    tags = []
    if schools > 0:
        tags.append(f"{schools} school{'s' if schools != 1 else ''}")
    if hospitals > 0:
        tags.append(f"{hospitals} health facilit{'ies' if hospitals != 1 else 'y'}")

    if tags:
        action = f"URGENT \u2014 {action} (near {', '.join(tags)})"

    return action


def enrich_alerts(alerts_df, radius_m: int = 2000):
    """
    Adds schools_nearby, hospitals_nearby, lookup_ok, and recommended_action
    columns to an alerts DataFrame (expects lat, lon, no2_mol_m2,
    aerosol_index columns - the same shape hotspot_detection.flag_hotspots()
    returns).

    Only called on already-thresholded alerts, not every grid cell, to
    keep Overpass API usage light and the dashboard responsive.
    """
    df = alerts_df.copy()
    schools_col, hospitals_col, ok_col, action_col = [], [], [], []

    for _, row in df.iterrows():
        ctx = fetch_nearby_institutions(row["lat"], row["lon"], radius_m)
        schools_col.append(ctx.get("schools"))
        hospitals_col.append(ctx.get("hospitals"))
        ok_col.append(ctx.get("ok", False))
        action_col.append(recommend_action(row, ctx))

    df["schools_nearby"] = schools_col
    df["hospitals_nearby"] = hospitals_col
    df["lookup_ok"] = ok_col
    df["recommended_action"] = action_col
    return df