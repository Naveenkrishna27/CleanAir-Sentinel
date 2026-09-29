"""
Streamlit dashboard for CleanAir Sentinel (Bengaluru).

Run with: streamlit run app.py
"""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecasting import build_features, load_model
from hotspot_detection import flag_hotspots, score_photo_severity
from impact_context import enrich_alerts
from aqi_fetch import fetch_bengaluru_aqi

st.set_page_config(page_title="CleanAir Sentinel", layout="wide", page_icon="🛰️")

ALERT_THRESHOLD = 0.7

DATA_DIR = "data"
SAMPLE_DIR = "data/sample"
PHOTOS_DIR = "data/citizen_photos"

CITIZEN_REPORTS_CSV = f"{DATA_DIR}/citizen_reports.csv"
SATELLITE_CSV = f"{DATA_DIR}/satellite_data.csv"
TIMESERIES_CSV = f"{DATA_DIR}/satellite_timeseries.csv"

os.makedirs(PHOTOS_DIR, exist_ok=True)

# --- Global styling -----------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  { font-family: 'Inter', -apple-system, sans-serif; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1280px; }

    /* Header — deep indigo/violet gradient into near-black */
    .cas-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #3730A3 50%, #0A0E17 100%);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 20px;
        border: 1px solid rgba(99, 102, 241, 0.25);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }
    .cas-header h1 {
        margin: 0; font-size: 2.2rem; font-weight: 800;
        color: #F8FAFC; letter-spacing: -0.03em;
    }
    .cas-header p { margin: 8px 0 0 0; color: #A5B4FC; font-size: 0.98rem; }

    /* KPI cards — each with a distinct accent top-bar for visual rhythm */
    .cas-kpi {
        background: #12182B;
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-top: 3px solid var(--accent, #6366F1);
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
        min-height: 172px;
    }
    .cas-kpi .kpi-label {
        color: #94A3B8; font-size: 0.8rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;
    }
    .cas-kpi .kpi-value {
        color: #F8FAFC; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;
    }
    .cas-kpi .kpi-sub { color: #64748B; font-size: 0.78rem; margin-top: 2px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; margin-bottom: 20px; border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #12182B;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 14px 28px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 1.02rem;
        transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1A2138;
        border-color: rgba(99, 102, 241, 0.4);
        color: #C7D2FE;
        transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] {
        background-color: #4338CA !important;
        border-color: #6366F1 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Status badges — semantic colors only (green=live, amber=sample, rose=synthetic) */
    .cas-badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.83rem; font-weight: 500; margin-bottom: 10px;
    }
    .cas-badge-live { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .cas-badge-sample { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .cas-badge-synthetic { background: rgba(244, 63, 94, 0.15); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.3); }

    /* Section cards (wraps charts/tables) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #0F1420; border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.10) !important;
    }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* Buttons — indigo brand accent */
    .stButton button, .stFormSubmitButton button {
        background-color: #6366F1 !important; border: none !important; color: white !important;
        font-weight: 600 !important;
    }
    .stButton button:hover, .stFormSubmitButton button:hover { background-color: #4F46E5 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #080B12; border-right: 1px solid rgba(148,163,184,0.08); }
    .cas-sidebar-title { color: #F8FAFC; font-weight: 700; font-size: 1.1rem; margin-bottom: 2px; }
    .cas-sidebar-sub { color: #64748B; font-size: 0.82rem; margin-bottom: 18px; }

    footer { visibility: hidden; }
    .cas-footer {
        text-align: center; color: #475569; font-size: 0.8rem;
        margin-top: 40px; padding-top: 16px; border-top: 1px solid rgba(148,163,184,0.1);
    }
</style>""", unsafe_allow_html=True)


# --- Data loading with a three-tier fallback: real -> sample -> synthetic ---

def _load_csv_with_fallback(real_path: str, sample_path: str, synthetic: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if os.path.exists(real_path):
        try:
            return pd.read_csv(real_path), "real"
        except Exception:
            pass
    if os.path.exists(sample_path):
        try:
            return pd.read_csv(sample_path), "sample"
        except Exception:
            pass
    return synthetic, "synthetic"


def _load_photo_reports() -> tuple[pd.DataFrame, str]:
    demo = pd.DataFrame({
        "lat": [12.97, 13.02, 12.92, 13.05],
        "lon": [77.59, 77.64, 77.55, 77.50],
        "severity_score": [0.8, 0.4, 0.9, 0.3],
        "timestamp": ["2026-09-20"] * 4,
    })
    df, source = _load_csv_with_fallback(CITIZEN_REPORTS_CSV, f"{SAMPLE_DIR}/citizen_reports.csv", demo)
    if source in ("real", "sample") and "severity_score" not in df.columns:
        # Windows saves paths with backslashes; Linux (the deployed app) needs
        # forward slashes. Forward slashes work on both.
        df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)

        # Score each photo on its own. If one can't be read, skip just that
        # report rather than inventing a severity for it.
        def _score(path):
            try:
                return score_photo_severity(path)
            except Exception:
                return None

        df["severity_score"] = df["image_path"].apply(_score)
        unreadable = int(df["severity_score"].isna().sum())
        if unreadable:
            st.warning(f"{unreadable} citizen report(s) skipped: photo file could not be read.")
        df = df.dropna(subset=["severity_score"]).reset_index(drop=True)
        if df.empty:
            return demo, "synthetic"
    return df, source


def _load_satellite_grid() -> tuple[pd.DataFrame, str]:
    demo = pd.DataFrame({
        "lat": [12.97, 13.02, 12.92, 13.05],
        "lon": [77.59, 77.64, 77.55, 77.50],
        "no2_mol_m2": [0.00012, 0.00008, 0.00015, 0.00006],
        "aerosol_index": [1.2, 0.8, 1.5, 0.5],
    })
    df, source = _load_csv_with_fallback(SATELLITE_CSV, f"{SAMPLE_DIR}/satellite_data.csv", demo)
    if {"lat", "lon"}.issubset(df.columns):
        return df, source
    return demo, "synthetic"


SOURCE_LABELS = {
    "real": ("cas-badge-live", "✅ Live data"),
    "sample": ("cas-badge-sample", "🟡 Sample snapshot"),
    "synthetic": ("cas-badge-synthetic", "⚠️ Synthetic placeholder"),
}


def badge_html(source_key: str, prefix: str = "") -> str:
    css_class, label = SOURCE_LABELS[source_key]
    return f'<span class="cas-badge {css_class}">{prefix}{label}</span>'


@st.cache_data(ttl=600, show_spinner=False)
def get_aqi() -> dict:
    """Streamlit reruns this script on every click, so cache the AQI for
    10 minutes instead of calling the API each time."""
    return fetch_bengaluru_aqi()


@st.cache_data(ttl=600, show_spinner=False)
def get_enriched_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """The school/hospital lookup hits OpenStreetMap once per alert, so this
    is cached - but only for 10 minutes, not longer: Overpass's free public
    server can fail transiently, and a failed lookup should self-heal on
    the next visit rather than showing "unavailable" for a long time."""
    return enrich_alerts(alerts_df)


def kpi_card(label: str, value: str, sub: str = "", accent: str = "#6366F1") -> str:
    return f"""
    <div class="cas-kpi" style="--accent: {accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


HOTSPOT_COLUMNS = ["hotspot_score", "lat", "lon", "citizen_severity", "no2_umol_m2", "aerosol_index"]


def hotspot_view(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Compact, readable version of the hotspot table for display.

    NO2 is shown in umol/m2: the raw mol/m2 values are around 0.00005, which
    round to 0.0001 at four decimals and hide any difference between cells.
    The diagnostic *_norm columns are left out; they are internal working."""
    out = df.copy()
    if "no2_mol_m2" in out.columns:
        out["no2_umol_m2"] = out["no2_mol_m2"] * 1e6
    out = out[[c for c in (columns or HOTSPOT_COLUMNS) if c in out.columns]]
    return out.round({"hotspot_score": 3, "lat": 4, "lon": 4, "citizen_severity": 3, "no2_umol_m2": 1, "aerosol_index": 2})


# --- Load data & compute hotspots once, up front -------------------------
photos_df, photos_source = _load_photo_reports()
satellite_df, satellite_source = _load_satellite_grid()

try:
    hotspots = flag_hotspots(photos_df, satellite_df)
except Exception:
    hotspots = pd.DataFrame()

# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="cas-sidebar-title">🛰️ CleanAir Sentinel</div>', unsafe_allow_html=True)
    st.markdown('<div class="cas-sidebar-sub">Bengaluru · Build with AI: Code for Communities</div>', unsafe_allow_html=True)
    st.markdown("**Data sources**")
    st.caption("Satellite (Sentinel-5P + NASA POWER)")
    st.markdown(badge_html(satellite_source), unsafe_allow_html=True)
    st.caption("Citizen reports")
    st.markdown(badge_html(photos_source), unsafe_allow_html=True)
    st.divider()
    st.markdown("**Alert sensitivity**")
    ALERT_THRESHOLD = st.slider(
        "Alert threshold", 0.3, 0.95, 0.7, 0.05,
        help="A location raises an alert when its hotspot score is above this. Lower it to flag more locations.",
    )
    st.divider()
    st.caption("Built on Google Earth Engine, NASA POWER, and OpenStreetMap — all free, open data sources.")
    st.caption("AQI: Open-Meteo / Copernicus CAMS model (CC BY 4.0)")

alerts = hotspots[hotspots["hotspot_score"] > ALERT_THRESHOLD] if not hotspots.empty else pd.DataFrame()

# --- Header ----------------------------------------------------------------
st.markdown("""
<div class="cas-header">
    <h1>🛰️ CleanAir Sentinel</h1>
    <p>Real-time air quality hotspot detection &amp; forecasting for Bengaluru — fusing satellite imagery, weather, and citizen reports into decision-ready alerts.</p>
</div>
""", unsafe_allow_html=True)

# --- KPI row -----------------------------------------------------------
aqi_data = get_aqi()
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(kpi_card("Grid cells monitored", str(len(hotspots)) if not hotspots.empty else "—", "5km resolution", "#6366F1"), unsafe_allow_html=True)
with k2:
    top_score = f"{hotspots['hotspot_score'].max():.2f}" if not hotspots.empty else "—"
    st.markdown(kpi_card("Peak hotspot score", top_score, "0–1 scale", "#22D3EE"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Active alerts", str(len(alerts)), f"threshold > {ALERT_THRESHOLD}", "#FB7185"), unsafe_allow_html=True)
with k4:
    avg_no2 = f"{hotspots['no2_mol_m2'].mean():.2e}" if not hotspots.empty else "—"
    st.markdown(kpi_card("Avg NO2 (grid)", avg_no2, "mol/m²", "#34D399"), unsafe_allow_html=True)
with k5:
    aqi_val = str(aqi_data["aqi"]) if aqi_data.get("aqi") is not None else "—"
    if aqi_data.get("aqi") is not None:
        # Always show WHEN it was measured, and only call it "live" if it is
        # fresh - a station can go quiet and keep returning its last value.
        aqi_label = "Bengaluru AQI (stale)" if aqi_data.get("stale") else "Bengaluru AQI"
        aqi_sub = f"{aqi_data['category']} · {aqi_data['updated_text']}" if aqi_data.get("updated_text") else aqi_data["category"]
        aqi_sub += "<br>Modelled · CAMS" if not aqi_data.get("stale") else "<br>Last reported reading"
    else:
        aqi_label = "Bengaluru AQI"
        aqi_sub = "Reading unavailable"
    st.markdown(kpi_card(aqi_label, aqi_val, aqi_sub, "#F59E0B"), unsafe_allow_html=True)

st.write("")

tab_map, tab_submit, tab_forecast, tab_alerts = st.tabs(
    ["Hotspot map", "Submit a report", "Forecast", "Alerts"]
)

# --- Hotspot map --------------------------------------------------------
with tab_map:
    with st.container(border=True):
        st.subheader("Pollution hotspots")
        st.caption("Citizen photo severity fused with satellite NO2 and aerosol index, on a 5km grid.")

        st.markdown(badge_html(satellite_source, "Satellite: "), unsafe_allow_html=True)
        st.markdown(badge_html(photos_source, "Citizen: "), unsafe_allow_html=True)
        if photos_source != "real":
            st.caption("👉 Go to **Submit a report** to add real citizen data.")

        if hotspots.empty:
            st.error("Couldn't compute hotspots — check data files.")
        else:
            fig = px.scatter_mapbox(
                hotspots, lat="lat", lon="lon", size="hotspot_score", color="hotspot_score",
                color_continuous_scale="OrRd", zoom=10, mapbox_style="open-street-map",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(hotspot_view(hotspots), use_container_width=True, hide_index=True)

# --- Submit a report (citizen photo intake) ---------------------------------
with tab_submit:
    with st.container(border=True):
        st.subheader("Submit a pollution report")
        st.caption("Upload a photo of local air conditions and tag its location. This feeds the citizen-sourced signal into the hotspot map.")

        with st.form("citizen_report_form", clear_on_submit=True):
            uploaded_photo = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitude", min_value=12.85, max_value=13.15, value=12.9716, format="%.4f")
            with col2:
                lon = st.number_input("Longitude", min_value=77.40, max_value=77.75, value=77.5946, format="%.4f")
            submitted = st.form_submit_button("Submit report", use_container_width=True)

            if submitted:
                if uploaded_photo is None:
                    st.error("Please attach a photo before submitting.")
                else:
                    try:
                        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                        safe_name = f"{timestamp.replace(':', '-')}_{uploaded_photo.name}"
                        image_path = os.path.join(PHOTOS_DIR, safe_name)
                        with open(image_path, "wb") as f:
                            f.write(uploaded_photo.getbuffer())

                        new_row = pd.DataFrame([{
                            "image_path": image_path, "lat": lat, "lon": lon, "timestamp": timestamp,
                        }])
                        if os.path.exists(CITIZEN_REPORTS_CSV):
                            new_row.to_csv(CITIZEN_REPORTS_CSV, mode="a", header=False, index=False)
                        else:
                            os.makedirs(DATA_DIR, exist_ok=True)
                            new_row.to_csv(CITIZEN_REPORTS_CSV, index=False)

                        st.success("Report submitted — refresh to see it on the hotspot map.")
                    except Exception as e:
                        st.error(f"Couldn't save report: {e}")

        if os.path.exists(CITIZEN_REPORTS_CSV):
            st.caption("Recent reports:")
            st.dataframe(pd.read_csv(CITIZEN_REPORTS_CSV).tail(10), use_container_width=True)

# --- Forecast -------------------------------------------------------------
with tab_forecast:
    with st.container(border=True):
        st.subheader("Air quality forecast")
        st.caption("Regression forecast selected via time-series cross-validation across multiple models.")

        ts_df, ts_source = _load_csv_with_fallback(TIMESERIES_CSV, f"{SAMPLE_DIR}/satellite_timeseries.csv", pd.DataFrame())

        if ts_df.empty:
            st.info("Run `python data_ingestion.py` then `python forecasting.py` to populate this tab.")
        else:
            st.markdown(badge_html(ts_source), unsafe_allow_html=True)
            try:
                model = load_model()
                feature_df = build_features(ts_df)
                feature_cols = [c for c in feature_df.columns if c not in ("date", "no2_mol_m2")]
                preds = model.predict(feature_df[feature_cols])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=feature_df["date"], y=feature_df["no2_mol_m2"], name="Actual", line=dict(color="#22D3EE", width=2.5)))
                fig.add_trace(go.Scatter(x=feature_df["date"], y=preds, name="Predicted", line=dict(color="#A5B4FC", dash="dot", width=2.5)))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E2E8F0", legend=dict(bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                    yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="NO2 (mol/m²)"),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                if "temp" in ts_df.columns:
                    st.caption("Temperature over the same period (°C, from NASA POWER)")
                    temp_fig = go.Figure()
                    temp_fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["temp"], line=dict(color="#FB7185", width=2), fill="tozeroy", fillcolor="rgba(251,113,133,0.08)"))
                    temp_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#E2E8F0", height=220,
                        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="°C"),
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(temp_fig, use_container_width=True)
            except FileNotFoundError:
                st.info("No trained model yet — run `python forecasting.py`.")
            except Exception as e:
                st.error(f"Forecast unavailable: {e}")

# --- Alerts -----------------------------------------------------------------
with tab_alerts:
    with st.container(border=True):
        st.subheader("Authority alerts")
        if hotspots.empty:
            st.info("No hotspot data available yet.")
        elif len(alerts) == 0:
            st.success("No hotspots above alert threshold right now.")
        else:
            st.error(f"{len(alerts)} location(s) above alert threshold ({ALERT_THRESHOLD})")
            with st.spinner("Checking nearby schools/hospitals via OpenStreetMap..."):
                enriched = get_enriched_alerts(alerts)

            if not enriched["lookup_ok"].all():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.warning("The nearby-institution lookup didn't respond for one or more alerts.")
                with col2:
                    if st.button("Retry lookup", use_container_width=True):
                        get_enriched_alerts.clear()
                        st.rerun()

            st.dataframe(
                hotspot_view(enriched, columns=[
                    "hotspot_score", "recommended_action", "schools_nearby", "hospitals_nearby",
                    "lat", "lon", "citizen_severity", "no2_umol_m2", "aerosol_index",
                ]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "hotspot_score": st.column_config.NumberColumn("Score"),
                    "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
                    "schools_nearby": st.column_config.NumberColumn("Schools nearby"),
                    "hospitals_nearby": st.column_config.NumberColumn("Health facilities nearby"),
                    "no2_umol_m2": st.column_config.NumberColumn("NO2 (µmol/m²)"),
                },
            )

st.markdown('<div class="cas-footer">CleanAir Sentinel · Built with Sentinel-5P, NASA POWER &amp; OpenStreetMap · Build with AI: Code for Communities, 2nd Edition</div>', unsafe_allow_html=True)