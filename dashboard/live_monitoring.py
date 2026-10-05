"""
Near-Real-Time Supply Chain Monitoring Module
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
Purpose: Live weather data + simulated operational events → updated XGBoost inference
         using the EXISTING prepare_feature_row pipeline (no new model, no retraining).

IMPORTANT NOTES:
- Weather data: LIVE from Open-Meteo (no API key required)
- Operational events: SIMULATED — clearly labelled as such
- All model inference uses the existing DisruptionExplainer.prepare_feature_row()
- Historical CSV is NEVER modified; only copies of records are used
- This module is designed to be imported by dashboard/app.py ONLY
"""

import datetime
import time
import math
import pandas as pd
import numpy as np

try:
    import requests as _requests_lib
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# RISK THRESHOLDS  (identical to Pages 4 & 5)
# ─────────────────────────────────────────────────────────────────────────────
RISK_LOW_THRESHOLD      = 0.35
RISK_HIGH_THRESHOLD     = 0.65

# ─────────────────────────────────────────────────────────────────────────────
# DEMO / DEFAULT PORT COORDINATES
# Derived from the dataset's Origin_Port values only (no invented locations).
# Labelled as demo locations in the UI.
# ─────────────────────────────────────────────────────────────────────────────
PORT_COORDINATES = {
    "US-LAX": {"name": "Los Angeles (US-LAX)",  "lat": 33.729,  "lon": -118.263},
    "US-LGB": {"name": "Long Beach (US-LGB)",   "lat": 33.754,  "lon": -118.216},
    "CN-SHA": {"name": "Shanghai (CN-SHA)",      "lat": 30.631,  "lon": 121.502},
    "CN-NGB": {"name": "Ningbo (CN-NGB)",        "lat": 29.867,  "lon": 121.550},
    "DE-HAM": {"name": "Hamburg (DE-HAM)",       "lat": 53.545,  "lon": 9.969},
    "VN-SGN": {"name": "Ho Chi Minh City (VN-SGN)", "lat": 10.780, "lon": 106.699},
}
DEFAULT_PORT_KEY = "US-LAX"

# WMO weather code → human-readable description
WMO_CODE_MAP = {
    0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy Fog",
    51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Dense Drizzle",
    61: "Light Rain", 63: "Moderate Rain", 65: "Heavy Rain",
    71: "Light Snow", 73: "Moderate Snow", 75: "Heavy Snow",
    80: "Light Showers", 81: "Moderate Showers", 82: "Violent Showers",
    95: "Thunderstorm", 96: "Thunderstorm + Hail", 99: "Thunderstorm + Heavy Hail",
}

def _wmo_to_risk(code: int) -> float:
    """Converts a WMO weather code to a simple 0–1 weather risk indicator.
    This is a heuristic only and is clearly labelled as such in the UI."""
    if code in (0, 1):
        return 0.05
    elif code in (2, 3):
        return 0.15
    elif code in (45, 48):
        return 0.35
    elif code in (51, 53, 55, 61, 63):
        return 0.45
    elif code in (65, 71, 73, 80, 81):
        return 0.60
    elif code in (75, 82, 95):
        return 0.75
    elif code in (96, 99):
        return 0.90
    else:
        return 0.20


@st.cache_data(ttl=360)   # Cache ~6 minutes per location to avoid hammering API
def get_live_weather(lat: float, lon: float) -> dict:
    """
    Fetches current weather from the Open-Meteo free API (no API key required).
    Returns a dict with available fields, or a fallback dict on any failure.
    Cached for ~6 minutes to prevent excessive API calls.
    """
    fallback = {
        "available":     False,
        "temperature_c": None,
        "precipitation_mm": None,
        "wind_speed_kmh": None,
        "weather_code":  None,
        "weather_desc":  "Unavailable",
        "weather_risk":  None,
        "source":        "UNAVAILABLE",
        "fetched_at":    datetime.datetime.utcnow().isoformat(),
    }

    if not _REQUESTS_AVAILABLE:
        fallback["source"] = "requests library not installed"
        return fallback

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,precipitation,wind_speed_10m,weather_code"
        "&timezone=auto&forecast_days=1"
    )
    try:
        resp = _requests_lib.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        curr = data.get("current", {})
        code = curr.get("weather_code", 0)
        return {
            "available":       True,
            "temperature_c":   curr.get("temperature_2m"),
            "precipitation_mm": curr.get("precipitation"),
            "wind_speed_kmh":  curr.get("wind_speed_10m"),
            "weather_code":    code,
            "weather_desc":    WMO_CODE_MAP.get(int(code) if code is not None else 0, "Unknown"),
            "weather_risk":    _wmo_to_risk(int(code) if code is not None else 0),
            "source":          "LIVE API (Open-Meteo)",
            "fetched_at":      datetime.datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        fallback["source"] = f"API error: {exc}"
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATED OPERATIONAL EVENTS
# Modifies only a COPY of the shipment record; never touches df_main.
# Changes are deterministic (not random on every rerun).
# ─────────────────────────────────────────────────────────────────────────────
LIVE_EVENTS = [
    "Normal Operations",
    "Supplier Reliability Drop",
    "Inventory Pressure",
    "Port Congestion Increase",
    "Route Risk Increase",
    "Carrier Reliability Drop",
    "Severe Weather Event",
    "Equipment Availability Drop",
]

def apply_live_event(record: dict, event_name: str, weather: dict) -> dict:
    """
    Returns a COPY of `record` with simulated operational changes applied.
    The original `record` is NEVER modified.
    Only modifies fields that exist in the record AND are part of the
    existing XGBoost feature pipeline.
    """
    live = record.copy()

    if event_name == "Normal Operations":
        pass  # No changes — baseline

    elif event_name == "Supplier Reliability Drop":
        if "Supplier_Reliability_Score" in live:
            live["Supplier_Reliability_Score"] = max(0.0, float(live["Supplier_Reliability_Score"]) - 0.35)

    elif event_name == "Inventory Pressure":
        if "Inventory_Level" in live:
            live["Inventory_Level"] = max(0.0, float(live["Inventory_Level"]) * 0.40)
        if "Safety_Stock" in live:
            live["Safety_Stock"] = max(0.0, float(live["Safety_Stock"]) * 0.50)

    elif event_name == "Port Congestion Increase":
        if "Port_Congestion_Level" in live:
            live["Port_Congestion_Level"] = "Critical"

    elif event_name == "Route Risk Increase":
        if "Route_Risk_Level" in live:
            live["Route_Risk_Level"] = "High"
        if "Geopolitical_Risk_Score" in live:
            live["Geopolitical_Risk_Score"] = min(1.0, float(live["Geopolitical_Risk_Score"]) + 0.40)

    elif event_name == "Carrier Reliability Drop":
        if "Carrier_Reliability_Score" in live:
            live["Carrier_Reliability_Score"] = max(0.0, float(live["Carrier_Reliability_Score"]) - 0.30)

    elif event_name == "Severe Weather Event":
        if "Weather_Risk_Score" in live:
            live["Weather_Risk_Score"] = min(1.0, float(live["Weather_Risk_Score"]) + 0.40)
        if "Weather_Condition" in live:
            live["Weather_Condition"] = "Storm"

    elif event_name == "Equipment Availability Drop":
        if "Handling_Equipment_Availability" in live:
            live["Handling_Equipment_Availability"] = max(0.0, float(live["Handling_Equipment_Availability"]) * 0.45)
        if "Capacity_Utilization" in live:
            live["Capacity_Utilization"] = min(100.0, float(live["Capacity_Utilization"]) + 20.0)

    return live


def calculate_live_risk(explainer_obj, live_record: dict) -> float:
    """Runs XGBoost inference using the EXISTING prepare_feature_row pipeline."""
    try:
        feat_row = explainer_obj.prepare_feature_row(live_record)
        return float(explainer_obj.model.predict_proba(feat_row)[0, 1])
    except Exception:
        return 0.0


def get_risk_level(prob: float) -> tuple:
    """Returns (label, css_class, hex_color) consistent with Pages 4–5."""
    if prob < RISK_LOW_THRESHOLD:
        return "LOW RISK", "risk-low", "#43A047"
    elif prob < RISK_HIGH_THRESHOLD:
        return "MODERATE RISK", "risk-medium", "#FB8C00"
    else:
        return "HIGH RISK", "risk-high", "#E53935"


def calculate_risk_change(baseline: float, current: float) -> float:
    """Returns risk change in percentage points (pp)."""
    return (current - baseline) * 100.0


def append_live_event_log(
    shipment_id: str,
    event: str,
    baseline_prob: float,
    current_prob: float,
    recommended_action: str,
) -> None:
    """Appends a record to the in-memory session-state event log (max 20 entries).
    NEVER writes to any CSV or training dataset."""
    if "live_event_log" not in st.session_state:
        st.session_state["live_event_log"] = []

    entry = {
        "Timestamp":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Shipment_ID":        shipment_id,
        "Event":              event,
        "Baseline_Risk":      f"{baseline_prob * 100:.1f}%",
        "Current_Risk":       f"{current_prob * 100:.1f}%",
        "Risk_Change":        f"{calculate_risk_change(baseline_prob, current_prob):+.1f} pp",
        "Risk_Level":         get_risk_level(current_prob)[0],
        "Recommended_Action": recommended_action,
    }
    log = st.session_state["live_event_log"]
    log.insert(0, entry)          # newest first
    st.session_state["live_event_log"] = log[:20]   # keep latest 20


def predict_impact(impact_artifacts: dict, record: dict) -> dict | None:
    """
    Safely calculates impact predictions using the existing impact_artifacts.
    Returns None if inference cannot be safely completed.
    """
    if impact_artifacts is None:
        return None
    try:
        import numpy as np
        feat_cols        = impact_artifacts["feature_cols"]
        cat_cols         = impact_artifacts["cat_cols"]
        cat_encoded_cols = impact_artifacts["cat_encoded_cols"]
        models           = impact_artifacts["models"]

        row_df  = pd.DataFrame([record])
        # Check all numerical feature cols are present
        missing_num = [c for c in feat_cols if c not in row_df.columns]
        if missing_num:
            return None

        X_num = row_df[feat_cols].copy()
        X_cat = pd.get_dummies(row_df[cat_cols], drop_first=False)
        X_cat = X_cat.reindex(columns=cat_encoded_cols, fill_value=0)
        X_imp = pd.concat([X_num, X_cat], axis=1).astype(np.float32)

        return {
            "delay":    float(models["Delivery_Delay_Days"].predict(X_imp)[0]),
            "shortage": float(models["Inventory_Shortage_Units"].predict(X_imp)[0]),
            "loss":     float(models["Financial_Loss_USD"].predict(X_imp)[0]),
            "score":    float(models["Overall_Impact_Score"].predict(X_imp)[0]),
            "prod_pct": float(models["Production_Impact_Pct"].predict(X_imp)[0]),
        }
    except Exception:
        return None
