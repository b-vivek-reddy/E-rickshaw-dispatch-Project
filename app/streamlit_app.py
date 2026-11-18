"""
AI-Based Campus E-Rickshaw Demand & Intelligent Dispatch System — Deployment App
==================================================================================
Mandatory Component 10 (Deployment): a single Streamlit app with three role-based
views (Student, Driver, Admin) so the full system can be demonstrated live, instead
of shipping three separate apps.

Run with:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import RAW_DIR, MODELS_DIR, FIGURES_DIR, load_locations, zone_distance_matrix  # noqa: E402
from dispatch import DispatchEngine  # noqa: E402

st.set_page_config(page_title="Campus E-Rickshaw Dispatch", page_icon="🛺", layout="wide")

# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_everything():
    locations = load_locations()
    zone_names, dist_matrix = zone_distance_matrix(locations)
    drivers = pd.read_csv(RAW_DIR / "drivers.csv")
    rides = pd.read_csv(RAW_DIR / "ride_requests.csv", parse_dates=["timestamp", "date"])
    engine = DispatchEngine()
    demand_model = joblib.load(MODELS_DIR / "demand_model.joblib")
    avail_model = joblib.load(MODELS_DIR / "availability_model.joblib")
    return locations, zone_names, dist_matrix, drivers, rides, engine, demand_model, avail_model


locations, zone_names, dist_matrix, drivers_df, rides_hist, engine, demand_model, avail_model = load_everything()
ZONE_TYPE = locations.set_index("zone_name")["zone_type"].to_dict()


def cyc(val, period):
    return np.sin(2 * np.pi * val / period), np.cos(2 * np.pi * val / period)


# ---------------------------------------------------------------------------
# Sidebar: role selector + shared scenario controls
# ---------------------------------------------------------------------------
st.sidebar.title("🛺 Campus E-Rickshaw Dispatch")
role = st.sidebar.radio("View as:", ["Student", "Driver", "Admin"])

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario context")
hour = st.sidebar.slider("Hour of day", 0, 23, 13)
weekday_name = st.sidebar.selectbox("Day of week",
                                     ["Monday", "Tuesday", "Wednesday", "Thursday",
                                      "Friday", "Saturday", "Sunday"], index=0)
weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(weekday_name)
is_weekend = weekday >= 5
is_exam = st.sidebar.checkbox("Exam period", value=False)
is_raining = st.sidebar.checkbox("Raining", value=False)
st.sidebar.caption("These controls simulate the live context a real deployment would "
                    "read from the system clock + a weather/calendar feed.")


def time_block_of(h):
    if h >= 23 or h < 6:
        return "late_night"
    if 6 <= h < 8:
        return "early_morning"
    if 8 <= h < 12:
        return "morning"
    if 12 <= h < 14:
        return "lunch"
    if 14 <= h < 17:
        return "afternoon"
    if 17 <= h < 20:
        return "evening"
    return "night"


# ---------------------------------------------------------------------------
# Simulate "current" driver availability for this scenario (since we don't have a
# live GPS feed, we sample a plausible snapshot using the trained availability model
# applied to the roster, for internal consistency with the training data).
# ---------------------------------------------------------------------------
def simulate_current_drivers():
    hs, hc = cyc(hour, 24)
    rows = []
    rng = np.random.default_rng(hash((hour, weekday, is_exam, is_raining)) % (2**32))
    for _, d in drivers_df.iterrows():
        feat = pd.DataFrame([{
            "hour_sin": hs, "hour_cos": hc, "prev_slot_available": 0.5,
            "is_weekend": int(is_weekend), "is_exam_period": int(is_exam),
            "is_raining": int(is_raining), "shift_type": d["shift_type"],
            "home_zone": d["home_zone"],
        }])
        p_avail = avail_model.predict_proba(feat)[0, 1]
        status = "available" if rng.random() < p_avail else "busy"
        rows.append({"driver_id": d["driver_id"], "home_zone": d["home_zone"],
                      "shift_type": d["shift_type"], "status": status, "p_available": round(p_avail, 2)})
    return pd.DataFrame(rows)


current_drivers = simulate_current_drivers()
available_now = current_drivers[current_drivers["status"] == "available"][["driver_id", "home_zone"]]

# =============================================================================
# STUDENT VIEW
# =============================================================================
if role == "Student":
    st.title("🎓 Student — Request a Ride")
    st.caption("Request an e-rickshaw and get an ML-predicted wait time before you book.")

    col1, col2 = st.columns(2)
    pickup_zone = col1.selectbox("Pickup location", zone_names, index=zone_names.index("Central Library"))
    dropoff_options = [z for z in zone_names if z != pickup_zone]
    dropoff_zone = col2.selectbox("Drop-off location", dropoff_options,
                                   index=dropoff_options.index("Boys Hostel Zone 1")
                                   if "Boys Hostel Zone 1" in dropoff_options else 0)

    if st.button("🔍 Request Ride", type="primary"):
        block = time_block_of(hour)
        pu_type = ZONE_TYPE[pickup_zone]
        do_type = ZONE_TYPE[dropoff_zone]
        i, j = zone_names.index(pickup_zone), zone_names.index(dropoff_zone)
        distance_km = float(dist_matrix[i, j])

        nearby = len(available_now[available_now["home_zone"].map(
            lambda z: dist_matrix[zone_names.index(z), i] <= 1.5)])
        zone_hist = rides_hist[rides_hist["pickup_zone"] == pickup_zone]
        rolling_wait = zone_hist["wait_time_minutes"].mean()
        rolling_fail = zone_hist["request_failed"].mean()

        hs, hc = cyc(hour, 24); ws, wc = cyc(weekday, 7)
        req_feat = pd.DataFrame([{
            "hour_sin": hs, "hour_cos": hc, "weekday_sin": ws, "weekday_cos": wc,
            "distance_km": distance_km, "nearby_available_drivers": nearby,
            "rolling_zone_wait_avg": rolling_wait, "rolling_zone_fail_rate": rolling_fail,
            "is_weekend": int(is_weekend), "is_exam_period": int(is_exam), "is_raining": int(is_raining),
            "pickup_zone_type": pu_type, "dropoff_zone_type": do_type, "time_block": block,
        }])

        failure_risk = engine.predict_failure_risk(req_feat)[0]
        result = engine.match_driver(pickup_zone, available_now, failure_risk=failure_risk)

        st.markdown("---")
        if result["matched"]:
            eta_pred = engine.predict_eta(req_feat)[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", "✅ Driver Found")
            c2.metric("Predicted Wait Time", f"{eta_pred:.1f} min")
            c3.metric("Assigned Driver", result["driver_id"])
            st.success(f"Matched within {result['radius_used_km']} km radius "
                       f"({result['expansions_used']} radius expansion(s) needed). "
                       f"Distance to destination: {distance_km:.2f} km.")
        else:
            st.error(f"⚠️ No driver available right now within the maximum search radius "
                     f"({result['radius_used_km']} km). Estimated failure risk was "
                     f"{failure_risk:.0%} for this request. Please try again shortly, "
                     f"or walk to a nearby pickup point with better coverage.")
        st.caption(f"Model-estimated a-priori failure risk for this request: {failure_risk:.0%} "
                   f"(used to decide the starting search radius — see Admin view for how this "
                   f"model was trained).")

    st.markdown("---")
    st.subheader("Live driver snapshot near you")
    st.dataframe(current_drivers.sort_values("status"), width='stretch', hide_index=True)

# =============================================================================
# DRIVER VIEW
# =============================================================================
elif role == "Driver":
    st.title("🧑‍✈️ Driver — Nearby Requests & Status")
    my_driver_id = st.sidebar.selectbox("You are:", drivers_df["driver_id"].tolist())
    my_row = drivers_df[drivers_df["driver_id"] == my_driver_id].iloc[0]
    my_status_row = current_drivers[current_drivers["driver_id"] == my_driver_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Home Zone", my_row["home_zone"])
    c2.metric("Current Status", my_status_row["status"].capitalize())
    c3.metric("Model P(available) this slot", f"{my_status_row['p_available']:.0%}")

    st.markdown("---")
    st.subheader("Nearby demand right now (simulated incoming requests for this hour)")
    i = zone_names.index(my_row["home_zone"])
    nearby_zones = [z for z, d in zip(zone_names, dist_matrix[i]) if d <= 2.0]
    block = time_block_of(hour)
    demand_estimates = []
    for z in nearby_zones:
        hs, hc = cyc(hour, 24); ws, wc = cyc(weekday, 7)
        zone_hist = rides_hist[(rides_hist["pickup_zone"] == z) & (rides_hist["hour"] == hour)]
        lag1 = zone_hist["request_id"].count() / max(rides_hist["date"].nunique(), 1)
        feat = pd.DataFrame([{
            "hour_sin": hs, "hour_cos": hc, "weekday_sin": ws, "weekday_cos": wc,
            "lag_1h_same_zone": lag1, "lag_24h_same_zone_hour": lag1,
            "rolling7_avg_zone_hour": lag1, "is_weekend": int(is_weekend),
            "is_exam_period": int(is_exam), "is_raining": int(is_raining),
            "pickup_zone": z, "pickup_zone_type": ZONE_TYPE[z],
        }])
        pred = max(0, demand_model.predict(feat)[0])
        demand_estimates.append({"Zone": z, "Distance (km)": round(float(dist_matrix[i, zone_names.index(z)]), 2),
                                  "Predicted requests (next hr)": round(pred, 2)})
    demand_df_view = pd.DataFrame(demand_estimates).sort_values("Predicted requests (next hr)", ascending=False)
    st.dataframe(demand_df_view, width='stretch', hide_index=True)
    st.caption("Ranks nearby zones by predicted near-term demand, so you can position "
               "yourself before requests come in rather than waiting idle.")

# =============================================================================
# ADMIN VIEW
# =============================================================================
else:
    st.title("🛠️ Admin — Operations Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Demand Heatmap", "Failed-Request / Route Analysis",
                                       "Model Performance", "Explainability (SHAP)"])

    with tab1:
        st.subheader("Historical demand by zone and hour")
        pivot = rides_hist.pivot_table(index="pickup_zone", columns="hour",
                                        values="request_id", aggfunc="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3)
        ax.set_title("Ride Requests by Pickup Zone × Hour of Day (full history)")
        st.pyplot(fig)

        st.subheader("Live current availability")
        st.dataframe(current_drivers, width='stretch', hide_index=True)
        st.metric("Drivers available right now", int((current_drivers["status"] == "available").sum()),
                   f"of {len(current_drivers)} total")

    with tab2:
        st.subheader("Worst-served pickup → dropoff routes")
        worst_routes = engine.route_analysis(rides_hist, top_n=15)
        st.dataframe(worst_routes, width='stretch', hide_index=True)
        st.caption("`service_score` blends failure rate (70%) and normalized average wait "
                   "time (30%) — these routes should be prioritized for additional driver "
                   "coverage or a dedicated shuttle.")

        st.subheader("Failed-request rate by pickup zone")
        fail_by_zone = rides_hist.groupby("pickup_zone")["request_failed"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        fail_by_zone.plot(kind="barh", ax=ax, color="#C44E52")
        ax.set_xlabel("Failure rate")
        st.pyplot(fig)

        st.subheader("Dispatch policy comparison")
        st.markdown("Evaluated on the held-out test period (`src/dispatch.py :: evaluate_policy`):")
        st.table(pd.DataFrame({
            "Policy": ["Static Radius (baseline, used to generate historical data)",
                       "ML-Informed Dynamic Radius (this system)"],
            "Fulfilment Rate": ["92.2%", "93.6%"],
        }))

    with tab3:
        st.subheader("Model comparison & final test metrics")
        import json
        model_files = {
            "Demand Prediction (regression)": "demand_model_metrics.json",
            "Driver Availability (classification)": "availability_model_metrics.json",
            "ETA Prediction (regression)": "eta_model_metrics.json",
            "Failed-Request Prediction (classification)": "failed_model_metrics.json",
        }
        chosen = st.selectbox("Select model:", list(model_files.keys()))
        with open(MODELS_DIR / model_files[chosen]) as f:
            m = json.load(f)
        st.markdown(f"**Best model selected:** `{m['best_model']}`")
        st.markdown("**Validation comparison across candidate models:**")
        st.dataframe(pd.DataFrame(m["validation_results"]).T, width='stretch')
        st.markdown("**Final held-out test metrics (best model, evaluated once):**")
        st.table(pd.DataFrame([m["test_metrics"]]))

        fig_map = {
            "Demand Prediction (regression)": "demand_model_comparison.png",
            "Driver Availability (classification)": "availability_model_comparison.png",
            "ETA Prediction (regression)": "eta_model_comparison.png",
            "Failed-Request Prediction (classification)": "failed_model_comparison.png",
        }
        st.image(str(FIGURES_DIR / fig_map[chosen]), width='stretch')

    with tab4:
        st.subheader("SHAP feature-importance explanations")
        shap_map = {
            "Demand Prediction": "demand_shap_summary.png",
            "Driver Availability": "availability_shap_summary.png",
            "ETA Prediction": "eta_shap_summary.png",
            "Failed-Request Prediction": "failed_shap_summary.png",
        }
        chosen_shap = st.selectbox("Select model for explanation:", list(shap_map.keys()))
        st.image(str(FIGURES_DIR / shap_map[chosen_shap]), width='stretch')
        st.caption("Each point is one test-set prediction; red = high feature value, "
                   "blue = low. Position on the x-axis shows whether that feature pushed "
                   "the prediction up or down for that case.")

st.sidebar.markdown("---")
st.sidebar.caption("INT395 Supervised Learning — Project 04. All data is simulated; see "
                    "`data/README.md` for full documentation of assumptions and limitations.")
