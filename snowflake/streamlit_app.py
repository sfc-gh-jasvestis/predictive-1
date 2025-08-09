import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import List

from snowflake.snowpark.context import get_active_session
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

st.set_page_config(page_title="Predictive Maintenance (Snowflake)", page_icon="🛠️", layout="wide")

session = get_active_session()

st.sidebar.header("Configuration")
num_machines = st.sidebar.slider("Number of machines (seed)", 1, 25, 8)
num_days = st.sidebar.slider("Historical days (seed)", 90, 720, 365, step=15)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
horizon_days = st.sidebar.slider("Prediction horizon (days)", 3, 30, 7)
risk_threshold = st.sidebar.slider("Service risk threshold", 0.1, 0.9, 0.5, 0.05)
forecast_days = st.sidebar.slider("Forecast window (days)", 7, 60, 30)

if st.sidebar.button("Reseed data in Snowflake"):
    with st.spinner("Seeding data in Snowflake…"):
        res = session.sql(f"CALL SEED_DATA({num_machines}, {num_days}, {seed})").collect()
        st.sidebar.success(str(res[0][0]) if res else "Seeded")

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = session.table("MACHINE_DAILY").to_pandas()
    df["date"] = pd.to_datetime(df["DATE"]) if "DATE" in df.columns else pd.to_datetime(df["date"])  # case handling
    # Normalize column names to lower
    df.columns = [c.lower() for c in df.columns]
    return df

raw = load_data()
if raw.empty:
    st.warning("No data found in MACHINE_DAILY. Use the 'Reseed data' button or call SEED_DATA in a worksheet.")
    st.stop()

# Ensure expected columns
expected_cols = {"date","machine_id","age_days","usage_hours","internal_temp","vibration","ambient_temp","humidity","rainfall_mm","breakdown","serviced"}
missing = expected_cols - set(raw.columns)
if missing:
    st.error(f"Missing expected columns in MACHINE_DAILY: {missing}")
    st.stop()

# Build rolling features and labels
@st.cache_data(show_spinner=False)
def make_supervised(df: pd.DataFrame, horizon_days: int):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    df.sort_values(["machine_id", "date"], inplace=True)

    df["y_fail_within_h"] = 0
    def label_machine(g: pd.DataFrame) -> pd.DataFrame:
        future_fail = g["breakdown"].rolling(window=horizon_days, min_periods=1).max().shift(-horizon_days + 1)
        g["y_fail_within_h"] = future_fail.fillna(0).astype(int)
        return g
    df = df.groupby("machine_id", as_index=False, group_keys=False).apply(label_machine)
    df = df.groupby("machine_id", group_keys=False).apply(lambda g: g.iloc[:-horizon_days] if len(g) > horizon_days else g.iloc[0:0])

    for col in ["usage_hours", "internal_temp", "vibration", "ambient_temp", "humidity", "rainfall_mm"]:
        df[f"{col}_ma7"] = df.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        df[f"{col}_std7"] = df.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))

    feature_cols = [
        "age_days", "usage_hours", "internal_temp", "vibration",
        "ambient_temp", "humidity", "rainfall_mm",
        "usage_hours_ma7", "internal_temp_ma7", "vibration_ma7",
        "ambient_temp_ma7", "humidity_ma7", "rainfall_mm_ma7",
        "usage_hours_std7", "internal_temp_std7", "vibration_std7",
        "ambient_temp_std7", "humidity_std7", "rainfall_mm_std7",
    ]
    X = df[feature_cols]
    y = df["y_fail_within_h"].astype(int)
    meta = df[["machine_id", "date", "breakdown", "serviced"]].reset_index(drop=True)
    return X, y, meta, feature_cols

X, y, meta, feature_cols = make_supervised(raw, horizon_days=horizon_days)

@st.cache_resource(show_spinner=True)
def train_classifier(X: pd.DataFrame, y: pd.Series, seed: int):
    if len(X) < 50 or len(np.unique(y)) < 2:
        return None, {"auc_roc": float("nan"), "prevalence_test": float(y.mean()) if len(y) else float("nan")}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, min_samples_split=4, min_samples_leaf=2, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
    return clf, {"auc_roc": roc, "prevalence_test": float(y_test.mean())}

model, metrics = train_classifier(X, y, seed=seed)

# Score historical risk
hist = raw.copy()
for col in ["usage_hours", "internal_temp", "vibration", "ambient_temp", "humidity", "rainfall_mm"]:
    hist[f"{col}_ma7"] = hist.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
    hist[f"{col}_std7"] = hist.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))
missing_cols = [c for c in feature_cols if c not in hist.columns]
for c in missing_cols:
    hist[c] = 0.0
if model is not None:
    hist["risk"] = model.predict_proba(hist[feature_cols])[:, 1]
else:
    hist["risk"] = 0.0

# Forecast future features for recommendations
def forecast_future_features(df: pd.DataFrame, days_ahead: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 4242)
    last_date = pd.to_datetime(df["date"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D")

    machines = df["machine_id"].unique()
    latest_by_machine = df.sort_values("date").groupby("machine_id").tail(7)

    frames: List[pd.DataFrame] = []
    for mid in machines:
        g = latest_by_machine[latest_by_machine["machine_id"] == mid].copy()
        baseline = g.mean(numeric_only=True)
        age_last = int(g["age_days"].iloc[-1])
        usage_base = float(baseline.get("usage_hours", 8.0))
        ambient = []
        humid = []
        rain = []
        day_index_start = (last_date.timetuple().tm_yday) % 365
        for i in range(days_ahead):
            di = (day_index_start + i + 1) % 365
            temp = 18 + 10 * math.sin(2 * math.pi * di / 365) + rng.normal(0, 1.2)
            h = 55 + 20 * math.sin(2 * math.pi * (di + 90) / 365) + rng.normal(0, 3.0)
            h = float(np.clip(h, 15, 100))
            rp = 0.25 + 0.15 * math.sin(2 * math.pi * (di + 45) / 365)
            r = (rng.random() < np.clip(rp, 0.05, 0.6)) * rng.gamma(2.0, 2.5)
            ambient.append(temp)
            humid.append(h)
            rain.append(float(r))
        usage = np.clip(rng.normal(usage_base, 0.6, days_ahead), 0, 24)
        internal = np.array(ambient) + 0.9 * usage + rng.normal(0, 1.0, days_ahead)
        vibration = 0.06 * usage + 0.0008 * (age_last + np.arange(1, days_ahead + 1)) + 0.01 * (np.array(humid) - 50) + rng.normal(0, 0.04, days_ahead)
        f = pd.DataFrame({
            "date": future_dates,
            "machine_id": mid,
            "age_days": age_last + np.arange(1, days_ahead + 1),
            "usage_hours": usage,
            "internal_temp": internal,
            "vibration": vibration,
            "ambient_temp": ambient,
            "humidity": humid,
            "rainfall_mm": rain,
            "breakdown": 0,
            "serviced": 0,
        })
        frames.append(f)
    future = pd.concat(frames, ignore_index=True)
    full = pd.concat([df, future], ignore_index=True)
    full.sort_values(["machine_id", "date"], inplace=True)
    for col in ["usage_hours", "internal_temp", "vibration", "ambient_temp", "humidity", "rainfall_mm"]:
        full[f"{col}_ma7"] = full.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        full[f"{col}_std7"] = full.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))
    future_enriched = full[full["date"].isin(future["date"])].copy()
    return future_enriched

future = forecast_future_features(raw, days_ahead=forecast_days, seed=seed)

def compute_recommendations(model, feature_cols, future_df: pd.DataFrame, risk_threshold: float) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame({"machine_id": sorted(raw["machine_id"].unique()), "recommended_service_date": None, "risk_at_recommendation": 0.0, "days_until_service": None})
    X_future = future_df[feature_cols]
    future_df = future_df.copy()
    future_df["risk"] = model.predict_proba(X_future)[:, 1]
    recs = []
    for mid, g in future_df.groupby("machine_id"):
        g_sorted = g.sort_values("date")
        above = g_sorted[g_sorted["risk"] >= risk_threshold]
        if len(above) > 0:
            first_date = pd.to_datetime(above.iloc[0]["date"]).date()
            first_risk = float(above.iloc[0]["risk"])
            recs.append({
                "machine_id": mid,
                "recommended_service_date": str(first_date),
                "risk_at_recommendation": first_risk,
                "days_until_service": int((pd.to_datetime(first_date) - pd.to_datetime(g_sorted.iloc[0]["date"]).normalize()).days),
            })
        else:
            recs.append({
                "machine_id": mid,
                "recommended_service_date": None,
                "risk_at_recommendation": float(g_sorted["risk"].max()) if "risk" in g_sorted else 0.0,
                "days_until_service": None,
            })
    return pd.DataFrame(recs)

recs = compute_recommendations(model, feature_cols, future, risk_threshold)

st.title("🛠️ Predictive Maintenance (Snowflake)")
st.caption("Synthetic data stored in Snowflake. Train a simple model and derive forward-looking service recommendations.")

col1, col2, col3, col4 = st.columns(4)
uptime = 1.0 - raw["breakdown"].mean()
col1.metric("Fleet Uptime", f"{100*uptime:.1f}%")
col2.metric("Total Breakdowns", int(raw["breakdown"].sum()))
col3.metric("AUC-ROC (test)", f"{metrics['auc_roc']:.3f}")
col4.metric("At-risk Machines (next window)", int((recs["recommended_service_date"].notna()).sum()))

st.divider()

overview_tab, machine_tab = st.tabs(["Overview", "Machine Explorer"]) 

with overview_tab:
    import altair as alt
    daily = raw.groupby("date", as_index=False)["breakdown"].sum().rename(columns={"breakdown": "breakdowns"})
    weather_daily = raw.groupby("date", as_index=False).agg({"ambient_temp": "mean", "humidity": "mean"})
    daily = daily.merge(weather_daily, on="date", how="left")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Breakdowns over time (fleet)")
        base = alt.Chart(daily).encode(x="date:T")
        bars = base.mark_bar(opacity=0.6, color="#e74c3c").encode(y="breakdowns:Q")
        line = base.mark_line(color="#2980b9").encode(y=alt.Y("ambient_temp:Q", axis=alt.Axis(title="Ambient Temp (°C)")))
        st.altair_chart((bars + line).resolve_scale(y="independent"), use_container_width=True)
    with right:
        st.subheader("Upcoming service recommendations")
        show_recs = recs.copy()
        show_recs["recommended_service_date"] = show_recs["recommended_service_date"].fillna("—")
        st.dataframe(show_recs.sort_values(["recommended_service_date"], na_position="last"), use_container_width=True)

with machine_tab:
    import altair as alt
    machine_ids = sorted(raw["machine_id"].unique().tolist())
    selected = st.selectbox("Choose a machine", machine_ids, index=0)
    g = raw[raw["machine_id"] == selected].copy()
    g_hist = hist[hist["machine_id"] == selected].copy()
    g_future = future[future["machine_id"] == selected].copy()

    st.subheader(f"Machine {selected}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Breakdowns (history)", int(g["breakdown"].sum()))
    k2.metric("Last Service (reactive)", str(pd.to_datetime(g[g["serviced"] == 1]["date"].max()).date()) if (g["serviced"] == 1).any() else "—")
    g_rec = recs[recs["machine_id"] == selected]
    next_service = g_rec.iloc[0]["recommended_service_date"] if len(g_rec) else None
    k3.metric("Recommended Service", next_service or "—")
    if next_service:
        days_until = g_rec.iloc[0]["days_until_service"]
        k4.metric("Days until service", int(days_until))
    else:
        k4.metric("Days until service", "—")

    base = alt.Chart(g_hist).encode(x="date:T")
    line_usage = base.mark_line(color="#2ecc71").encode(y=alt.Y("usage_hours:Q", axis=alt.Axis(title="Usage Hours")))
    line_vib = base.mark_line(color="#f39c12").encode(y=alt.Y("vibration:Q", axis=alt.Axis(title="Vibration")))
    line_temp = base.mark_line(color="#e67e22").encode(y=alt.Y("internal_temp:Q", axis=alt.Axis(title="Internal Temp (°C)")))
    line_risk = base.mark_line(color="#8e44ad").encode(y=alt.Y("risk:Q", axis=alt.Axis(title="Risk (0-1)")))
    points_fail = alt.Chart(g).mark_point(color="#c0392b", size=60).encode(x="date:T", y=alt.value(0)).transform_filter("datum.breakdown == 1")
    st.altair_chart((line_usage + line_vib + line_temp + line_risk + points_fail).resolve_scale(y="independent"), use_container_width=True)

    st.markdown("**Forecasted risk (next days)**")
    base_f = alt.Chart(g_future).encode(x="date:T")
    if model is not None:
        Xf = g_future[[c for c in feature_cols if c in g_future.columns]]
        g_future = g_future.copy()
        g_future["risk"] = model.predict_proba(Xf)[:, 1]
    line_frisk = base_f.mark_line(color="#8e44ad").encode(y="risk:Q")
    rule_thr = alt.Chart(pd.DataFrame({"y": [risk_threshold]})).mark_rule(color="#e74c3c", strokeDash=[6, 6]).encode(y="y:Q")
    st.altair_chart((line_frisk + rule_thr), use_container_width=True)
