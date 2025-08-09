import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Tuple, Dict, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Demo",
    page_icon="🛠️",
    layout="wide"
)


# -----------------------------
# Utility: Reproducible Random
# -----------------------------
st.session_state.setdefault("app_init_time", datetime.utcnow().isoformat())


def get_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# -----------------------------
# Synthetic Data Generation
# -----------------------------
@st.cache_data(show_spinner=False)
def generate_weather(num_days: int, seed: int) -> pd.DataFrame:
    """Generate daily weather with seasonality and noise."""
    rng = get_rng(seed + 17)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=num_days, freq="D")

    day_index = np.arange(num_days)
    # Temperature seasonal pattern: mean 18C, amplitude 10C
    ambient_temp = 18 + 10 * np.sin(2 * np.pi * day_index / 365) + rng.normal(0, 2, num_days)
    humidity = 55 + 20 * np.sin(2 * np.pi * (day_index + 90) / 365) + rng.normal(0, 5, num_days)
    humidity = np.clip(humidity, 15, 100)

    # Rainfall: chance varies with season + noise
    rain_prob = 0.25 + 0.15 * np.sin(2 * np.pi * (day_index + 45) / 365)
    rainfall_mm = rng.binomial(1, np.clip(rain_prob, 0.05, 0.6), num_days) * rng.gamma(2.0, 3.0, num_days)

    weather = pd.DataFrame({
        "date": dates,
        "ambient_temp": ambient_temp,
        "humidity": humidity,
        "rainfall_mm": rainfall_mm,
    })
    return weather


def _simulate_machine_series(
    machine_id: int,
    num_days: int,
    weather: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = get_rng(seed + 1000 + machine_id)

    dates = weather["date"].values
    age_days = np.arange(num_days) + rng.integers(0, 365)

    # Base usage hours with weekly pattern and noise
    weekly = 8 + 2 * np.sin(2 * np.pi * (np.arange(num_days) % 7) / 7)
    usage_hours = weekly + rng.normal(0, 1.0, num_days)
    usage_hours = np.clip(usage_hours, 0, 24)

    # Internal temp depends on ambient + usage
    internal_temp = weather["ambient_temp"].values + 0.9 * usage_hours + rng.normal(0, 1.5, num_days)

    # Vibration depends on usage, age, humidity
    vibration = 0.06 * usage_hours + 0.0008 * age_days + 0.01 * (weather["humidity"].values - 50)
    vibration += rng.normal(0, 0.05, num_days)

    # Failure probability via logistic function
    z = (
        -6.0
        + 0.25 * (usage_hours - 8)
        + 0.05 * (internal_temp - 35)
        + 8.0 * np.maximum(vibration - 1.0, 0.0)
        + 0.012 * (age_days - 200)
        + 0.012 * (weather["humidity"].values - 60)
        + 0.03 * (weather["rainfall_mm"].values > 0).astype(float)
    )
    p_fail = 1 / (1 + np.exp(-z))

    breakdown = rng.binomial(1, np.clip(p_fail, 0.0, 0.85), num_days)

    df = pd.DataFrame({
        "date": dates,
        "machine_id": machine_id,
        "age_days": age_days,
        "usage_hours": usage_hours,
        "internal_temp": internal_temp,
        "vibration": vibration,
        "ambient_temp": weather["ambient_temp"].values,
        "humidity": weather["humidity"].values,
        "rainfall_mm": weather["rainfall_mm"].values,
        "breakdown": breakdown,
    })

    # Reactive service the day after a breakdown: reset age partially and reduce vibration
    df["serviced"] = 0
    for i in range(1, num_days):
        if df.loc[i - 1, "breakdown"] == 1:
            df.loc[i, "serviced"] = 1
            df.loc[i:, "age_days"] = df.loc[i:, "age_days"].values - 90
            df.loc[i:, "vibration"] = df.loc[i:, "vibration"].values - 0.15

    df["age_days"] = np.maximum(df["age_days"], 0)
    df["vibration"] = np.maximum(df["vibration"], 0.0)
    return df


@st.cache_data(show_spinner=True)
def generate_dataset(num_machines: int, num_days: int, seed: int) -> pd.DataFrame:
    weather = generate_weather(num_days=num_days, seed=seed)
    frames: List[pd.DataFrame] = []
    for mid in range(1, num_machines + 1):
        frames.append(_simulate_machine_series(mid, num_days, weather, seed))
    data = pd.concat(frames, ignore_index=True)
    data.sort_values(["machine_id", "date"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


# -----------------------------
# Supervised Learning Dataset
# -----------------------------
@st.cache_data(show_spinner=False)
def make_supervised(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime

    # For each machine/day, label 1 if any breakdown in next horizon_days
    df["y_fail_within_h"] = 0
    by_machine = df.groupby("machine_id", group_keys=False)
    def label_group(g: pd.DataFrame) -> pd.DataFrame:
        future_fail = g["breakdown"].rolling(window=horizon_days, min_periods=1).max().shift(-horizon_days + 1)
        g["y_fail_within_h"] = future_fail.fillna(0).astype(int)
        return g
    df = by_machine.apply(label_group)

    # Simple rolling features
    for col in ["usage_hours", "internal_temp", "vibration", "ambient_temp", "humidity", "rainfall_mm"]:
        df[f"{col}_ma7"] = df.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        df[f"{col}_std7"] = df.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))

    # Drop rows at the tail that have no future label
    df = df.groupby("machine_id", group_keys=False).apply(lambda g: g.iloc[:-horizon_days] if len(g) > horizon_days else g.iloc[0:0])

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


# -----------------------------
# Model Training
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_classifier(X: pd.DataFrame, y: pd.Series, seed: int) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")

    metrics = {
        "auc_roc": roc,
        "prevalence_test": float(y_test.mean()),
    }
    return clf, metrics


# -----------------------------
# Forecast next N days for recommendations
# -----------------------------
def forecast_future_features(
    df: pd.DataFrame,
    days_ahead: int,
    seed: int,
) -> pd.DataFrame:
    rng = get_rng(seed + 4242)

    last_date = pd.to_datetime(df["date"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D")

    machines = df["machine_id"].unique()
    latest_by_machine = df.sort_values("date").groupby("machine_id").tail(7)

    frames: List[pd.DataFrame] = []
    for mid in machines:
        g = latest_by_machine[latest_by_machine["machine_id"] == mid].copy()
        # Use last-week averages as baseline, then add small drift and seasonal weather continuation
        baseline = g.mean(numeric_only=True)
        age_last = int(g["age_days"].iloc[-1])
        usage_base = float(baseline.get("usage_hours", 8.0))
        vib_base = float(baseline.get("vibration", 0.6))
        internal_base = float(baseline.get("internal_temp", 40.0))

        # Continue weather seasonally
        day_index_start = (last_date.timetuple().tm_yday) % 365
        ambient = []
        humid = []
        rain = []
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
        vibration = (
            0.06 * usage + 0.0008 * (age_last + np.arange(1, days_ahead + 1)) + 0.01 * (np.array(humid) - 50)
            + rng.normal(0, 0.04, days_ahead)
        )

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

    # Add rolling features based on expanding last 7 days
    full = pd.concat([df, future], ignore_index=True)
    full.sort_values(["machine_id", "date"], inplace=True)
    for col in ["usage_hours", "internal_temp", "vibration", "ambient_temp", "humidity", "rainfall_mm"]:
        full[f"{col}_ma7"] = full.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        full[f"{col}_std7"] = full.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))

    future_enriched = full[full["date"].isin(future_dates)].copy()
    return future_enriched


def compute_recommendations(
    model: RandomForestClassifier,
    feature_cols: List[str],
    future_df: pd.DataFrame,
    risk_threshold: float,
    horizon_days: int,
) -> pd.DataFrame:
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
                "risk_at_recommendation": float(g_sorted["risk"].max()),
                "days_until_service": None,
            })

    return pd.DataFrame(recs)


# -----------------------------
# UI Helpers
# -----------------------------
def kpi(label: str, value, help_text: str | None = None, color: str = "default"):
    st.metric(label, value=value, help=help_text)


def format_percent(p: float) -> str:
    if p != p or p is None:
        return "NA"
    return f"{100*p:.1f}%"


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Configuration")
num_machines = st.sidebar.slider("Number of machines", 1, 25, 8)
num_days = st.sidebar.slider("Historical days", 90, 720, 365, step=15)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
horizon_days = st.sidebar.slider("Prediction horizon (days)", 3, 30, 7)
risk_threshold = st.sidebar.slider("Service risk threshold", 0.1, 0.9, 0.5, 0.05)
forecast_days = st.sidebar.slider("Forecast window (days)", 7, 60, 30)

# -----------------------------
# Data + Model
# -----------------------------
st.sidebar.caption("Generating synthetic data and training model…")
raw = generate_dataset(num_machines=num_machines, num_days=num_days, seed=seed)
X, y, meta, feature_cols = make_supervised(raw, horizon_days=horizon_days)
model, metrics = train_classifier(X, y, seed=seed)

# Risk on historical rows (for charts)
with st.spinner("Scoring historical risk…"):
    hist = raw.copy()
    # Build rolling features aligned to raw
    for col in ["usage_hours", "internal_temp", "vibration", "ambient_temp", "humidity", "rainfall_mm"]:
        hist[f"{col}_ma7"] = hist.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        hist[f"{col}_std7"] = hist.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))
    # Align feature columns and score
    missing_cols = [c for c in feature_cols if c not in hist.columns]
    for c in missing_cols:
        hist[c] = 0.0
    hist["risk"] = model.predict_proba(hist[feature_cols])[:, 1]

future = forecast_future_features(raw, days_ahead=forecast_days, seed=seed)
recs = compute_recommendations(model, feature_cols, future, risk_threshold, horizon_days)

# -----------------------------
# Layout
# -----------------------------
st.title("🛠️ Predictive Maintenance Demo")
st.caption("Synthetic demo showing how predictive maintenance prevents breakdowns. Adjust controls in the sidebar.")

# KPIs row
col1, col2, col3, col4 = st.columns(4)

uptime = 1.0 - raw["breakdown"].mean()
col1.metric("Fleet Uptime", format_percent(uptime))
col2.metric("Total Breakdowns", int(raw["breakdown"].sum()))
col3.metric("AUC-ROC (test)", f"{metrics['auc_roc']:.3f}")
col4.metric("Next 30d: At-risk Machines (≥ thr)", int((recs["recommended_service_date"].notna()).sum()))

st.divider()

# Tabs
overview_tab, machine_tab, modeling_tab, qa_tab = st.tabs([
    "Overview", "Machine Explorer", "Modeling", "Q&A"
])

# -----------------------------
# Overview Tab
# -----------------------------
with overview_tab:
    import altair as alt

    # Fleet breakdowns per day
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

    st.markdown("**Weather effect vs failures**")
    corr_df = raw[["breakdown", "ambient_temp", "humidity", "rainfall_mm", "usage_hours", "vibration", "internal_temp"]].copy()
    corr = corr_df.corr(numeric_only=True)
    corr_long = corr.reset_index().melt(id_vars="index")
    corr_long.columns = ["feature_x", "feature_y", "corr"]
    heat = alt.Chart(corr_long).mark_rect().encode(
        x="feature_x:N", y="feature_y:N",
        color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]))
    )
    st.altair_chart(heat, use_container_width=True)

# -----------------------------
# Machine Explorer Tab
# -----------------------------
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

    # Time series: usage, vibration, internal temp, risk, breakdown markers
    base = alt.Chart(g_hist).encode(x="date:T")

    line_usage = base.mark_line(color="#2ecc71").encode(y=alt.Y("usage_hours:Q", axis=alt.Axis(title="Usage Hours")))
    line_vib = base.mark_line(color="#f39c12").encode(y=alt.Y("vibration:Q", axis=alt.Axis(title="Vibration")))
    line_temp = base.mark_line(color="#e67e22").encode(y=alt.Y("internal_temp:Q", axis=alt.Axis(title="Internal Temp (°C)")))
    line_risk = base.mark_line(color="#8e44ad").encode(y=alt.Y("risk:Q", axis=alt.Axis(title="Risk (0-1)")))
    points_fail = alt.Chart(g).mark_point(color="#c0392b", size=60).encode(x="date:T", y=alt.value(0), tooltip=["date", "breakdown"]).transform_filter("datum.breakdown == 1")

    st.altair_chart((line_usage + line_vib + line_temp + line_risk + points_fail).resolve_scale(y="independent"), use_container_width=True)

    st.markdown("**Forecasted risk (next days)**")
    gf = g_future.copy()
    gf_features = gf.copy()
    Xf = gf_features[[c for c in feature_cols if c in gf_features.columns]]
    gf["risk"] = model.predict_proba(Xf)[:, 1]

    base_f = alt.Chart(gf).encode(x="date:T")
    line_frisk = base_f.mark_line(color="#8e44ad").encode(y="risk:Q")
    rule_thr = alt.Chart(pd.DataFrame({"y": [risk_threshold]})).mark_rule(color="#e74c3c", strokeDash=[6, 6]).encode(y="y:Q")
    st.altair_chart((line_frisk + rule_thr), use_container_width=True)

    st.markdown("**What-if: adjust weather to see risk sensitivity**")
    temp_delta = st.slider("Ambient temperature delta (°C)", -10.0, 10.0, 0.0, 0.5)
    hum_delta = st.slider("Humidity delta (%)", -30.0, 30.0, 0.0, 1.0)

    gf_whatif = gf_features.copy()
    for col in ["ambient_temp", "humidity"]:
        if col == "ambient_temp":
            gf_whatif[col] = gf_whatif[col] + temp_delta
        if col == "humidity":
            gf_whatif[col] = np.clip(gf_whatif[col] + hum_delta, 0, 100)
    # Recompute rolling features influenced by direct columns
    for col in ["ambient_temp", "humidity"]:
        gf_whatif[f"{col}_ma7"] = gf_whatif.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        gf_whatif[f"{col}_std7"] = gf_whatif.groupby("machine_id")[col].transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))

    X_whatif = gf_whatif[[c for c in feature_cols if c in gf_whatif.columns]]
    gf_whatif["risk"] = model.predict_proba(X_whatif)[:, 1]

    base_w = alt.Chart(gf_whatif).encode(x="date:T")
    line_wrisk = base_w.mark_line(color="#d35400").encode(y="risk:Q")
    st.altair_chart((line_frisk + line_wrisk + rule_thr).resolve_scale(y="shared"), use_container_width=True)
    st.caption("Purple: baseline forecast risk; Orange: what-if adjusted weather risk")

# -----------------------------
# Modeling Tab
# -----------------------------
with modeling_tab:
    import altair as alt

    st.subheader("Feature importance")
    try:
        perm = permutation_importance(model, X, y, n_repeats=5, random_state=seed, n_jobs=-1)
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": perm.importances_mean})
    except Exception:
        # Fallback to impurity-based importance
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})

    imp_df = imp_df.sort_values("importance", ascending=False)
    st.altair_chart(
        alt.Chart(imp_df.head(20)).mark_bar().encode(x="importance:Q", y=alt.Y("feature:N", sort="-x")),
        use_container_width=True,
    )

    st.subheader("Confusion matrix (threshold=0.5 on test split)")
    # Build quick test split to show CM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    y_hat = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_hat)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).reset_index().melt(id_vars="index")
    cm_df.columns = ["actual", "predicted", "count"]
    st.altair_chart(
        alt.Chart(cm_df).mark_rect().encode(x="predicted:N", y="actual:N", color="count:Q", tooltip=["actual", "predicted", "count"]),
        use_container_width=True,
    )

    st.markdown(
        f"AUC-ROC on held-out test set: **{metrics['auc_roc']:.3f}** (prevalence: {metrics['prevalence_test']:.3f})."
    )

# -----------------------------
# Q&A Tab
# -----------------------------
with qa_tab:
    st.subheader("Forward-looking questions")

    st.markdown("**When should a machine be serviced next?**")
    st.write("Recommendations are based on when predicted failure risk exceeds the chosen threshold within the forecast window.")
    st.dataframe(recs.sort_values(["recommended_service_date"], na_position="last"), use_container_width=True)

    st.markdown("**Does weather play a part in breakdowns?**")
    # Summarize relative importance of weather features vs. operational
    try:
        perm = permutation_importance(model, X, y, n_repeats=5, random_state=seed, n_jobs=-1)
        imp = pd.DataFrame({"feature": feature_cols, "importance": perm.importances_mean})
    except Exception:
        imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})

    imp["group"] = imp["feature"].apply(lambda f: "weather" if any(k in f for k in ["ambient_temp", "humidity", "rainfall"]) else "operational")
    grp = imp.groupby("group")["importance"].sum()
    weather_share = float(grp.get("weather", 0.0)) / max(float(grp.sum()), 1e-9)
    st.write(f"Weather explains approximately {weather_share*100:.1f}% of the model's importance mass in this run.")

    st.markdown("Adjust the sidebar seed and parameters to see how sensitivity changes across scenarios.")
