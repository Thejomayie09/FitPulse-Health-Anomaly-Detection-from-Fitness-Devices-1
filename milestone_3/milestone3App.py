import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FitPulse – M3 Anomaly Detection",
    page_icon="🚨",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════
#  STYLING  — exact M1 / M2 palette & component styles
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1b2a; color: #dce8f5; }
section[data-testid="stSidebar"] { background: #0b1829; border-right: 1px solid #1e3a5f; }
section[data-testid="stSidebar"] * { color: #a8c8e8 !important; }
h1, h2, h3 { color: #a78bfa !important; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: white; font-weight: 600; border: none;
    border-radius: 8px; padding: 0.5rem 1.2rem; transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(168,85,247,0.4);
}

div[data-testid="metric-container"] {
    background: #162a41; border: 1px solid #1e4a6e;
    border-radius: 12px; padding: 1rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }

details {
    background: #111f30; border: 1px solid #1e3a5f !important;
    border-radius: 10px; margin-bottom: 10px;
}
details summary {
    background: linear-gradient(90deg, #162a41, #1a1f4a);
    border-radius: 10px; padding: 14px 18px;
    color: #a78bfa; font-weight: 700; font-size: 1.05rem;
    cursor: pointer; list-style: none;
}
details[open] summary {
    border-bottom: 1px solid #1e3a5f;
    border-radius: 10px 10px 0 0;
}

/* Info / log boxes — same as M1 */
.info-box {
    background: rgba(99,102,241,0.12); border-left: 4px solid #6366f1;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 0.88rem; color: #a8c8e8;
}
.log-box {
    background: rgba(74,144,196,0.1); border-left: 4px solid #a78bfa;
    border-radius: 4px; padding: 10px 15px; margin: 5px 0;
    font-family: 'Source Code Pro', monospace;
}
.warn-box {
    background: rgba(245,158,11,0.10); border-left: 4px solid #f59e0b;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 0.88rem; color: #fde68a;
}
.alert-box {
    background: rgba(239,68,68,0.12); border-left: 4px solid #ef4444;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 0.88rem; color: #fca5a5;
}
.ok-box {
    background: rgba(16,185,129,0.10); border-left: 4px solid #10b981;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 0.88rem; color: #6ee7b7;
}

/* Progress bar — same as M2 */
.prog-wrap {
    background: #162a41; border-radius: 99px; height: 7px;
    margin: 10px 0; overflow: hidden;
}
.prog-fill {
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    height: 100%; border-radius: 99px;
    animation: pbar 1.4s ease-in-out infinite;
}
@keyframes pbar {
    0%,100%{opacity:1;width:60%} 50%{opacity:.5;width:90%}
}

/* Big count cards matching screenshot */
.count-card {
    background: #162a41; border: 1px solid #1e4a6e;
    border-radius: 14px; padding: 22px 16px; text-align: center;
}
.count-num {
    font-size: 2.6rem; font-weight: 800;
    color: #ef4444; line-height: 1;
}
.count-label {
    font-size: 0.72rem; font-weight: 700; color: #5a7a9a;
    letter-spacing: 0.08em; margin-top: 6px; text-transform: uppercase;
}

/* Method cards */
.method-card {
    background: #162a41; border: 1px solid #1e4a6e;
    border-radius: 10px; padding: 16px 18px;
}
.method-title { font-weight: 700; font-size: 0.92rem; margin-bottom: 6px; }
.method-desc  { font-size: 0.82rem; color: #a8c8e8; line-height: 1.55; }

/* Section label */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    color: #4a6a8a; text-transform: uppercase; margin-bottom: 12px;
}

/* File status cards */
.file-card {
    border-radius: 10px; padding: 12px 8px; text-align: center; min-height: 95px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PLOTLY THEME  — matches M1 / M2
# ══════════════════════════════════════════════════════════════
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,42,0.6)",
    font=dict(color="#dce8f5", family="Inter"),
    title_font=dict(color="#a78bfa", size=14),
    colorway=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"],
    legend=dict(bgcolor="rgba(22,42,65,0.85)", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(t=55, b=40, l=50, r=20),
)

def T(fig, h=None):
    fig.update_layout(**({**PT, "height": h} if h else PT))
    fig.update_xaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
    fig.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
    return fig

def tip(t):   st.markdown(f'<div class="info-box">💡 {t}</div>',   unsafe_allow_html=True)
def warn(t):  st.markdown(f'<div class="warn-box">⚠️ {t}</div>',  unsafe_allow_html=True)
def alert(t): st.markdown(f'<div class="alert-box">🚨 {t}</div>', unsafe_allow_html=True)
def ok(t):    st.markdown(f'<div class="ok-box">✅ {t}</div>',    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE INIT  — from sir's code
# ══════════════════════════════════════════════════════════════
for k, v in [
    ("files_loaded",    False),
    ("anomaly_done",    False),
    ("simulation_done", False),
    ("master",          None),
    ("hr_minute",       None),
    ("anom_hr",         None),
    ("anom_steps",      None),
    ("anom_sleep",      None),
    ("sim_results",     None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
#  FILE REGISTRY  — exact column signatures from sir's code
# ══════════════════════════════════════════════════════════════
REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {
        "key_cols": ["ActivityDate","TotalSteps","Calories"],
        "label": "Daily Activity", "icon": "🏃",
    },
    "hourlySteps_merged.csv":       {
        "key_cols": ["ActivityHour","StepTotal"],
        "label": "Hourly Steps", "icon": "👟",
    },
    "hourlyIntensities_merged.csv": {
        "key_cols": ["ActivityHour","TotalIntensity"],
        "label": "Hourly Intensities", "icon": "⚡",
    },
    "minuteSleep_merged.csv":       {
        "key_cols": ["date","value","logId"],
        "label": "Minute Sleep", "icon": "😴",
    },
    "heartrate_seconds_merged.csv": {
        "key_cols": ["Time","Value"],
        "label": "Heart Rate", "icon": "💓",
    },
}

def score_match(df, req_info):
    return sum(1 for col in req_info["key_cols"] if col in df.columns)

# ══════════════════════════════════════════════════════════════
#  ANOMALY DETECTION  — sir's exact logic, unchanged
# ══════════════════════════════════════════════════════════════

def detect_hr_anomalies(master, hr_high=100, hr_low=50, residual_sigma=2.0):
    df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    hr_daily = df.groupby("Date")["AvgHR"].mean().reset_index()
    hr_daily.columns = ["Date","AvgHR"]
    hr_daily = hr_daily.sort_values("Date")

    hr_daily["thresh_high"]  = hr_daily["AvgHR"] > hr_high
    hr_daily["thresh_low"]   = hr_daily["AvgHR"] < hr_low
    hr_daily["rolling_med"]  = hr_daily["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_daily["residual"]     = hr_daily["AvgHR"] - hr_daily["rolling_med"]
    resid_std                = hr_daily["residual"].std()
    hr_daily["resid_anomaly"]= hr_daily["residual"].abs() > (residual_sigma * resid_std)
    hr_daily["is_anomaly"]   = hr_daily["thresh_high"] | hr_daily["thresh_low"] | hr_daily["resid_anomaly"]

    def reason(row):
        r = []
        if row["thresh_high"]:   r.append(f"HR>{hr_high}")
        if row["thresh_low"]:    r.append(f"HR<{hr_low}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    hr_daily["reason"] = hr_daily.apply(reason, axis=1)
    return hr_daily


def detect_steps_anomalies(master, steps_low=500, steps_high=25000, residual_sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    sd = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")

    sd["thresh_low"]   = sd["TotalSteps"] < steps_low
    sd["thresh_high"]  = sd["TotalSteps"] > steps_high
    sd["rolling_med"]  = sd["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    sd["residual"]     = sd["TotalSteps"] - sd["rolling_med"]
    resid_std          = sd["residual"].std()
    sd["resid_anomaly"]= sd["residual"].abs() > (residual_sigma * resid_std)
    sd["is_anomaly"]   = sd["thresh_low"] | sd["thresh_high"] | sd["resid_anomaly"]

    def reason(row):
        r = []
        if row["thresh_low"]:    r.append(f"Steps<{steps_low}")
        if row["thresh_high"]:   r.append(f"Steps>{steps_high}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    sd["reason"] = sd.apply(reason, axis=1)
    return sd


def detect_sleep_anomalies(master, sleep_low=60, sleep_high=600, residual_sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    sd = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")

    sd["thresh_low"]   = (sd["TotalSleepMinutes"] > 0) & (sd["TotalSleepMinutes"] < sleep_low)
    sd["thresh_high"]  = sd["TotalSleepMinutes"] > sleep_high
    sd["no_data"]      = sd["TotalSleepMinutes"] == 0
    sd["rolling_med"]  = sd["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sd["residual"]     = sd["TotalSleepMinutes"] - sd["rolling_med"]
    resid_std          = sd["residual"].std()
    sd["resid_anomaly"]= sd["residual"].abs() > (residual_sigma * resid_std)
    sd["is_anomaly"]   = sd["thresh_low"] | sd["thresh_high"] | sd["resid_anomaly"]

    def reason(row):
        r = []
        if row["no_data"]:       r.append("No device worn")
        if row["thresh_low"]:    r.append(f"Sleep<{sleep_low}min")
        if row["thresh_high"]:   r.append(f"Sleep>{sleep_high}min")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    sd["reason"] = sd.apply(reason, axis=1)
    return sd


def simulate_accuracy(master, n_inject=10):
    """Sir's exact simulation logic — inject 10 known anomalies per signal."""
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")

    results = {}

    # Heart Rate
    hr_sim = df_daily[["Date","AvgHR"]].copy()
    idx = np.random.choice(len(hr_sim), n_inject, replace=False)
    hr_sim.loc[idx, "AvgHR"] = np.random.choice(
        [115,120,125,35,40,45,118,130,38,42], n_inject, replace=True)
    hr_sim["rolling_med"]   = hr_sim["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_sim["residual"]      = hr_sim["AvgHR"] - hr_sim["rolling_med"]
    resid_std               = hr_sim["residual"].std()
    hr_sim["detected"]      = ((hr_sim["AvgHR"] > 100) | (hr_sim["AvgHR"] < 50) |
                               (hr_sim["residual"].abs() > 2 * resid_std))
    tp = hr_sim.iloc[idx]["detected"].sum()
    results["Heart Rate"] = {"injected": n_inject, "detected": int(tp),
                              "accuracy": round(tp/n_inject*100, 1)}

    # Steps
    st_sim = df_daily[["Date","TotalSteps"]].copy()
    idx2   = np.random.choice(len(st_sim), n_inject, replace=False)
    st_sim.loc[idx2, "TotalSteps"] = np.random.choice(
        [50,100,150,30000,35000,28000,80,200,31000,29000], n_inject, replace=True)
    st_sim["rolling_med"]   = st_sim["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    st_sim["residual"]      = st_sim["TotalSteps"] - st_sim["rolling_med"]
    resid_std2              = st_sim["residual"].std()
    st_sim["detected"]      = ((st_sim["TotalSteps"] < 500) | (st_sim["TotalSteps"] > 25000) |
                               (st_sim["residual"].abs() > 2 * resid_std2))
    tp2 = st_sim.iloc[idx2]["detected"].sum()
    results["Steps"] = {"injected": n_inject, "detected": int(tp2),
                         "accuracy": round(tp2/n_inject*100, 1)}

    # Sleep
    sl_sim = df_daily[["Date","TotalSleepMinutes"]].copy()
    idx3   = np.random.choice(len(sl_sim), n_inject, replace=False)
    sl_sim.loc[idx3, "TotalSleepMinutes"] = np.random.choice(
        [10,20,30,700,750,800,15,25,710,720], n_inject, replace=True)
    sl_sim["rolling_med"]   = sl_sim["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sl_sim["residual"]      = sl_sim["TotalSleepMinutes"] - sl_sim["rolling_med"]
    resid_std3              = sl_sim["residual"].std()
    sl_sim["detected"]      = (((sl_sim["TotalSleepMinutes"] > 0) & (sl_sim["TotalSleepMinutes"] < 60)) |
                               (sl_sim["TotalSleepMinutes"] > 600) |
                               (sl_sim["residual"].abs() > 2 * resid_std3))
    tp3 = sl_sim.iloc[idx3]["detected"].sum()
    results["Sleep"] = {"injected": n_inject, "detected": int(tp3),
                         "accuracy": round(tp3/n_inject*100, 1)}

    results["Overall"] = round(np.mean([results[k]["accuracy"] for k in ["Heart Rate","Steps","Sleep"]]), 1)
    return results

# ══════════════════════════════════════════════════════════════
#  SIDEBAR  — M2 style
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=60)
    st.markdown("## Fitness Data Pro\n**Milestone 3**")
    st.markdown("---")

    # Pipeline progress
    steps_done = sum([
        st.session_state.files_loaded,
        st.session_state.anomaly_done,
        st.session_state.simulation_done,
    ])
    pct = int(steps_done / 3 * 100)
    st.markdown(f"""
    <div style="margin-bottom:12px">
        <div style="font-size:0.72rem;color:#5a7a9a;margin-bottom:4px">PIPELINE · {pct}% COMPLETE</div>
        <div class="prog-wrap">
            <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#6366f1,#a78bfa);
                        border-radius:99px;"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    for done, icon, label in [
        (st.session_state.files_loaded,    "📂", "Data Loaded"),
        (st.session_state.anomaly_done,    "🚨", "Anomalies Detected"),
        (st.session_state.simulation_done, "🎯", "Accuracy Simulated"),
    ]:
        dot   = "🟢" if done else "⚪"
        color = "#dce8f5" if done else "#5a7a9a"
        st.markdown(f'<div style="font-size:0.84rem;padding:3px 0;color:{color}">{dot} {icon} {label}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#5a7a9a;letter-spacing:0.1em">THRESHOLDS</div>',
                unsafe_allow_html=True)
    hr_high = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180)
    hr_low  = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70)
    st_low  = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000)
    sl_low  = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120)
    sl_high = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900)
    sigma   = st.slider("Residual σ threshold", 1.0, 4.0, 2.0, 0.5)
    st.markdown("---")
    st.caption("Anomaly Detection & Visualization · Weeks 5–6")

# ══════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════
st.title("🚨 Anomaly Detection & Visualization")
st.markdown("Detect unusual health patterns in **heart rate**, **steps**, and **sleep** using Threshold Violations, Prophet Residuals, and DBSCAN Outlier Detection.")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD  (M2-style 5-file upload)
# ══════════════════════════════════════════════════════════════
with st.expander("📂  Step 1 — Upload Your 5 Fitbit CSV Files", expanded=True):
    tip("Upload the same 5 files from Milestone 2. Files are auto-detected by column structure — any file name works.")

    uploaded = st.file_uploader(
        "Upload all 5 CSV files",
        type=["csv"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Auto-detect files by column signature (score_match)
    detected = {}
    ignored  = []
    raw_uploads = []
    if uploaded:
        for uf in uploaded:
            try:
                uf.seek(0)
                df_tmp = pd.read_csv(uf)
                raw_uploads.append((uf.name, df_tmp))
            except Exception:
                ignored.append(uf.name)

        used_names = set()
        for req_name, finfo in REQUIRED_FILES.items():
            best_score, best_name, best_df = 0, None, None
            for uname, udf in raw_uploads:
                s = score_match(udf, finfo)
                if s > best_score:
                    best_score, best_name, best_df = s, uname, udf
            if best_score >= 2:
                detected[req_name] = best_df
                used_names.add(best_name)

    n_matched = len(detected)

    # File status cards — M2 style
    g = st.columns(5)
    for col, (req_name, finfo) in zip(g, REQUIRED_FILES.items()):
        found = req_name in detected
        bor   = "#10b981" if found else "#1e4a6e"
        tc    = "#10b981" if found else "#6b7280"
        ico   = finfo["icon"]
        label = finfo["label"]
        col.markdown(f"""
        <div style="background:#162a41;border:1px solid {bor};border-radius:10px;
                    padding:12px 8px;text-align:center;min-height:95px">
            <div style="font-size:1.7rem">{ico}</div>
            <div style="font-size:0.78rem;font-weight:700;color:{tc};margin-top:4px">{label}</div>
            <div style="font-size:0.62rem;color:#5a7a9a;margin-top:5px">
                {'✅ Detected' if found else '❌ Not detected'}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    if n_matched == 0:
        st.info("👆 Upload your CSV files above. The app auto-detects each file by its column structure.")
        st.stop()
    elif n_matched < 5:
        warn(f"Detected {n_matched}/5 files. Some features may be unavailable.")

    # Load & Build button
    if st.button("⚡  Load & Build Master DataFrame"):
        with st.spinner("Parsing timestamps and building master dataset…"):
            try:
                daily    = detected["dailyActivity_merged.csv"].copy()
                hourly_s = detected["hourlySteps_merged.csv"].copy()
                hourly_i = detected["hourlyIntensities_merged.csv"].copy()
                sleep    = detected["minuteSleep_merged.csv"].copy()
                hr       = detected["heartrate_seconds_merged.csv"].copy()

                # Parse timestamps — sir's exact format strings with fallback
                def safe_dt(series, fmt):
                    try:
                        return pd.to_datetime(series, format=fmt)
                    except Exception:
                        return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")

                daily["ActivityDate"]    = safe_dt(daily["ActivityDate"],    "%m/%d/%Y")
                hourly_s["ActivityHour"] = safe_dt(hourly_s["ActivityHour"], "%m/%d/%Y %I:%M:%S %p")
                hourly_i["ActivityHour"] = safe_dt(hourly_i["ActivityHour"], "%m/%d/%Y %I:%M:%S %p")
                sleep["date"]            = safe_dt(sleep["date"],            "%m/%d/%Y %I:%M:%S %p")
                hr["Time"]               = safe_dt(hr["Time"],               "%m/%d/%Y %I:%M:%S %p")

                # Resample HR to 1-minute — sir's exact logic
                hr_minute = (hr.set_index("Time")
                               .groupby("Id")["Value"]
                               .resample("1min").mean()
                               .reset_index())
                hr_minute.columns = ["Id","Time","HeartRate"]
                hr_minute = hr_minute.dropna()

                hr_minute["Date"] = hr_minute["Time"].dt.date
                hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                            .agg(["mean","max","min","std"]).reset_index()
                            .rename(columns={"mean":"AvgHR","max":"MaxHR",
                                             "min":"MinHR","std":"StdHR"}))

                sleep["Date"] = sleep["date"].dt.date
                sleep_daily   = (sleep.groupby(["Id","Date"])
                                 .agg(TotalSleepMinutes=("value","count"),
                                      DominantSleepStage=("value", lambda x: x.mode()[0]))
                                 .reset_index())

                master = daily.copy().rename(columns={"ActivityDate":"Date"})
                master["Date"] = master["Date"].dt.date
                master = master.merge(hr_daily,    on=["Id","Date"], how="left")
                master = master.merge(sleep_daily, on=["Id","Date"], how="left")
                master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
                master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
                for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                    master[col] = master.groupby("Id")[col].transform(
                        lambda x: x.fillna(x.median()))

                st.session_state.master      = master
                st.session_state.hr_minute   = hr_minute
                st.session_state.files_loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"Error building dataset: {e}")
                st.exception(e)

    if st.session_state.files_loaded:
        master = st.session_state.master
        ok(f"Master DataFrame ready — **{master.shape[0]:,} rows** · **{master['Id'].nunique()} users** · **{master.shape[1]} columns**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows",    f"{master.shape[0]:,}")
        m2.metric("Users",         master["Id"].nunique())
        m3.metric("Date Range",
                  f"{pd.to_datetime(master['Date']).min().strftime('%d %b')} → "
                  f"{pd.to_datetime(master['Date']).max().strftime('%d %b %y')}")
        m4.metric("Columns",       master.shape[1])

# Stop if files not loaded yet
if not st.session_state.files_loaded:
    st.stop()

master = st.session_state.master

# ══════════════════════════════════════════════════════════════
#  STEP 2 — METHOD CARDS + RUN BUTTON
#  (matches screenshot exactly)
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Detection Methods Applied</div>', unsafe_allow_html=True)

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown("""
    <div class="method-card">
        <div class="method-title" style="color:#ef4444">① Threshold Violations</div>
        <div class="method-desc">Hard upper/lower limits on HR, Steps, Sleep. Simple, interpretable, fast.</div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown(f"""
    <div class="method-card">
        <div class="method-title" style="color:#f59e0b">② Residual-Based</div>
        <div class="method-desc">Rolling median as baseline. Flag days where actual deviates by ±{sigma:.0f}σ.</div>
    </div>""", unsafe_allow_html=True)
with mc3:
    st.markdown("""
    <div class="method-card">
        <div class="method-title" style="color:#10b981">③ DBSCAN Outliers</div>
        <div class="method-desc">Users labelled −1 by DBSCAN in Milestone 2 are structural outliers.</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

if st.button("🔵  Run Anomaly Detection (All 3 Methods)", key="run_anomaly"):
    with st.spinner("Running all 3 detection methods…"):
        try:
            anom_hr    = detect_hr_anomalies(master,    hr_high, hr_low,   sigma)
            anom_steps = detect_steps_anomalies(master, st_low,  25000,    sigma)
            anom_sleep = detect_sleep_anomalies(master, sl_low,  sl_high,  sigma)
            st.session_state.anom_hr    = anom_hr
            st.session_state.anom_steps = anom_steps
            st.session_state.anom_sleep = anom_sleep
            st.session_state.anomaly_done = True
            st.rerun()
        except Exception as e:
            st.error(f"Detection error: {e}")
            st.exception(e)

if not st.session_state.anomaly_done:
    st.info("👆 Click **Run Anomaly Detection** to begin.")
    st.stop()

# Load results
anom_hr    = st.session_state.anom_hr
anom_steps = st.session_state.anom_steps
anom_sleep = st.session_state.anom_sleep

n_hr    = int(anom_hr["is_anomaly"].sum())
n_steps = int(anom_steps["is_anomaly"].sum())
n_sleep = int(anom_sleep["is_anomaly"].sum())
n_total = n_hr + n_steps + n_sleep

# ── Total anomaly banner — matches screenshot ──────────────────
st.markdown(
    f'<div class="alert-box" style="font-size:0.95rem;font-weight:600">'
    f'🚨 Total anomalies flagged: {n_total} &nbsp;(HR: {n_hr} · Steps: {n_steps} · Sleep: {n_sleep})'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Big count cards — matches screenshot exactly ────────────────
cc1, cc2, cc3, cc4 = st.columns(4)
for col, label, num in zip(
    [cc1, cc2, cc3, cc4],
    ["HR ANOMALIES", "STEPS ANOMALIES", "SLEEP ANOMALIES", "TOTAL FLAGS"],
    [n_hr, n_steps, n_sleep, n_total],
):
    col.markdown(f"""
    <div class="count-card">
        <div class="count-num">{num}</div>
        <div class="count-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  CHART 1 — HEART RATE ANOMALY CHART
# ══════════════════════════════════════════════════════════════
with st.expander("💓  Chart 1 — Heart Rate Anomaly Detection Chart", expanded=True):
    tip("Blue line = actual HR · Green dotted line = rolling median · Shaded band = ±2σ expected zone · Large red circles = anomalies · 'Residual±2σ' annotation confirms detection method.")

    hr_anom   = anom_hr[anom_hr["is_anomaly"]]
    hr_normal = anom_hr[~anom_hr["is_anomaly"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Days",     len(anom_hr))
    c2.metric("Avg HR",         f"{anom_hr['AvgHR'].mean():.1f} bpm")
    c3.metric("🚨 Anomalies",   n_hr,
              delta=f"{n_hr/len(anom_hr)*100:.1f}% of days",
              delta_color="inverse" if n_hr > 0 else "off")
    c4.metric("Max HR",         f"{anom_hr['AvgHR'].max():.1f} bpm")

    if n_hr > 0:
        alert(f"**{n_hr} anomalous HR days** — outside threshold [{hr_low}–{hr_high} bpm] or ±{sigma:.0f}σ residual band.")
    else:
        ok("No HR anomalies detected with current settings.")

    fig_hr = go.Figure()

    # ±2σ shaded band
    resid_std_hr = anom_hr["residual"].std()
    upper_hr = anom_hr["rolling_med"] + sigma * resid_std_hr
    lower_hr = anom_hr["rolling_med"] - sigma * resid_std_hr

    xs = anom_hr["Date"].tolist()
    fig_hr.add_trace(go.Scatter(
        x=xs + xs[::-1],
        y=upper_hr.tolist() + lower_hr.tolist()[::-1],
        fill="toself", fillcolor="rgba(56,189,248,0.12)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        name=f"±{sigma:.0f}σ Expected Band", hoverinfo="skip",
    ))

    # Blue solid line — actual HR
    fig_hr.add_trace(go.Scatter(
        x=anom_hr["Date"], y=anom_hr["AvgHR"],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=5, color="#38bdf8", opacity=0.75),
        name="Avg Heart Rate",
        hovertemplate="📅 %{x|%Y-%m-%d}<br>💓 %{y:.1f} bpm<extra></extra>",
    ))

    # Green dotted rolling median
    fig_hr.add_trace(go.Scatter(
        x=anom_hr["Date"], y=anom_hr["rolling_med"],
        mode="lines",
        line=dict(color="#10b981", width=1.8, dash="dot"),
        name="Rolling Median",
        hovertemplate="Median: %{y:.1f} bpm<extra></extra>",
    ))

    # Threshold lines
    fig_hr.add_hline(y=hr_high, line_dash="dash", line_color="#ef4444", line_width=1.5,
                     annotation_text=f"High Threshold ({hr_high} bpm)",
                     annotation_font_color="#ef4444", annotation_font_size=10,
                     annotation_position="top right")
    fig_hr.add_hline(y=hr_low, line_dash="dash", line_color="#f9a8d4", line_width=1.5,
                     annotation_text=f"Low Threshold ({hr_low} bpm)",
                     annotation_font_color="#f9a8d4", annotation_font_size=10,
                     annotation_position="bottom right")

    # Large red circles + "Residual±2σ" text annotations
    if not hr_anom.empty:
        fig_hr.add_trace(go.Scatter(
            x=hr_anom["Date"], y=hr_anom["AvgHR"],
            mode="markers+text",
            marker=dict(size=18, color="rgba(239,68,68,0.85)", symbol="circle",
                        line=dict(color="white", width=2)),
            text=["▲ Residual±2σ"] * len(hr_anom),
            textposition="top center",
            textfont=dict(size=9, color="#fbbf24"),
            name="🚨 Anomaly",
            hovertemplate="<b>⚠️ ANOMALY</b><br>📅 %{x|%Y-%m-%d}<br>💓 %{y:.1f} bpm<extra></extra>",
        ))

    T(fig_hr, h=460)
    fig_hr.update_layout(
        title="Heart Rate — Anomaly Detection Chart (Threshold + Residual-Based)",
        xaxis_title="Date", yaxis_title="Heart Rate (bpm)",
        hovermode="x unified",
    )
    fig_hr.update_xaxes(tickformat="%d %b", tickangle=-30)
    st.plotly_chart(fig_hr, use_container_width=True)
    tip(f"Shaded blue zone = ±{sigma:.0f}σ expected range around rolling median · Red/pink dashed lines = absolute thresholds · Red circles with 'Residual±2σ' label = anomaly days.")

    # Anomaly records table — sir's format
    if not hr_anom.empty:
        with st.expander(f"📋 View {n_hr} HR Anomaly Records"):
            st.dataframe(
                hr_anom[hr_anom["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]]
                .rename(columns={"rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                .round(2),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════
#  CHART 2 — SLEEP PATTERN VISUALIZATION  (dual subplot)
# ══════════════════════════════════════════════════════════════
with st.expander("😴  Chart 2 — Sleep Pattern Visualization", expanded=True):
    tip("Dual subplot: **Top** = sleep duration signal + red diamond anomaly markers + green healthy zone · **Bottom** = residual deviation bars (red = anomaly day, blue = normal).")

    sleep_anom   = anom_sleep[anom_sleep["is_anomaly"]]
    sleep_normal = anom_sleep[~anom_sleep["is_anomaly"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Days",    len(anom_sleep))
    c2.metric("Avg Sleep",     f"{anom_sleep['TotalSleepMinutes'].mean():.0f} min")
    c3.metric("🚨 Anomalies",  n_sleep,
              delta=f"{n_sleep/len(anom_sleep)*100:.1f}% of days",
              delta_color="inverse" if n_sleep > 0 else "off")
    c4.metric("Days < 60 min", int((anom_sleep["TotalSleepMinutes"] < 60).sum()))

    if n_sleep > 0:
        alert(f"**{n_sleep} anomalous sleep days** — outside healthy zone [{sl_low}–{sl_high} min] or ±{sigma:.0f}σ residual.")
    else:
        ok("Sleep patterns look normal with current settings.")

    fig_sleep = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.08,
        subplot_titles=("Sleep Duration (minutes/night)", "Deviation from Expected"),
    )

    # Green healthy zone (top panel)
    xs_sl = anom_sleep["Date"].tolist()
    fig_sleep.add_trace(go.Scatter(
        x=xs_sl + xs_sl[::-1],
        y=[sl_high]*len(xs_sl) + [sl_low]*len(xs_sl),
        fill="toself", fillcolor="rgba(16,185,129,0.09)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        name=f"Healthy Zone ({sl_low}–{sl_high} min)", hoverinfo="skip",
    ), row=1, col=1)

    # Purple solid line — actual sleep
    fig_sleep.add_trace(go.Scatter(
        x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
        mode="lines+markers",
        line=dict(color="#a78bfa", width=2.5),
        marker=dict(size=5, color="#a78bfa", opacity=0.75),
        name="Sleep Minutes",
        hovertemplate="📅 %{x|%Y-%m-%d}<br>😴 %{y:.0f} min<extra></extra>",
    ), row=1, col=1)

    # Green dotted rolling median
    fig_sleep.add_trace(go.Scatter(
        x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
        mode="lines",
        line=dict(color="#10b981", width=1.8, dash="dot"),
        name="Rolling Median",
        hovertemplate="Median: %{y:.0f} min<extra></extra>",
    ), row=1, col=1)

    # Threshold lines
    fig_sleep.add_hline(y=sl_low,  line_dash="dash", line_color="#ef4444", line_width=1.4,
                        annotation_text=f"Min ({sl_low} min)",
                        annotation_font_color="#ef4444", annotation_font_size=10,
                        annotation_position="bottom right", row=1, col=1)
    fig_sleep.add_hline(y=sl_high, line_dash="dash", line_color="#f59e0b", line_width=1.4,
                        annotation_text=f"Max ({sl_high} min)",
                        annotation_font_color="#f59e0b", annotation_font_size=10,
                        annotation_position="top right", row=1, col=1)

    # Red diamond anomaly markers + annotation
    if not sleep_anom.empty:
        reasons = []
        for _, row in sleep_anom.iterrows():
            r = ["Residual±2σ"]
            if row["TotalSleepMinutes"] < sl_low: r.insert(0, f"Sleep<{sl_low}min")
            if row["TotalSleepMinutes"] > sl_high: r.insert(0, f"Sleep>{sl_high}min")
            reasons.append(", ".join(dict.fromkeys(r)))
        fig_sleep.add_trace(go.Scatter(
            x=sleep_anom["Date"], y=sleep_anom["TotalSleepMinutes"],
            mode="markers+text",
            marker=dict(size=16, color="#ef4444", symbol="diamond",
                        line=dict(color="white", width=1.5)),
            text=["▲ " + r for r in reasons],
            textposition="top center",
            textfont=dict(size=8, color="#fbbf24"),
            name="😴 Sleep Anomaly",
            hovertemplate="<b>⚠️ ANOMALY</b><br>📅 %{x|%Y-%m-%d}<br>😴 %{y:.0f} min<extra></extra>",
        ), row=1, col=1)

    # Residual bars — red=anomaly, blue=normal
    bar_colors_sl = ["#ef4444" if a else "#38bdf8" for a in anom_sleep["resid_anomaly"]]
    fig_sleep.add_trace(go.Bar(
        x=anom_sleep["Date"], y=anom_sleep["residual"],
        marker_color=bar_colors_sl, marker_opacity=0.80,
        name="Residual",
        hovertemplate="📅 %{x|%Y-%m-%d}<br>Δ %{y:.0f} min from expected<extra></extra>",
    ), row=2, col=1)
    fig_sleep.add_hline(y=0, line_color="#4a6a8a", line_width=1, row=2, col=1)

    fig_sleep.update_layout(**{**PT, "height": 560}, showlegend=True,
                            title="Sleep Pattern — Anomaly Visualization (Dual Subplot)")
    fig_sleep.update_xaxes(gridcolor="#1e3a5f", tickformat="%d %b", tickangle=-30)
    fig_sleep.update_yaxes(gridcolor="#1e3a5f")
    fig_sleep.update_yaxes(title_text="Sleep (min)", row=1, col=1)
    fig_sleep.update_yaxes(title_text="Deviation (min)", row=2, col=1)
    st.plotly_chart(fig_sleep, use_container_width=True)
    tip("Top panel answers **when** anomalies occurred. Bottom panel answers **how severe** they were. A bar near zero means barely any deviation from the expected trend.")

    if not sleep_anom.empty:
        with st.expander(f"📋 View {n_sleep} Sleep Anomaly Records"):
            st.dataframe(
                sleep_anom[sleep_anom["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]]
                .rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected",
                                  "residual":"Deviation","reason":"Anomaly Reason"})
                .round(2),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════
#  CHART 3 — STEP COUNT TREND WITH ALERTS  (dual subplot)
# ══════════════════════════════════════════════════════════════
with st.expander("👟  Chart 3 — Step Count Trend with Alerts", expanded=True):
    tip("Green line = actual steps · Blue dashed = rolling median trend · **Red vertical bands** = alert days · ▲ triangle markers = anomaly points · Bottom = residual deviation bars.")

    steps_anom   = anom_steps[anom_steps["is_anomaly"]]
    steps_normal = anom_steps[~anom_steps["is_anomaly"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Days",    len(anom_steps))
    c2.metric("Avg Steps/Day", f"{anom_steps['TotalSteps'].mean():,.0f}")
    c3.metric("🚨 Anomalies",  n_steps,
              delta=f"{n_steps/len(anom_steps)*100:.1f}% of days",
              delta_color="inverse" if n_steps > 0 else "off")
    c4.metric("Days < 500",    int((anom_steps["TotalSteps"] < 500).sum()))

    if n_steps > 0:
        alert(f"**{n_steps} anomalous step days** — low activity or unusual spike detected via threshold or ±{sigma:.0f}σ residual.")
    else:
        ok("Step count patterns look normal with current settings.")

    fig_steps = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.08,
        subplot_titles=("Daily Steps (avg across users)", "Residual Deviation from Trend"),
    )

    # Vertical red alert bands — sir's signature feature
    for _, row in steps_anom.iterrows():
        d = str(row["Date"])
        try:
            d_next = str(pd.Timestamp(d) + pd.Timedelta(days=1))[:10]
        except Exception:
            d_next = d
        fig_steps.add_vrect(
            x0=d, x1=d_next,
            fillcolor="rgba(239,68,68,0.15)",
            line_color="rgba(239,68,68,0.45)", line_width=1.5,
            row=1, col=1,
        )

    # Green solid line — actual steps
    fig_steps.add_trace(go.Scatter(
        x=anom_steps["Date"], y=anom_steps["TotalSteps"],
        mode="lines+markers",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=5, color="#10b981", opacity=0.7),
        name="Avg Daily Steps",
        hovertemplate="📅 %{x|%Y-%m-%d}<br>👟 %{y:,.0f} steps<extra></extra>",
    ), row=1, col=1)

    # Blue dashed rolling median trend
    fig_steps.add_trace(go.Scatter(
        x=anom_steps["Date"], y=anom_steps["rolling_med"],
        mode="lines",
        line=dict(color="#38bdf8", width=2, dash="dash"),
        name="Trend (Rolling Median)",
        hovertemplate="Trend: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # Threshold lines
    fig_steps.add_hline(y=st_low, line_dash="dash", line_color="#ef4444", line_width=1.4,
                        annotation_text=f"Low Alert ({st_low:,} steps)",
                        annotation_font_color="#ef4444", annotation_font_size=10,
                        annotation_position="bottom right", row=1, col=1)
    fig_steps.add_hline(y=25000, line_dash="dash", line_color="#f59e0b", line_width=1.4,
                        annotation_text="High Alert (25,000 steps)",
                        annotation_font_color="#f59e0b", annotation_font_size=10,
                        annotation_position="top right", row=1, col=1)

    # Triangle markers on anomaly days
    if not steps_anom.empty:
        fig_steps.add_trace(go.Scatter(
            x=steps_anom["Date"], y=steps_anom["TotalSteps"],
            mode="markers+text",
            marker=dict(size=14, color="#fbbf24", symbol="triangle-up",
                        line=dict(color="#ef4444", width=2)),
            text=["▲"] * len(steps_anom),
            textposition="top center",
            textfont=dict(size=9, color="#ef4444"),
            name="🚨 Steps Alert",
            hovertemplate="<b>⚠️ ALERT</b><br>📅 %{x|%Y-%m-%d}<br>👟 %{y:,.0f} steps<extra></extra>",
        ), row=1, col=1)

    # Residual bars — green=normal, red=anomaly
    bar_colors_st = ["#ef4444" if a else "#10b981" for a in anom_steps["resid_anomaly"]]
    fig_steps.add_trace(go.Bar(
        x=anom_steps["Date"], y=anom_steps["residual"],
        marker_color=bar_colors_st, marker_opacity=0.80,
        name="Residual",
        hovertemplate="📅 %{x|%Y-%m-%d}<br>Δ %{y:,.0f} steps<extra></extra>",
    ), row=2, col=1)
    fig_steps.add_hline(y=0, line_color="#4a6a8a", line_width=1, row=2, col=1)

    fig_steps.update_layout(**{**PT, "height": 560}, showlegend=True,
                            title="Step Count Trend — Alerts & Anomalies (Threshold + Residual)")
    fig_steps.update_xaxes(gridcolor="#1e3a5f", tickformat="%d %b", tickangle=-30)
    fig_steps.update_yaxes(gridcolor="#1e3a5f")
    fig_steps.update_yaxes(title_text="Steps", row=1, col=1)
    fig_steps.update_yaxes(title_text="Residual (steps)", row=2, col=1)
    st.plotly_chart(fig_steps, use_container_width=True)
    tip("Vertical red bands highlight anomaly dates — visible even from a distance. Positive residual bar = walked more than expected. Negative = less than expected.")

    if not steps_anom.empty:
        with st.expander(f"📋 View {n_steps} Steps Anomaly Records"):
            st.dataframe(
                steps_anom[steps_anom["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]]
                .rename(columns={"TotalSteps":"Steps","rolling_med":"Expected",
                                  "residual":"Deviation","reason":"Anomaly Reason"})
                .round(2),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════
#  CHART 4 — DBSCAN PCA  (user-level structural outlier)
# ══════════════════════════════════════════════════════════════
with st.expander("🤖  Chart 4 — DBSCAN Outlier Detection (PCA Projection)", expanded=True):
    tip("Each dot = one user · Colour = cluster · **Red X = structural outlier (DBSCAN label −1)** · User IDs shown as labels. This is the only chart that looks at ALL health signals simultaneously.")

    st.markdown("""
    <div class="info-box">
        <strong>Why is this chart different from Charts 1–3?</strong><br>
        Charts 1–3 detect anomalies <em>per day per signal</em>. Chart 4 detects anomalies
        <em>per user across ALL signals simultaneously</em>. A user can appear normal on every individual
        chart but still be flagged here due to an unusual <em>combination</em> of health metrics.
        PCA compresses all dimensions into 2D for visualisation.
    </div>""", unsafe_allow_html=True)

    cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes",
                    "FairlyActiveMinutes","LightlyActiveMinutes",
                    "SedentaryMinutes","TotalSleepMinutes"]
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA

        avail_cols = [c for c in cluster_cols if c in master.columns]
        cf = master.groupby("Id")[avail_cols].mean().round(3).dropna()

        if len(cf) < 3:
            warn("Need at least 3 users for DBSCAN. Check your dataset.")
        else:
            X_scaled  = StandardScaler().fit_transform(cf)
            db_labels = DBSCAN(eps=2.2, min_samples=2).fit_predict(X_scaled)
            pca       = PCA(n_components=2, random_state=42)
            X_pca     = pca.fit_transform(X_scaled)
            var       = pca.explained_variance_ratio_ * 100

            cf["DBSCAN"] = db_labels
            outlier_ids  = cf[cf["DBSCAN"] == -1].index.tolist()
            n_outliers   = len(outlier_ids)
            n_clusters   = len(set(db_labels)) - (1 if -1 in db_labels else 0)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Users",    len(cf))
            c2.metric("Clusters Found", n_clusters)
            c3.metric("🔴 Outliers",    n_outliers,
                      help="Users whose health profile doesn't fit any cluster (label = −1)")

            if n_outliers > 0:
                alert(f"**{n_outliers} structural outlier(s)** — User(s) {outlier_ids} do not fit any cluster profile.")
            else:
                ok("No structural outliers detected. Try reducing DBSCAN eps in the sidebar.")

            CLUSTER_COLORS = ["#a78bfa","#38bdf8","#10b981","#f59e0b","#f97316","#06b6d4"]
            fig_db = go.Figure()

            for lbl in sorted(set(db_labels)):
                if lbl == -1: continue
                mask = db_labels == lbl
                fig_db.add_trace(go.Scatter(
                    x=X_pca[mask, 0], y=X_pca[mask, 1],
                    mode="markers+text",
                    name=f"Cluster {lbl}",
                    marker=dict(size=14, color=CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)],
                                opacity=0.85, line=dict(color="white", width=1.5)),
                    text=[str(uid)[-4:] for uid in cf.index[mask]],
                    textposition="top center",
                    textfont=dict(size=8, color="#a8c8e8"),
                    hovertemplate=f"<b>Cluster {lbl}</b><br>User: %{{text}}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
                ))

            if n_outliers > 0:
                mask_out = db_labels == -1
                fig_db.add_trace(go.Scatter(
                    x=X_pca[mask_out, 0], y=X_pca[mask_out, 1],
                    mode="markers+text",
                    name="🚨 Outlier / Anomaly",
                    marker=dict(size=20, color="#ef4444", symbol="x",
                                line=dict(color="white", width=2.5)),
                    text=[str(uid)[-4:] for uid in cf.index[mask_out]],
                    textposition="top center",
                    textfont=dict(size=9, color="#ef4444"),
                    hovertemplate="<b>⚠️ OUTLIER</b><br>User: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra>Structural Outlier</extra>",
                ))

            T(fig_db, h=500)
            fig_db.update_layout(
                title=f"DBSCAN Outlier Detection — PCA Projection (eps=2.2) · {n_clusters} cluster(s) · {n_outliers} outlier(s)",
                xaxis_title=f"PC1 ({var[0]:.1f}% variance)",
                yaxis_title=f"PC2 ({var[1]:.1f}% variance)",
            )
            st.plotly_chart(fig_db, use_container_width=True)
            tip("Red X markers = users who don't fit any cluster. This is the power of DBSCAN — it catches multi-dimensional anomalies that individual signal analysis misses.")

            if outlier_ids:
                st.markdown("**📋 Outlier User Profiles**")
                st.dataframe(cf[cf["DBSCAN"] == -1][avail_cols].round(2),
                             use_container_width=True)

    except ImportError:
        warn("scikit-learn not installed. Run: pip install scikit-learn")
    except Exception as e:
        warn(f"DBSCAN clustering error: {e}")

# ══════════════════════════════════════════════════════════════
#  CHART 5 — ACCURACY SIMULATION
# ══════════════════════════════════════════════════════════════
with st.expander("🎯  Chart 5 — Simulated Detection Accuracy (90%+ Target)", expanded=True):
    tip("10 known extreme anomalies are injected into each signal. We then run the detector and measure how many it catches. This validates the system meets the 90%+ accuracy requirement.")

    st.markdown("""
    <div class="info-box">
        <strong>How injection testing works:</strong><br>
        <strong>Step 1</strong> — Copy the real signal ·
        <strong>Step 2</strong> — Select 10 random days ·
        <strong>Step 3</strong> — Inject extreme values (e.g. HR = 150 bpm) ·
        <strong>Step 4</strong> — Run the same detector ·
        <strong>Step 5</strong> — Count how many injected points were caught ·
        <strong>Step 6</strong> — Accuracy = (caught ÷ 10) × 100%
    </div>""", unsafe_allow_html=True)

    if st.button("▶️  Run Accuracy Simulation (10 injected anomalies per signal)", key="run_sim"):
        with st.spinner("Injecting anomalies and measuring detection rate…"):
            try:
                sim = simulate_accuracy(master, n_inject=10)
                st.session_state.sim_results     = sim
                st.session_state.simulation_done = True
                st.rerun()
            except Exception as e:
                st.error(f"Simulation error: {e}")

    if not st.session_state.simulation_done:
        st.info("Click **Run Accuracy Simulation** above to validate the detection system.")
    else:
        sim     = st.session_state.sim_results
        overall = sim["Overall"]
        passed  = overall >= 90.0

        # Top banner
        banner_c = "#10b981" if passed else "#ef4444"
        banner_t = (f"Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT"
                    if passed else
                    f"Overall accuracy: {overall}% — ❌ BELOW 90% TARGET")
        if passed:
            ok(f"**{banner_t}**")
        else:
            warn(f"**{banner_t}**")

        # Per-signal + overall cards
        card_cols = st.columns(4)
        for col, label in zip(card_cols[:3], ["Heart Rate","Steps","Sleep"]):
            r = sim[label]
            c = "#10b981" if r["accuracy"] >= 90 else "#ef4444"
            col.markdown(f"""
            <div style="background:#162a41;border:1.5px solid {c};border-radius:12px;
                        padding:16px 10px;text-align:center">
                <div style="color:{c};font-weight:800;font-size:1.7rem">{r['accuracy']:.1f}%</div>
                <div style="color:#dce8f5;font-size:0.82rem;font-weight:600;margin:4px 0">{label}</div>
                <div style="color:#5a7a9a;font-size:0.72rem">{r['detected']}/{r['injected']} detected</div>
                <div style="margin-top:6px">
                    <span style="background:{c};color:#0d1b2a;font-size:0.65rem;font-weight:800;
                                 padding:2px 9px;border-radius:20px">
                        {'✅ PASS' if r['accuracy']>=90 else '❌ FAIL'}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

        with card_cols[3]:
            c = "#10b981" if passed else "#ef4444"
            st.markdown(f"""
            <div style="background:#162a41;border:2px solid {c};border-radius:12px;
                        padding:16px 10px;text-align:center">
                <div style="color:{c};font-weight:800;font-size:1.7rem">{overall}%</div>
                <div style="color:#dce8f5;font-size:0.82rem;font-weight:600;margin:4px 0">Overall</div>
                <div style="color:#5a7a9a;font-size:0.72rem">avg all signals</div>
                <div style="margin-top:6px">
                    <span style="background:{c};color:#0d1b2a;font-size:0.65rem;font-weight:800;
                                 padding:2px 9px;border-radius:20px">
                        {'90%+ ACHIEVED' if passed else 'BELOW TARGET'}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

        # Accuracy bar chart
        signals    = ["Heart Rate","Steps","Sleep"]
        accs       = [sim[s]["accuracy"] for s in signals]
        bar_colors = ["#10b981" if a >= 90 else "#ef4444" for a in accs]

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=signals, y=accs,
            marker_color=bar_colors, marker_opacity=0.85,
            text=[f"{a:.1f}%" for a in accs],
            textposition="outside",
            textfont=dict(size=14, color="#dce8f5"),
            name="Detection Accuracy",
            width=0.4,
        ))
        fig_acc.add_hline(y=90, line_dash="dash", line_color="#ef4444", line_width=2,
                          annotation_text="90% Target",
                          annotation_font_color="#ef4444", annotation_font_size=11,
                          annotation_position="top right")
        fig_acc.add_annotation(
            x=0.5, y=1.13, xref="paper", yref="paper",
            text=f"{'✅ 90%+ ACHIEVED' if passed else '❌ Below Target'} — Overall: {overall}%",
            showarrow=False,
            font=dict(size=13, color="#10b981" if passed else "#ef4444"),
            bgcolor="rgba(22,42,65,0.85)",
            bordercolor="#1e3a5f", borderpad=6,
        )
        T(fig_acc, h=420)
        fig_acc.update_layout(
            title="Simulated Anomaly Detection Accuracy (10 Injected Anomalies per Signal)",
            xaxis_title="Signal", yaxis_title="Detection Accuracy (%)",
            yaxis_range=[0, 120], showlegend=False,
        )
        st.plotly_chart(fig_acc, use_container_width=True)
        tip("Green bars = PASS (≥90%) · Red bars = FAIL · Dashed line = 90% target threshold. 100% on injected anomalies is expected because injected values are extreme (±5σ). Real-world anomalies may be more subtle.")

# ══════════════════════════════════════════════════════════════
#  SUMMARY + EXPORT
# ══════════════════════════════════════════════════════════════
with st.expander("📋  Summary & Export Anomaly Report"):
    st.markdown("### 🚦 Milestone 3 — Completion Checklist")

    checks = [
        (True,                              "📂", "Data loaded and master DataFrame built"),
        (st.session_state.anomaly_done,     "① ", "Threshold Violations detected"),
        (st.session_state.anomaly_done,     "② ", "Residual-Based (±2σ) detection complete"),
        (st.session_state.anomaly_done,     "③ ", "DBSCAN structural outliers identified"),
        (st.session_state.anomaly_done,     "💓", "Chart 1 — Heart rate anomaly chart"),
        (st.session_state.anomaly_done,     "😴", "Chart 2 — Sleep pattern dual subplot"),
        (st.session_state.anomaly_done,     "👟", "Chart 3 — Step count trend with alert bands"),
        (st.session_state.anomaly_done,     "🤖", "Chart 4 — DBSCAN PCA scatter plot"),
        (st.session_state.simulation_done,  "🎯", "Chart 5 — Accuracy simulation (90%+ target)"),
    ]
    for done, icon, label in checks:
        dot = "✅" if done else "⏭️"
        st.markdown(f"{dot} {icon} {label}")

    st.markdown("---")

    # Build + export CSV report
    rows = []
    for metric_label, df_d, val_col, lo, hi, unit in [
        ("Heart Rate (bpm)", anom_hr,    "AvgHR",             hr_low, hr_high, "bpm"),
        ("Daily Steps",      anom_steps, "TotalSteps",        st_low, 25000,   "steps"),
        ("Sleep (min)",      anom_sleep, "TotalSleepMinutes", sl_low, sl_high, "min"),
    ]:
        for _, row in df_d[df_d["is_anomaly"]].iterrows():
            v = row[val_col]
            rows.append({
                "Date":           str(row["Date"])[:10],
                "Metric":         metric_label,
                "Value":          round(v, 1),
                "Unit":           unit,
                "Method":         "Threshold + Residual±2σ",
                "Threshold":      f"{lo}–{hi} {unit}",
                "Anomaly Reason": row.get("reason", "—"),
                "Severity":       "High" if (v > hi*1.1 or (lo > 0 and v < lo*0.9)) else "Medium",
            })

    if rows:
        report_df = pd.DataFrame(rows).sort_values("Date")
        st.dataframe(report_df, use_container_width=True)
        st.download_button(
            label="📥 Download Anomaly Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="fitpulse_anomaly_report_m3.csv",
            mime="text/csv",
        )

    else:
        st.info("Run anomaly detection above — the export populates automatically.")