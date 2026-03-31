import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import io
from datetime import datetime

from preprocessing import (
    load_all_files, parse_timestamps, resample_heartrate,
    build_master_df, prepare_tsfresh_input, build_clustering_features,
)

st.set_page_config(page_title="FitPulse — All Milestones", page_icon="💪", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1b2a; color: #dce8f5; }
section[data-testid="stSidebar"] { background: #0b1829; border-right: 1px solid #1e3a5f; }
section[data-testid="stSidebar"] * { color: #a8c8e8 !important; }
h1, h2, h3 { color: #a78bfa !important; }
.stButton > button { background: linear-gradient(135deg, #6366f1, #a855f7); color: white; font-weight: 600; border: none; border-radius: 8px; padding: 0.5rem 1.2rem; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(168,85,247,0.4); }
.stButton > button:disabled { opacity: 0.4; transform: none !important; }
div[data-testid="metric-container"] { background: #162a41; border: 1px solid #1e4a6e; border-radius: 12px; padding: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }
details { background: #111f30; border: 1px solid #1e3a5f !important; border-radius: 10px; margin-bottom: 10px; }
details summary { background: linear-gradient(90deg, #162a41, #1a1f4a); border-radius: 10px; padding: 14px 18px; color: #a78bfa; font-weight: 700; font-size: 1.05rem; cursor: pointer; list-style: none; }
details[open] summary { border-bottom: 1px solid #1e3a5f; border-radius: 10px 10px 0 0; }
.info-box { background: rgba(99,102,241,0.12); border-left: 4px solid #6366f1; border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 0.88rem; color: #a8c8e8; }
.log-box { background: rgba(74,144,196,0.1); border-left: 4px solid #a78bfa; border-radius: 4px; padding: 10px 15px; margin: 5px 0; font-family: 'Source Code Pro', monospace; }
.warn-box { background: rgba(245,158,11,0.10); border-left: 4px solid #f59e0b; border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 0.88rem; color: #fde68a; }
.alert-box { background: rgba(239,68,68,0.12); border-left: 4px solid #ef4444; border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 0.88rem; color: #fca5a5; }
.ok-box { background: rgba(16,185,129,0.10); border-left: 4px solid #10b981; border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 0.88rem; color: #6ee7b7; }
.shared-box { background: rgba(56,189,248,0.10); border-left: 4px solid #38bdf8; border-radius: 6px; padding: 12px 16px; margin: 10px 0; font-size: 0.88rem; color: #7dd3fc; }
.prog-wrap { background: #162a41; border-radius: 99px; height: 7px; margin: 10px 0; overflow: hidden; }
.prog-fill { background: linear-gradient(90deg, #6366f1, #a78bfa); height: 100%; border-radius: 99px; animation: pbar 1.4s ease-in-out infinite; }
@keyframes pbar { 0%,100%{opacity:1;width:60%} 50%{opacity:.5;width:90%} }
.count-card { background: #162a41; border: 1px solid #1e4a6e; border-radius: 14px; padding: 22px 16px; text-align: center; }
.count-num { font-size: 2.6rem; font-weight: 800; color: #ef4444; line-height: 1; }
.count-label { font-size: 0.72rem; font-weight: 700; color: #5a7a9a; letter-spacing: 0.08em; margin-top: 6px; text-transform: uppercase; }
.kpi-card { background:#162a41; border:1px solid #1e4a6e; border-radius:14px; padding:1.2rem 0.8rem; text-align:center; }
.kpi-num  { font-size:2.4rem; font-weight:800; line-height:1; margin-bottom:0.25rem; }
.kpi-lbl  { font-size:0.68rem; color:#5a7a9a; text-transform:uppercase; letter-spacing:0.08em; }
.kpi-sub  { font-size:0.65rem; color:#4a6a8a; margin-top:0.2rem; }
.stat-card { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px 18px; font-size: 0.85rem; line-height: 2; }
.stat-title { font-size: 0.7rem; font-weight: 700; color: #4a6a8a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
.anom-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0; border-bottom: 1px solid rgba(30,58,95,0.5); font-size: 0.82rem; }
.method-card { background: #162a41; border: 1px solid #1e4a6e; border-radius: 10px; padding: 16px 18px; }
.method-title { font-weight: 700; font-size: 0.92rem; margin-bottom: 6px; }
.method-desc  { font-size: 0.82rem; color: #a8c8e8; line-height: 1.55; }
.section-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; color: #4a6a8a; text-transform: uppercase; margin-bottom: 12px; }
[data-testid="stDecoration"] { display: none !important; }
.stStatusWidget { opacity: 0 !important; pointer-events: none; }

/* Home page cards */
.home-milestone-card {
    background: linear-gradient(135deg, #162a41, #1a1f4a);
    border: 1px solid #1e4a6e;
    border-radius: 16px;
    padding: 28px 22px;
    text-align: center;
    transition: all 0.25s ease;
    cursor: pointer;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
}
.home-milestone-card:hover {
    border-color: #a78bfa;
    box-shadow: 0 6px 24px rgba(167,139,250,0.25);
    transform: translateY(-3px);
}
.home-card-icon { font-size: 3rem; line-height: 1; }
.home-card-title { font-size: 1.05rem; font-weight: 700; color: #dce8f5; margin: 6px 0 4px; }
.home-card-sub { font-size: 0.78rem; color: #5a7a9a; line-height: 1.5; }
.home-card-badge { font-size: 0.68rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,27,42,0.6)",
    font=dict(color="#dce8f5", family="Inter"), title_font=dict(color="#a78bfa", size=14),
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
def shared_banner(files_dict):
    names = [v[0] for v in files_dict.values() if v]
    st.markdown(
        f'<div class="shared-box">📂 <strong>Files carried over from Milestone 2</strong> — '
        f'{len(names)} files loaded automatically. '
        f'<em>No re-upload needed.</em></div>',
        unsafe_allow_html=True)

def convert_df(df): return df.to_csv(index=False).encode('utf-8')

# ── Cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load(file_keys):
    file_objs = {}
    for role, (name, data) in file_keys.items():
        if data is not None:
            buf = BytesIO(data); buf.name = name
            file_objs[role] = buf
    dfs_raw = load_all_files(file_objs)
    dfs, _  = parse_timestamps(dfs_raw)
    return dfs

@st.cache_data(show_spinner=False)
def cached_resample(hr_bytes, hr_name):
    buf = BytesIO(hr_bytes); buf.name = hr_name
    df  = pd.read_csv(buf)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], infer_datetime_format=True, errors="coerce")
        df = df.dropna(subset=["Time"])
    result, _ = resample_heartrate(df)
    return result

@st.cache_data(show_spinner=False)
def cached_master(file_keys):
    dfs = cached_load(file_keys)
    master, _ = build_master_df(dfs)
    return master

@st.cache_data(show_spinner=False)
def cached_elbow(feat_bytes, k_max):
    from sklearn.cluster import KMeans
    import pickle
    feat = pickle.loads(feat_bytes)
    K_range  = range(2, min(k_max + 1, len(feat)))
    inertias = [KMeans(n_clusters=k, random_state=42, n_init="auto").fit(feat).inertia_ for k in K_range]
    return list(K_range), inertias

@st.cache_data(show_spinner=False)
def cached_pca(feat_bytes):
    from sklearn.decomposition import PCA
    import pickle
    feat   = pickle.loads(feat_bytes)
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(feat)
    return coords, pca.explained_variance_ratio_

# ── File registry (M3/M4 column-structure detection) ──────────────────────────
M3_REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {"key_cols": ["ActivityDate","TotalSteps","Calories"],  "label": "Daily Activity",    "icon": "🏃"},
    "hourlySteps_merged.csv":       {"key_cols": ["ActivityHour","StepTotal"],              "label": "Hourly Steps",      "icon": "👟"},
    "hourlyIntensities_merged.csv": {"key_cols": ["ActivityHour","TotalIntensity"],         "label": "Hourly Intensities","icon": "⚡"},
    "minuteSleep_merged.csv":       {"key_cols": ["date","value","logId"],                  "label": "Minute Sleep",      "icon": "😴"},
    "heartrate_seconds_merged.csv": {"key_cols": ["Time","Value"],                          "label": "Heart Rate",        "icon": "💓"},
}
M4_FILES = M3_REQUIRED_FILES

def score_match(df, req_info):
    return sum(1 for col in req_info["key_cols"] if col in df.columns)

def m2_matched_to_file_keys(matched):
    fk = {}
    for role, f in matched.items():
        if f is not None:
            f.seek(0)
            fk[role] = (f.name, f.read())
            f.seek(0)
    return fk

def file_keys_to_m3_detected(file_keys):
    raw_uploads = []
    for role, (name, data) in file_keys.items():
        try:
            df = pd.read_csv(BytesIO(data))
            raw_uploads.append((name, df))
        except Exception:
            pass
    detected = {}
    for req_name, finfo in M3_REQUIRED_FILES.items():
        best_s, best_d = 0, None
        for uname, udf in raw_uploads:
            s = score_match(udf, finfo)
            if s > best_s:
                best_s, best_d = s, udf
        if best_s >= 2:
            detected[req_name] = best_d
    return detected


# ── M3 anomaly detection functions ────────────────────────────────────────────
def detect_hr_anomalies(master, hr_high=100, hr_low=50, residual_sigma=2.0):
    df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    hr_daily = df.groupby("Date")["AvgHR"].mean().reset_index()
    hr_daily.columns = ["Date","AvgHR"]
    hr_daily = hr_daily.sort_values("Date")
    hr_daily["thresh_high"]   = hr_daily["AvgHR"] > hr_high
    hr_daily["thresh_low"]    = hr_daily["AvgHR"] < hr_low
    hr_daily["rolling_med"]   = hr_daily["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_daily["residual"]      = hr_daily["AvgHR"] - hr_daily["rolling_med"]
    resid_std                 = hr_daily["residual"].std()
    hr_daily["resid_anomaly"] = hr_daily["residual"].abs() > (residual_sigma * resid_std)
    hr_daily["is_anomaly"]    = hr_daily["thresh_high"] | hr_daily["thresh_low"] | hr_daily["resid_anomaly"]
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
    sd["thresh_low"]    = sd["TotalSteps"] < steps_low
    sd["thresh_high"]   = sd["TotalSteps"] > steps_high
    sd["rolling_med"]   = sd["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    sd["residual"]      = sd["TotalSteps"] - sd["rolling_med"]
    resid_std           = sd["residual"].std()
    sd["resid_anomaly"] = sd["residual"].abs() > (residual_sigma * resid_std)
    sd["is_anomaly"]    = sd["thresh_low"] | sd["thresh_high"] | sd["resid_anomaly"]
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
    sd["thresh_low"]    = (sd["TotalSleepMinutes"] > 0) & (sd["TotalSleepMinutes"] < sleep_low)
    sd["thresh_high"]   = sd["TotalSleepMinutes"] > sleep_high
    sd["no_data"]       = sd["TotalSleepMinutes"] == 0
    sd["rolling_med"]   = sd["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sd["residual"]      = sd["TotalSleepMinutes"] - sd["rolling_med"]
    resid_std           = sd["residual"].std()
    sd["resid_anomaly"] = sd["residual"].abs() > (residual_sigma * resid_std)
    sd["is_anomaly"]    = sd["thresh_low"] | sd["thresh_high"] | sd["resid_anomaly"]
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
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}
    hr_sim = df_daily[["Date","AvgHR"]].copy()
    idx = np.random.choice(len(hr_sim), n_inject, replace=False)
    hr_sim.loc[idx, "AvgHR"] = np.random.choice([115,120,125,35,40,45,118,130,38,42], n_inject, replace=True)
    hr_sim["rolling_med"] = hr_sim["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_sim["residual"]    = hr_sim["AvgHR"] - hr_sim["rolling_med"]
    hr_sim["detected"]    = ((hr_sim["AvgHR"] > 100) | (hr_sim["AvgHR"] < 50) | (hr_sim["residual"].abs() > 2 * hr_sim["residual"].std()))
    tp = hr_sim.iloc[idx]["detected"].sum()
    results["Heart Rate"] = {"injected": n_inject, "detected": int(tp), "accuracy": round(tp/n_inject*100, 1)}
    st_sim = df_daily[["Date","TotalSteps"]].copy()
    idx2 = np.random.choice(len(st_sim), n_inject, replace=False)
    st_sim.loc[idx2, "TotalSteps"] = np.random.choice([50,100,150,30000,35000,28000,80,200,31000,29000], n_inject, replace=True)
    st_sim["rolling_med"] = st_sim["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    st_sim["residual"]    = st_sim["TotalSteps"] - st_sim["rolling_med"]
    st_sim["detected"]    = ((st_sim["TotalSteps"] < 500) | (st_sim["TotalSteps"] > 25000) | (st_sim["residual"].abs() > 2 * st_sim["residual"].std()))
    tp2 = st_sim.iloc[idx2]["detected"].sum()
    results["Steps"] = {"injected": n_inject, "detected": int(tp2), "accuracy": round(tp2/n_inject*100, 1)}
    sl_sim = df_daily[["Date","TotalSleepMinutes"]].copy()
    idx3 = np.random.choice(len(sl_sim), n_inject, replace=False)
    sl_sim.loc[idx3, "TotalSleepMinutes"] = np.random.choice([10,20,30,700,750,800,15,25,710,720], n_inject, replace=True)
    sl_sim["rolling_med"] = sl_sim["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sl_sim["residual"]    = sl_sim["TotalSleepMinutes"] - sl_sim["rolling_med"]
    sl_sim["detected"]    = (((sl_sim["TotalSleepMinutes"] > 0) & (sl_sim["TotalSleepMinutes"] < 60)) | (sl_sim["TotalSleepMinutes"] > 600) | (sl_sim["residual"].abs() > 2 * sl_sim["residual"].std()))
    tp3 = sl_sim.iloc[idx3]["detected"].sum()
    results["Sleep"] = {"injected": n_inject, "detected": int(tp3), "accuracy": round(tp3/n_inject*100, 1)}
    results["Overall"] = round(np.mean([results[k]["accuracy"] for k in ["Heart Rate","Steps","Sleep"]]), 1)
    return results

# ── M4 detection functions ─────────────────────────────────────────────────────
def m4_detect_hr(master, hr_high=100, hr_low=50, sigma=2.0):
    df = master[["Id","Date","AvgHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["AvgHR"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["AvgHR"].rolling(3, center=True, min_periods=1).median()
    d["residual"]    = d["AvgHR"] - d["rolling_med"]
    std = d["residual"].std() if d["residual"].std() > 0 else 1.0
    d["thresh_high"] = d["AvgHR"] > hr_high; d["thresh_low"] = d["AvgHR"] < hr_low
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_high"] | d["thresh_low"] | d["resid_anom"]
    def reason(r):
        p = []
        if r.thresh_high: p.append(f"HR>{hr_high}")
        if r.thresh_low:  p.append(f"HR<{hr_low}")
        if r.resid_anom:  p.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(p)
    d["reason"] = d.apply(reason, axis=1)
    return d

def m4_detect_steps(master, st_low=500, st_high=25000, sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    d["residual"]    = d["TotalSteps"] - d["rolling_med"]
    std = d["residual"].std() if d["residual"].std() > 0 else 1.0
    d["thresh_low"]  = d["TotalSteps"] < st_low; d["thresh_high"] = d["TotalSteps"] > st_high
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(r):
        p = []
        if r.thresh_low:  p.append(f"Steps<{int(st_low):,}")
        if r.thresh_high: p.append(f"Steps>{int(st_high):,}")
        if r.resid_anom:  p.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(p)
    d["reason"] = d.apply(reason, axis=1)
    return d

def m4_detect_sleep(master, sl_low=60, sl_high=600, sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    d["residual"]    = d["TotalSleepMinutes"] - d["rolling_med"]
    std = d["residual"].std() if d["residual"].std() > 0 else 1.0
    d["thresh_low"]  = (d["TotalSleepMinutes"] > 0) & (d["TotalSleepMinutes"] < sl_low)
    d["thresh_high"] = d["TotalSleepMinutes"] > sl_high
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(r):
        p = []
        if r.thresh_low:  p.append(f"Sleep<{int(sl_low)}min")
        if r.thresh_high: p.append(f"Sleep>{int(sl_high)}min")
        if r.resid_anom:  p.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(p)
    d["reason"] = d.apply(reason, axis=1)
    return d

def m4_chart_timeline(anom_hr_f, anom_steps_f, anom_sleep_f, h=300):
    all_anoms = []
    for df_, sig in [(anom_hr_f,"Heart Rate"),(anom_steps_f,"Steps"),(anom_sleep_f,"Sleep")]:
        a = df_[df_["is_anomaly"]].copy(); a["signal"] = sig
        val_col = {"Heart Rate":"AvgHR","Steps":"TotalSteps","Sleep":"TotalSleepMinutes"}[sig]
        a["value"] = a[val_col] if val_col in a.columns else 0
        all_anoms.append(a[["Date","signal","reason","value"]])
    if not all_anoms: return None
    combined = pd.concat(all_anoms, ignore_index=True); combined["Date"] = pd.to_datetime(combined["Date"])
    color_map = {"Heart Rate":"#38bdf8","Steps":"#10b981","Sleep":"#a78bfa"}
    fig = go.Figure()
    for sig, col in color_map.items():
        sub = combined[combined["signal"]==sig]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub["Date"], y=sub["signal"], mode="markers", name=sig,
                marker=dict(color=col, size=14, symbol="diamond", line=dict(color="white",width=2)),
                hovertemplate=f"<b>{sig}</b><br>📅 %{{x|%d %b %Y}}<br>%{{customdata}}<extra>⚠️ ANOMALY</extra>",
                customdata=sub["reason"].values))
    T(fig, h)
    fig.update_layout(title="📅 Combined Anomaly Timeline — All Signals", xaxis_title="Date", yaxis_title="Signal",
        yaxis=dict(categoryorder="array", categoryarray=["Sleep","Steps","Heart Rate"], gridcolor="#1e3a5f"), hovermode="closest")
    fig.update_xaxes(tickformat="%d %b %Y", tickangle=-30)
    return fig

def m4_generate_pdf(master, anom_hr, anom_steps, anom_sleep, hr_high, hr_low, st_low, sl_low, sl_high, sigma):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER
    buf = io.BytesIO()
    C_ACCENT=colors.HexColor("#a78bfa"); C_BLUE=colors.HexColor("#38bdf8"); C_GREEN=colors.HexColor("#10b981")
    C_RED=colors.HexColor("#ef4444"); C_TEXT=colors.HexColor("#dce8f5"); C_MUTED=colors.HexColor("#5a7a9a")
    C_BORDER=colors.HexColor("#1e3a5f"); C_ROW1=colors.HexColor("#0d1b2a"); C_ROW2=colors.HexColor("#111f30")
    PAGE_W, PAGE_H = A4
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=18*mm, bottomMargin=18*mm)
    def S(name, **kw): return ParagraphStyle(name, **kw)
    s_title = S("t", fontName="Helvetica-Bold", fontSize=18, textColor=C_ACCENT, spaceAfter=4, leading=22, alignment=TA_CENTER)
    s_sub   = S("s", fontName="Helvetica", fontSize=9, textColor=C_MUTED, spaceAfter=8, alignment=TA_CENTER)
    s_body  = S("b", fontName="Helvetica", fontSize=8.5, textColor=C_TEXT, leading=13, spaceAfter=4)
    s_small = S("sm", fontName="Helvetica", fontSize=7.5, textColor=C_MUTED, leading=11)
    def hr_line(): return HRFlowable(width="100%", thickness=0.5, color=C_ACCENT, spaceAfter=6, spaceBefore=6)
    def section_hdr(text, color=C_ACCENT):
        return [Spacer(1,4*mm), Paragraph(text, S("sh", fontName="Helvetica-Bold", fontSize=11, textColor=color, spaceBefore=0, spaceAfter=2)), hr_line()]
    def kv_table(rows):
        data = [[Paragraph(k, S("kk", fontName="Helvetica-Bold", fontSize=9, textColor=C_MUTED)), Paragraph(str(v), S("kv", fontName="Helvetica-Bold", fontSize=9, textColor=C_TEXT))] for k,v in rows]
        t = Table(data, colWidths=[55*mm,110*mm])
        t.setStyle(TableStyle([("ROWBACKGROUNDS",(0,0),(-1,-1),[C_ROW1,C_ROW2]),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("GRID",(0,0),(-1,-1),0.25,C_BORDER)]))
        return t
    def anom_table(df, val_col, val_label):
        sub = df[df["is_anomaly"]][["Date",val_col,"rolling_med","residual","reason"]].copy().round(2)
        if sub.empty: return Paragraph("No anomalies detected.", s_body)
        sub["Date"] = sub["Date"].astype(str).str[:10]
        td = [["Date", val_label, "Expected", "Deviation", "Reason"]] + [[str(r["Date"]), f"{r[val_col]:.1f}", f"{r['rolling_med']:.1f}", f"{r['residual']:.1f}", str(r["reason"])[:55]] for _,r in sub.head(25).iterrows()]
        t = Table(td, colWidths=[25*mm,22*mm,22*mm,22*mm,89*mm], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C_BORDER),("TEXTCOLOR",(0,0),(-1,0),C_ACCENT),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),("ROWBACKGROUNDS",(0,1),(-1,-1),[C_ROW1,C_ROW2]),("TEXTCOLOR",(0,1),(-1,-1),C_TEXT),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("GRID",(0,0),(-1,-1),0.3,C_BORDER),("ALIGN",(1,0),(3,-1),"CENTER")]))
        elems = [t]
        if len(sub) > 25: elems.append(Paragraph(f"... and {len(sub)-25} more records.", s_small))
        return elems
    n_hr=int(anom_hr["is_anomaly"].sum()); n_steps=int(anom_steps["is_anomaly"].sum()); n_sleep=int(anom_sleep["is_anomaly"].sum()); n_total=n_hr+n_steps+n_sleep
    n_users=master["Id"].nunique(); n_days=master["Date"].nunique()
    d_min=pd.to_datetime(master["Date"]).min().strftime("%d %b %Y"); d_max=pd.to_datetime(master["Date"]).max().strftime("%d %b %Y")
    kpi_data=[["Metric","Count","% of Days"],["Heart Rate Anomalies",str(n_hr),f"{n_hr/max(len(anom_hr),1)*100:.1f}%"],["Steps Anomalies",str(n_steps),f"{n_steps/max(len(anom_steps),1)*100:.1f}%"],["Sleep Anomalies",str(n_sleep),f"{n_sleep/max(len(anom_sleep),1)*100:.1f}%"],["TOTAL FLAGS",str(n_total),"—"]]
    kpi_t=Table(kpi_data,colWidths=[70*mm,55*mm,55*mm])
    kpi_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C_BORDER),("TEXTCOLOR",(0,0),(-1,0),C_ACCENT),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("ROWBACKGROUNDS",(0,1),(-1,-1),[C_ROW1,C_ROW2]),("TEXTCOLOR",(0,1),(-1,-1),C_TEXT),("FONTNAME",(0,4),(-1,4),"Helvetica-Bold"),("TEXTCOLOR",(1,4),(-1,4),C_RED),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),0.3,C_BORDER),("ALIGN",(1,0),(-1,-1),"CENTER")]))
    profile_cols=["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
    avail_p=[c for c in profile_cols if c in master.columns]; user_p=master.groupby("Id")[avail_p].mean().round(0).reset_index(); user_p["Id"]=user_p["Id"].astype(str).str[-6:]
    ph=["User (last 6)"]+[c[:12] for c in avail_p]; pw=[180*mm/(len(avail_p)+1)]*(len(avail_p)+1)
    pd_data=[ph]+[[row["Id"]]+[f"{row[c]:,.0f}" for c in avail_p] for _,row in user_p.iterrows()]
    prof_t=Table(pd_data,colWidths=pw,repeatRows=1)
    prof_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C_BORDER),("TEXTCOLOR",(0,0),(-1,0),C_ACCENT),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),("ROWBACKGROUNDS",(0,1),(-1,-1),[C_ROW1,C_ROW2]),("TEXTCOLOR",(0,1),(-1,-1),C_TEXT),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),5),("GRID",(0,0),(-1,-1),0.3,C_BORDER),("ALIGN",(1,0),(-1,-1),"CENTER")]))
    story=[Spacer(1,8*mm),Paragraph("FitPulse — Anomaly Detection Report",s_title),Paragraph("Milestone 4  ·  Insights & Export Dashboard",s_sub),Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}",s_sub),hr_line(),Spacer(1,4*mm)]
    story+=section_hdr("1.  DATASET OVERVIEW",C_ACCENT); story.append(kv_table([("Dataset Source","Real Fitbit Device Data — Kaggle (arashnic/fitbit)"),("Total Users",f"{n_users} participants"),("Date Range",f"{d_min}  →  {d_max}"),("Total Days",f"{n_days} observation days"),("Pipeline","Milestone 4 — Anomaly Detection Dashboard")])); story.append(Spacer(1,5*mm))
    story+=section_hdr("2.  ANOMALY SUMMARY",C_RED); story.append(kpi_t); story.append(Spacer(1,5*mm))
    story+=section_hdr("3.  DETECTION THRESHOLDS USED",C_GREEN); story.append(kv_table([("Heart Rate High",f"> {hr_high} bpm"),("Heart Rate Low",f"< {hr_low} bpm"),("Steps Low Alert",f"< {st_low:,} steps/day"),("Sleep Low",f"< {sl_low} minutes/night"),("Sleep High",f"> {sl_high} minutes/night"),("Residual Sigma",f"+/- {sigma:.1f}sigma from 3-day rolling median")])); story.append(Spacer(1,5*mm))
    story+=section_hdr("4.  DETECTION METHODOLOGY",C_BLUE); story.append(Paragraph(f"<b>Three complementary methods:</b><br/><br/><b>① Threshold Violations</b> — Hard upper/lower bounds.<br/><br/><b>② Residual-Based Detection</b> — 3-day rolling median baseline. Days deviating by +/- {sigma:.1f}sigma are flagged.<br/><br/><b>③ DBSCAN Structural Outliers</b> — Users labelled -1 are structural outliers.",s_body)); story.append(Spacer(1,5*mm))
    story+=section_hdr("5.  HEART RATE ANOMALY RECORDS",C_RED); hr_e=anom_table(anom_hr,"AvgHR","Avg HR (bpm)"); story+=hr_e if isinstance(hr_e,list) else [hr_e]; story.append(Spacer(1,5*mm))
    story+=section_hdr("6.  STEP COUNT ANOMALY RECORDS",C_GREEN); st_e=anom_table(anom_steps,"TotalSteps","Steps"); story+=st_e if isinstance(st_e,list) else [st_e]; story.append(Spacer(1,5*mm))
    story+=section_hdr("7.  SLEEP ANOMALY RECORDS",C_ACCENT); sl_e=anom_table(anom_sleep,"TotalSleepMinutes","Sleep (min)"); story+=sl_e if isinstance(sl_e,list) else [sl_e]; story.append(Spacer(1,5*mm))
    story+=section_hdr("8.  USER ACTIVITY PROFILES",C_BLUE); story.append(Paragraph("Average daily metrics per user across the observation period:",s_body)); story.append(prof_t); story.append(Spacer(1,5*mm))
    story+=section_hdr("9.  CONCLUSION",C_GREEN); story.append(Paragraph(f"FitPulse Milestone 4 processed <b>{n_users} users</b> over <b>{n_days} days</b>. Total: <b>{n_total} anomalous events</b> — {n_hr} HR, {n_steps} steps, {n_sleep} sleep anomalies.",s_body))
    def page_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#0a0e1a")); canvas.rect(0,0,PAGE_W,PAGE_H,fill=True,stroke=False)
        canvas.setFillColor(colors.HexColor("#0d1b2a")); canvas.rect(0,PAGE_H-12*mm,PAGE_W,12*mm,fill=True,stroke=False)
        canvas.setFillColor(colors.HexColor("#a78bfa")); canvas.rect(0,PAGE_H-12*mm,PAGE_W,0.8,fill=True,stroke=False)
        canvas.setFillColor(colors.HexColor("#5a7a9a")); canvas.setFont("Helvetica",7)
        canvas.drawCentredString(PAGE_W/2,8*mm,f"FitPulse ML Pipeline  ·  Anomaly Detection Report  ·  Page {doc.page}")
        canvas.setFillColor(colors.HexColor("#1e3a5f")); canvas.rect(0,6*mm,PAGE_W,0.5,fill=True,stroke=False)
        canvas.restoreState()
    doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)
    buf.seek(0); return buf

def m4_generate_csv(anom_hr, anom_steps, anom_sleep):
    parts = []
    for df_, sig, vc in [(anom_hr,"Heart Rate","AvgHR"),(anom_steps,"Steps","TotalSteps"),(anom_sleep,"Sleep","TotalSleepMinutes")]:
        sub = df_[df_["is_anomaly"]][["Date",vc,"rolling_med","residual","reason"]].copy()
        sub["signal"] = sig; sub = sub.rename(columns={vc:"value","rolling_med":"expected"})
        parts.append(sub)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined[["signal","Date","value","expected","residual","reason"]].sort_values(["signal","Date"]).round(2)
    buf = io.StringIO(); combined.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ══════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
for k, v in [
    ("shared_file_keys",   None),
    ("m3_files_loaded",    False), ("m3_anomaly_done",    False), ("m3_simulation_done", False),
    ("m3_master",          None),  ("m3_hr_minute",       None),
    ("m3_anom_hr",         None),  ("m3_anom_steps",      None),  ("m3_anom_sleep",      None),
    ("m3_sim_results",     None),
    ("m4_pipeline_done",   False), ("m4_master",          None),
    ("m4_anom_hr",         None),  ("m4_anom_steps",      None),  ("m4_anom_sleep",      None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=60)
    st.markdown("## FitPulse")
    st.markdown("---")

    if st.session_state.shared_file_keys:
        n_shared = len(st.session_state.shared_file_keys)
        st.markdown(f'<div style="background:rgba(56,189,248,0.12);border:1px solid #38bdf8;border-radius:8px;padding:8px 12px;margin-bottom:10px;font-size:0.78rem;color:#7dd3fc">📂 <strong>{n_shared} files</strong> shared from M2</div>', unsafe_allow_html=True)

    milestone = st.selectbox(
        "📌 Navigate To",
        ["🏠 Home",
         "📊 Milestone 1 — Data Preprocessing",
         "🧬 Milestone 2 — Feature Extraction & Modelling",
         "🚨 Milestone 3 — Anomaly Detection",
         "📈 Milestone 4 — Insights Dashboard"],
        key="milestone_selector",
        help="🏠 Home  |  🏋️ M1: Clean & inspect  |  🧬 M2: Features & model  |  🚨 M3: Detect anomalies  |  📊 M4: Insights & export"
    )
    st.markdown("---")

    if "Home" in milestone:
        st.markdown("**Welcome to FitPulse!**")
        st.caption("Select any milestone from the dropdown or click a card on the Home page.")

    elif "Milestone 1" in milestone:
        st.markdown("**M1 Pipeline:**\n1. Upload CSV 📂\n2. Inspect 🔍\n3. Clean ⚙️\n4. Export ⬇️")
        st.caption("v1.0 | Preprocessing")

    elif "Milestone 2" in milestone:
        st.markdown("**M2 Pipeline:**\n1. Upload 5 CSVs 📂\n2. TSFresh features 🔬\n3. Prophet forecast 📈\n4. Clustering 🔵")
        st.markdown('<div style="background:rgba(16,185,129,0.10);border-left:3px solid #10b981;border-radius:4px;padding:6px 10px;font-size:0.78rem;color:#6ee7b7;margin-top:6px">✅ Files uploaded here will be<br>auto-shared with M3 & M4</div>', unsafe_allow_html=True)
        st.caption("v2.0 | Feature Extraction")

    elif "Milestone 3" in milestone:
        steps_done = sum([st.session_state.m3_files_loaded, st.session_state.m3_anomaly_done, st.session_state.m3_simulation_done])
        pct = int(steps_done / 3 * 100)
        st.markdown(f'<div style="margin-bottom:12px"><div style="font-size:0.72rem;color:#5a7a9a;margin-bottom:4px">PIPELINE · {pct}% COMPLETE</div><div class="prog-wrap"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#6366f1,#a78bfa);border-radius:99px;"></div></div></div>', unsafe_allow_html=True)
        for done, icon, label in [(st.session_state.m3_files_loaded,"📂","Data Loaded"),(st.session_state.m3_anomaly_done,"🚨","Anomalies Detected"),(st.session_state.m3_simulation_done,"🎯","Accuracy Simulated")]:
            dot = "🟢" if done else "⚪"; color = "#dce8f5" if done else "#5a7a9a"
            st.markdown(f'<div style="font-size:0.84rem;padding:3px 0;color:{color}">{dot} {icon} {label}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div style="font-size:0.72rem;color:#5a7a9a;letter-spacing:0.1em">THRESHOLDS</div>', unsafe_allow_html=True)
        m3_hr_high = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="m3_hr_high")
        m3_hr_low  = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="m3_hr_low")
        m3_st_low  = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000, key="m3_st_low")
        m3_sl_low  = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120,  key="m3_sl_low")
        m3_sl_high = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900,  key="m3_sl_high")
        m3_sigma   = st.slider("Residual σ threshold", 1.0, 4.0, 2.0, 0.5, key="m3_sigma")
        st.caption("v3.0 | Anomaly Detection")

    else:  # M4
        pct = 100 if st.session_state.m4_pipeline_done else 0
        st.markdown(f'<div style="margin-bottom:14px"><div style="font-size:0.72rem;color:#5a7a9a;margin-bottom:4px">PIPELINE · {pct}% COMPLETE</div><div class="prog-wrap"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#6366f1,#a78bfa);border-radius:99px;"></div></div></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div style="font-size:0.72rem;color:#5a7a9a;letter-spacing:0.1em">DETECTION THRESHOLDS</div>', unsafe_allow_html=True)
        hr_high = int(st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="m4_hr_high"))
        hr_low  = int(st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="m4_hr_low"))
        st_low  = int(st.number_input("Steps Low/day",    value=500, min_value=0,   max_value=2000, key="m4_st_low"))
        sl_low  = int(st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120,  key="m4_sl_low"))
        sl_high = int(st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900,  key="m4_sl_high"))
        sigma   = float(st.slider("Residual sigma", 1.0, 4.0, 2.0, 0.5, key="m4_sigma"))
        st.markdown("---")
        date_range = None; sel_user = None
        if st.session_state.m4_pipeline_done and st.session_state.m4_master is not None:
            _m = st.session_state.m4_master
            all_dates = pd.to_datetime(_m["Date"])
            d_min_g = all_dates.min().date(); d_max_g = all_dates.max().date()
            st.markdown('<div style="font-size:0.72rem;color:#5a7a9a;letter-spacing:0.1em">DATE FILTER</div>', unsafe_allow_html=True)
            date_range = st.date_input("Date range", value=(d_min_g,d_max_g), min_value=d_min_g, max_value=d_max_g, key="m4_daterange", label_visibility="collapsed")
            st.markdown('<div style="font-size:0.72rem;color:#5a7a9a;letter-spacing:0.1em;margin-top:8px">USER FILTER</div>', unsafe_allow_html=True)
            all_users = sorted(_m["Id"].unique()); user_options = ["All Users"] + [f"...{str(u)[-6:]}" for u in all_users]
            sel_lbl = st.selectbox("User", user_options, key="m4_user", label_visibility="collapsed")
            sel_user = None if sel_lbl=="All Users" else all_users[user_options.index(sel_lbl)-1]
        st.markdown("---")
        st.caption("v4.0 | Dashboard & Export")


# ══════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════
if "Home" in milestone:
    # Hero section
    st.markdown("""
    <div style="text-align:center; padding: 48px 20px 32px;">
        <div style="font-size:4rem; line-height:1; margin-bottom:16px;">💪</div>
        <h1 style="font-size:2.8rem; font-weight:800; color:#a78bfa !important; margin:0 0 12px;">FitPulse</h1>
        <p style="font-size:1.15rem; color:#a8c8e8; max-width:560px; margin:0 auto 8px;">
            Your end-to-end Fitbit data analytics pipeline — from raw CSV to anomaly detection.
        </p>
        <p style="font-size:0.88rem; color:#4a6a8a;">
            Select a milestone below to get started
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="text-align:center; font-size:0.72rem; font-weight:700; letter-spacing:0.14em; color:#4a6a8a; text-transform:uppercase; margin-bottom:20px;">Choose a Milestone</div>', unsafe_allow_html=True)

    # Milestone cards
    cards = [
        {
            "icon": "📊",
            "title": "Milestone 1",
            "subtitle": "Data Preprocessing",
            "desc": "Upload a CSV, clean missing values, detect outliers, and export a polished dataset.",
            "badge": "Start Here",
            "badge_color": "#6366f1",
            "milestone_val": "📊 Milestone 1 — Data Preprocessing",
        },
        {
            "icon": "🧬",
            "title": "Milestone 2",
            "subtitle": "Feature Extraction & Modelling",
            "desc": "Upload 5 Fitbit files, extract TSFresh features, forecast trends, and cluster users.",
            "badge": "Core Pipeline",
            "badge_color": "#10b981",
            "milestone_val": "🧬 Milestone 2 — Feature Extraction & Modelling",
        },
        {
            "icon": "🚨",
            "title": "Milestone 3",
            "subtitle": "Anomaly Detection",
            "desc": "Detect unusual HR, steps, and sleep patterns using threshold, residual & DBSCAN methods.",
            "badge": "Detection",
            "badge_color": "#ef4444",
            "milestone_val": "🚨 Milestone 3 — Anomaly Detection",
        },
        {
            "icon": "📈",
            "title": "Milestone 4",
            "subtitle": "Insights Dashboard",
            "desc": "Interactive dashboard with date & user filters. Export a full PDF or CSV anomaly report.",
            "badge": "Export",
            "badge_color": "#f59e0b",
            "milestone_val": "📈 Milestone 4 — Insights Dashboard",
        },
    ]

    def navigate_to_milestone(m_val):
        st.session_state["milestone_selector"] = m_val

    cols = st.columns(4, gap="medium")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="home-milestone-card">
                <div class="home-card-icon">{card["icon"]}</div>
                <div class="home-card-title">{card["title"]}</div>
                <div style="font-size:0.82rem; font-weight:600; color:#a78bfa; margin-bottom:4px;">{card["subtitle"]}</div>
                <div class="home-card-sub">{card["desc"]}</div>
                <div class="home-card-badge" style="background:rgba(255,255,255,0.07); border:1px solid {card["badge_color"]}; color:{card["badge_color"]};">
                    {card["badge"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            st.button(f"Go to {card['title']} →", key=f"home_btn_{card['title']}", use_container_width=True, on_click=navigate_to_milestone, args=(card["milestone_val"],))

    st.markdown("---")

    # Pipeline flow diagram
    st.markdown('<div style="text-align:center; font-size:0.72rem; font-weight:700; letter-spacing:0.14em; color:#4a6a8a; text-transform:uppercase; margin-bottom:16px;">How It Works</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:center; gap:0; flex-wrap:wrap; margin-bottom:24px;">
        <div style="background:#162a41; border:1px solid #6366f1; border-radius:10px; padding:14px 18px; text-align:center; min-width:130px;">
            <div style="font-size:1.4rem;">📂</div>
            <div style="font-size:0.8rem; font-weight:700; color:#6366f1; margin-top:4px;">Upload CSV</div>
            <div style="font-size:0.68rem; color:#5a7a9a; margin-top:2px;">M1 or M2</div>
        </div>
        <div style="color:#1e4a6e; font-size:1.4rem; padding:0 6px;">→</div>
        <div style="background:#162a41; border:1px solid #10b981; border-radius:10px; padding:14px 18px; text-align:center; min-width:130px;">
            <div style="font-size:1.4rem;">🔬</div>
            <div style="font-size:0.8rem; font-weight:700; color:#10b981; margin-top:4px;">Extract Features</div>
            <div style="font-size:0.68rem; color:#5a7a9a; margin-top:2px;">TSFresh + Prophet</div>
        </div>
        <div style="color:#1e4a6e; font-size:1.4rem; padding:0 6px;">→</div>
        <div style="background:#162a41; border:1px solid #ef4444; border-radius:10px; padding:14px 18px; text-align:center; min-width:130px;">
            <div style="font-size:1.4rem;">🚨</div>
            <div style="font-size:0.8rem; font-weight:700; color:#ef4444; margin-top:4px;">Detect Anomalies</div>
            <div style="font-size:0.68rem; color:#5a7a9a; margin-top:2px;">3 Methods</div>
        </div>
        <div style="color:#1e4a6e; font-size:1.4rem; padding:0 6px;">→</div>
        <div style="background:#162a41; border:1px solid #f59e0b; border-radius:10px; padding:14px 18px; text-align:center; min-width:130px;">
            <div style="font-size:1.4rem;">📄</div>
            <div style="font-size:0.8rem; font-weight:700; color:#f59e0b; margin-top:4px;">Export Report</div>
            <div style="font-size:0.68rem; color:#5a7a9a; margin-top:2px;">PDF + CSV</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick tips
    st.markdown('<div style="text-align:center; font-size:0.72rem; font-weight:700; letter-spacing:0.14em; color:#4a6a8a; text-transform:uppercase; margin-bottom:12px;">Quick Tips</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown('<div class="info-box">💡 <strong>Upload files in M2</strong> — they\'re automatically shared with M3 and M4. No re-uploading needed when you switch milestones.</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="info-box">💡 <strong>Adjust thresholds</strong> in the sidebar (HR, steps, sleep limits and sigma) to tune how sensitive anomaly detection is.</div>', unsafe_allow_html=True)
    with t3:
        st.markdown('<div class="info-box">💡 <strong>M4 has date & user filters</strong> in the sidebar — use them to drill into a specific user or time range before exporting.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MILESTONE 1
# ══════════════════════════════════════════════════════════════
elif "Milestone 1" in milestone:
    st.title("📊 Data Collection & Preprocessing")
    st.markdown("Clean, normalise, and visualise your fitness tracking data easily.")
    st.markdown("---")
    uploaded_file = st.file_uploader("Drop your fitness CSV here", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'processed_df' not in st.session_state: st.session_state.processed_df = None
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Total Rows",    f"{len(df):,}")
        with m2: st.metric("Dimensions",    f"{df.shape[1]} Cols")
        with m3: st.metric("Missing Cells", int(df.isnull().sum().sum()))
        with m4:
            comp = 100 - (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
            st.metric("Completeness", f"{comp:.1f}%")
        tab1, tab2, tab3 = st.tabs(["🔍 Inspection", "⚙️ Processing", "📈 Visualization"])
        with tab1:
            st.subheader("Raw Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            col_nulls, col_types = st.columns(2)
            with col_nulls:
                st.markdown("**Missing Values per Column**")
                null_counts = df.isnull().sum()
                if null_counts[null_counts > 0].empty: st.success("✅ No missing values found!")
                else:
                    fig_null = px.bar(x=null_counts[null_counts > 0].index, y=null_counts[null_counts > 0].values, labels={"x":"Column","y":"Missing Count"}, color_discrete_sequence=["#a78bfa"])
                    fig_null.update_traces(showlegend=False)
                    fig_null.update_layout(title_text="")
                    fig_null.update_xaxes(tickangle=-30)
                    st.plotly_chart(T(fig_null, h=400), use_container_width=True)
            with col_types:
                st.markdown("**Data Types**")
                st.dataframe(df.dtypes.to_frame(name='Type').astype(str), use_container_width=True)
        with tab2:
            st.subheader("Data Cleaning Engine")
            tip("Choose what to fix and click Execute.")
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown("### Settings")
                handle_dates   = st.checkbox("Normalise Dates",     value=True)
                handle_numeric = st.checkbox("Interpolate Numbers", value=True)
                handle_cat     = st.checkbox("Fill Categories",     value=True)
                run_btn        = st.button("🚀 Execute Pipeline")
            with c2:
                if run_btn:
                    with st.spinner("Cleaning your data..."):
                        df_p = df.copy(); logs = []
                        if handle_dates:
                            date_cols = [c for c in df_p.columns if 'date' in c.lower()]
                            for col in date_cols:
                                df_p[col] = pd.to_datetime(df_p[col], dayfirst=True, errors='coerce').ffill(); df_p[col] = df_p[col].dt.date; logs.append(f"Fixed timestamps in: `{col}`")
                        if handle_numeric:
                            num_cols = df_p.select_dtypes(include=[np.number]).columns.tolist()
                            df_p[num_cols] = df_p[num_cols].interpolate().ffill().bfill(); logs.append("Interpolated numeric gaps (Linear Method)")
                        if handle_cat:
                            cat_cols = df_p.select_dtypes(include=['object']).columns
                            for col in cat_cols: df_p[col] = df_p[col].fillna("Unknown")
                            logs.append("Filled missing categories with 'Unknown'")
                        st.session_state.processed_df = df_p; st.success("✅ Preprocessing Complete!")
                        for l in logs: st.markdown(f'<div class="log-box">✅ {l}</div>', unsafe_allow_html=True)
                if st.session_state.processed_df is not None:
                    st.markdown("---"); st.subheader("Cleaned Result Preview")
                    st.dataframe(st.session_state.processed_df.head(5), use_container_width=True)
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Rows", f"{len(st.session_state.processed_df):,}")
                    b2.metric("Missing Before", int(df.isnull().sum().sum()))
                    b3.metric("Missing After", int(st.session_state.processed_df.isnull().sum().sum()), delta=f"-{int(df.isnull().sum().sum()) - int(st.session_state.processed_df.isnull().sum().sum())} fixed", delta_color="inverse")
                    st.download_button("📥 Download Cleaned CSV", data=convert_df(st.session_state.processed_df), file_name='fitness_cleaned.csv', mime='text/csv')
        with tab3:
            if st.session_state.processed_df is not None:
                processed_df = st.session_state.processed_df; st.subheader("🕵️ Outlier Detector")
                tip("Select any numeric column to check if it has unusual values.")
                num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    col_to_plot = st.selectbox("Select a metric to check for unusual values:", num_cols)
                    col_data = processed_df[col_to_plot].dropna()
                    Q1 = col_data.quantile(0.25); Q3 = col_data.quantile(0.75); IQR = Q3 - Q1
                    outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
                    s1,s2,s3,s4,s5 = st.columns(5)
                    s1.metric("Min",f"{col_data.min():,.1f}"); s2.metric("Max",f"{col_data.max():,.1f}"); s3.metric("Mean",f"{col_data.mean():,.1f}"); s4.metric("Median",f"{col_data.median():,.1f}"); s5.metric("Outliers",len(outliers),delta=f"{len(outliers)/len(col_data)*100:.1f}% of data",delta_color="inverse")
                    v1, v2 = st.columns(2)
                    with v1:
                        st.markdown(f"**📦 Box Plot — {col_to_plot}**")
                        fig_box = px.box(processed_df, y=col_to_plot, points="all", color_discrete_sequence=["#a78bfa"])
                        fig_box.update_traces(marker=dict(size=4,opacity=0.5,color="#38bdf8"),line=dict(color="#a78bfa"),fillcolor="rgba(167,139,250,0.3)",boxmean=True)
                        fig_box.update_layout(**{**PT,"height":420},title=dict(text=f"Range View — {col_to_plot}",font=dict(color="#a78bfa",size=13)),yaxis_title=col_to_plot,xaxis=dict(showticklabels=False)); fig_box.update_yaxes(gridcolor="#1e3a5f",zerolinecolor="#1e3a5f")
                        st.plotly_chart(fig_box, use_container_width=True)
                    with v2:
                        st.markdown(f"**📊 Histogram — {col_to_plot}**")
                        lower_bound = Q1 - 1.5*IQR; upper_bound = Q3 + 1.5*IQR
                        fig_hist = px.histogram(processed_df, x=col_to_plot, nbins=40, color_discrete_sequence=["#6366f1"])
                        fig_hist.update_traces(marker_color="#6366f1",marker_line_color="#a78bfa",marker_line_width=1,opacity=0.85)
                        if len(outliers) > 0:
                            fig_hist.add_vrect(x0=col_data.min(),x1=lower_bound,fillcolor="rgba(239,68,68,0.12)",layer="below",line_width=0,annotation_text="Outlier zone",annotation_position="top left",annotation_font_color="#ef4444",annotation_font_size=10)
                            fig_hist.add_vrect(x0=upper_bound,x1=col_data.max(),fillcolor="rgba(239,68,68,0.12)",layer="below",line_width=0,annotation_text="Outlier zone",annotation_position="top right",annotation_font_color="#ef4444",annotation_font_size=10)
                        fig_hist.add_vline(x=col_data.mean(),line_dash="dash",line_color="#f59e0b",line_width=2); fig_hist.add_annotation(x=col_data.mean(),y=1,yref="paper",text=f"Mean: {col_data.mean():,.1f}",showarrow=False,font=dict(color="#f59e0b",size=11),xanchor="left",yanchor="top",bgcolor="rgba(13,27,42,0.7)",bordercolor="#f59e0b")
                        fig_hist.add_vline(x=col_data.median(),line_dash="dot",line_color="#10b981",line_width=2); fig_hist.add_annotation(x=col_data.median(),y=0.88,yref="paper",text=f"Median: {col_data.median():,.1f}",showarrow=False,font=dict(color="#10b981",size=11),xanchor="left",yanchor="top",bgcolor="rgba(13,27,42,0.7)",bordercolor="#10b981")
                        fig_hist.update_layout(**{**PT,"height":420},title=dict(text=f"Frequency Distribution — {col_to_plot}",font=dict(color="#a78bfa",size=13)),xaxis_title=col_to_plot,yaxis_title="Number of Records",bargap=0.03)
                        fig_hist.update_xaxes(gridcolor="#1e3a5f",zerolinecolor="#1e3a5f"); fig_hist.update_yaxes(gridcolor="#1e3a5f",zerolinecolor="#1e3a5f")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    if len(outliers) > 0:
                        st.markdown("---"); st.markdown(f"### 🚨 Outlier Rows — {len(outliers)} found")
                        tip(f"These are the actual rows where **{col_to_plot}** has an unusual value.")
                        outlier_rows = processed_df[(processed_df[col_to_plot] < Q1 - 1.5*IQR) | (processed_df[col_to_plot] > Q3 + 1.5*IQR)].copy()
                        st.dataframe(outlier_rows, use_container_width=True)
                        st.markdown(f'<div class="info-box">⚠️ <strong>Outlier range for {col_to_plot}:</strong><br>Any value below <strong>{lower_bound:,.1f}</strong> or above <strong>{upper_bound:,.1f}</strong> is an outlier.<br>Normal range: <strong>{Q1:,.1f}</strong> (Q1) to <strong>{Q3:,.1f}</strong> (Q3)</div>', unsafe_allow_html=True)
                    else:
                        st.success(f"✅ No outliers found in **{col_to_plot}** — all values are within the normal range.")
                else: st.warning("No numeric columns found.")
            else: st.warning("⚠️ Run the **Processing** tab first to see visualisations.")
    else:
        st.markdown('<div style="text-align:center;padding:100px 20px;border:2px dashed #1e3a5f;border-radius:20px;"><h2 style="color:#4a90c4 !important;">Awaiting Dataset...</h2><p>Upload a CSV file to begin the fitness data transformation.</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MILESTONE 2  — uploads stored to shared_file_keys
# ══════════════════════════════════════════════════════════════
elif "Milestone 2" in milestone:
    st.title("🧬 Fitness Data — Feature Extraction & Modelling")
    st.markdown("Load your Fitbit data, extract patterns, forecast trends, and group users by behaviour.")
    st.markdown("---")

    with st.expander("📂  Step 1 — Upload Your 5 Data Files", expanded=True):
        tip("Upload all 5 files here. They will be <strong>automatically shared</strong> with Milestone 3 and Milestone 4 — no re-uploading needed!")
        uploaded = st.file_uploader("Upload all 5 CSV files", type=["csv"], accept_multiple_files=True, label_visibility="collapsed")
        ROLES = {
            "daily_activity":     ("Daily Activity",    "🏃", ["dailyactivity","daily_activity"]),
            "heartrate":          ("Heart Rate",         "💓", ["heartrate","heart_rate"]),
            "hourly_intensities": ("Hourly Intensities", "⚡", ["hourlyintensities","hourly_intensities"]),
            "hourly_steps":       ("Hourly Steps",       "👟", ["hourlysteps","hourly_steps"]),
            "minute_sleep":       ("Minute Sleep",       "😴", ["minutesleep","minute_sleep"]),
        }
        def match_files(files):
            m = {k: None for k in ROLES}
            for f in files:
                nl = f.name.lower().replace(" ","").replace("-","")
                for role, (_, __, kws) in ROLES.items():
                    if any(kw in nl for kw in kws) and m[role] is None: m[role] = f; break
            return m

        matched   = match_files(uploaded) if uploaded else {k: None for k in ROLES}
        n_matched = sum(1 for v in matched.values() if v is not None)

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        g = st.columns(5)
        for col, (role, (label, icon, _)) in zip(g, ROLES.items()):
            ok_f = matched[role] is not None; fname = matched[role].name if ok_f else "Not found"
            short = (fname[:18] + "…") if len(fname) > 18 else fname
            col.markdown(f'<div style="background:#162a41;border:1px solid {"#10b981" if ok_f else "#1e4a6e"};border-radius:10px;padding:12px 8px;text-align:center;min-height:95px"><div style="font-size:1.7rem">{icon}</div><div style="font-size:0.78rem;font-weight:700;color:{"#10b981" if ok_f else "#6b7280"};margin-top:4px">{label}</div><div style="font-size:0.62rem;color:#5a7a9a;margin-top:5px">{"✅ " + short if ok_f else "❌ not detected"}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        if n_matched == 0:
            st.info("👆 Upload your CSV files above. File names must contain: dailyActivity, heartrate, hourlyIntensities, hourlySteps, minuteSleep.")
            st.stop()
        elif n_matched < 5:
            st.warning(f"Detected {n_matched}/5 files. Check file names contain the keywords above.")

    file_keys = {}
    for role, f in matched.items():
        if f is not None:
            f.seek(0); file_keys[role] = (f.name, f.read()); f.seek(0)

    if file_keys:
        st.session_state["shared_file_keys"] = file_keys

    _cache_key = str(sorted([(r, n) for r, (n, _) in file_keys.items()]))
    _is_first  = st.session_state.get("_last_cache_key") != _cache_key
    status_ph  = st.empty()
    if _is_first:
        status_ph.markdown('<div style="background:linear-gradient(90deg,#1a2744,#1a1f4a);border:1px solid #a78bfa;border-radius:10px;padding:18px 24px;margin:12px 0;text-align:center"><div style="font-size:1.1rem;margin-bottom:8px">⏳ <strong style="color:#a78bfa">Loading and processing your files… please wait</strong></div><div class="prog-wrap"><div class="prog-fill"></div></div><div style="font-size:0.8rem;color:#5a7a9a;margin-top:6px">This only runs once per upload — subsequent interactions will be instant. Files will be auto-shared with M3 and M4.</div></div>', unsafe_allow_html=True)

    dfs = cached_load(file_keys); master_df = cached_master(file_keys)
    hr_resampled = None
    if "heartrate" in file_keys:
        hr_name, hr_bytes = file_keys["heartrate"]; hr_resampled = cached_resample(hr_bytes, hr_name)
    st.session_state["_last_cache_key"] = _cache_key; status_ph.empty()

    st.markdown(f'<div class="shared-box">📂 <strong>Files uploaded and saved.</strong> Milestone 3 and Milestone 4 will automatically use these {n_matched} files — no re-upload needed when you switch milestones.</div>', unsafe_allow_html=True)

    with st.expander("🧹  Data Cleaning Report", expanded=False):
        ROLES_LABELS = {"daily_activity":"Daily Activity","heartrate":"Heart Rate","hourly_intensities":"Hourly Intensities","hourly_steps":"Hourly Steps","minute_sleep":"Minute Sleep"}
        for label, df in dfs.items():
            st.markdown(f"**{ROLES_LABELS.get(label, label)}**")
            raw_df = pd.read_csv(io.BytesIO(file_keys[label][1]))
            dupes=int(raw_df.duplicated().sum()); nulls_b=int(raw_df.isnull().sum().sum()); nulls_a=int(df.isnull().sum().sum()); rows_b=len(raw_df); rows_a=len(df)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Rows (original)",f"{rows_b:,}"); c2.metric("Rows (after clean)",f"{rows_a:,}",delta=f"-{rows_b-rows_a} removed" if rows_b!=rows_a else "none removed",delta_color="inverse")
            c3.metric("Duplicates removed",dupes); c4.metric("Nulls before → after",f"{nulls_b} → {nulls_a}")
            null_cols=raw_df.isnull().sum(); null_cols=null_cols[null_cols>0]
            if not null_cols.empty:
                num_in=raw_df.select_dtypes(include=[np.number]).columns.tolist(); cat_in=raw_df.select_dtypes(include=["object"]).columns.tolist()
                for col,cnt in null_cols.items():
                    action="filled — interpolation" if col in num_in else ("filled with 'Unknown'" if col in cat_in else "rows with bad timestamps dropped")
                    st.markdown(f'<div class="info-box">🔧 <code>{col}</code>: {cnt} missing → {action}</div>', unsafe_allow_html=True)
            else: st.success("No missing values in this file.")
            st.markdown("---")

    if master_df.empty: st.error("Could not build the master dataset."); st.stop()
    steps_col = next((c for c in master_df.columns if "step" in c.lower()), None)
    sleep_col  = next((c for c in master_df.columns if "sleep" in c.lower() or "Sleep" in c), None)

    import pickle
    from sklearn.preprocessing import StandardScaler
    if "Id" in master_df.columns:
        _num_cols = [c for c in master_df.select_dtypes(include=[np.number]).columns if c not in {"KMeans_Cluster","DBSCAN_Cluster","Id","id","logId"}]
        user_df   = master_df.groupby("Id")[_num_cols].mean().reset_index(drop=True)
    else:
        user_df = master_df.copy()
    feat_matrix  = build_clustering_features(user_df)
    feat_scaled  = StandardScaler().fit_transform(feat_matrix.fillna(0))
    feat_pkl     = pickle.dumps(feat_scaled)
    k_range_list, inertias = cached_elbow(feat_pkl, k_max=min(10, len(feat_scaled)-1))
    pca_coords, pca_var    = cached_pca(feat_pkl)

    with st.expander("🔍  Step 2 — Data Preview & Quality Check"):
        tip("Preview each file and check whether any data is missing.")
        tabs = st.tabs([ROLES[k][0] for k in dfs.keys()])
        for tab, (role, df) in zip(tabs, dfs.items()):
            with tab:
                r1,r2,r3 = st.columns(3); r1.metric("Rows",f"{df.shape[0]:,}"); r2.metric("Columns",df.shape[1]); r3.metric("Missing",int(df.isnull().sum().sum()))
                st.dataframe(df.head(6), use_container_width=True)
        m1_,m2_,m3_,m4_ = st.columns(4)
        m1_.metric("Total Rows",f"{master_df.shape[0]:,}"); m2_.metric("Columns",master_df.shape[1]); m3_.metric("Missing Cells",int(master_df.isnull().sum().sum())); m4_.metric("Completeness",f"{100-master_df.isnull().sum().sum()/master_df.size*100:.1f}%")
        st.dataframe(master_df.head(6), use_container_width=True)
        st.download_button("📥 Download Merged Dataset", master_df.to_csv(index=False).encode(), "fitness_master.csv", "text/csv")

    with st.expander("🔬  Step 3 — Automatic Feature Extraction (TSFresh)"):
        tip("TSFresh reads heart rate over time and automatically calculates useful numbers per user.")
        tsf_input = prepare_tsfresh_input(hr_resampled) if hr_resampled is not None else None
        tsf_features = st.session_state.get("tsf_features")
        if tsf_input is None: st.warning("Heart rate file not found — upload it to enable TSFresh.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Users",tsf_input["id"].nunique()); c2.metric("Heart Rate Readings",f"{len(tsf_input):,}"); c3.metric("Features Extracted",tsf_features.shape[1] if tsf_features is not None else "—")
            if tsf_features is None:
                if st.button("▶️  Extract Features", key="run_tsf"):
                    prog_ph = st.empty(); prog_ph.markdown('<div class="info-box">🔬 Extracting heart rate features… this takes 30–90 seconds.</div>', unsafe_allow_html=True)
                    try:
                        from tsfresh import extract_features; from tsfresh.utilities.dataframe_functions import impute; from tsfresh.feature_extraction import MinimalFCParameters
                        feat = extract_features(tsf_input,column_id="id",column_sort="time",column_value="value",default_fc_parameters=MinimalFCParameters(),n_jobs=1,disable_progressbar=True)
                        impute(feat); st.session_state["tsf_features"]=feat; tsf_features=feat; prog_ph.empty(); st.success(f"✅ Done! Extracted {feat.shape[1]} features for {feat.shape[0]} users.")
                    except ImportError: prog_ph.empty(); st.error("Run: pip install tsfresh")
                    except Exception as e: prog_ph.empty(); st.error(str(e))
            else: st.success(f"✅ {tsf_features.shape[1]} features extracted for {tsf_features.shape[0]} users.")
            if tsf_features is not None and not tsf_features.empty:
                n_show=min(12,tsf_features.shape[1]); sub=tsf_features.iloc[:,:n_show].copy()
                norm=(sub-sub.min())/(sub.max()-sub.min()+1e-9); norm.columns=norm.columns.str.replace("value__","",regex=False).str.replace("_"," ",regex=False).str.title(); norm.index=["User "+str(i) for i in norm.index]
                fig_h=px.imshow(norm,color_continuous_scale="RdBu_r",zmin=0,zmax=1,aspect="auto",text_auto=".2f",title="Heart Rate Feature Scores per User  (0 = low · 1 = high)")
                fig_h.update_layout(height=max(320,len(norm)*40+120),xaxis=dict(tickangle=-35),**PT); st.plotly_chart(fig_h,use_container_width=True)

    with st.expander("📈  Step 4 — Trend Forecasting (Prophet)"):
        tip("Prophet looks at past data and predicts future values.")
        def prophet_plot(ds_series,y_series,label,key,unit="",color="#10b981"):
            tmp=pd.DataFrame({"ds":pd.to_datetime(ds_series,errors="coerce"),"y":pd.to_numeric(y_series,errors="coerce")}).dropna().sort_values("ds")
            if len(tmp)<10: st.warning(f"Not enough data to forecast {label}."); return
            run=st.button(f"▶️  Forecast {label}",key=f"btn_{key}")
            if run:
                ph=st.empty(); ph.markdown(f'<div class="info-box">📈 Fitting Prophet for {label}… please wait.</div>',unsafe_allow_html=True)
                try:
                    from prophet import Prophet; m=Prophet(weekly_seasonality=True,daily_seasonality=False,interval_width=0.80); m.fit(tmp); fc=m.predict(m.make_future_dataframe(periods=30,freq="D")); st.session_state[f"fc_{key}"]=(fc,tmp); ph.empty()
                except ImportError: ph.empty(); st.error("Run: pip install prophet")
                except Exception as e: ph.empty(); st.error(str(e))
            saved=st.session_state.get(f"fc_{key}")
            if saved:
                fc,tmp2=saved; cut_str=str(tmp2["ds"].max())
                def hex_rgba(h,a): h=h.lstrip("#"); return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"
                ci_fill=hex_rgba(color,0.30); ci_line=hex_rgba(color,0.0)
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=fc["ds"],y=fc["yhat_upper"],mode="lines",line=dict(color=ci_line,width=0),name="CI Upper",showlegend=False))
                fig.add_trace(go.Scatter(x=fc["ds"],y=fc["yhat_lower"],mode="lines",fill="tonexty",fillcolor=ci_fill,line=dict(color=ci_line,width=0),name="CI Lower"))
                fig.add_trace(go.Scatter(x=fc["ds"],y=fc["yhat"],mode="lines",line=dict(color="#111827",width=2),name="Trend"))
                y_lbl=f"Actual {label}{' (' + unit + ')' if unit else ''}"
                fig.add_trace(go.Scatter(x=tmp2["ds"],y=tmp2["y"],mode="markers",marker=dict(color=color,size=7,opacity=0.85,line=dict(color="white",width=0.5)),name=y_lbl))
                PT_fc={k:v for k,v in PT.items() if k!="legend"}
                fig.update_layout(title=f"{label} — Prophet Trend Forecast",xaxis_title="Date",yaxis_title=f"{label}{' (' + unit + ')' if unit else ''}",legend=dict(orientation="v",x=0.01,y=0.99,bgcolor="rgba(22,42,65,0.85)",bordercolor="#1e3a5f",borderwidth=1,font=dict(size=11,color="#dce8f5")),shapes=[dict(type="line",x0=cut_str,x1=cut_str,y0=0,y1=1,xref="x",yref="paper",line=dict(color="#f59e0b",width=1.8,dash="dash"))],annotations=[dict(x=cut_str,y=0.99,xref="x",yref="paper",text="Forecast Start",showarrow=False,font=dict(color="#f59e0b",size=10),xanchor="left",yanchor="top",bgcolor="rgba(22,42,65,0.7)")],hovermode="x unified",height=420,**PT_fc)
                fig.update_xaxes(gridcolor="#1e3a5f",zerolinecolor="#1e3a5f",tickformat="%Y-%m-%d",tickangle=-20); fig.update_yaxes(gridcolor="#1e3a5f",zerolinecolor="#1e3a5f")
                st.plotly_chart(fig,use_container_width=True)
                tip(f"Dots = actual {label.lower()}. Black line = Prophet trend. Shaded band = 80% CI. Orange dashed = forecast start.")

        if hr_resampled is not None and "Time" in hr_resampled.columns:
            st.markdown("### 💓 Heart Rate")
            _hr=hr_resampled.copy(); _hr["Time"]=pd.to_datetime(_hr["Time"],errors="coerce"); _hr["ActivityDate"]=_hr["Time"].dt.normalize()
            _hr_daily=(_hr.groupby(["Id","ActivityDate"])["Value"].mean().reset_index().groupby("ActivityDate")["Value"].mean().reset_index() if "Id" in _hr.columns else _hr.groupby("ActivityDate")["Value"].mean().reset_index())
            _hr_daily=_hr_daily.sort_values("ActivityDate").reset_index(drop=True)
            prophet_plot(_hr_daily["ActivityDate"],_hr_daily["Value"],"Heart Rate","hr","bpm",color="#38bdf8")
        else: st.info("Heart rate data not available.")
        st.markdown("---")
        if steps_col and "ActivityDate" in master_df.columns:
            st.markdown("### 👟 Daily Steps"); agg=master_df.groupby("ActivityDate")[steps_col].mean().reset_index(); prophet_plot(agg["ActivityDate"],agg[steps_col],"Daily Steps",steps_col,"steps",color="#10b981")
        else: st.info("Steps data not available.")
        st.markdown("---")
        if sleep_col and "ActivityDate" in master_df.columns:
            st.markdown("### 😴 Sleep Duration"); agg=master_df.groupby("ActivityDate")[sleep_col].mean().reset_index(); prophet_plot(agg["ActivityDate"],agg[sleep_col],"Sleep (minutes)",sleep_col,"min",color="#a78bfa")
        else: st.info("Sleep data not available.")

    with st.expander("🔵  Step 5 — User Grouping (Clustering)"):
        from sklearn.cluster import KMeans, DBSCAN
        st.markdown("### 📐 Elbow Chart — How Many Groups?")
        tip("Look for the **elbow** — where the line bends and stops dropping steeply.")
        fig_el=px.line(x=k_range_list,y=inertias,markers=True,labels={"x":"Number of Groups (k)","y":"Spread Score"},title="Elbow Chart",color_discrete_sequence=["#a78bfa"])
        fig_el.update_traces(marker=dict(size=10,color="#38bdf8",line=dict(color="white",width=1.5))); st.plotly_chart(T(fig_el),use_container_width=True)
        st.markdown("---"); st.markdown("### ⚙️ Clustering Settings")
        cc1,cc2,cc3=st.columns(3)
        n_k=cc1.slider("KMeans — number of groups (k)",2,min(10,len(feat_scaled)-1),3,key="k"); eps=cc2.slider("DBSCAN — neighbourhood size (eps)",0.3,5.0,2.5,0.1,key="eps"); msamp=cc3.slider("DBSCAN — minimum group size",2,15,3,key="ms")
        km_labels=KMeans(n_clusters=n_k,random_state=42,n_init="auto").fit_predict(feat_scaled); db_labels=DBSCAN(eps=eps,min_samples=msamp).fit_predict(feat_scaled)
        user_df["KMeans_Cluster"]=km_labels; user_df["DBSCAN_Cluster"]=db_labels
        n_db_c=len(set(db_labels))-(1 if -1 in db_labels else 0); n_noise=int((db_labels==-1).sum())
        v1,v2=pca_var; pca_df=pd.DataFrame(pca_coords,columns=["PC1","PC2"]); pca_df["KMeans"]=["Group "+str(l) for l in km_labels]; pca_df["DBSCAN"]=["Outlier" if l==-1 else "Group "+str(l) for l in db_labels]
        st.markdown("---"); st.markdown("### 🗺️ User Group Maps (PCA)"); tip("Each **dot = one user**. Same colour = same group.")
        p1,p2=st.columns(2)
        with p1:
            fig_km=px.scatter(pca_df,x="PC1",y="PC2",color="KMeans",title=f"KMeans: {n_k} User Groups",labels={"PC1":f"Dim 1 ({v1:.0%})","PC2":f"Dim 2 ({v2:.0%})","KMeans":"Group"},color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
            fig_km.update_traces(marker=dict(size=13,opacity=0.85,line=dict(color="white",width=1))); st.plotly_chart(T(fig_km),use_container_width=True)
        with p2:
            _db_unique=sorted([x for x in pca_df["DBSCAN"].unique() if x!="Outlier"])+(["Outlier"] if "Outlier" in pca_df["DBSCAN"].values else [])
            _db_pal=["#a78bfa","#38bdf8","#10b981","#f59e0b","#f97316","#06b6d4","#84cc16","#e879f9","#facc15","#4ade80","#60a5fa","#fb923c"]
            _db_cm={g:_db_pal[i%len(_db_pal)] for i,g in enumerate([x for x in _db_unique if x!="Outlier"])}
            if "Outlier" in _db_unique: _db_cm["Outlier"]="#94a3b8"
            fig_db=px.scatter(pca_df,x="PC1",y="PC2",color="DBSCAN",title=f"DBSCAN: {n_db_c} Groups + {n_noise} Outliers",labels={"PC1":f"Dim 1 ({v1:.0%})","PC2":f"Dim 2 ({v2:.0%})","DBSCAN":"Group"},category_orders={"DBSCAN":_db_unique},color_discrete_map=_db_cm)
            for trace in fig_db.data:
                if trace.name=="Outlier": trace.marker.update(size=7,opacity=0.35,line=dict(color="white",width=0.3))
                else: trace.marker.update(size=13,opacity=0.9,line=dict(color="white",width=1))
            st.plotly_chart(T(fig_db),use_container_width=True)
        tip("**Outlier** (small grey dots) = a user whose habits don't fit any group.")
        st.markdown("---"); st.markdown("### 🌐 Advanced Group Map (t-SNE)")
        if "tsne_df" not in st.session_state:
            if st.button("▶️  Run t-SNE",key="run_tsne"):
                ph=st.empty(); ph.markdown('<div class="info-box">🌐 Building t-SNE map… takes ~20 seconds.</div>',unsafe_allow_html=True)
                try:
                    from sklearn.manifold import TSNE; perp=min(30,max(5,len(feat_scaled)-1))
                    tc=TSNE(n_components=2,random_state=42,perplexity=perp,max_iter=1000).fit_transform(feat_scaled)
                    tsne_df=pd.DataFrame(tc,columns=["tSNE1","tSNE2"]); tsne_df["KMeans"]=["Group "+str(l) for l in km_labels]; tsne_df["DBSCAN"]=["Outlier" if l==-1 else "Group "+str(l) for l in db_labels]
                    st.session_state["tsne_df"]=tsne_df; ph.empty(); st.rerun()
                except Exception as e: ph.empty(); st.error(str(e))
        else:
            tsne_df=st.session_state["tsne_df"]; tsne_df["KMeans"]=["Group "+str(l) for l in km_labels]; tsne_df["DBSCAN"]=["Outlier" if l==-1 else "Group "+str(l) for l in db_labels]
            t1,t2=st.columns(2)
            with t1:
                fig_t1=px.scatter(tsne_df,x="tSNE1",y="tSNE2",color="KMeans",title="t-SNE — KMeans Groups",labels={"tSNE1":"X","tSNE2":"Y","KMeans":"Group"},color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
                fig_t1.update_traces(marker=dict(size=12,opacity=0.85,line=dict(color="white",width=1))); st.plotly_chart(T(fig_t1),use_container_width=True)
            with t2:
                _t_unique=sorted([x for x in tsne_df["DBSCAN"].unique() if x!="Outlier"])+(["Outlier"] if "Outlier" in tsne_df["DBSCAN"].values else [])
                _t_pal=["#a78bfa","#38bdf8","#10b981","#f59e0b","#f97316","#06b6d4","#84cc16","#e879f9","#facc15","#4ade80","#60a5fa","#fb923c"]
                _t_cm={g:_t_pal[i%len(_t_pal)] for i,g in enumerate([x for x in _t_unique if x!="Outlier"])}
                if "Outlier" in _t_unique: _t_cm["Outlier"]="#94a3b8"
                fig_t2=px.scatter(tsne_df,x="tSNE1",y="tSNE2",color="DBSCAN",title="t-SNE — DBSCAN Groups",labels={"tSNE1":"X","tSNE2":"Y","DBSCAN":"Group"},category_orders={"DBSCAN":_t_unique},color_discrete_map=_t_cm)
                for trace in fig_t2.data:
                    if trace.name=="Outlier": trace.marker.update(size=7,opacity=0.35,line=dict(color="white",width=0.3))
                    else: trace.marker.update(size=12,opacity=0.9,line=dict(color="white",width=1))
                st.plotly_chart(T(fig_t2),use_container_width=True)
            if st.button("🔄 Re-run t-SNE",key="rerun_tsne"): del st.session_state["tsne_df"]; st.rerun()
        st.markdown("---"); st.markdown("### 📊 What Does Each Group Look Like?")
        _excl2={"KMeans_Cluster","DBSCAN_Cluster","Id","id","logId"}; num_cols2=[c for c in user_df.select_dtypes(include=[np.number]).columns if c not in _excl2]
        _priority=["TotalSteps","TotalDistance","Calories","AvgHeartRate","TotalSleepMinutes","TotalMinutesAsleep","VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes"]
        ordered=[c for c in _priority if c in num_cols2]; remaining=[c for c in num_cols2 if c not in ordered]; plot_cols=(ordered+remaining)[:8]
        if plot_cols:
            profile=user_df.groupby("KMeans_Cluster")[plot_cols].mean().reset_index(); long=profile.melt(id_vars="KMeans_Cluster",var_name="Metric",value_name="Average"); long["Group"]="Group "+long["KMeans_Cluster"].astype(str)
            fig_bar=px.bar(long,x="Metric",y="Average",color="Group",barmode="group",title="Average Metric per User Group",color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
            fig_bar.update_xaxes(tickangle=-20); fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>Average: %{y:,.1f}<extra>%{fullData.name}</extra>")
            st.plotly_chart(T(fig_bar,h=420),use_container_width=True)
        _gc=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444"]
        for i,row in user_df.groupby("KMeans_Cluster")[plot_cols if plot_cols else num_cols2].mean().iterrows():
            parts=[]; al="Unknown"
            if steps_col and steps_col in row.index:
                v=row[steps_col]; al=("🏃 Highly Active" if v>10000 else "🚶 Moderately Active" if v>7000 else "🧘 Lightly Active" if v>4000 else "🛋️ Sedentary"); parts.append(f"Avg <strong>{v:,.0f} steps/day</strong>")
            if "AvgHeartRate" in row.index: hr_v=row["AvgHeartRate"]; parts.append(f"HR <strong>{hr_v:.0f} bpm</strong>")
            if sleep_col and sleep_col in row.index: v=row[sleep_col]; parts.append(f"<strong>{v:.0f} min</strong> sleep")
            color=_gc[i%len(_gc)]; n_u=int((user_df["KMeans_Cluster"]==i).sum()); desc="  &nbsp;·&nbsp;  ".join(parts) if parts else "See chart above."
            st.markdown(f'<div style="background:#162a41;border-left:4px solid {color};border-radius:8px;padding:12px 16px;margin:8px 0;"><div style="display:flex;align-items:center;gap:10px;margin-bottom:6px"><span style="background:{color};color:#0d1b2a;font-weight:800;font-size:0.78rem;padding:2px 8px;border-radius:20px">Group {i}</span><span style="color:{color};font-weight:700;font-size:1rem">{al}</span><span style="color:#5a7a9a;font-size:0.78rem;margin-left:auto">{n_u} user(s)</span></div><div style="color:#dce8f5;font-size:0.88rem">{desc}</div></div>',unsafe_allow_html=True)

    with st.expander("🏁  Summary & Download"):
        st.success("🎉 Milestone 2 pipeline complete!")
        tsf_features=st.session_state.get("tsf_features")
        r1,r2,r3=st.columns(3); r1.metric("Files Loaded",f"{n_matched}/5"); r2.metric("Users",master_df["Id"].nunique() if "Id" in master_df.columns else "—"); r3.metric("Days of Data",f"{master_df.shape[0]:,} rows")
        ok(f"Files are saved and shared. Switch to Milestone 3 or 4 to continue — no re-upload needed.")
        st.download_button("📥 Download Master Dataset (CSV)", master_df.to_csv(index=False).encode(), "fitness_m2_master.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
#  MILESTONE 3  — uses shared files from M2, or own upload
# ══════════════════════════════════════════════════════════════
elif "Milestone 3" in milestone:
    st.title("🚨 Anomaly Detection & Visualization")
    st.markdown("Detect unusual health patterns using Threshold Violations, Residual-Based detection, and DBSCAN Structural Outliers.")
    st.markdown("---")

    shared_fk = st.session_state.get("shared_file_keys")
    use_shared = shared_fk is not None and len(shared_fk) >= 5

    with st.expander("📂  Step 1 — Data Files", expanded=True):
        if use_shared:
            st.markdown(f"""
            <div class="shared-box">
                📂 <strong>Files automatically loaded from Milestone 2</strong> — {len(shared_fk)} files ready.
                No upload needed. You can proceed directly to detection below.
            </div>""", unsafe_allow_html=True)
            shared_detected = file_keys_to_m3_detected(shared_fk)
            g = st.columns(5)
            for col, (req_name, finfo) in zip(g, M3_REQUIRED_FILES.items()):
                found = req_name in shared_detected
                bor = "#10b981" if found else "#f59e0b"; tc = "#10b981" if found else "#f59e0b"
                col.markdown(f'<div style="background:#162a41;border:1px solid {bor};border-radius:10px;padding:12px 8px;text-align:center;min-height:95px"><div style="font-size:1.7rem">{finfo["icon"]}</div><div style="font-size:0.78rem;font-weight:700;color:{tc};margin-top:4px">{finfo["label"]}</div><div style="font-size:0.62rem;color:#5a7a9a;margin-top:5px">{"✅ From M2" if found else "⚠️ Not matched"}</div></div>', unsafe_allow_html=True)
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            if st.checkbox("🔄 Upload different files instead", key="m3_override_upload"):
                use_shared = False

        if not use_shared:
            tip("Upload the same 5 files from Milestone 2. Files are auto-detected by column structure.")
            m3_uploaded = st.file_uploader("Upload all 5 CSV files", type=["csv"], accept_multiple_files=True, label_visibility="collapsed", key="m3_file_uploader")
            m3_detected = {}; m3_raw_uploads = []
            if m3_uploaded:
                for uf in m3_uploaded:
                    try: uf.seek(0); df_tmp = pd.read_csv(uf); m3_raw_uploads.append((uf.name, df_tmp))
                    except Exception: pass
                for req_name, finfo in M3_REQUIRED_FILES.items():
                    best_score, best_df = 0, None
                    for uname, udf in m3_raw_uploads:
                        s = score_match(udf, finfo)
                        if s > best_score: best_score, best_df = s, udf
                    if best_score >= 2: m3_detected[req_name] = best_df
            m3_n_matched = len(m3_detected)
            g = st.columns(5)
            for col, (req_name, finfo) in zip(g, M3_REQUIRED_FILES.items()):
                found = req_name in m3_detected; bor = "#10b981" if found else "#1e4a6e"; tc = "#10b981" if found else "#6b7280"
                col.markdown(f'<div style="background:#162a41;border:1px solid {bor};border-radius:10px;padding:12px 8px;text-align:center;min-height:95px"><div style="font-size:1.7rem">{finfo["icon"]}</div><div style="font-size:0.78rem;font-weight:700;color:{tc};margin-top:4px">{finfo["label"]}</div><div style="font-size:0.62rem;color:#5a7a9a;margin-top:5px">{"✅ Detected" if found else "❌ Not detected"}</div></div>', unsafe_allow_html=True)
            st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
            if m3_n_matched == 0: st.info("👆 Upload your 5 CSV files above."); st.stop()
            elif m3_n_matched < 5: warn(f"Detected {m3_n_matched}/5 files.")
        else:
            m3_detected = file_keys_to_m3_detected(shared_fk)

        if st.button("⚡  Load & Build Master DataFrame", key="m3_load_btn"):
            prog_ph = st.empty()
            def update_prog(txt, pct):
                prog_ph.markdown(f'<div style="background:linear-gradient(90deg,#1a2744,#1a1f4a);border:1px solid #10b981;border-radius:12px;padding:16px 20px;margin:12px 0"><div style="font-size:0.95rem;color:#10b981;font-weight:700;margin-bottom:8px">⏳ {txt}</div><div class="prog-wrap"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#10b981,#38bdf8);border-radius:99px;"></div></div></div>',unsafe_allow_html=True)
            try:
                update_prog("Loading data files...", 15)
                daily = m3_detected["dailyActivity_merged.csv"].copy()
                sleep = m3_detected.get("minuteSleep_merged.csv", pd.DataFrame()).copy()
                hr    = m3_detected.get("heartrate_seconds_merged.csv", pd.DataFrame()).copy()
                def safe_dt(series, fmt):
                    try: return pd.to_datetime(series, format=fmt)
                    except: return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
                update_prog("Formatting dates...", 30)
                daily["ActivityDate"] = safe_dt(daily["ActivityDate"], "%m/%d/%Y")
                hr_minute = pd.DataFrame(); hr_daily = pd.DataFrame()
                if not hr.empty and "Time" in hr.columns and "Value" in hr.columns:
                    update_prog("Processing heart rate data...", 50)
                    hr["Time"] = safe_dt(hr["Time"], "%m/%d/%Y %I:%M:%S %p")
                    hr_minute = hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index()
                    hr_minute.columns = ["Id","Time","HeartRate"]; hr_minute = hr_minute.dropna(); hr_minute["Date"] = hr_minute["Time"].dt.date
                    hr_daily = hr_minute.groupby(["Id","Date"])["HeartRate"].agg(["mean","max","min","std"]).reset_index().rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"})
                sleep_daily = pd.DataFrame()
                if not sleep.empty and "date" in sleep.columns and "value" in sleep.columns:
                    update_prog("Processing sleep data...", 70)
                    sleep["date"] = safe_dt(sleep["date"], "%m/%d/%Y %I:%M:%S %p"); sleep["Date"] = sleep["date"].dt.date
                    sleep_daily = sleep.groupby(["Id","Date"]).agg(TotalSleepMinutes=("value","count")).reset_index()
                update_prog("Merging all data into master table...", 85)
                master = daily.copy().rename(columns={"ActivityDate":"Date"}); master["Date"] = master["Date"].dt.date
                if not hr_daily.empty:
                    master = master.merge(hr_daily, on=["Id","Date"], how="left")
                    for col in ["AvgHR","MaxHR","MinHR","StdHR"]: master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))
                if not sleep_daily.empty:
                    master = master.merge(sleep_daily, on=["Id","Date"], how="left"); master["TotalSleepMinutes"] = master["TotalSleepMinutes"].fillna(0)
                update_prog("Finalising...", 100)
                st.session_state.m3_master = master; st.session_state.m3_hr_minute = hr_minute; st.session_state.m3_files_loaded = True; prog_ph.empty(); st.rerun()
            except Exception as e: prog_ph.empty(); st.error(f"Error: {e}"); st.exception(e)

        if st.session_state.m3_files_loaded:
            master = st.session_state.m3_master
            ok(f"Master DataFrame ready — **{master.shape[0]:,} rows** · **{master['Id'].nunique()} users** · **{master.shape[1]} columns**")
            m1, m2_, m3__, m4_ = st.columns(4)
            m1.metric("Total Rows",f"{master.shape[0]:,}"); m2_.metric("Users",master["Id"].nunique())
            m3__.metric("Date Range",f"{pd.to_datetime(master['Date']).min().strftime('%d %b')} → {pd.to_datetime(master['Date']).max().strftime('%d %b %y')}"); m4_.metric("Columns",master.shape[1])

    if not st.session_state.m3_files_loaded: st.stop()
    master = st.session_state.m3_master
    m3_hr_high = st.session_state.get("m3_hr_high", 100); m3_hr_low = st.session_state.get("m3_hr_low", 50)
    m3_st_low  = st.session_state.get("m3_st_low",  500); m3_sl_low = st.session_state.get("m3_sl_low", 60)
    m3_sl_high = st.session_state.get("m3_sl_high", 600); m3_sigma  = st.session_state.get("m3_sigma",  2.0)

    st.markdown('<div class="section-label">Detection Methods Applied</div>', unsafe_allow_html=True)
    mc1, mc2, mc3_ = st.columns(3)
    with mc1: st.markdown(f'<div class="method-card"><div class="method-title" style="color:#ef4444">① Threshold Violations</div><div class="method-desc">Hard upper/lower limits on HR, Steps, Sleep.</div></div>', unsafe_allow_html=True)
    with mc2: st.markdown(f'<div class="method-card"><div class="method-title" style="color:#f59e0b">② Residual-Based</div><div class="method-desc">Rolling median baseline. Flag days deviating by ±{m3_sigma:.0f}σ.</div></div>', unsafe_allow_html=True)
    with mc3_: st.markdown('<div class="method-card"><div class="method-title" style="color:#10b981">③ DBSCAN Outliers</div><div class="method-desc">Users labelled −1 are structural outliers.</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    if st.button("🔵  Run Anomaly Detection (All 3 Methods)", key="run_m3_anomaly"):
        with st.spinner("Running all 3 detection methods…"):
            try:
                anom_hr = detect_hr_anomalies(master, m3_hr_high, m3_hr_low, m3_sigma)
                anom_steps = detect_steps_anomalies(master, m3_st_low, 25000, m3_sigma)
                anom_sleep = detect_sleep_anomalies(master, m3_sl_low, m3_sl_high, m3_sigma)
                st.session_state.m3_anom_hr = anom_hr; st.session_state.m3_anom_steps = anom_steps; st.session_state.m3_anom_sleep = anom_sleep; st.session_state.m3_anomaly_done = True; st.rerun()
            except Exception as e: st.error(f"Detection error: {e}"); st.exception(e)

    if not st.session_state.m3_anomaly_done: st.info("👆 Click **Run Anomaly Detection** to begin."); st.stop()
    anom_hr = st.session_state.m3_anom_hr; anom_steps = st.session_state.m3_anom_steps; anom_sleep = st.session_state.m3_anom_sleep
    n_hr = int(anom_hr["is_anomaly"].sum()); n_steps = int(anom_steps["is_anomaly"].sum()); n_sleep = int(anom_sleep["is_anomaly"].sum()); n_total = n_hr + n_steps + n_sleep

    st.markdown(f'<div class="alert-box" style="font-size:0.95rem;font-weight:600">🚨 Total anomalies flagged: {n_total} &nbsp;(HR: {n_hr} · Steps: {n_steps} · Sleep: {n_sleep})</div>', unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns(4)
    for col, label, num in zip([cc1,cc2,cc3,cc4],["HR ANOMALIES","STEPS ANOMALIES","SLEEP ANOMALIES","TOTAL FLAGS"],[n_hr,n_steps,n_sleep,n_total]):
        col.markdown(f'<div class="count-card"><div class="count-num">{num}</div><div class="count-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("💓  Chart 1 — Heart Rate Anomaly Detection Chart", expanded=True):
        tip("Blue line = actual HR · Green dotted = rolling median · Shaded band = ±σ expected zone · Red circles = anomalies.")
        hr_anom = anom_hr[anom_hr["is_anomaly"]]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Days",len(anom_hr)); c2.metric("Avg HR",f"{anom_hr['AvgHR'].mean():.1f} bpm"); c3.metric("🚨 Anomalies",n_hr,delta=f"{n_hr/len(anom_hr)*100:.1f}% of days",delta_color="inverse" if n_hr>0 else "off"); c4.metric("Max HR",f"{anom_hr['AvgHR'].max():.1f} bpm")
        if n_hr > 0: alert(f"**{n_hr} anomalous HR days** — outside [{m3_hr_low}–{m3_hr_high} bpm] or ±{m3_sigma:.0f}σ.")
        else: ok("No HR anomalies detected.")
        resid_std_hr = anom_hr["residual"].std(); upper_hr = anom_hr["rolling_med"] + m3_sigma * resid_std_hr; lower_hr = anom_hr["rolling_med"] - m3_sigma * resid_std_hr; xs = anom_hr["Date"].tolist()
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(x=xs+xs[::-1],y=upper_hr.tolist()+lower_hr.tolist()[::-1],fill="toself",fillcolor="rgba(56,189,248,0.12)",line=dict(color="rgba(0,0,0,0)",width=0),name=f"±{m3_sigma:.0f}σ Band",hoverinfo="skip"))
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"],y=anom_hr["AvgHR"],mode="lines+markers",line=dict(color="#38bdf8",width=2.5),marker=dict(size=5,color="#38bdf8",opacity=0.75),name="Avg Heart Rate",hovertemplate="📅 %{x|%Y-%m-%d}<br>💓 %{y:.1f} bpm<extra></extra>"))
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"],y=anom_hr["rolling_med"],mode="lines",line=dict(color="#10b981",width=1.8,dash="dot"),name="Rolling Median"))
        fig_hr.add_hline(y=m3_hr_high,line_dash="dash",line_color="#ef4444",line_width=1.5,annotation_text=f"High ({m3_hr_high} bpm)",annotation_font_color="#ef4444",annotation_font_size=10,annotation_position="top right")
        fig_hr.add_hline(y=m3_hr_low,line_dash="dash",line_color="#f9a8d4",line_width=1.5,annotation_text=f"Low ({m3_hr_low} bpm)",annotation_font_color="#f9a8d4",annotation_font_size=10,annotation_position="bottom right")
        if not hr_anom.empty:
            fig_hr.add_trace(go.Scatter(x=hr_anom["Date"],y=hr_anom["AvgHR"],mode="markers+text",marker=dict(size=18,color="rgba(239,68,68,0.85)",symbol="circle",line=dict(color="white",width=2)),text=["▲ Residual±2σ"]*len(hr_anom),textposition="top center",textfont=dict(size=9,color="#fbbf24"),name="🚨 Anomaly"))
        T(fig_hr,h=460); fig_hr.update_layout(title="Heart Rate — Anomaly Detection Chart",xaxis_title="Date",yaxis_title="Heart Rate (bpm)",hovermode="x unified"); fig_hr.update_xaxes(tickformat="%d %b",tickangle=-30)
        st.plotly_chart(fig_hr,use_container_width=True)
        if not hr_anom.empty:
            with st.expander(f"📋 View {n_hr} HR Anomaly Records"):
                st.dataframe(hr_anom[["Date","AvgHR","rolling_med","residual","reason"]].rename(columns={"rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"}).round(2),use_container_width=True)

    with st.expander("😴  Chart 2 — Sleep Pattern Visualization", expanded=True):
        tip("Dual subplot: Top = sleep + anomaly markers · Bottom = residual bars (red = anomaly).")
        sleep_anom = anom_sleep[anom_sleep["is_anomaly"]]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Days",len(anom_sleep)); c2.metric("Avg Sleep",f"{anom_sleep['TotalSleepMinutes'].mean():.0f} min"); c3.metric("🚨 Anomalies",n_sleep,delta=f"{n_sleep/len(anom_sleep)*100:.1f}% of days",delta_color="inverse" if n_sleep>0 else "off"); c4.metric("Days < 60 min",int((anom_sleep["TotalSleepMinutes"]<60).sum()))
        if n_sleep > 0: alert(f"**{n_sleep} anomalous sleep days** — outside [{m3_sl_low}–{m3_sl_high} min] or ±{m3_sigma:.0f}σ.")
        else: ok("Sleep patterns look normal.")
        fig_sleep = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.08,subplot_titles=("Sleep Duration (minutes/night)","Deviation from Expected"))
        xs_sl = anom_sleep["Date"].tolist()
        fig_sleep.add_trace(go.Scatter(x=xs_sl+xs_sl[::-1],y=[m3_sl_high]*len(xs_sl)+[m3_sl_low]*len(xs_sl),fill="toself",fillcolor="rgba(16,185,129,0.09)",line=dict(color="rgba(0,0,0,0)",width=0),name=f"Healthy Zone",hoverinfo="skip"),row=1,col=1)
        fig_sleep.add_trace(go.Scatter(x=anom_sleep["Date"],y=anom_sleep["TotalSleepMinutes"],mode="lines+markers",line=dict(color="#a78bfa",width=2.5),marker=dict(size=5,color="#a78bfa",opacity=0.75),name="Sleep Minutes"),row=1,col=1)
        fig_sleep.add_trace(go.Scatter(x=anom_sleep["Date"],y=anom_sleep["rolling_med"],mode="lines",line=dict(color="#10b981",width=1.8,dash="dot"),name="Rolling Median"),row=1,col=1)
        fig_sleep.add_hline(y=m3_sl_low,line_dash="dash",line_color="#ef4444",line_width=1.4,annotation_text=f"Min ({m3_sl_low} min)",annotation_font_color="#ef4444",annotation_font_size=10,annotation_position="bottom right",row=1,col=1)
        fig_sleep.add_hline(y=m3_sl_high,line_dash="dash",line_color="#f59e0b",line_width=1.4,annotation_text=f"Max ({m3_sl_high} min)",annotation_font_color="#f59e0b",annotation_font_size=10,annotation_position="top right",row=1,col=1)
        if not sleep_anom.empty:
            fig_sleep.add_trace(go.Scatter(x=sleep_anom["Date"],y=sleep_anom["TotalSleepMinutes"],mode="markers+text",marker=dict(size=16,color="#ef4444",symbol="diamond",line=dict(color="white",width=1.5)),text=["▲ Residual±2σ"]*len(sleep_anom),textposition="top center",textfont=dict(size=8,color="#fbbf24"),name="😴 Sleep Anomaly"),row=1,col=1)
        bar_colors_sl=["#ef4444" if a else "#38bdf8" for a in anom_sleep["resid_anomaly"]]
        fig_sleep.add_trace(go.Bar(x=anom_sleep["Date"],y=anom_sleep["residual"],marker_color=bar_colors_sl,marker_opacity=0.80,name="Residual"),row=2,col=1)
        fig_sleep.add_hline(y=0,line_color="#4a6a8a",line_width=1,row=2,col=1)
        fig_sleep.update_layout(**{**PT,"height":560},showlegend=True,title="Sleep Pattern — Anomaly Visualization"); fig_sleep.update_xaxes(gridcolor="#1e3a5f",tickformat="%d %b",tickangle=-30); fig_sleep.update_yaxes(gridcolor="#1e3a5f"); fig_sleep.update_yaxes(title_text="Sleep (min)",row=1,col=1); fig_sleep.update_yaxes(title_text="Deviation (min)",row=2,col=1)
        st.plotly_chart(fig_sleep,use_container_width=True)
        if not sleep_anom.empty:
            with st.expander(f"📋 View {n_sleep} Sleep Anomaly Records"):
                st.dataframe(sleep_anom[["Date","TotalSleepMinutes","rolling_med","residual","reason"]].rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"}).round(2),use_container_width=True)

    with st.expander("👟  Chart 3 — Step Count Trend with Alerts", expanded=True):
        tip("Green line = actual steps · Blue dashed = rolling median · **Red vertical bands** = alert days.")
        steps_anom = anom_steps[anom_steps["is_anomaly"]]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Days",len(anom_steps)); c2.metric("Avg Steps/Day",f"{anom_steps['TotalSteps'].mean():,.0f}"); c3.metric("🚨 Anomalies",n_steps,delta=f"{n_steps/len(anom_steps)*100:.1f}% of days",delta_color="inverse" if n_steps>0 else "off"); c4.metric("Days < 500",int((anom_steps["TotalSteps"]<500).sum()))
        if n_steps > 0: alert(f"**{n_steps} anomalous step days** — threshold or ±{m3_sigma:.0f}σ residual.")
        else: ok("Step count patterns look normal.")
        fig_steps = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.08,subplot_titles=("Daily Steps (avg across users)","Residual Deviation from Trend"))
        for _, row in steps_anom.iterrows():
            d=str(row["Date"])
            try: d_next=str(pd.Timestamp(d)+pd.Timedelta(days=1))[:10]
            except: d_next=d
            fig_steps.add_vrect(x0=d,x1=d_next,fillcolor="rgba(239,68,68,0.15)",line_color="rgba(239,68,68,0.45)",line_width=1.5,row=1,col=1)
        fig_steps.add_trace(go.Scatter(x=anom_steps["Date"],y=anom_steps["TotalSteps"],mode="lines+markers",line=dict(color="#10b981",width=2.5),marker=dict(size=5,color="#10b981",opacity=0.7),name="Avg Daily Steps"),row=1,col=1)
        fig_steps.add_trace(go.Scatter(x=anom_steps["Date"],y=anom_steps["rolling_med"],mode="lines",line=dict(color="#38bdf8",width=2,dash="dash"),name="Trend (Rolling Median)"),row=1,col=1)
        fig_steps.add_hline(y=m3_st_low,line_dash="dash",line_color="#ef4444",line_width=1.4,annotation_text=f"Low Alert ({m3_st_low:,})",annotation_font_color="#ef4444",annotation_font_size=10,annotation_position="bottom right",row=1,col=1)
        fig_steps.add_hline(y=25000,line_dash="dash",line_color="#f59e0b",line_width=1.4,annotation_text="High Alert (25,000)",annotation_font_color="#f59e0b",annotation_font_size=10,annotation_position="top right",row=1,col=1)
        if not steps_anom.empty:
            fig_steps.add_trace(go.Scatter(x=steps_anom["Date"],y=steps_anom["TotalSteps"],mode="markers+text",marker=dict(size=14,color="#fbbf24",symbol="triangle-up",line=dict(color="#ef4444",width=2)),text=["▲"]*len(steps_anom),textposition="top center",textfont=dict(size=9,color="#ef4444"),name="🚨 Steps Alert"),row=1,col=1)
        bar_colors_st=["#ef4444" if a else "#10b981" for a in anom_steps["resid_anomaly"]]
        fig_steps.add_trace(go.Bar(x=anom_steps["Date"],y=anom_steps["residual"],marker_color=bar_colors_st,marker_opacity=0.80,name="Residual"),row=2,col=1)
        fig_steps.add_hline(y=0,line_color="#4a6a8a",line_width=1,row=2,col=1)
        fig_steps.update_layout(**{**PT,"height":560},showlegend=True,title="Step Count Trend — Alerts & Anomalies"); fig_steps.update_xaxes(gridcolor="#1e3a5f",tickformat="%d %b",tickangle=-30); fig_steps.update_yaxes(gridcolor="#1e3a5f"); fig_steps.update_yaxes(title_text="Steps",row=1,col=1); fig_steps.update_yaxes(title_text="Residual (steps)",row=2,col=1)
        st.plotly_chart(fig_steps,use_container_width=True)
        if not steps_anom.empty:
            with st.expander(f"📋 View {n_steps} Steps Anomaly Records"):
                st.dataframe(steps_anom[["Date","TotalSteps","rolling_med","residual","reason"]].rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"}).round(2),use_container_width=True)

    with st.expander("🤖  Chart 4 — DBSCAN Outlier Detection (PCA Projection)", expanded=True):
        tip("Each dot = one user · **Red X = structural outlier (DBSCAN label −1)**.")
        cluster_cols=["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
        try:
            from sklearn.preprocessing import StandardScaler; from sklearn.cluster import DBSCAN as DBSCAN_sk; from sklearn.decomposition import PCA as PCA_sk
            avail_cols=[c for c in cluster_cols if c in master.columns]; cf=master.groupby("Id")[avail_cols].mean().round(3).dropna()
            if len(cf) < 3: warn("Need at least 3 users for DBSCAN.")
            else:
                X_scaled=StandardScaler().fit_transform(cf); db_labels=DBSCAN_sk(eps=2.2,min_samples=2).fit_predict(X_scaled)
                pca_=PCA_sk(n_components=2,random_state=42); X_pca=pca_.fit_transform(X_scaled); var=pca_.explained_variance_ratio_*100
                cf["DBSCAN"]=db_labels; outlier_ids=cf[cf["DBSCAN"]==-1].index.tolist(); n_outliers=len(outlier_ids); n_clusters=len(set(db_labels))-(1 if -1 in db_labels else 0)
                c1,c2,c3=st.columns(3); c1.metric("Total Users",len(cf)); c2.metric("Clusters Found",n_clusters); c3.metric("🔴 Outliers",n_outliers)
                if n_outliers > 0: alert(f"**{n_outliers} structural outlier(s)** — User(s) {outlier_ids} do not fit any cluster profile.")
                else: ok("No structural outliers detected.")
                CLUSTER_COLORS=["#a78bfa","#38bdf8","#10b981","#f59e0b","#f97316","#06b6d4"]; fig_db4=go.Figure()
                for lbl in sorted(set(db_labels)):
                    if lbl==-1: continue
                    mask=db_labels==lbl
                    fig_db4.add_trace(go.Scatter(x=X_pca[mask,0],y=X_pca[mask,1],mode="markers+text",name=f"Cluster {lbl}",marker=dict(size=14,color=CLUSTER_COLORS[lbl%len(CLUSTER_COLORS)],opacity=0.85,line=dict(color="white",width=1.5)),text=[str(uid)[-4:] for uid in cf.index[mask]],textposition="top center",textfont=dict(size=8,color="#a8c8e8")))
                if n_outliers > 0:
                    mask_out=db_labels==-1
                    fig_db4.add_trace(go.Scatter(x=X_pca[mask_out,0],y=X_pca[mask_out,1],mode="markers+text",name="🚨 Outlier",marker=dict(size=20,color="#ef4444",symbol="x",line=dict(color="white",width=2.5)),text=[str(uid)[-4:] for uid in cf.index[mask_out]],textposition="top center",textfont=dict(size=9,color="#ef4444"),hovertemplate="<b>⚠️ OUTLIER</b><br>User: %{text}<extra>Structural Outlier</extra>"))
                T(fig_db4,h=500); fig_db4.update_layout(title=f"DBSCAN Outlier Detection — {n_clusters} cluster(s) · {n_outliers} outlier(s)",xaxis_title=f"PC1 ({var[0]:.1f}%)",yaxis_title=f"PC2 ({var[1]:.1f}%)")
                st.plotly_chart(fig_db4,use_container_width=True)
                if outlier_ids: st.markdown("**📋 Outlier User Profiles**"); st.dataframe(cf[cf["DBSCAN"]==-1][avail_cols].round(2),use_container_width=True)
        except Exception as e: warn(f"DBSCAN error: {e}")

    with st.expander("🎯  Chart 5 — Simulated Detection Accuracy (90%+ Target)", expanded=True):
        tip("10 known extreme anomalies are injected per signal. We measure how many the detector catches.")
        if st.button("▶️  Run Accuracy Simulation", key="run_m3_sim"):
            with st.spinner("Running simulation…"):
                try:
                    sim=simulate_accuracy(master,n_inject=10); st.session_state.m3_sim_results=sim; st.session_state.m3_simulation_done=True; st.rerun()
                except Exception as e: st.error(f"Simulation error: {e}")
        if not st.session_state.m3_simulation_done: st.info("Click above to validate the detection system.")
        else:
            sim=st.session_state.m3_sim_results; overall=sim["Overall"]; passed=overall>=90.0
            if passed: ok(f"**Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT**")
            else: warn(f"**Overall accuracy: {overall}% — ❌ BELOW 90% TARGET**")
            card_cols=st.columns(4)
            for col,label in zip(card_cols[:3],["Heart Rate","Steps","Sleep"]):
                r=sim[label]; c="#10b981" if r["accuracy"]>=90 else "#ef4444"
                col.markdown(f'<div style="background:#162a41;border:1.5px solid {c};border-radius:12px;padding:16px 10px;text-align:center"><div style="color:{c};font-weight:800;font-size:1.7rem">{r["accuracy"]:.1f}%</div><div style="color:#dce8f5;font-size:0.82rem;font-weight:600;margin:4px 0">{label}</div><div style="color:#5a7a9a;font-size:0.72rem">{r["detected"]}/{r["injected"]} detected</div></div>',unsafe_allow_html=True)
            with card_cols[3]:
                c="#10b981" if passed else "#ef4444"
                st.markdown(f'<div style="background:#162a41;border:2px solid {c};border-radius:12px;padding:16px 10px;text-align:center"><div style="color:{c};font-weight:800;font-size:1.7rem">{overall}%</div><div style="color:#dce8f5;font-size:0.82rem;font-weight:600;margin:4px 0">Overall</div></div>',unsafe_allow_html=True)
            signals=["Heart Rate","Steps","Sleep"]; accs=[sim[s]["accuracy"] for s in signals]; bar_colors=["#10b981" if a>=90 else "#ef4444" for a in accs]
            fig_acc=go.Figure()
            fig_acc.add_trace(go.Bar(x=signals,y=accs,marker_color=bar_colors,marker_opacity=0.85,text=[f"{a:.1f}%" for a in accs],textposition="outside",textfont=dict(size=14,color="#dce8f5"),width=0.4))
            fig_acc.add_hline(y=90,line_dash="dash",line_color="#ef4444",line_width=2,annotation_text="90% Target",annotation_font_color="#ef4444",annotation_font_size=11,annotation_position="top right")
            T(fig_acc,h=420); fig_acc.update_layout(title="Simulated Anomaly Detection Accuracy",xaxis_title="Signal",yaxis_title="Detection Accuracy (%)",yaxis_range=[0,120],showlegend=False)
            st.plotly_chart(fig_acc,use_container_width=True)

    with st.expander("📋  Summary & Export Anomaly Report"):
        rows=[]
        for metric_label,df_d,val_col,lo,hi,unit in [("Heart Rate (bpm)",anom_hr,"AvgHR",m3_hr_low,m3_hr_high,"bpm"),("Daily Steps",anom_steps,"TotalSteps",m3_st_low,25000,"steps"),("Sleep (min)",anom_sleep,"TotalSleepMinutes",m3_sl_low,m3_sl_high,"min")]:
            if val_col not in df_d.columns: continue
            for _,row in df_d[df_d["is_anomaly"]].iterrows():
                v=row[val_col]; rows.append({"Date":str(row["Date"])[:10],"Metric":metric_label,"Value":round(v,1),"Unit":unit,"Method":"Threshold + Residual±2σ","Threshold":f"{lo}–{hi} {unit}","Anomaly Reason":row.get("reason","—"),"Severity":"High" if (v>hi*1.1 or (lo>0 and v<lo*0.9)) else "Medium"})
        if rows:
            report_df=pd.DataFrame(rows).sort_values("Date"); st.dataframe(report_df,use_container_width=True)
            st.download_button("📥 Download Anomaly Report (CSV)",data=report_df.to_csv(index=False).encode("utf-8"),file_name="fitpulse_anomaly_report_m3.csv",mime="text/csv")
        else: st.info("Run detection above to populate the export.")


# ══════════════════════════════════════════════════════════════
#  MILESTONE 4  — uses shared files from M2, or own upload
# ══════════════════════════════════════════════════════════════
else:
    st.title("📊 FitPulse — Insights Dashboard")
    st.markdown("Upload your 5 Fitbit CSV files, run detection, then filter by date & user and export your report.")
    st.markdown("---")

    shared_fk = st.session_state.get("shared_file_keys")
    use_shared_m4 = shared_fk is not None and len(shared_fk) >= 5

    with st.expander("📂  Step 1 — Data Files", expanded=True):
        if use_shared_m4:
            st.markdown(f"""
            <div class="shared-box">
                📂 <strong>Files automatically loaded from Milestone 2</strong> — {len(shared_fk)} files ready.
                No upload needed. Click the pipeline button below to start.
            </div>""", unsafe_allow_html=True)
            shared_detected_m4 = file_keys_to_m3_detected(shared_fk)
            g = st.columns(5)
            for col_ui, (req_name, finfo) in zip(g, M4_FILES.items()):
                found = req_name in shared_detected_m4; bor="#10b981" if found else "#f59e0b"; tc="#10b981" if found else "#f59e0b"
                col_ui.markdown(f'<div style="background:#162a41;border:1px solid {bor};border-radius:12px;padding:14px 8px;text-align:center;min-height:100px"><div style="font-size:1.8rem">{finfo["icon"]}</div><div style="font-size:0.78rem;font-weight:700;color:{tc};margin-top:6px">{finfo["label"]}</div><div style="font-size:0.62rem;color:#5a7a9a;margin-top:5px">{"✅ From M2" if found else "⚠️ Not matched"}</div></div>',unsafe_allow_html=True)
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
            if st.checkbox("🔄 Upload different files instead", key="m4_override_upload"):
                use_shared_m4 = False

        if not use_shared_m4:
            tip("Upload all 5 CSV files at once — hold <b>Ctrl</b> (Windows) or <b>Cmd</b> (Mac) while clicking.")
            m4_uploaded = st.file_uploader("Upload all 5 CSV files", type=["csv"], accept_multiple_files=True, key="m4_uploader", label_visibility="collapsed")
            m4_detected = {}; m4_raw = []
            if m4_uploaded:
                for uf in m4_uploaded:
                    try: uf.seek(0); m4_raw.append((uf.name, pd.read_csv(uf)))
                    except Exception: pass
                for req_name, finfo in M4_FILES.items():
                    best_s, best_d = 0, None
                    for uname, udf in m4_raw:
                        s = score_match(udf, finfo)
                        if s > best_s: best_s, best_d = s, udf
                    if best_s >= 2: m4_detected[req_name] = best_d
            n_up = len(m4_detected)
            g = st.columns(5)
            for col_ui, (req_name, finfo) in zip(g, M4_FILES.items()):
                found = req_name in m4_detected; bor="#10b981" if found else "#1e4a6e"; tc="#10b981" if found else "#6b7280"
                col_ui.markdown(f'<div style="background:#162a41;border:1px solid {bor};border-radius:12px;padding:14px 8px;text-align:center;min-height:100px"><div style="font-size:1.8rem">{finfo["icon"]}</div><div style="font-size:0.78rem;font-weight:700;color:{tc};margin-top:6px">{finfo["label"]}</div><div style="font-size:0.62rem;color:#5a7a9a;margin-top:5px">{"✅ Detected" if found else "❌ Not found"}</div></div>',unsafe_allow_html=True)
            if n_up == 0: st.info("👆 Upload all 5 CSV files above.")
            elif n_up < 5: warn(f"{n_up}/5 files detected.")
            else: ok("All 5 files detected and ready.")
        else:
            m4_detected = file_keys_to_m3_detected(shared_fk); n_up = len(m4_detected)

        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        run_disabled = (n_up < 5)
        run_clicked  = st.button("⚡  Load & Run Full Detection Pipeline", disabled=run_disabled, key="m4_run")

        if run_clicked and n_up == 5:
            prog_ph = st.empty()
            def update_m4_prog(step_txt, pct_val):
                prog_ph.markdown(f'<div style="background:linear-gradient(90deg,#1a2744,#1a1f4a);border:1px solid #a78bfa;border-radius:12px;padding:20px 24px;margin:12px 0"><div style="font-size:1.0rem;color:#a78bfa;font-weight:700;margin-bottom:10px">⏳ Running Pipeline...</div><div style="font-size:0.85rem;color:#dce8f5;margin-bottom:10px">{step_txt}</div><div class="prog-wrap"><div style="width:{pct_val}%;height:100%;background:linear-gradient(90deg,#6366f1,#a78bfa);border-radius:99px;"></div></div><div style="font-size:0.72rem;color:#4a6a8a;margin-top:6px">{pct_val}% complete</div></div>',unsafe_allow_html=True)
            try:
                update_m4_prog("📂 Loading CSV files...", 15)
                daily = m4_detected["dailyActivity_merged.csv"].copy()
                sleep = m4_detected.get("minuteSleep_merged.csv", pd.DataFrame()).copy()
                hr    = m4_detected.get("heartrate_seconds_merged.csv", pd.DataFrame()).copy()
                def safe_dt(s, fmt):
                    try: return pd.to_datetime(s, format=fmt)
                    except: return pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
                update_m4_prog("🔗 Merging daily activity data...", 30)
                daily["ActivityDate"] = safe_dt(daily["ActivityDate"], "%m/%d/%Y")
                hr_daily = pd.DataFrame()
                if not hr.empty and "Time" in hr.columns and "Value" in hr.columns:
                    update_m4_prog("💓 Processing heart rate data...", 50)
                    hr["Time"] = safe_dt(hr["Time"], "%m/%d/%Y %I:%M:%S %p")
                    hr_min = hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index()
                    hr_min.columns = ["Id","Time","HeartRate"]; hr_min = hr_min.dropna(); hr_min["Date"] = hr_min["Time"].dt.date
                    hr_daily = hr_min.groupby(["Id","Date"])["HeartRate"].agg(["mean","max","min","std"]).reset_index().rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"})
                sleep_daily = pd.DataFrame()
                if not sleep.empty and "date" in sleep.columns and "value" in sleep.columns:
                    update_m4_prog("😴 Processing sleep data...", 65)
                    sleep["date"] = safe_dt(sleep["date"], "%m/%d/%Y %I:%M:%S %p"); sleep["Date"] = sleep["date"].dt.date
                    sleep_daily = sleep.groupby(["Id","Date"]).agg(TotalSleepMinutes=("value","count")).reset_index()
                update_m4_prog("🔗 Building master dataset...", 80)
                master = daily.copy().rename(columns={"ActivityDate":"Date"}); master["Date"] = master["Date"].dt.date
                if not hr_daily.empty:
                    master = master.merge(hr_daily, on=["Id","Date"], how="left")
                    for c in ["AvgHR","MaxHR","MinHR","StdHR"]: master[c] = master.groupby("Id")[c].transform(lambda x: x.fillna(x.median()))
                if not sleep_daily.empty:
                    master = master.merge(sleep_daily, on=["Id","Date"], how="left"); master["TotalSleepMinutes"] = master["TotalSleepMinutes"].fillna(0)
                update_m4_prog("🚨 Running anomaly detection...", 95)
                _hr_high=st.session_state.get("m4_hr_high",100); _hr_low=st.session_state.get("m4_hr_low",50)
                _st_low=st.session_state.get("m4_st_low",500); _sl_low=st.session_state.get("m4_sl_low",60)
                _sl_high=st.session_state.get("m4_sl_high",600); _sigma=st.session_state.get("m4_sigma",2.0)
                anom_hr_r=m4_detect_hr(master,_hr_high,_hr_low,_sigma); anom_steps_r=m4_detect_steps(master,_st_low,25000,_sigma); anom_sleep_r=m4_detect_sleep(master,_sl_low,_sl_high,_sigma)
                update_m4_prog("✅ Complete!", 100)
                st.session_state.m4_master=master; st.session_state.m4_anom_hr=anom_hr_r; st.session_state.m4_anom_steps=anom_steps_r; st.session_state.m4_anom_sleep=anom_sleep_r; st.session_state.m4_pipeline_done=True; prog_ph.empty(); st.rerun()
            except Exception as e: prog_ph.empty(); st.error(f"Pipeline error: {e}"); st.exception(e)

        if st.session_state.m4_pipeline_done:
            _m = st.session_state.m4_master
            ok(f"Pipeline complete — **{_m.shape[0]:,} rows** · **{_m['Id'].nunique()} users** · **{_m.shape[1]} columns**")
            mm1,mm2,mm3,mm4 = st.columns(4)
            mm1.metric("Total Rows",f"{_m.shape[0]:,}"); mm2.metric("Users",_m["Id"].nunique())
            mm3.metric("Date Range",f"{pd.to_datetime(_m['Date']).min().strftime('%d %b')} → {pd.to_datetime(_m['Date']).max().strftime('%d %b %y')}"); mm4.metric("Columns",_m.shape[1])

    if not st.session_state.m4_pipeline_done:
        st.stop()

    master=st.session_state.m4_master; anom_hr=st.session_state.m4_anom_hr; anom_steps=st.session_state.m4_anom_steps; anom_sleep=st.session_state.m4_anom_sleep
    hr_high=st.session_state.get("m4_hr_high",100); hr_low=st.session_state.get("m4_hr_low",50); st_low=st.session_state.get("m4_st_low",500); sl_low=st.session_state.get("m4_sl_low",60); sl_high=st.session_state.get("m4_sl_high",600); sigma=st.session_state.get("m4_sigma",2.0)

    try:
        if date_range and isinstance(date_range, tuple) and len(date_range)==2: d_from=pd.Timestamp(date_range[0]); d_to=pd.Timestamp(date_range[1])
        else: all_d=pd.to_datetime(master["Date"]); d_from,d_to=all_d.min(),all_d.max()
    except: all_d=pd.to_datetime(master["Date"]); d_from,d_to=all_d.min(),all_d.max()

    def filt(df, dc="Date"):
        df2=df.copy(); df2[dc]=pd.to_datetime(df2[dc]); return df2[(df2[dc]>=d_from)&(df2[dc]<=d_to)]

    anom_hr_f=filt(anom_hr); anom_steps_f=filt(anom_steps); anom_sleep_f=filt(anom_sleep); master_f=filt(master)
    if sel_user: master_f=master_f[master_f["Id"]==sel_user]

    n_hr_f=int(anom_hr_f["is_anomaly"].sum()); n_steps_f=int(anom_steps_f["is_anomaly"].sum()); n_sleep_f=int(anom_sleep_f["is_anomaly"].sum()); n_total_f=n_hr_f+n_steps_f+n_sleep_f; n_users_f=master_f["Id"].nunique(); n_days_f=master_f["Date"].nunique()
    worst_hr_row=anom_hr_f[anom_hr_f["is_anomaly"]]
    worst_hr_str=(pd.Timestamp(worst_hr_row.iloc[worst_hr_row["residual"].abs().values.argmax()]["Date"]).strftime("%d %b") if not worst_hr_row.empty else "—")

    st.markdown("---")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi_items = [(k1,str(n_total_f),"TOTAL FLAGS","across all signals","#ef4444"),(k2,str(n_hr_f),"HR FLAGS","heart rate anomalies","#f472b6"),(k3,str(n_steps_f),"STEPS ALERTS","step count anomalies","#10b981"),(k4,str(n_sleep_f),"SLEEP FLAGS","sleep anomalies","#a78bfa"),(k5,str(n_users_f),"USERS","in selected range","#38bdf8"),(k6,worst_hr_str,"PEAK HR ANOMALY","highest deviation day","#f59e0b")]
    for col,num,lbl,sub,color in kpi_items:
        col.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:{color}">{num}</div><div class="kpi-lbl">{lbl}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    ok(f"Pipeline complete · {n_users_f} users · {n_days_f} days · {n_total_f} total anomalies flagged")
    st.markdown("---")

    tab_overview,tab_hr,tab_steps,tab_sleep,tab_export = st.tabs(["📊 Overview","💓 Heart Rate","👟 Steps","😴 Sleep","📥 Export Report"])

    with tab_overview:
        with st.expander("📅  Combined Anomaly Timeline", expanded=True):
            tip("Each diamond = one anomaly event. Hover to see the date and reason.")
            fig_tl=m4_chart_timeline(anom_hr_f,anom_steps_f,anom_sleep_f)
            if fig_tl: st.plotly_chart(fig_tl,use_container_width=True)
            else: ok("No anomalies in the selected date range.")
        with st.expander("🗂️  Recent Anomaly Log", expanded=True):
            all_log=[]
            for df_,sig,clr in [(anom_hr_f,"Heart Rate","#38bdf8"),(anom_steps_f,"Steps","#10b981"),(anom_sleep_f,"Sleep","#a78bfa")]:
                a=df_[df_["is_anomaly"]].copy(); a["signal"]=sig; a["color"]=clr; all_log.append(a[["Date","signal","color","reason"]])
            if all_log:
                log_df=pd.concat(all_log,ignore_index=True); log_df["Date"]=pd.to_datetime(log_df["Date"]); log_df=log_df.sort_values("Date",ascending=False).head(15)
                st.markdown('<div style="display:flex;gap:14px;padding:7px 12px;background:#0d1b2a;border-radius:6px 6px 0 0;border:1px solid #1e3a5f;border-bottom:2px solid #1e3a5f;font-size:0.72rem;font-weight:700;color:#4a6a8a;text-transform:uppercase;letter-spacing:0.07em;"><span style="width:24px"></span><span style="min-width:95px">Signal</span><span style="min-width:100px">Date</span><span>Reason</span></div>',unsafe_allow_html=True)
                for _,row in log_df.iterrows():
                    st.markdown(f'<div class="anom-row" style="padding-left:12px;border:1px solid #1a2f44;border-top:none;background:#111f30;"><span>🚨</span><span style="color:{row["color"]};font-weight:700;font-size:0.82rem;min-width:95px">{row["signal"]}</span><span style="color:#a8c8e8;font-size:0.8rem;min-width:100px">{row["Date"].strftime("%d %b %Y")}</span><span style="color:#64748b;font-size:0.78rem;font-style:italic">{row["reason"]}</span></div>',unsafe_allow_html=True)
            else: ok("No anomalies detected in selected range.")

    with tab_hr:
        badge_hr=f'<span style="margin-left:auto;background:rgba(239,68,68,0.15);border:1px solid #ef4444;border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#ef4444;font-weight:700">{n_hr_f} anomalies</span>'
        st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin:1rem 0 0.8rem"><div style="background:rgba(239,68,68,0.15);border:1px solid #ef4444;border-radius:8px;width:38px;height:38px;display:flex;align-items:center;justify-content:center;font-size:1.3rem">💓</div><span style="font-size:1.15rem;font-weight:700;color:#dce8f5">Heart Rate — Deep Dive</span>{badge_hr}</div>',unsafe_allow_html=True)
        col_stat,col_rec=st.columns(2)
        with col_stat:
            st.markdown('<div class="stat-title">HR STATISTICS</div>',unsafe_allow_html=True)
            mean_hr=anom_hr_f["AvgHR"].mean(); max_hr=anom_hr_f["AvgHR"].max(); min_hr=anom_hr_f["AvgHR"].min()
            st.markdown(f'<div class="stat-card"><div>Mean HR: <b style="color:#38bdf8">{mean_hr:.1f} bpm</b></div><div>Max HR: <b style="color:#ef4444">{max_hr:.1f} bpm</b></div><div>Min HR: <b style="color:#f9a8d4">{min_hr:.1f} bpm</b></div><div>Anomaly days: <b style="color:#ef4444">{n_hr_f}</b> of {len(anom_hr_f)} total</div><div>Anomaly rate: <b style="color:#ef4444">{n_hr_f/max(len(anom_hr_f),1)*100:.1f}%</b></div><div>Normal range: <b style="color:#a8c8e8">{hr_low} – {hr_high} bpm</b></div></div>',unsafe_allow_html=True)
            if n_hr_f>0:
                alert(f"{n_hr_f} HR anomaly days detected")
            else:
                ok("No HR anomalies in selected range")
        with col_rec:
            st.markdown('<div class="stat-title">HR ANOMALY RECORDS</div>',unsafe_allow_html=True)
            hr_disp=anom_hr_f[anom_hr_f["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].round(2)
            if not hr_disp.empty: st.dataframe(hr_disp.rename(columns={"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),use_container_width=True,height=260)
            else: ok("No HR anomalies in selected range.")

    with tab_steps:
        badge_st=f'<span style="margin-left:auto;background:rgba(16,185,129,0.15);border:1px solid #10b981;border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#10b981;font-weight:700">{n_steps_f} alerts</span>'
        st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin:1rem 0 0.8rem"><div style="background:rgba(16,185,129,0.15);border:1px solid #10b981;border-radius:8px;width:38px;height:38px;display:flex;align-items:center;justify-content:center;font-size:1.3rem">👟</div><span style="font-size:1.15rem;font-weight:700;color:#dce8f5">Steps — Deep Dive</span>{badge_st}</div>',unsafe_allow_html=True)
        col_stat2,col_rec2=st.columns(2)
        with col_stat2:
            st.markdown('<div class="stat-title">STEPS STATISTICS</div>',unsafe_allow_html=True)
            mean_st=anom_steps_f["TotalSteps"].mean(); max_st=anom_steps_f["TotalSteps"].max(); min_st=anom_steps_f["TotalSteps"].min(); days_lt=int((anom_steps_f["TotalSteps"]<500).sum())
            st.markdown(f'<div class="stat-card"><div>Mean steps/day: <b style="color:#10b981">{mean_st:,.0f}</b></div><div>Max steps/day: <b style="color:#38bdf8">{max_st:,.0f}</b></div><div>Min steps/day: <b style="color:#ef4444">{min_st:,.0f}</b></div><div>Anomaly days: <b style="color:#ef4444">{n_steps_f}</b> of {len(anom_steps_f)} total</div><div>Anomaly rate: <b style="color:#ef4444">{n_steps_f/max(len(anom_steps_f),1)*100:.1f}%</b></div><div>Days &lt; 500 steps: <b style="color:#ef4444">{days_lt}</b></div><div>Alert range: <b style="color:#a8c8e8">{st_low:,} – 25,000 steps/day</b></div></div>',unsafe_allow_html=True)
            if n_steps_f>0:
                alert(f"{n_steps_f} step alert days detected")
            else:
                ok("No step anomalies in selected range")
        with col_rec2:
            st.markdown('<div class="stat-title">STEPS ANOMALY RECORDS</div>',unsafe_allow_html=True)
            st_disp=anom_steps_f[anom_steps_f["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].round(2)
            if not st_disp.empty: st.dataframe(st_disp.rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),use_container_width=True,height=260)
            else: ok("No step anomalies in selected range.")

    with tab_sleep:
        badge_sl=f'<span style="margin-left:auto;background:rgba(167,139,250,0.15);border:1px solid #a78bfa;border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#a78bfa;font-weight:700">{n_sleep_f} anomalies</span>'
        st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin:1rem 0 0.8rem"><div style="background:rgba(167,139,250,0.15);border:1px solid #a78bfa;border-radius:8px;width:38px;height:38px;display:flex;align-items:center;justify-content:center;font-size:1.3rem">😴</div><span style="font-size:1.15rem;font-weight:700;color:#dce8f5">Sleep — Deep Dive</span>{badge_sl}</div>',unsafe_allow_html=True)
        col_stat3,col_rec3=st.columns(2)
        with col_stat3:
            st.markdown('<div class="stat-title">SLEEP STATISTICS</div>',unsafe_allow_html=True)
            mean_sl=anom_sleep_f["TotalSleepMinutes"].mean(); max_sl=anom_sleep_f["TotalSleepMinutes"].max()
            nonzero=anom_sleep_f[anom_sleep_f["TotalSleepMinutes"]>0]["TotalSleepMinutes"]; min_sl=nonzero.min() if not nonzero.empty else 0
            days_lt_sl=int((anom_sleep_f["TotalSleepMinutes"]<sl_low).sum())
            st.markdown(f'<div class="stat-card"><div>Mean sleep/night: <b style="color:#a78bfa">{mean_sl:.0f} min</b></div><div>Max sleep/night: <b style="color:#38bdf8">{max_sl:.0f} min</b></div><div>Min (non-zero): <b style="color:#ef4444">{min_sl:.0f} min</b></div><div>Anomaly days: <b style="color:#ef4444">{n_sleep_f}</b> of {len(anom_sleep_f)} total</div><div>Anomaly rate: <b style="color:#ef4444">{n_sleep_f/max(len(anom_sleep_f),1)*100:.1f}%</b></div><div>Days &lt; {sl_low} min: <b style="color:#ef4444">{days_lt_sl}</b></div><div>Healthy range: <b style="color:#a8c8e8">{sl_low} – {sl_high} min/night</b></div></div>',unsafe_allow_html=True)
            if n_sleep_f>0:
                alert(f"{n_sleep_f} sleep anomaly days detected")
            else:
                ok("No sleep anomalies in selected range")
        with col_rec3:
            st.markdown('<div class="stat-title">SLEEP ANOMALY RECORDS</div>',unsafe_allow_html=True)
            sl_disp=anom_sleep_f[anom_sleep_f["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].round(2)
            if not sl_disp.empty: st.dataframe(sl_disp.rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),use_container_width=True,height=260)
            else: ok("No sleep anomalies in selected range.")

    with tab_export:
        with st.expander("📄  PDF Report", expanded=True):
            tip("Full multi-page PDF with executive summary, anomaly tables, user profiles, and methodology.")
            st.markdown('<div style="background:#162a41;border:1px solid #1e4a6e;border-radius:12px;padding:16px 18px;margin-bottom:1rem"><div style="font-weight:700;color:#a78bfa;margin-bottom:8px">📄 What\'s in the PDF (9 sections):</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.82rem;color:#a8c8e8"><div>✅ Dataset overview</div><div>✅ Anomaly summary table</div><div>✅ Detection thresholds</div><div>✅ Methodology</div><div>✅ HR anomaly records</div><div>✅ Steps anomaly records</div><div>✅ Sleep anomaly records</div><div>✅ User activity profiles</div><div style="grid-column:1/-1">✅ Conclusion & findings</div></div></div>',unsafe_allow_html=True)
            if st.button("📄  Generate & Download PDF Report", key="m4_gen_pdf"):
                with st.spinner("Generating PDF…"):
                    try:
                        pdf_buf=m4_generate_pdf(master_f,anom_hr_f,anom_steps_f,anom_sleep_f,hr_high,hr_low,st_low,sl_low,sl_high,sigma)
                        fname_pdf=f"FitPulse_M4_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                        st.download_button(label="⬇️  Download PDF Report",data=pdf_buf,file_name=fname_pdf,mime="application/pdf",key="m4_dl_pdf")
                        ok("PDF ready — click the button above to download.")
                    except Exception as e: st.error(f"PDF error: {e}"); st.exception(e)
        with st.expander("📊  CSV Export", expanded=True):
            tip("All anomaly records from all three signals combined into one CSV.")
            csv_data=m4_generate_csv(anom_hr_f,anom_steps_f,anom_sleep_f)
            st.download_button(label="⬇️  Download Anomaly CSV",data=csv_data,file_name=f"FitPulse_M4_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",mime="text/csv",key="m4_dl_csv")
            with st.expander("👁️  Preview CSV (first 20 rows)"):
                try: st.dataframe(pd.read_csv(io.StringIO(csv_data.decode())).head(20),use_container_width=True)
                except: st.info("Preview unavailable.")
        with st.expander("🏁  Milestone 4 Completion Checklist", expanded=False):
            ok("🎉 Milestone 4 pipeline complete!")
            for icon,label in [("📂","5 Fitbit CSV files loaded and merged"),("💓","Heart rate processed (1-min resampling)"),("😴","Sleep data processed"),("🔗","Master dataset built"),("①","Threshold violations detected"),("②","Residual-based detection applied"),("📅","Combined anomaly timeline"),("💓","HR statistics + records"),("👟","Steps statistics + records"),("😴","Sleep statistics + records"),("🔍","Date & user filter in sidebar"),("📄","PDF report export (ReportLab)"),("📊","CSV anomaly export")]:
                st.markdown(f"✅ {icon} {label}")