import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import (
    load_all_files, parse_timestamps, resample_heartrate,
    build_master_df, prepare_tsfresh_input, build_clustering_features,
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="FitPulse – M2", page_icon="💪", layout="wide")

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
        border-radius: 8px; padding: 0.45rem 1.2rem; transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(168,85,247,0.4); }
    div[data-testid="metric-container"] {
        background: #162a41; border: 1px solid #1e4a6e;
        border-radius: 10px; padding: 0.9rem;
    }
    .stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 8px; }
    details { background: #111f30; border: 1px solid #1e3a5f !important; border-radius: 10px; margin-bottom: 10px; }
    details summary {
        background: linear-gradient(90deg, #162a41, #1a1f4a);
        border-radius: 10px; padding: 14px 18px;
        color: #a78bfa; font-weight: 700; font-size: 1.05rem;
        cursor: pointer; list-style: none;
    }
    details[open] summary { border-bottom: 1px solid #1e3a5f; border-radius: 10px 10px 0 0; }
    .info-box {
        background: rgba(99,102,241,0.12); border-left: 4px solid #6366f1;
        border-radius: 6px; padding: 10px 14px; margin: 8px 0;
        font-size: 0.88rem; color: #a8c8e8;
    }
    .loading-banner {
        background: linear-gradient(90deg, #1a2744, #1a1f4a);
        border: 1px solid #a78bfa; border-radius: 10px;
        padding: 18px 24px; margin: 12px 0; text-align: center;
    }
    /* ── Prevent Streamlit from dimming the page on rerun ── */
    .stSpinner > div { border-top-color: #a78bfa !important; }
    [data-testid="stAppViewContainer"] > section { opacity: 1 !important; transition: none !important; }
    .main .block-container { opacity: 1 !important; }
    /* Hide the small running indicator top-right so it's less alarming */
    [data-testid="stDecoration"] { display: none !important; }
    .stStatusWidget { opacity: 0 !important; pointer-events: none; }
    /* Animated progress bar for our custom banners */
    .prog-wrap { background:#162a41; border-radius:99px; height:7px; margin:10px 0; overflow:hidden; }
    .prog-fill  { background:linear-gradient(90deg,#6366f1,#a78bfa); height:100%; border-radius:99px;
                  animation:pbar 1.4s ease-in-out infinite; }
    @keyframes pbar { 0%,100%{opacity:1;width:60%} 50%{opacity:.5;width:90%} }
</style>
""", unsafe_allow_html=True)

# ── Theme helper ───────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,27,42,0.6)",
    font=dict(color="#dce8f5", family="Inter"), title_font=dict(color="#a78bfa", size=14),
    colorway=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"],
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=50, b=36, l=40, r=20),
)
def T(fig, h=None):
    fig.update_layout(**({**PT, "height": h} if h else PT))
    fig.update_xaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
    fig.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
    return fig

def tip(text):
    st.markdown(f'<div class="info-box">💡 {text}</div>', unsafe_allow_html=True)

def loading_msg(icon, text):
    st.markdown(f'<div class="loading-banner"><span style="font-size:1.4rem">{icon}</span><br>'
                f'<strong style="color:#a78bfa">{text}</strong></div>', unsafe_allow_html=True)

# ── Cached heavy functions ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load(file_keys):
    # file_keys is a dict of {role: (name, bytes)} — hashable for cache
    from io import BytesIO
    import types
    file_objs = {}
    for role, (name, data) in file_keys.items():
        if data is not None:
            buf = BytesIO(data)
            buf.name = name
            file_objs[role] = buf
    dfs_raw = load_all_files(file_objs)
    dfs, _  = parse_timestamps(dfs_raw)
    return dfs

@st.cache_data(show_spinner=False)
def cached_resample(hr_bytes, hr_name):
    from io import BytesIO
    buf = BytesIO(hr_bytes); buf.name = hr_name
    df  = pd.read_csv(buf)
    # Parse timestamp directly — skip full parse_timestamps overhead
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
def cached_elbow(feat_array_bytes, k_max):
    from sklearn.cluster import KMeans
    import pickle
    feat = pickle.loads(feat_array_bytes)
    K_range  = range(2, min(k_max + 1, len(feat)))
    inertias = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(feat)
        inertias.append(km.inertia_)
    return list(K_range), inertias

@st.cache_data(show_spinner=False)
def cached_pca(feat_array_bytes):
    from sklearn.decomposition import PCA
    import pickle
    feat   = pickle.loads(feat_array_bytes)
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(feat)
    return coords, pca.explained_variance_ratio_

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=60)
    st.markdown("## Fitness Data Pro\n**Milestone 2**")
    st.markdown("---")
    st.markdown("""
**How to use:**
1. Upload your 5 CSV files
2. Open each section below
3. Click buttons to run heavy steps
4. Download results at the end
""")
    st.caption("Feature Extraction & Modelling")

st.title("🧬 Fitness Data — Feature Extraction & Modelling")
st.markdown("Load your Fitbit data, extract patterns, forecast trends, and group users by behaviour.")
st.markdown("---")


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — UPLOAD
# ══════════════════════════════════════════════════════════════
with st.expander("📂  Step 1 — Upload Your Data Files", expanded=True):
    tip("Select all 5 files at once — hold Ctrl (Windows) or Cmd (Mac) while clicking.")

    uploaded = st.file_uploader(
        "Upload all 5 CSV files",
        type=["csv"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    ROLES = {
        "daily_activity":     ("Daily Activity",      "🏃", ["dailyactivity","daily_activity"]),
        "heartrate":          ("Heart Rate",           "💓", ["heartrate","heart_rate"]),
        "hourly_intensities": ("Hourly Intensities",   "⚡", ["hourlyintensities","hourly_intensities"]),
        "hourly_steps":       ("Hourly Steps",         "👟", ["hourlysteps","hourly_steps"]),
        "minute_sleep":       ("Minute Sleep",         "😴", ["minutesleep","minute_sleep"]),
    }

    def match_files(files):
        m = {k: None for k in ROLES}
        for f in files:
            nl = f.name.lower().replace(" ","").replace("-","")
            for role, (_, __, kws) in ROLES.items():
                if any(kw in nl for kw in kws) and m[role] is None:
                    m[role] = f; break
        return m

    matched   = match_files(uploaded) if uploaded else {k: None for k in ROLES}
    n_matched = sum(1 for v in matched.values() if v is not None)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    g = st.columns(5)
    for col, (role, (label, icon, _)) in zip(g, ROLES.items()):
        ok    = matched[role] is not None
        fname = matched[role].name if ok else "Not found"
        short = (fname[:18] + "…") if len(fname) > 18 else fname
        col.markdown(f"""
        <div style="background:#162a41;border:1px solid {'#10b981' if ok else '#1e4a6e'};
                    border-radius:10px;padding:12px 8px;text-align:center;min-height:95px">
            <div style="font-size:1.7rem">{icon}</div>
            <div style="font-size:0.78rem;font-weight:700;color:{'#10b981' if ok else '#6b7280'};margin-top:4px">{label}</div>
            <div style="font-size:0.62rem;color:#5a7a9a;margin-top:5px;word-break:break-all">
                {'✅ ' + short if ok else '❌ not detected'}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    if n_matched == 0:
        st.info("👆 Upload your CSV files above. File names must contain: dailyActivity, heartrate, hourlyIntensities, hourlySteps, minuteSleep.")
        st.stop()
    elif n_matched < 5:
        st.warning(f"Detected {n_matched}/5 files. Check that file names contain the keywords above.")


# ── Build cache key (hashable) ─────────────────────────────────
file_keys = {}
for role, f in matched.items():
    if f is not None:
        f.seek(0)
        file_keys[role] = (f.name, f.read())
        f.seek(0)

# ── Load + merge (cached — only shows banner on first load) ───
# Use a session key based on file hashes to know if this is a fresh upload
_cache_key = str(sorted([(r, n) for r, (n, _) in file_keys.items()]))
_is_first_load = st.session_state.get("_last_cache_key") != _cache_key

status_ph = st.empty()

if _is_first_load:
    status_ph.markdown('''
    <div class="loading-banner">
        <div style="font-size:1.1rem;margin-bottom:8px">⏳ <strong style="color:#a78bfa">Loading and processing your files… please wait</strong></div>
        <div class="prog-wrap"><div class="prog-fill"></div></div>
        <div style="font-size:0.8rem;color:#5a7a9a;margin-top:6px">This only runs once per upload — subsequent interactions will be instant</div>
    </div>''', unsafe_allow_html=True)

dfs        = cached_load(file_keys)
master_df  = cached_master(file_keys)

hr_resampled = None
if "heartrate" in file_keys:
    hr_name, hr_bytes = file_keys["heartrate"]
    hr_resampled = cached_resample(hr_bytes, hr_name)

# Mark this upload set as cached
st.session_state["_last_cache_key"] = _cache_key
status_ph.empty()

# ── Preprocessing report ──────────────────────────────────────
with st.expander("🧹  Data Cleaning Report — what was fixed automatically", expanded=False):
    tip("Every time you upload files, the pipeline automatically cleans them before any analysis. Here is exactly what was done.")

    ROLES_LABELS = {
        "daily_activity": "Daily Activity", "heartrate": "Heart Rate",
        "hourly_intensities": "Hourly Intensities", "hourly_steps": "Hourly Steps",
        "minute_sleep": "Minute Sleep",
    }

    for label, df in dfs.items():
        st.markdown(f"**{ROLES_LABELS.get(label, label)}**")
        import io
        raw_df = pd.read_csv(io.BytesIO(file_keys[label][1]))

        dupes   = int(raw_df.duplicated().sum())
        nulls_b = int(raw_df.isnull().sum().sum())
        nulls_a = int(df.isnull().sum().sum())
        rows_b  = len(raw_df)
        rows_a  = len(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (original)",    f"{rows_b:,}")
        c2.metric("Rows (after clean)", f"{rows_a:,}",
                  delta=f"-{rows_b - rows_a} removed" if rows_b != rows_a else "none removed",
                  delta_color="inverse")
        c3.metric("Duplicates removed", dupes)
        c4.metric("Nulls before → after", f"{nulls_b} → {nulls_a}")

        null_cols = raw_df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        if not null_cols.empty:
            num_in_file = raw_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_in_file = raw_df.select_dtypes(include=["object"]).columns.tolist()
            for col, cnt in null_cols.items():
                if col in num_in_file:
                    action = "filled — interpolation per user, then median fallback"
                elif col in cat_in_file:
                    action = "filled with ‘Unknown’"
                else:
                    action = "rows with bad timestamps were dropped"
                st.markdown(f'<div class="info-box">🔧 <code>{col}</code>: {cnt} missing → {action}</div>',
                            unsafe_allow_html=True)
        else:
            st.success("  No missing values in this file.")
        st.markdown("---")

if master_df.empty:
    st.error("Could not build the master dataset. Check that the Daily Activity file is uploaded.")
    st.stop()

steps_col = next((c for c in master_df.columns if "step" in c.lower()), None)
sleep_col  = next((c for c in master_df.columns if "sleep" in c.lower() or "Sleep" in c), None)

# Pre-compute clustering inputs once (cached)
# Aggregate to ONE ROW PER USER — clustering should group users not days
import pickle
from sklearn.preprocessing import StandardScaler

if "Id" in master_df.columns:
    # Average all daily metrics per user → one row per user
    _num_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
    _excl     = {"KMeans_Cluster", "DBSCAN_Cluster", "Id", "id", "logId"}
    _num_cols = [c for c in _num_cols if c not in _excl]
    user_df   = master_df.groupby("Id")[_num_cols].mean().reset_index(drop=True)
else:
    user_df = master_df.copy()

feat_matrix = build_clustering_features(user_df)
feat_scaled = StandardScaler().fit_transform(feat_matrix.fillna(0))
feat_pkl    = pickle.dumps(feat_scaled)   # bytes → hashable for cache

# Pre-compute elbow + PCA once
k_range_list, inertias = cached_elbow(feat_pkl, k_max=min(10, len(feat_scaled)-1))
pca_coords, pca_var    = cached_pca(feat_pkl)


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with st.expander("🔍  Step 2 — Data Preview & Quality Check"):
    tip("Here you can see a preview of each file and check whether any data is missing.")

    tabs = st.tabs([ROLES[k][0] for k in dfs.keys()])
    for tab, (role, df) in zip(tabs, dfs.items()):
        with tab:
            r1, r2, r3 = st.columns(3)
            r1.metric("Rows",    f"{df.shape[0]:,}")
            r2.metric("Columns", df.shape[1])
            r3.metric("Missing", int(df.isnull().sum().sum()))
            st.dataframe(df.head(6), use_container_width=True)

    any_nulls = any(df.isnull().sum().sum() > 0 for df in dfs.values())
    if any_nulls:
        st.markdown("**Where is data missing?** *(taller bar = more missing values)*")
        nc = st.columns(min(3, sum(1 for df in dfs.values() if df.isnull().sum().sum() > 0)))
        ci = 0
        for role, df in dfs.items():
            ns = df.isnull().sum(); ns = ns[ns > 0]
            if not ns.empty:
                fig = px.bar(x=ns.index, y=ns.values, title=ROLES[role][0],
                             labels={"x":"Column","y":"Missing Count"},
                             color_discrete_sequence=["#a78bfa"])
                fig.update_xaxes(tickangle=-30)
                nc[ci % len(nc)].plotly_chart(T(fig, h=230), use_container_width=True)
                ci += 1
    else:
        st.success("✅ No missing values found across all files.")

    st.markdown("---")
    st.markdown("**Merged Master Dataset** — all 5 files combined into one table:")
    tip("Each row = one day for one user. All files are joined by User ID + Date.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows",    f"{master_df.shape[0]:,}")
    m2.metric("Columns",       master_df.shape[1])
    m3.metric("Missing Cells", int(master_df.isnull().sum().sum()))
    m4.metric("Completeness",  f"{100 - master_df.isnull().sum().sum()/master_df.size*100:.1f}%")
    st.dataframe(master_df.head(6), use_container_width=True)
    st.download_button("📥 Download Merged Dataset", master_df.to_csv(index=False).encode(),
                       "fitness_master.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — TSFRESH
# ══════════════════════════════════════════════════════════════
with st.expander("🔬  Step 3 — Automatic Feature Extraction (TSFresh)"):
    tip("TSFresh reads heart rate over time and automatically calculates useful numbers per user — like average, variability, peaks. These **features** are what machine learning models use.")

    tsf_input    = prepare_tsfresh_input(hr_resampled) if hr_resampled is not None else None
    tsf_features = st.session_state.get("tsf_features")

    if tsf_input is None:
        st.warning("Heart rate file not found — upload it to enable TSFresh.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Users",               tsf_input["id"].nunique())
        c2.metric("Heart Rate Readings", f"{len(tsf_input):,}")
        c3.metric("Features Extracted",  tsf_features.shape[1] if tsf_features is not None else "—")

        if tsf_features is None:
            if st.button("▶️  Extract Features", key="run_tsf"):
                prog_ph = st.empty()
                prog_ph.markdown('''<div class="loading-banner">
                    <div style="font-size:1.1rem;margin-bottom:8px">🔬 <strong style="color:#a78bfa">Extracting heart rate features…</strong></div>
                    <div class="prog-wrap"><div class="prog-fill"></div></div>
                    <div style="font-size:0.8rem;color:#5a7a9a;margin-top:6px">This takes 30–90 seconds. Keep this tab open — it will update automatically when done.</div>
                </div>''', unsafe_allow_html=True)
                try:
                    from tsfresh import extract_features
                    from tsfresh.utilities.dataframe_functions import impute
                    from tsfresh.feature_extraction import MinimalFCParameters
                    feat = extract_features(
                        tsf_input, column_id="id", column_sort="time", column_value="value",
                        default_fc_parameters=MinimalFCParameters(), n_jobs=1, disable_progressbar=True,
                    )
                    impute(feat)
                    st.session_state["tsf_features"] = feat
                    tsf_features = feat
                    prog_ph.empty()
                    st.success(f"✅ Done! Extracted {feat.shape[1]} features for {feat.shape[0]} users.")
                except ImportError:
                    prog_ph.empty(); st.error("Run: pip install tsfresh")
                except Exception as e:
                    prog_ph.empty(); st.error(str(e))
        else:
            st.success(f"✅ {tsf_features.shape[1]} features already extracted for {tsf_features.shape[0]} users.")

        if tsf_features is not None and not tsf_features.empty:
            st.markdown("**Feature Heatmap — each row is a user, each column is a feature**")
            tip("🟥 Red = high value · 🟦 Blue = low value · Users with similar colours have similar heart rate patterns.")
            n_show = min(12, tsf_features.shape[1])
            sub    = tsf_features.iloc[:, :n_show].copy()
            norm   = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)
            norm.columns = (norm.columns.str.replace("value__","",regex=False)
                                        .str.replace("_"," ",regex=False).str.title())
            norm.index = ["User " + str(i) for i in norm.index]
            fig_h = px.imshow(norm, color_continuous_scale="RdBu_r", zmin=0, zmax=1,
                              aspect="auto", text_auto=".2f",
                              labels={"x":"Heart Rate Feature","y":"User","color":"Score (0–1)"},
                              title="Heart Rate Feature Scores per User  (0 = low · 1 = high)")
            fig_h.update_layout(height=max(320, len(norm)*40+120), xaxis=dict(tickangle=-35), **PT)
            st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — PROPHET FORECASTING
# ══════════════════════════════════════════════════════════════
with st.expander("📈  Step 4 — Trend Forecasting (Prophet)"):
    tip("Prophet looks at your past data and predicts future values. The **shaded area** = range of likely outcomes. The **dashed line** separates real data from forecast.")

    def prophet_plot(ds_series, y_series, label, key, unit="", color="#10b981"):
        tmp = pd.DataFrame({
            "ds": pd.to_datetime(ds_series, errors="coerce"),
            "y":  pd.to_numeric(y_series, errors="coerce"),
        }).dropna().sort_values("ds")
        if len(tmp) < 10:
            st.warning(f"Not enough data to forecast {label}.")
            return

        days = 30
        run  = st.button(f"▶️  Forecast {label}", key=f"btn_{key}")

        if run:
            ph = st.empty()
            ph.markdown(f'<div class="loading-banner">📈 <strong style="color:#a78bfa">Fitting Prophet model for {label}… please wait.</strong></div>',
                        unsafe_allow_html=True)
            try:
                from prophet import Prophet
                m = Prophet(weekly_seasonality=True, daily_seasonality=False,
                            interval_width=0.80)
                m.fit(tmp)
                fc = m.predict(m.make_future_dataframe(periods=days, freq="D"))
                st.session_state[f"fc_{key}"] = (fc, tmp)
                ph.empty()
            except ImportError:
                ph.empty(); st.error("Run: pip install prophet")
            except Exception as e:
                ph.empty(); st.error(str(e))

        saved = st.session_state.get(f"fc_{key}")
        if saved:
            fc, tmp2 = saved
            cut     = tmp2["ds"].max()
            cut_str = str(cut)

            # Parse hex color -> rgba for fill
            def hex_to_rgba(h, a):
                h = h.lstrip("#")
                r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                return f"rgba({r},{g},{b},{a})"

            ci_fill  = hex_to_rgba(color, 0.30)
            ci_line  = hex_to_rgba(color, 0.0)
            dot_col  = color

            fig = go.Figure()

            # 1a. CI upper boundary — invisible line, carries hover info
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat_upper"],
                mode="lines",
                line=dict(color=ci_line, width=0),
                name="CI Upper",
                hovertemplate="CI Upper: %{y:.1f}<extra>80% CI</extra>",
                showlegend=False,
            ))

            # 1b. CI lower boundary — fills back to upper, carries hover info
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat_lower"],
                mode="lines",
                fill="tonexty",
                fillcolor=ci_fill,
                line=dict(color=ci_line, width=0),
                name="CI Lower",
                hovertemplate="CI Lower: %{y:.1f}<extra>80% CI</extra>",
                showlegend=False,
            ))

            # 1c. Legend entry for CI (invisible trace just for legend)
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color=hex_to_rgba(color, 0.6), width=10),
                name="80% CI",
                hoverinfo="skip",
            ))

            # 2. Trend line — black, full range (history + forecast)
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat"],
                mode="lines",
                line=dict(color="#111827", width=2),
                name="Trend",
                hovertemplate="Trend: %{y:.1f}<extra>Trend</extra>",
            ))

            # 3. Actual dots — coloured scatter
            y_lbl = f"Actual {label}{' (' + unit + ')' if unit else ''}"
            fig.add_trace(go.Scatter(
                x=tmp2["ds"], y=tmp2["y"],
                mode="markers",
                marker=dict(color=dot_col, size=7, opacity=0.85,
                            line=dict(color="white", width=0.5)),
                name=y_lbl,
                hovertemplate=label + ": %{y:.1f}<extra>" + y_lbl + "</extra>",
            ))

            # 4. Forecast start vertical dashed line + annotation
            PT_fc = {k: v for k, v in PT.items() if k != "legend"}
            fig.update_layout(
                title=f"{label} — Prophet Trend Forecast",
                xaxis_title="Date",
                yaxis_title=f"{label}{' (' + unit + ')' if unit else ''}",
                legend=dict(
                    orientation="v", x=0.01, y=0.99,
                    bgcolor="rgba(22,42,65,0.85)", bordercolor="#1e3a5f",
                    borderwidth=1, font=dict(size=11, color="#dce8f5"),
                ),
                shapes=[dict(
                    type="line", x0=cut_str, x1=cut_str, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="#f59e0b", width=1.8, dash="dash"),
                )],
                annotations=[dict(
                    x=cut_str, y=0.99, xref="x", yref="paper",
                    text="Forecast Start", showarrow=False,
                    font=dict(color="#f59e0b", size=10),
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(22,42,65,0.7)",
                )],
                hovermode="x unified",
                height=420,
                **PT_fc,
            )
            fig.update_xaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f",
                             tickformat="%Y-%m-%d", tickangle=-20)
            fig.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
            st.plotly_chart(fig, use_container_width=True)
            tip(f"Dots = actual recorded {label.lower()}. Black line = Prophet’s predicted trend. Shaded band = 80% confidence interval (likely range). Orange dashed line = where forecast begins.")

    if hr_resampled is not None and "Time" in hr_resampled.columns:
        st.markdown("### 💓 Heart Rate")
        _hr = hr_resampled.copy()
        _hr["Time"] = pd.to_datetime(_hr["Time"], errors="coerce")
        _hr["ActivityDate"] = _hr["Time"].dt.normalize()
        if "Id" in _hr.columns:
            _hr_daily = (_hr.groupby(["Id","ActivityDate"])["Value"].mean()
                           .reset_index()
                           .groupby("ActivityDate")["Value"].mean()
                           .reset_index())
        else:
            _hr_daily = _hr.groupby("ActivityDate")["Value"].mean().reset_index()
        _hr_daily = _hr_daily.sort_values("ActivityDate").reset_index(drop=True)

        prophet_plot(_hr_daily["ActivityDate"], _hr_daily["Value"], "Heart Rate", "hr", "bpm", color="#38bdf8")
    else:
        st.info("Heart rate data not available.")

    st.markdown("---")
    if steps_col and "ActivityDate" in master_df.columns:
        st.markdown("### 👟 Daily Steps")
        agg = master_df.groupby("ActivityDate")[steps_col].mean().reset_index()
        prophet_plot(agg["ActivityDate"], agg[steps_col], "Daily Steps", steps_col, "steps", color="#10b981")
    else:
        st.info("Steps data not available.")

    st.markdown("---")
    if sleep_col and "ActivityDate" in master_df.columns:
        st.markdown("### 😴 Sleep Duration")
        agg = master_df.groupby("ActivityDate")[sleep_col].mean().reset_index()
        prophet_plot(agg["ActivityDate"], agg[sleep_col], "Sleep (minutes)", sleep_col, "min", color="#a78bfa")
    else:
        st.info("Sleep data not available.")


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — CLUSTERING
# ══════════════════════════════════════════════════════════════
with st.expander("🔵  Step 5 — User Grouping (Clustering)"):
    st.markdown("""
    <div style="background:#162a41;border:1px solid #1e4a6e;border-radius:12px;padding:18px 20px;margin-bottom:12px">
        <h4 style="color:#a78bfa;margin:0 0 10px 0">🤔 What does a "Group" mean here?</h4>
        <p style="color:#dce8f5;margin:0 0 10px 0">
            Imagine you have 30 Fitbit users. Some walk 10,000 steps a day, sleep 8 hours, and have a steady heart rate.
            Others barely hit 3,000 steps, sleep poorly, and have irregular heart rates. These are <strong style="color:#a78bfa">naturally different types of users</strong>.
        </p>
        <p style="color:#dce8f5;margin:0 0 10px 0">
            Clustering is an algorithm that <strong style="color:#38bdf8">reads all the numbers</strong> (steps, sleep, calories, heart rate, etc.)
            for every user and automatically puts similar users into the same group — <strong style="color:#38bdf8">without you telling it anything</strong>.
        </p>
        <p style="color:#dce8f5;margin:0 0 8px 0"><strong style="color:#f59e0b">Example of what groups might look like:</strong></p>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:6px">
            <div style="background:#0d1b2a;border-left:3px solid #10b981;border-radius:6px;padding:8px 10px">
                <div style="color:#10b981;font-weight:700;font-size:0.85rem">🏃 Group 0 — Active</div>
                <div style="color:#a8c8e8;font-size:0.78rem;margin-top:3px">High steps · Low sedentary time · Good sleep</div>
            </div>
            <div style="background:#0d1b2a;border-left:3px solid #a78bfa;border-radius:6px;padding:8px 10px">
                <div style="color:#a78bfa;font-weight:700;font-size:0.85rem">🛋️ Group 1 — Sedentary</div>
                <div style="color:#a8c8e8;font-size:0.78rem;margin-top:3px">Low steps · High sedentary time · Irregular sleep</div>
            </div>
            <div style="background:#0d1b2a;border-left:3px solid #38bdf8;border-radius:6px;padding:8px 10px">
                <div style="color:#38bdf8;font-weight:700;font-size:0.85rem">🚶 Group 2 — Moderate</div>
                <div style="color:#a8c8e8;font-size:0.78rem;margin-top:3px">Average steps · Some active days · Average sleep</div>
            </div>
        </div>
        <p style="color:#6b7a9a;font-size:0.8rem;margin:10px 0 0 0">
            ⚠️ The group <strong>numbers (0, 1, 2…) have no special meaning</strong> — they are just labels. What matters is the <strong>profile chart below</strong> which shows what each group's average looks like.
        </p>
    </div>
    """, unsafe_allow_html=True)
    tip("We use two methods: **KMeans** (you pick how many groups) and **DBSCAN** (auto-detects groups and flags unusual users who don't fit anywhere).")

    from sklearn.cluster import KMeans, DBSCAN

    # ── Elbow (pre-computed, instant) ─────────────────────────
    st.markdown("### 📐 How Many Groups? — Elbow Chart")
    tip("Look for the **elbow** — where the line bends and stops dropping steeply. That's the best number of groups (k).")
    fig_el = px.line(x=k_range_list, y=inertias, markers=True,
                     labels={"x":"Number of Groups (k)","y":"Spread Score (lower = tighter groups)"},
                     title="Elbow Chart — Choose the best number of groups",
                     color_discrete_sequence=["#a78bfa"])
    fig_el.update_traces(marker=dict(size=10, color="#38bdf8", line=dict(color="white", width=1.5)))
    st.plotly_chart(T(fig_el), use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚙️ Clustering Settings")
    cc1, cc2, cc3 = st.columns(3)
    n_k   = cc1.slider("KMeans — number of groups (k)", 2, min(10, len(feat_scaled)-1), 3, key="k",
                        help="How many groups KMeans should create. Use the Elbow Chart above to pick the best value.")
    eps   = cc2.slider("DBSCAN — neighbourhood size (eps)", 0.3, 5.0, 2.5, 0.1, key="eps",
                        help="Controls how close two users must be to be placed in the same group. SMALL value = strict = many tiny groups. LARGE value = relaxed = fewer bigger groups. If you see too many groups, increase this.")
    msamp = cc3.slider("DBSCAN — minimum group size", 2, 15, 3, key="ms",
                        help="Minimum number of users needed to form a group. If a cluster has fewer users than this, they become Outliers. Increase this to reduce noise.")

    st.markdown("""
    <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:12px 16px;margin-top:4px;font-size:0.82rem;color:#a8c8e8">
        <strong style="color:#a78bfa">What do these settings do?</strong><br><br>
        🔵 <strong>KMeans k</strong> — You decide how many groups. Every user is forced into exactly one group. Use the Elbow Chart to pick k.<br>
        🟠 <strong>DBSCAN eps</strong> — Think of it as a "friendship radius". If two users are within this radius, they belong to the same group.
        Too small → everyone is an outlier. Too large → everyone is in one group. <em>Start at 1.5, increase if you see too many outliers.</em><br>
        🟡 <strong>DBSCAN min size</strong> — A group must have at least this many users to count. Small clusters below this size become outliers.
    </div>""", unsafe_allow_html=True)

    # KMeans & DBSCAN — run on user-level features (one row per user)
    km_labels = KMeans(n_clusters=n_k, random_state=42, n_init="auto").fit_predict(feat_scaled)
    db_labels = DBSCAN(eps=eps, min_samples=msamp).fit_predict(feat_scaled)
    user_df["KMeans_Cluster"] = km_labels
    user_df["DBSCAN_Cluster"] = db_labels
    n_db_c  = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = int((db_labels == -1).sum())

    # PCA coords pre-computed — instant
    v1, v2 = pca_var
    pca_df = pd.DataFrame(pca_coords, columns=["PC1","PC2"])
    pca_df["KMeans"] = ["Group " + str(l) for l in km_labels]
    pca_df["DBSCAN"] = [("Outlier" if l == -1 else "Group " + str(l)) for l in db_labels]

    # ── PCA scatter ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ User Group Maps (PCA)")
    tip("Each **dot = one user**. Same colour = same group. The chart squashes many dimensions into 2 so we can see it — focus on the colour, not the exact position.")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"**KMeans — {n_k} groups** (you chose this)")
        fig_km = px.scatter(pca_df, x="PC1", y="PC2", color="KMeans",
                            title=f"KMeans: {n_k} User Groups",
                            labels={"PC1":f"Dimension 1 ({v1:.0%} info)","PC2":f"Dimension 2 ({v2:.0%} info)","KMeans":"Group"},
                            color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
        fig_km.update_traces(marker=dict(size=13, opacity=0.85, line=dict(color="white", width=1)))
        st.plotly_chart(T(fig_km), use_container_width=True)
    with p2:
        st.markdown(f"**DBSCAN — auto-detected {n_db_c} group(s), {n_noise} outlier(s)**")
        # Build colour map: groups get bold colours, outliers get muted grey
        _db_unique = sorted([x for x in pca_df["DBSCAN"].unique() if x != "Outlier"]) + (["Outlier"] if "Outlier" in pca_df["DBSCAN"].values else [])
        # 12 clearly distinct colours — none of them red (red = outlier confusion risk)
        _db_palette = [
            "#a78bfa","#38bdf8","#10b981","#f59e0b","#f97316",
            "#06b6d4","#84cc16","#e879f9","#facc15","#4ade80",
            "#60a5fa","#fb923c"
        ]
        _db_colormap = {grp: _db_palette[i % len(_db_palette)] for i, grp in enumerate([x for x in _db_unique if x != "Outlier"])}
        if "Outlier" in _db_unique:
            _db_colormap["Outlier"] = "#94a3b8"   # clear light grey — unmistakably different
        _db_colors = [_db_colormap[g] for g in _db_unique]

        fig_db = px.scatter(pca_df, x="PC1", y="PC2", color="DBSCAN",
                            title=f"DBSCAN: {n_db_c} Groups + {n_noise} Outliers",
                            labels={"PC1":f"Dimension 1 ({v1:.0%} info)","PC2":f"Dimension 2 ({v2:.0%} info)","DBSCAN":"Group"},
                            category_orders={"DBSCAN": _db_unique},
                            color_discrete_map=_db_colormap)
        # Groups get larger dots, outliers smaller and transparent
        for trace in fig_db.data:
            if trace.name == "Outlier":
                trace.marker.update(size=7, opacity=0.35, line=dict(color="white", width=0.3))
            else:
                trace.marker.update(size=13, opacity=0.9, line=dict(color="white", width=1))
        st.plotly_chart(T(fig_db), use_container_width=True)

    tip("**Outlier** (shown as small grey dots in DBSCAN) = a user whose habits don't fit any group — could be a data issue or a genuinely unusual person.")

    # ── t-SNE ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌐 Advanced Group Map (t-SNE)")
    tip("t-SNE is better at separating groups that are close together. It takes ~20 seconds to run once.")

    if "tsne_df" not in st.session_state:
        if st.button("▶️  Run t-SNE", key="run_tsne"):
            ph = st.empty()
            ph.markdown('''<div class="loading-banner">
                <div style="font-size:1.1rem;margin-bottom:8px">🌐 <strong style="color:#a78bfa">Building t-SNE map…</strong></div>
                <div class="prog-wrap"><div class="prog-fill"></div></div>
                <div style="font-size:0.8rem;color:#5a7a9a;margin-top:6px">Takes ~20 seconds. The page will refresh automatically when done.</div>
            </div>''', unsafe_allow_html=True)
            try:
                from sklearn.manifold import TSNE
                perp = min(30, max(5, len(feat_scaled) - 1))
                tc   = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000).fit_transform(feat_scaled)
                tsne_df = pd.DataFrame(tc, columns=["tSNE1","tSNE2"])
                tsne_df["KMeans"] = ["Group " + str(l) for l in km_labels]
                tsne_df["DBSCAN"] = [("Outlier" if l == -1 else "Group " + str(l)) for l in db_labels]
                st.session_state["tsne_df"]      = tsne_df
                st.session_state["tsne_km_snap"] = km_labels.copy()
                st.session_state["tsne_db_snap"] = db_labels.copy()
                ph.empty()
                st.rerun()
            except Exception as e:
                ph.empty(); st.error(str(e))
    else:
        tsne_df = st.session_state["tsne_df"]
        # Update labels if sliders changed
        tsne_df["KMeans"] = ["Group " + str(l) for l in km_labels]
        tsne_df["DBSCAN"] = [("Outlier" if l == -1 else "Group " + str(l)) for l in db_labels]

        t1, t2 = st.columns(2)
        with t1:
            fig_t1 = px.scatter(tsne_df, x="tSNE1", y="tSNE2", color="KMeans",
                                title="t-SNE — KMeans Groups",
                                labels={"tSNE1":"t-SNE X","tSNE2":"t-SNE Y","KMeans":"Group"},
                                color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
            fig_t1.update_traces(marker=dict(size=12, opacity=0.85, line=dict(color="white", width=1)))
            st.plotly_chart(T(fig_t1), use_container_width=True)
        with t2:
            _t_unique = sorted([x for x in tsne_df["DBSCAN"].unique() if x != "Outlier"]) + (["Outlier"] if "Outlier" in tsne_df["DBSCAN"].values else [])
            _t_palette = [
                "#a78bfa","#38bdf8","#10b981","#f59e0b","#f97316",
                "#06b6d4","#84cc16","#e879f9","#facc15","#4ade80",
                "#60a5fa","#fb923c"
            ]
            _t_colormap = {grp: _t_palette[i % len(_t_palette)] for i, grp in enumerate([x for x in _t_unique if x != "Outlier"])}
            if "Outlier" in _t_unique:
                _t_colormap["Outlier"] = "#94a3b8"
            fig_t2 = px.scatter(tsne_df, x="tSNE1", y="tSNE2", color="DBSCAN",
                                title="t-SNE — DBSCAN Groups",
                                labels={"tSNE1":"t-SNE X","tSNE2":"t-SNE Y","DBSCAN":"Group"},
                                category_orders={"DBSCAN": _t_unique},
                                color_discrete_map=_t_colormap)
            for trace in fig_t2.data:
                if trace.name == "Outlier":
                    trace.marker.update(size=7, opacity=0.35, line=dict(color="white", width=0.3))
                else:
                    trace.marker.update(size=12, opacity=0.9, line=dict(color="white", width=1))
            st.plotly_chart(T(fig_t2), use_container_width=True)

        if st.button("🔄 Re-run t-SNE with current settings", key="rerun_tsne"):
            del st.session_state["tsne_df"]
            st.rerun()

    # ── Cluster profiles ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 What Does Each Group Look Like?")
    tip("Taller bars = higher average for that metric in that group. Use this to understand what makes each group different.")

    # Exclude Id, cluster labels, and any col with huge scale variance (like raw Id numbers)
    _exclude = {"KMeans_Cluster", "DBSCAN_Cluster", "Id", "id", "logId"}
    num_cols = [c for c in user_df.select_dtypes(include=[np.number]).columns
                if c not in _exclude]
    # Pick the most meaningful fitness cols first, then fallback to whatever is available
    _priority = ["TotalSteps","TotalDistance","Calories","AvgHeartRate",
                 "TotalSleepMinutes","TotalMinutesAsleep","VeryActiveMinutes",
                 "FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes",
                 "TotalIntensity","AverageIntensity","StepTotal"]
    ordered   = [c for c in _priority if c in num_cols]
    remaining = [c for c in num_cols  if c not in ordered]
    plot_cols  = (ordered + remaining)[:8]   # max 8 cols for readability

    if plot_cols:
        profile  = user_df.groupby("KMeans_Cluster")[plot_cols].mean().reset_index()
        long     = profile.melt(id_vars="KMeans_Cluster", var_name="Metric", value_name="Average")
        long["Group"] = "Group " + long["KMeans_Cluster"].astype(str)

        fig_bar = px.bar(long, x="Metric", y="Average", color="Group", barmode="group",
                         title="Average Metric per User Group",
                         labels={"Average":"Average Value","Metric":""},
                         color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
        fig_bar.update_xaxes(tickangle=-20)
        fig_bar.update_traces(
            hovertemplate="<b>%{x}</b><br>Average: %{y:,.1f}<extra>%{fullData.name}</extra>"
        )
        st.plotly_chart(T(fig_bar, h=420), use_container_width=True)
    else:
        st.info("No numeric metric columns found for profiling.")

    st.markdown("**What each group looks like — in plain English:**")
    tip("These descriptions are auto-generated from the data averages. The group name (Active / Moderate / Sedentary) is assigned based on step count.")
    _grp_colors = ["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444"]
    for i, row in user_df.groupby("KMeans_Cluster")[plot_cols if plot_cols else num_cols].mean().iterrows():
        parts = []
        activity_label = "Unknown"
        if steps_col and steps_col in row.index:
            v = row[steps_col]
            if v > 10000:
                activity_label = "🏃 Highly Active"
            elif v > 7000:
                activity_label = "🚶 Moderately Active"
            elif v > 4000:
                activity_label = "🧘 Lightly Active"
            else:
                activity_label = "🛋️ Sedentary"
            parts.append(f"Avg <strong>{v:,.0f} steps/day</strong>")
        if "AvgHeartRate" in row.index:
            hr = row["AvgHeartRate"]
            hr_label = "elevated" if hr > 80 else ("normal" if hr > 60 else "low")
            parts.append(f"Heart rate <strong>{hr:.0f} bpm</strong> ({hr_label})")
        if sleep_col and sleep_col in row.index:
            v = row[sleep_col]
            slp_label = "😴 Good sleep" if v >= 420 else "⚠️ Low sleep"
            parts.append(f"<strong>{v:.0f} min</strong> sleep/day ({slp_label})")
        if "Calories" in row.index:
            parts.append(f"Avg <strong>{row['Calories']:,.0f} calories</strong> burned/day")

        color = _grp_colors[i % len(_grp_colors)]
        n_users = int((user_df["KMeans_Cluster"] == i).sum())
        desc_html = "  &nbsp;·&nbsp;  ".join(parts) if parts else "See the bar chart above for details."

        st.markdown(f"""
        <div style="background:#162a41;border-left:4px solid {color};border-radius:8px;
                    padding:12px 16px;margin:8px 0;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                <span style="background:{color};color:#0d1b2a;font-weight:800;font-size:0.78rem;
                             padding:2px 8px;border-radius:20px">Group {i}</span>
                <span style="color:{color};font-weight:700;font-size:1rem">{activity_label}</span>
                <span style="color:#5a7a9a;font-size:0.78rem;margin-left:auto">{n_users} user(s) in this group</span>
            </div>
            <div style="color:#dce8f5;font-size:0.88rem">{desc_html}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(99,102,241,0.08);border:1px dashed #3a4a7e;border-radius:8px;
                padding:10px 14px;margin-top:10px;font-size:0.82rem;color:#7a8aaa">
        💡 <strong style="color:#a78bfa">Remember:</strong> Group numbers (0, 1, 2…) are just labels assigned automatically —
        Group 0 is not "better" than Group 2. What matters is the <strong>step count, heart rate, and sleep</strong> shown above.
        Use the bar chart to compare groups side by side.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — SUMMARY
# ══════════════════════════════════════════════════════════════
with st.expander("🏁  Summary & Download"):
    st.success("🎉 Pipeline complete!")
    tsf_features = st.session_state.get("tsf_features")

    r1, r2, r3 = st.columns(3)
    r1.metric("Files Loaded",     f"{n_matched} / 5")
    r2.metric("Users in Dataset", master_df["Id"].nunique() if "Id" in master_df.columns else "—")
    r3.metric("Days of Data",     f"{master_df.shape[0]:,} rows")

    r4, r5, r6 = st.columns(3)
    r4.metric("HR Data Points",   f"{len(hr_resampled):,}" if hr_resampled is not None else "—")
    r5.metric("TSFresh Features", str(tsf_features.shape[1]) if tsf_features is not None else "Not run")
    r6.metric("KMeans Groups",    str(n_k))

    st.markdown("---")
    checks = [
        ("✅", "Loaded and merged 5 Fitbit CSV files"),
        ("✅", "Parsed timestamps, resampled heart rate to 1-minute"),
        ("✅", "Built master daily dataset"),
        ("✅", "Filled missing values automatically"),
        ("✅" if tsf_features is not None else "⏭️", "TSFresh: heart rate feature extraction"),
        ("✅" if st.session_state.get("fc_hr") else "⏭️", "Prophet: heart rate forecast"),
        ("✅" if steps_col and st.session_state.get(f"fc_{steps_col}") else "⏭️", "Prophet: steps forecast"),
        ("✅" if sleep_col and st.session_state.get(f"fc_{sleep_col}") else "⏭️", "Prophet: sleep forecast"),
        ("✅", "KMeans clustering"),
        ("✅", "DBSCAN clustering"),
        ("✅", "PCA visualisation"),
        ("✅" if "tsne_df" in st.session_state else "⏭️", "t-SNE visualisation"),
    ]
    for icon, text in checks:
        st.markdown(f"{icon} {text}")

    st.markdown("---")
    st.download_button("📥 Download Final Master Dataset (CSV)",
                       master_df.to_csv(index=False).encode(),
                       "fitness_m2_master.csv", "text/csv")