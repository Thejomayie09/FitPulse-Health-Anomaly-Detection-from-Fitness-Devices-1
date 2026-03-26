import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from preprocessing import (
    load_all_files, parse_timestamps, resample_heartrate,
    build_master_df, prepare_tsfresh_input, build_clustering_features,
)

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(page_title="FitPulse", page_icon="💪", layout="wide")

# ── Shared CSS ──────────────────────────────────────────────────
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
    .log-box {
        background: rgba(74,144,196,0.1); border-left: 4px solid #a78bfa;
        border-radius: 4px; padding: 10px 15px; margin: 5px 0;
        font-family: 'Source Code Pro', monospace;
    }
    .loading-banner {
        background: linear-gradient(90deg, #1a2744, #1a1f4a);
        border: 1px solid #a78bfa; border-radius: 10px;
        padding: 18px 24px; margin: 12px 0; text-align: center;
    }
    .stSpinner > div { border-top-color: #a78bfa !important; }
    [data-testid="stAppViewContainer"] > section { opacity: 1 !important; transition: none !important; }
    .main .block-container { opacity: 1 !important; }
    [data-testid="stDecoration"] { display: none !important; }
    .stStatusWidget { opacity: 0 !important; pointer-events: none; }
    /* Pointer cursor on all interactive elements */
    .stSelectbox > div, .stSelectbox div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] * { cursor: pointer !important; }
    .stSlider, .stSlider * { cursor: pointer !important; }
    .stCheckbox, .stCheckbox * { cursor: pointer !important; }
    .stFileUploader, .stFileUploader * { cursor: pointer !important; }
    .stTabs [data-baseweb="tab"] { cursor: pointer !important; }
    .prog-wrap { background:#162a41; border-radius:99px; height:7px; margin:10px 0; overflow:hidden; }
    .prog-fill  { background:linear-gradient(90deg,#6366f1,#a78bfa); height:100%; border-radius:99px;
                  animation:pbar 1.4s ease-in-out infinite; }
    @keyframes pbar { 0%,100%{opacity:1;width:60%} 50%{opacity:.5;width:90%} }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ────────────────────────────────────────────────
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

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# ── Cached functions (Milestone 2) ─────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load(file_keys):
    from io import BytesIO
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
    from io import BytesIO
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
    inertias = [KMeans(n_clusters=k, random_state=42, n_init="auto").fit(feat).inertia_
                for k in K_range]
    return list(K_range), inertias

@st.cache_data(show_spinner=False)
def cached_pca(feat_bytes):
    from sklearn.decomposition import PCA
    import pickle
    feat   = pickle.loads(feat_bytes)
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(feat)
    return coords, pca.explained_variance_ratio_

# ══════════════════════════════════════════════════════════════
#  SIDEBAR — MILESTONE SELECTOR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=60)
    st.markdown("## Fitness Data Pro")
    st.markdown("---")
    milestone = st.selectbox(
        "📌 Select Milestone",
        ["📊 Milestone 1 — Data Preprocessing",
         "🧬 Milestone 2 — Feature Extraction & Modelling"],
        key="milestone_selector"
    )
    st.markdown("---")
    if "Milestone 1" in milestone:
        st.markdown("""
**Milestone 1 Pipeline:**
1. Upload CSV 📂
2. Inspect data 🔍
3. Clean & preprocess ⚙️
4. Export cleaned CSV ⬇️
""")
        st.caption("v1.0 | Preprocessing Tool")
    else:
        st.markdown("""
**Milestone 2 Pipeline:**
1. Upload 5 CSV files 📂
2. Preview & quality check 🔍
3. TSFresh feature extraction 🔬
4. Prophet forecasting 📈
5. Clustering 🔵
6. Download results ⬇️
""")
        st.caption("v2.0 | Feature Extraction & Modelling")


# ══════════════════════════════════════════════════════════════
#  MILESTONE 1
# ══════════════════════════════════════════════════════════════
if "Milestone 1" in milestone:

    st.title("📊 Data Collection & Preprocessing")
    st.markdown("Clean, normalise, and visualise your fitness tracking data easily.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Drop your fitness CSV here", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'processed_df' not in st.session_state:
            st.session_state.processed_df = None

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
                st.write("**Missing Values per Column**")
                null_counts = df.isnull().sum()
                st.bar_chart(null_counts[null_counts > 0])
            with col_types:
                st.write("**Data Types**")
                st.write(df.dtypes.to_frame(name='Type').astype(str))

        with tab2:
            st.subheader("Data Cleaning Engine")
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown("### Settings")
                handle_dates   = st.checkbox("Normalise Dates",    value=True)
                handle_numeric = st.checkbox("Interpolate Numbers", value=True)
                handle_cat     = st.checkbox("Fill Categories",     value=True)
                run_btn        = st.button("🚀 Execute Pipeline")
            with c2:
                if run_btn:
                    with st.spinner("Refining your data..."):
                        df_p = df.copy(); logs = []
                        if handle_dates:
                            date_cols = [c for c in df_p.columns if 'date' in c.lower()]
                            for col in date_cols:
                                df_p[col] = pd.to_datetime(df_p[col], dayfirst=True, errors='coerce').ffill()
                                df_p[col] = df_p[col].dt.date
                                logs.append(f"Fixed timestamps in: `{col}`")
                        if handle_numeric:
                            num_cols = df_p.select_dtypes(include=[np.number]).columns.tolist()
                            df_p[num_cols] = df_p[num_cols].interpolate().ffill().bfill()
                            logs.append("Interpolated numeric gaps (Linear Method)")
                        if handle_cat:
                            cat_cols = df_p.select_dtypes(include=['object']).columns
                            for col in cat_cols:
                                df_p[col] = df_p[col].fillna("Unknown")
                            logs.append("Filled missing categories with 'Unknown'")
                        st.session_state.processed_df = df_p
                        st.success("Preprocessing Complete!")
                        for l in logs:
                            st.markdown(f'<div class="log-box">✅ {l}</div>', unsafe_allow_html=True)
            if st.session_state.processed_df is not None:
                st.markdown("---")
                st.subheader("Cleaned Result")
                st.dataframe(st.session_state.processed_df.head(5), use_container_width=True)
                st.download_button("📥 Download Cleaned CSV",
                                   data=convert_df(st.session_state.processed_df),
                                   file_name='fitness_cleaned.csv', mime='text/csv')

        with tab3:
            if st.session_state.processed_df is not None:
                processed_df = st.session_state.processed_df
                st.subheader("🕵️ Outlier Detector")
                tip("Select any numeric column to check if it has unusual values. Outliers are values that are much higher or lower than the rest.")
                num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    col_to_plot = st.selectbox("Select a metric to check for unusual values:", num_cols)

                    # ── Summary stats strip ──────────────────────
                    col_data = processed_df[col_to_plot].dropna()
                    Q1  = col_data.quantile(0.25)
                    Q3  = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
                    s1,s2,s3,s4,s5 = st.columns(5)
                    s1.metric("Min",      f"{col_data.min():,.1f}")
                    s2.metric("Max",      f"{col_data.max():,.1f}")
                    s3.metric("Mean",     f"{col_data.mean():,.1f}")
                    s4.metric("Median",   f"{col_data.median():,.1f}")
                    s5.metric("Outliers", len(outliers),
                              delta=f"{len(outliers)/len(col_data)*100:.1f}% of data",
                              delta_color="inverse")

                    st.markdown("")
                    v1, v2 = st.columns(2)
                    with v1:
                        st.markdown(f"**📦 Box Plot — {col_to_plot}**")
                        fig_box = px.box(
                            processed_df, y=col_to_plot,
                            points="all",
                            color_discrete_sequence=["#a78bfa"],
                            labels={col_to_plot: col_to_plot},
                        )
                        fig_box.update_traces(
                            marker=dict(size=4, opacity=0.5, color="#38bdf8"),
                            line=dict(color="#a78bfa"),
                            fillcolor="rgba(167,139,250,0.3)",
                            boxmean=True,
                        )
                        fig_box.update_layout(
                            **{**PT, "height": 420},
                            title=dict(text=f"Range View — {col_to_plot}", font=dict(color="#a78bfa", size=13)),
                            yaxis_title=col_to_plot,
                            xaxis=dict(showticklabels=False),
                        )
                        fig_box.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
                        st.plotly_chart(fig_box, use_container_width=True)


                    with v2:
                        st.markdown(f"**📊 Histogram — {col_to_plot}**")
                        # Colour bars: outliers in red, normal in purple
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        fig_hist = px.histogram(
                            processed_df, x=col_to_plot,
                            nbins=40,
                            color_discrete_sequence=["#6366f1"],
                            labels={col_to_plot: col_to_plot, "count": "Number of Records"},
                        )
                        fig_hist.update_traces(
                            marker_color="#6366f1",
                            marker_line_color="#a78bfa",
                            marker_line_width=1,
                            opacity=0.85,
                        )
                        # Outlier zone shading
                        if len(outliers) > 0:
                            fig_hist.add_vrect(
                                x0=col_data.min(), x1=lower_bound,
                                fillcolor="rgba(239,68,68,0.12)",
                                layer="below", line_width=0,
                                annotation_text="Outlier zone",
                                annotation_position="top left",
                                annotation_font_color="#ef4444",
                                annotation_font_size=10,
                            )
                            fig_hist.add_vrect(
                                x0=upper_bound, x1=col_data.max(),
                                fillcolor="rgba(239,68,68,0.12)",
                                layer="below", line_width=0,
                                annotation_text="Outlier zone",
                                annotation_position="top right",
                                annotation_font_color="#ef4444",
                                annotation_font_size=10,
                            )
                        # Mean and Median lines — positioned at different heights to avoid overlap
                        fig_hist.add_vline(x=col_data.mean(), line_dash="dash",
                                           line_color="#f59e0b", line_width=2)
                        fig_hist.add_annotation(
                            x=col_data.mean(), y=1, yref="paper",
                            text=f"Mean: {col_data.mean():,.1f}",
                            showarrow=False, font=dict(color="#f59e0b", size=11),
                            xanchor="left", yanchor="top",
                            bgcolor="rgba(13,27,42,0.7)", bordercolor="#f59e0b",
                        )
                        fig_hist.add_vline(x=col_data.median(), line_dash="dot",
                                           line_color="#10b981", line_width=2)
                        fig_hist.add_annotation(
                            x=col_data.median(), y=0.88, yref="paper",
                            text=f"Median: {col_data.median():,.1f}",
                            showarrow=False, font=dict(color="#10b981", size=11),
                            xanchor="left", yanchor="top",
                            bgcolor="rgba(13,27,42,0.7)", bordercolor="#10b981",
                        )
                        fig_hist.update_layout(
                            **{**PT, "height": 420},
                            title=dict(text=f"Frequency Distribution — {col_to_plot}", font=dict(color="#a78bfa", size=13)),
                            xaxis_title=col_to_plot,
                            yaxis_title="Number of Records",
                            bargap=0.03,
                        )
                        fig_hist.update_xaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
                        fig_hist.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # ── Outlier table ────────────────────────────
                    if len(outliers) > 0:
                        st.markdown("---")
                        st.markdown(f"### 🚨 Outlier Rows — {len(outliers)} found")
                        tip(f"These are the actual rows where **{col_to_plot}** has an unusual value. The highlighted column shows the outlier value.")
                        outlier_rows = processed_df[
                            (processed_df[col_to_plot] < Q1 - 1.5*IQR) |
                            (processed_df[col_to_plot] > Q3 + 1.5*IQR)
                        ].copy()
                        outlier_rows["⚠️ Outlier Value"] = outlier_rows[col_to_plot]
                        st.dataframe(outlier_rows, use_container_width=True)
                        st.markdown(f"""
                        <div class="info-box">
                            ⚠️ <strong>What is an outlier?</strong><br>
                            Any value below <strong>{Q1 - 1.5*IQR:,.1f}</strong> or above <strong>{Q3 + 1.5*IQR:,.1f}</strong>
                            is considered an outlier for <strong>{col_to_plot}</strong>.<br>
                            Normal range: <strong>{Q1:,.1f}</strong> (Q1) to <strong>{Q3:,.1f}</strong> (Q3)
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.success(f"✅ No outliers found in **{col_to_plot}** — all values are within the normal range.")
                else:
                    st.warning("No numeric columns found.")
            else:
                st.warning("⚠️ Run the **Processing** tab first to see visualisations.")
    else:
        st.markdown("""
            <div style="text-align:center;padding:100px 20px;border:2px dashed #1e3a5f;border-radius:20px;">
                <h2 style="color:#4a90c4 !important;">Awaiting Dataset...</h2>
                <p>Upload a CSV file to begin the fitness data transformation.</p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MILESTONE 2
# ══════════════════════════════════════════════════════════════
else:

    st.title("🧬 Fitness Data — Feature Extraction & Modelling")
    st.markdown("Load your Fitbit data, extract patterns, forecast trends, and group users by behaviour.")
    st.markdown("---")

    # ── SECTION 1: UPLOAD ──────────────────────────────────────
    with st.expander("📂  Step 1 — Upload Your Data Files", expanded=True):
        tip("Select all 5 files at once — hold Ctrl (Windows) or Cmd (Mac) while clicking.")
        uploaded = st.file_uploader("Upload all 5 CSV files", type=["csv"],
                                    accept_multiple_files=True, label_visibility="collapsed")
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
            st.warning(f"Detected {n_matched}/5 files. Check file names contain the keywords above.")

    # ── Build cache key ────────────────────────────────────────
    file_keys = {}
    for role, f in matched.items():
        if f is not None:
            f.seek(0); file_keys[role] = (f.name, f.read()); f.seek(0)

    # ── Load data ──────────────────────────────────────────────
    _cache_key    = str(sorted([(r, n) for r, (n, _) in file_keys.items()]))
    _is_first     = st.session_state.get("_last_cache_key") != _cache_key
    status_ph     = st.empty()
    if _is_first:
        status_ph.markdown('''
        <div class="loading-banner">
            <div style="font-size:1.1rem;margin-bottom:8px">⏳ <strong style="color:#a78bfa">Loading and processing your files… please wait</strong></div>
            <div class="prog-wrap"><div class="prog-fill"></div></div>
            <div style="font-size:0.8rem;color:#5a7a9a;margin-top:6px">This only runs once per upload — subsequent interactions will be instant</div>
        </div>''', unsafe_allow_html=True)

    dfs       = cached_load(file_keys)
    master_df = cached_master(file_keys)
    hr_resampled = None
    if "heartrate" in file_keys:
        hr_name, hr_bytes = file_keys["heartrate"]
        hr_resampled = cached_resample(hr_bytes, hr_name)

    st.session_state["_last_cache_key"] = _cache_key
    status_ph.empty()

    # ── Cleaning report ────────────────────────────────────────
    with st.expander("🧹  Data Cleaning Report — what was fixed automatically", expanded=False):
        tip("Every time you upload files, the pipeline automatically cleans them. Here is exactly what was done.")
        ROLES_LABELS = {"daily_activity":"Daily Activity","heartrate":"Heart Rate",
                        "hourly_intensities":"Hourly Intensities","hourly_steps":"Hourly Steps","minute_sleep":"Minute Sleep"}
        for label, df in dfs.items():
            st.markdown(f"**{ROLES_LABELS.get(label, label)}**")
            import io
            raw_df  = pd.read_csv(io.BytesIO(file_keys[label][1]))
            dupes   = int(raw_df.duplicated().sum())
            nulls_b = int(raw_df.isnull().sum().sum())
            nulls_a = int(df.isnull().sum().sum())
            rows_b  = len(raw_df); rows_a = len(df)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Rows (original)",    f"{rows_b:,}")
            c2.metric("Rows (after clean)", f"{rows_a:,}",
                      delta=f"-{rows_b-rows_a} removed" if rows_b!=rows_a else "none removed",
                      delta_color="inverse")
            c3.metric("Duplicates removed", dupes)
            c4.metric("Nulls before → after", f"{nulls_b} → {nulls_a}")
            null_cols = raw_df.isnull().sum(); null_cols = null_cols[null_cols > 0]
            if not null_cols.empty:
                num_in = raw_df.select_dtypes(include=[np.number]).columns.tolist()
                cat_in = raw_df.select_dtypes(include=["object"]).columns.tolist()
                for col, cnt in null_cols.items():
                    if col in num_in:   action = "filled — interpolation per user, then median fallback"
                    elif col in cat_in: action = "filled with 'Unknown'"
                    else:               action = "rows with bad timestamps were dropped"
                    st.markdown(f'<div class="info-box">🔧 <code>{col}</code>: {cnt} missing → {action}</div>',
                                unsafe_allow_html=True)
            else:
                st.success("No missing values in this file.")
            st.markdown("---")

    if master_df.empty:
        st.error("Could not build the master dataset. Check that Daily Activity file is uploaded.")
        st.stop()

    steps_col = next((c for c in master_df.columns if "step" in c.lower()), None)
    sleep_col  = next((c for c in master_df.columns if "sleep" in c.lower() or "Sleep" in c), None)

    import pickle
    from sklearn.preprocessing import StandardScaler
    if "Id" in master_df.columns:
        _num_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
        _excl     = {"KMeans_Cluster","DBSCAN_Cluster","Id","id","logId"}
        _num_cols = [c for c in _num_cols if c not in _excl]
        user_df   = master_df.groupby("Id")[_num_cols].mean().reset_index(drop=True)
    else:
        user_df = master_df.copy()

    feat_matrix  = build_clustering_features(user_df)
    feat_scaled  = StandardScaler().fit_transform(feat_matrix.fillna(0))
    feat_pkl     = pickle.dumps(feat_scaled)
    k_range_list, inertias = cached_elbow(feat_pkl, k_max=min(10, len(feat_scaled)-1))
    pca_coords, pca_var    = cached_pca(feat_pkl)

    # ── SECTION 2: DATA OVERVIEW ───────────────────────────────
    with st.expander("🔍  Step 2 — Data Preview & Quality Check"):
        tip("Preview each file and check whether any data is missing.")
        tabs = st.tabs([ROLES[k][0] for k in dfs.keys()])
        for tab, (role, df) in zip(tabs, dfs.items()):
            with tab:
                r1,r2,r3 = st.columns(3)
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
                                 labels={"x":"Column","y":"Missing Count"}, color_discrete_sequence=["#a78bfa"])
                    fig.update_xaxes(tickangle=-30)
                    nc[ci % len(nc)].plotly_chart(T(fig, h=230), use_container_width=True)
                    ci += 1
        else:
            st.success("✅ No missing values found across all files.")
        st.markdown("---")
        st.markdown("**Merged Master Dataset** — all 5 files combined into one table:")
        tip("Each row = one day for one user. All files are joined by User ID + Date.")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total Rows",    f"{master_df.shape[0]:,}")
        m2.metric("Columns",       master_df.shape[1])
        m3.metric("Missing Cells", int(master_df.isnull().sum().sum()))
        m4.metric("Completeness",  f"{100 - master_df.isnull().sum().sum()/master_df.size*100:.1f}%")
        st.dataframe(master_df.head(6), use_container_width=True)
        st.download_button("📥 Download Merged Dataset", master_df.to_csv(index=False).encode(),
                           "fitness_master.csv", "text/csv")

    # ── SECTION 3: TSFRESH ────────────────────────────────────
    with st.expander("🔬  Step 3 — Automatic Feature Extraction (TSFresh)"):
        tip("TSFresh reads heart rate over time and automatically calculates useful numbers per user — like average, variability, peaks. These **features** are what machine learning models use.")
        tsf_input    = prepare_tsfresh_input(hr_resampled) if hr_resampled is not None else None
        tsf_features = st.session_state.get("tsf_features")
        if tsf_input is None:
            st.warning("Heart rate file not found — upload it to enable TSFresh.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Users",               tsf_input["id"].nunique())
            c2.metric("Heart Rate Readings", f"{len(tsf_input):,}")
            c3.metric("Features Extracted",  tsf_features.shape[1] if tsf_features is not None else "—")
            if tsf_features is None:
                if st.button("▶️  Extract Features", key="run_tsf"):
                    prog_ph = st.empty()
                    prog_ph.markdown('''<div class="loading-banner">
                        <div style="font-size:1.1rem;margin-bottom:8px">🔬 <strong style="color:#a78bfa">Extracting heart rate features…</strong></div>
                        <div class="prog-wrap"><div class="prog-fill"></div></div>
                        <div style="font-size:0.8rem;color:#5a7a9a;margin-top:6px">This takes 30–90 seconds. Keep this tab open.</div>
                    </div>''', unsafe_allow_html=True)
                    try:
                        from tsfresh import extract_features
                        from tsfresh.utilities.dataframe_functions import impute
                        from tsfresh.feature_extraction import MinimalFCParameters
                        feat = extract_features(tsf_input, column_id="id", column_sort="time",
                                                column_value="value",
                                                default_fc_parameters=MinimalFCParameters(),
                                                n_jobs=1, disable_progressbar=True)
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

    # ── SECTION 4: PROPHET ────────────────────────────────────
    with st.expander("📈  Step 4 — Trend Forecasting (Prophet)"):
        tip("Prophet looks at past data and predicts future values. Shaded area = likely range. Dashed line = where forecast begins.")

        def prophet_plot(ds_series, y_series, label, key, unit="", color="#10b981"):
            tmp = pd.DataFrame({
                "ds": pd.to_datetime(ds_series, errors="coerce"),
                "y":  pd.to_numeric(y_series, errors="coerce"),
            }).dropna().sort_values("ds")
            if len(tmp) < 10:
                st.warning(f"Not enough data to forecast {label}."); return

            run = st.button(f"▶️  Forecast {label}", key=f"btn_{key}")
            if run:
                ph = st.empty()
                ph.markdown(f'<div class="loading-banner">📈 <strong style="color:#a78bfa">Fitting Prophet for {label}… please wait.</strong></div>',
                            unsafe_allow_html=True)
                try:
                    from prophet import Prophet
                    m = Prophet(weekly_seasonality=True, daily_seasonality=False, interval_width=0.80)
                    m.fit(tmp)
                    fc = m.predict(m.make_future_dataframe(periods=30, freq="D"))
                    st.session_state[f"fc_{key}"] = (fc, tmp)
                    ph.empty()
                except ImportError:
                    ph.empty(); st.error("Run: pip install prophet")
                except Exception as e:
                    ph.empty(); st.error(str(e))

            saved = st.session_state.get(f"fc_{key}")
            if saved:
                fc, tmp2 = saved
                cut = tmp2["ds"].max(); cut_str = str(cut)
                def hex_to_rgba(h, a):
                    h = h.lstrip("#"); r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
                    return f"rgba({r},{g},{b},{a})"
                ci_fill = hex_to_rgba(color, 0.30)
                ci_line = hex_to_rgba(color, 0.0)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines",
                    line=dict(color=ci_line, width=0), name="CI Upper", showlegend=False,
                    hovertemplate="CI Upper: %{y:.1f}<extra>80% CI</extra>"))
                fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines",
                    fill="tonexty", fillcolor=ci_fill, line=dict(color=ci_line, width=0),
                    name="CI Lower", showlegend=False,
                    hovertemplate="CI Lower: %{y:.1f}<extra>80% CI</extra>"))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                    line=dict(color=hex_to_rgba(color, 0.6), width=10),
                    name="80% CI", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines",
                    line=dict(color="#111827", width=2), name="Trend",
                    hovertemplate="Trend: %{y:.1f}<extra>Trend</extra>"))
                y_lbl = f"Actual {label}{' (' + unit + ')' if unit else ''}"
                fig.add_trace(go.Scatter(x=tmp2["ds"], y=tmp2["y"], mode="markers",
                    marker=dict(color=color, size=7, opacity=0.85, line=dict(color="white", width=0.5)),
                    name=y_lbl,
                    hovertemplate=label + ": %{y:.1f}<extra>" + y_lbl + "</extra>"))
                PT_fc = {k: v for k, v in PT.items() if k != "legend"}
                fig.update_layout(
                    title=f"{label} — Prophet Trend Forecast",
                    xaxis_title="Date", yaxis_title=f"{label}{' (' + unit + ')' if unit else ''}",
                    legend=dict(orientation="v", x=0.01, y=0.99,
                                bgcolor="rgba(22,42,65,0.85)", bordercolor="#1e3a5f",
                                borderwidth=1, font=dict(size=11, color="#dce8f5")),
                    shapes=[dict(type="line", x0=cut_str, x1=cut_str, y0=0, y1=1,
                                 xref="x", yref="paper", line=dict(color="#f59e0b", width=1.8, dash="dash"))],
                    annotations=[dict(x=cut_str, y=0.99, xref="x", yref="paper",
                                      text="Forecast Start", showarrow=False,
                                      font=dict(color="#f59e0b", size=10),
                                      xanchor="left", yanchor="top", bgcolor="rgba(22,42,65,0.7)")],
                    hovermode="x unified", height=420, **PT_fc)
                fig.update_xaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f",
                                 tickformat="%Y-%m-%d", tickangle=-20)
                fig.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
                st.plotly_chart(fig, use_container_width=True)
                tip(f"Dots = actual recorded {label.lower()}. Black line = Prophet's predicted trend. Shaded band = 80% CI (CI Upper/Lower on hover). Orange dashed line = where forecast begins.")

        if hr_resampled is not None and "Time" in hr_resampled.columns:
            st.markdown("### 💓 Heart Rate")
            _hr = hr_resampled.copy()
            _hr["Time"] = pd.to_datetime(_hr["Time"], errors="coerce")
            _hr["ActivityDate"] = _hr["Time"].dt.normalize()
            if "Id" in _hr.columns:
                _hr_daily = (_hr.groupby(["Id","ActivityDate"])["Value"].mean()
                               .reset_index().groupby("ActivityDate")["Value"].mean().reset_index())
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

    # ── SECTION 5: CLUSTERING ─────────────────────────────────
    with st.expander("🔵  Step 5 — User Grouping (Clustering)"):
        # st.markdown("""
        # <div style="background:#162a41;border:1px solid #1e4a6e;border-radius:12px;padding:18px 20px;margin-bottom:12px">
        #     <h4 style="color:#a78bfa;margin:0 0 10px 0">🤔 What does a "Group" mean here?</h4>
        #     <p style="color:#dce8f5;margin:0 0 10px 0">
        #         Clustering reads all the numbers (steps, sleep, calories, heart rate) for every user and
        #         automatically puts similar users into the same group — <strong style="color:#38bdf8">without you telling it anything</strong>.
        #     </p>
        #     <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:6px">
        #         <div style="background:#0d1b2a;border-left:3px solid #10b981;border-radius:6px;padding:8px 10px">
        #             <div style="color:#10b981;font-weight:700;font-size:0.85rem">🏃 Active</div>
        #             <div style="color:#a8c8e8;font-size:0.78rem;margin-top:3px">High steps · Low sedentary · Good sleep</div>
        #         </div>
        #         <div style="background:#0d1b2a;border-left:3px solid #a78bfa;border-radius:6px;padding:8px 10px">
        #             <div style="color:#a78bfa;font-weight:700;font-size:0.85rem">🛋️ Sedentary</div>
        #             <div style="color:#a8c8e8;font-size:0.78rem;margin-top:3px">Low steps · High sedentary · Poor sleep</div>
        #         </div>
        #         <div style="background:#0d1b2a;border-left:3px solid #38bdf8;border-radius:6px;padding:8px 10px">
        #             <div style="color:#38bdf8;font-weight:700;font-size:0.85rem">🚶 Moderate</div>
        #             <div style="color:#a8c8e8;font-size:0.78rem;margin-top:3px">Average steps · Some active days</div>
        #         </div>
        #     </div>
        #     <p style="color:#6b7a9a;font-size:0.8rem;margin:10px 0 0 0">
        #         ⚠️ Group numbers (0,1,2…) are just labels. What matters is the profile chart below.
        #     </p>
        # </div>""", unsafe_allow_html=True)

        from sklearn.cluster import KMeans, DBSCAN

        st.markdown("### 📐 Elbow Chart — How Many Groups?")
        tip("Look for the **elbow** — where the line bends and stops dropping steeply. That's the best number of groups.")
        fig_el = px.line(x=k_range_list, y=inertias, markers=True,
                         labels={"x":"Number of Groups (k)","y":"Spread Score (lower = tighter groups)"},
                         title="Elbow Chart — Choose the best number of groups",
                         color_discrete_sequence=["#a78bfa"])
        fig_el.update_traces(marker=dict(size=10, color="#38bdf8", line=dict(color="white", width=1.5)))
        st.plotly_chart(T(fig_el), use_container_width=True)

        st.markdown("---")
        st.markdown("### ⚙️ Clustering Settings")
        cc1,cc2,cc3 = st.columns(3)
        n_k   = cc1.slider("KMeans — number of groups (k)", 2, min(10,len(feat_scaled)-1), 3, key="k",
                            help="How many groups KMeans should create. Use the Elbow Chart to pick the best value.")
        eps   = cc2.slider("DBSCAN — neighbourhood size (eps)", 0.3, 5.0, 2.5, 0.1, key="eps",
                            help="SMALL = strict = many groups. LARGE = relaxed = fewer groups. Increase if too many outliers.")
        msamp = cc3.slider("DBSCAN — minimum group size", 2, 15, 3, key="ms",
                            help="Groups smaller than this become outliers.")

        st.markdown("""
        <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:12px 16px;margin-top:4px;font-size:0.82rem;color:#a8c8e8">
            <strong style="color:#a78bfa">What do these settings do?</strong><br><br>
            🔵 <strong>KMeans k</strong> — You decide how many groups. Every user is forced into exactly one group.<br>
            🟠 <strong>DBSCAN eps</strong> — "Friendship radius". Small → everyone is outlier. Large → one big group. <em>Start at 2.5.</em><br>
            🟡 <strong>DBSCAN min size</strong> — Groups smaller than this become outliers.
        </div>""", unsafe_allow_html=True)

        km_labels = KMeans(n_clusters=n_k, random_state=42, n_init="auto").fit_predict(feat_scaled)
        db_labels = DBSCAN(eps=eps, min_samples=msamp).fit_predict(feat_scaled)
        user_df["KMeans_Cluster"] = km_labels
        user_df["DBSCAN_Cluster"] = db_labels
        n_db_c  = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = int((db_labels == -1).sum())

        v1, v2 = pca_var
        pca_df = pd.DataFrame(pca_coords, columns=["PC1","PC2"])
        pca_df["KMeans"] = ["Group " + str(l) for l in km_labels]
        pca_df["DBSCAN"] = [("Outlier" if l==-1 else "Group " + str(l)) for l in db_labels]

        st.markdown("---")
        st.markdown("### 🗺️ User Group Maps (PCA)")
        tip("Each **dot = one user**. Same colour = same group. Focus on colour, not exact position.")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(f"**KMeans — {n_k} groups**")
            fig_km = px.scatter(pca_df, x="PC1", y="PC2", color="KMeans",
                                title=f"KMeans: {n_k} User Groups",
                                labels={"PC1":f"Dimension 1 ({v1:.0%} info)","PC2":f"Dimension 2 ({v2:.0%} info)","KMeans":"Group"},
                                color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
            fig_km.update_traces(marker=dict(size=13, opacity=0.85, line=dict(color="white", width=1)))
            st.plotly_chart(T(fig_km), use_container_width=True)
        with p2:
            st.markdown(f"**DBSCAN — {n_db_c} group(s), {n_noise} outlier(s)**")
            _db_unique = sorted([x for x in pca_df["DBSCAN"].unique() if x != "Outlier"]) + \
                         (["Outlier"] if "Outlier" in pca_df["DBSCAN"].values else [])
            _db_palette = ["#a78bfa","#38bdf8","#10b981","#f59e0b","#f97316","#06b6d4",
                           "#84cc16","#e879f9","#facc15","#4ade80","#60a5fa","#fb923c"]
            _db_colormap = {g: _db_palette[i % len(_db_palette)]
                            for i,g in enumerate([x for x in _db_unique if x!="Outlier"])}
            if "Outlier" in _db_unique: _db_colormap["Outlier"] = "#94a3b8"
            fig_db = px.scatter(pca_df, x="PC1", y="PC2", color="DBSCAN",
                                title=f"DBSCAN: {n_db_c} Groups + {n_noise} Outliers",
                                labels={"PC1":f"Dimension 1 ({v1:.0%} info)","PC2":f"Dimension 2 ({v2:.0%} info)","DBSCAN":"Group"},
                                category_orders={"DBSCAN": _db_unique}, color_discrete_map=_db_colormap)
            for trace in fig_db.data:
                if trace.name == "Outlier":
                    trace.marker.update(size=7, opacity=0.35, line=dict(color="white", width=0.3))
                else:
                    trace.marker.update(size=13, opacity=0.9, line=dict(color="white", width=1))
            st.plotly_chart(T(fig_db), use_container_width=True)
        tip("**Outlier** (small grey dots) = a user whose habits don't fit any group.")

        st.markdown("---")
        st.markdown("### 🌐 Advanced Group Map (t-SNE)")
        tip("t-SNE compresses 20+ dimensions into 2 for better visual separation. Takes ~20 seconds.")
        if "tsne_df" not in st.session_state:
            if st.button("▶️  Run t-SNE", key="run_tsne"):
                ph = st.empty()
                ph.markdown('''<div class="loading-banner">
                    <div style="font-size:1.1rem;margin-bottom:8px">🌐 <strong style="color:#a78bfa">Building t-SNE map…</strong></div>
                    <div class="prog-wrap"><div class="prog-fill"></div></div>
                    <div style="font-size:0.8rem;color:#5a7a9a;margin-top:6px">Takes ~20 seconds. Page refreshes automatically when done.</div>
                </div>''', unsafe_allow_html=True)
                try:
                    from sklearn.manifold import TSNE
                    perp = min(30, max(5, len(feat_scaled)-1))
                    tc   = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000).fit_transform(feat_scaled)
                    tsne_df = pd.DataFrame(tc, columns=["tSNE1","tSNE2"])
                    tsne_df["KMeans"] = ["Group " + str(l) for l in km_labels]
                    tsne_df["DBSCAN"] = [("Outlier" if l==-1 else "Group " + str(l)) for l in db_labels]
                    st.session_state["tsne_df"] = tsne_df
                    ph.empty(); st.rerun()
                except Exception as e:
                    ph.empty(); st.error(str(e))
        else:
            tsne_df = st.session_state["tsne_df"]
            tsne_df["KMeans"] = ["Group " + str(l) for l in km_labels]
            tsne_df["DBSCAN"] = [("Outlier" if l==-1 else "Group " + str(l)) for l in db_labels]
            t1, t2 = st.columns(2)
            with t1:
                fig_t1 = px.scatter(tsne_df, x="tSNE1", y="tSNE2", color="KMeans",
                                    title="t-SNE — KMeans Groups",
                                    labels={"tSNE1":"t-SNE X","tSNE2":"t-SNE Y","KMeans":"Group"},
                                    color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
                fig_t1.update_traces(marker=dict(size=12, opacity=0.85, line=dict(color="white", width=1)))
                st.plotly_chart(T(fig_t1), use_container_width=True)
            with t2:
                _t_unique = sorted([x for x in tsne_df["DBSCAN"].unique() if x!="Outlier"]) + \
                            (["Outlier"] if "Outlier" in tsne_df["DBSCAN"].values else [])
                _t_pal = ["#a78bfa","#38bdf8","#10b981","#f59e0b","#f97316","#06b6d4",
                          "#84cc16","#e879f9","#facc15","#4ade80","#60a5fa","#fb923c"]
                _t_cmap = {g: _t_pal[i%len(_t_pal)] for i,g in enumerate([x for x in _t_unique if x!="Outlier"])}
                if "Outlier" in _t_unique: _t_cmap["Outlier"] = "#94a3b8"
                fig_t2 = px.scatter(tsne_df, x="tSNE1", y="tSNE2", color="DBSCAN",
                                    title="t-SNE — DBSCAN Groups",
                                    labels={"tSNE1":"t-SNE X","tSNE2":"t-SNE Y","DBSCAN":"Group"},
                                    category_orders={"DBSCAN": _t_unique}, color_discrete_map=_t_cmap)
                for trace in fig_t2.data:
                    if trace.name == "Outlier":
                        trace.marker.update(size=7, opacity=0.35, line=dict(color="white", width=0.3))
                    else:
                        trace.marker.update(size=12, opacity=0.9, line=dict(color="white", width=1))
                st.plotly_chart(T(fig_t2), use_container_width=True)
            if st.button("🔄 Re-run t-SNE with current settings", key="rerun_tsne"):
                del st.session_state["tsne_df"]; st.rerun()

        st.markdown("---")
        st.markdown("### 📊 What Does Each Group Look Like?")
        tip("Taller bars = higher average for that metric in that group.")
        _excl2    = {"KMeans_Cluster","DBSCAN_Cluster","Id","id","logId"}
        num_cols2 = [c for c in user_df.select_dtypes(include=[np.number]).columns if c not in _excl2]
        _priority = ["TotalSteps","TotalDistance","Calories","AvgHeartRate","TotalSleepMinutes",
                     "TotalMinutesAsleep","VeryActiveMinutes","FairlyActiveMinutes",
                     "LightlyActiveMinutes","SedentaryMinutes","TotalIntensity","AverageIntensity","StepTotal"]
        ordered   = [c for c in _priority if c in num_cols2]
        remaining = [c for c in num_cols2 if c not in ordered]
        plot_cols = (ordered + remaining)[:8]

        if plot_cols:
            profile = user_df.groupby("KMeans_Cluster")[plot_cols].mean().reset_index()
            long    = profile.melt(id_vars="KMeans_Cluster", var_name="Metric", value_name="Average")
            long["Group"] = "Group " + long["KMeans_Cluster"].astype(str)
            fig_bar = px.bar(long, x="Metric", y="Average", color="Group", barmode="group",
                             title="Average Metric per User Group",
                             labels={"Average":"Average Value","Metric":""},
                             color_discrete_sequence=["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444","#6366f1"])
            fig_bar.update_xaxes(tickangle=-20)
            fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>Average: %{y:,.1f}<extra>%{fullData.name}</extra>")
            st.plotly_chart(T(fig_bar, h=420), use_container_width=True)

        st.markdown("**Plain-English Group Descriptions:**")
        _gc = ["#a78bfa","#38bdf8","#10b981","#f59e0b","#ef4444"]
        for i, row in user_df.groupby("KMeans_Cluster")[plot_cols if plot_cols else num_cols2].mean().iterrows():
            parts = []; al = "Unknown"
            if steps_col and steps_col in row.index:
                v = row[steps_col]
                al = ("🏃 Highly Active" if v>10000 else "🚶 Moderately Active" if v>7000
                      else "🧘 Lightly Active" if v>4000 else "🛋️ Sedentary")
                parts.append(f"Avg <strong>{v:,.0f} steps/day</strong>")
            if "AvgHeartRate" in row.index:
                hr = row["AvgHeartRate"]
                parts.append(f"HR <strong>{hr:.0f} bpm</strong> ({'elevated' if hr>80 else 'normal' if hr>60 else 'low'})")
            if sleep_col and sleep_col in row.index:
                v = row[sleep_col]
                parts.append(f"<strong>{v:.0f} min</strong> sleep ({'😴 Good' if v>=420 else '⚠️ Low'})")
            if "Calories" in row.index:
                parts.append(f"<strong>{row['Calories']:,.0f} cal</strong>/day")
            color  = _gc[i % len(_gc)]
            n_u    = int((user_df["KMeans_Cluster"] == i).sum())
            desc   = "  &nbsp;·&nbsp;  ".join(parts) if parts else "See chart above."
            st.markdown(f"""
            <div style="background:#162a41;border-left:4px solid {color};border-radius:8px;padding:12px 16px;margin:8px 0;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                    <span style="background:{color};color:#0d1b2a;font-weight:800;font-size:0.78rem;padding:2px 8px;border-radius:20px">Group {i}</span>
                    <span style="color:{color};font-weight:700;font-size:1rem">{al}</span>
                    <span style="color:#5a7a9a;font-size:0.78rem;margin-left:auto">{n_u} user(s)</span>
                </div>
                <div style="color:#dce8f5;font-size:0.88rem">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(99,102,241,0.08);border:1px dashed #3a4a7e;border-radius:8px;
                    padding:10px 14px;margin-top:10px;font-size:0.82rem;color:#7a8aaa">
            💡 <strong style="color:#a78bfa">Remember:</strong> Group numbers are just labels — Group 0 is not better than Group 2.
            What matters is the step count, heart rate, and sleep shown above.
        </div>""", unsafe_allow_html=True)

    # ── SECTION 6: SUMMARY ────────────────────────────────────
    with st.expander("🏁  Summary & Download"):
        st.success("🎉 Pipeline complete!")
        tsf_features = st.session_state.get("tsf_features")
        r1,r2,r3 = st.columns(3)
        r1.metric("Files Loaded",     f"{n_matched} / 5")
        r2.metric("Users in Dataset", master_df["Id"].nunique() if "Id" in master_df.columns else "—")
        r3.metric("Days of Data",     f"{master_df.shape[0]:,} rows")
        r4,r5,r6 = st.columns(3)
        r4.metric("HR Data Points",   f"{len(hr_resampled):,}" if hr_resampled is not None else "—")
        r5.metric("TSFresh Features", str(tsf_features.shape[1]) if tsf_features is not None else "Not run")
        r6.metric("KMeans Groups",    str(n_k))
        st.markdown("---")
        checks = [
            ("✅","Loaded and merged 5 Fitbit CSV files"),
            ("✅","Parsed timestamps, resampled heart rate to 1-minute"),
            ("✅","Built master daily dataset"),
            ("✅","Filled missing values automatically"),
            ("✅" if tsf_features is not None else "⏭️","TSFresh: heart rate feature extraction"),
            ("✅" if st.session_state.get("fc_hr") else "⏭️","Prophet: heart rate forecast"),
            ("✅" if steps_col and st.session_state.get(f"fc_{steps_col}") else "⏭️","Prophet: steps forecast"),
            ("✅" if sleep_col and st.session_state.get(f"fc_{sleep_col}") else "⏭️","Prophet: sleep forecast"),
            ("✅","KMeans clustering"), ("✅","DBSCAN clustering"),
            ("✅","PCA visualisation"),
            ("✅" if "tsne_df" in st.session_state else "⏭️","t-SNE visualisation"),
        ]
        for icon, text in checks:
            st.markdown(f"{icon} {text}")
        st.markdown("---")
        st.download_button("📥 Download Final Master Dataset (CSV)",
                           master_df.to_csv(index=False).encode(),
                           "fitness_m2_master.csv", "text/csv")