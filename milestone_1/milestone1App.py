import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(page_title="FitPulse – M1", page_icon="💪", layout="wide")

# ── Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0d1b2a; color: #dce8f5; }
    section[data-testid="stSidebar"] { background: #112236; border-right: 1px solid #1e3a5f; }
    section[data-testid="stSidebar"] * { color: #a8c8e8; }
    h1, h2, h3 { color: #a78bfa !important; }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white; font-weight: 600; border: none;
        border-radius: 8px; padding: 0.5rem 1rem; transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(168,85,247,0.4); }
    div[data-testid="metric-container"] {
        background: #162a41; border: 1px solid #1e4a6e;
        border-radius: 12px; padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .log-box {
        background: rgba(74,144,196,0.1); border-left: 4px solid #a78bfa;
        border-radius: 4px; padding: 10px 15px; margin: 5px 0;
        font-family: 'Source Code Pro', monospace;
    }
    .info-box {
        background: rgba(99,102,241,0.12); border-left: 4px solid #6366f1;
        border-radius: 6px; padding: 10px 14px; margin: 8px 0;
        font-size: 0.88rem; color: #a8c8e8;
    }
    .stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }
    .stSelectbox > div, .stSelectbox div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] * { cursor: pointer !important; }
    .stCheckbox, .stCheckbox * { cursor: pointer !important; }
    .stFileUploader, .stFileUploader * { cursor: pointer !important; }
    .stTabs [data-baseweb="tab"] { cursor: pointer !important; }
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

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80)
    st.markdown("## Fitness Data Pro")
    st.markdown("### Milestone 1")
    st.markdown("---")
    st.subheader("Pipeline")
    st.info("1. Upload CSV 📂\n2. Inspect Data 🔍\n3. Clean & Process ⚙️\n4. Export CSV ⬇️")
    st.markdown("---")
    st.caption("v1.0 | Data Collection & Preprocessing")

# ── Title ────────────────────────────────────────────────────────
st.title("📊 Data Collection & Preprocessing")
st.markdown("Clean, normalise, and visualise your fitness tracking data easily.")
st.markdown("---")

# ── File Upload ──────────────────────────────────────────────────
uploaded_file = st.file_uploader("Drop your fitness CSV here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None

    # ── Overview metrics ─────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Total Rows",    f"{len(df):,}")
    with m2: st.metric("Dimensions",    f"{df.shape[1]} Cols")
    with m3: st.metric("Missing Cells", int(df.isnull().sum().sum()))
    with m4:
        comp = 100 - (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
        st.metric("Completeness", f"{comp:.1f}%")

    # ── Tabs ─────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Inspection", "⚙️ Processing", "📈 Visualization"])

    # ════════════════════════════════════════
    #  TAB 1 — INSPECTION
    # ════════════════════════════════════════
    with tab1:
        st.subheader("Raw Dataset Preview")
        tip("This is your data exactly as it came in — before any cleaning. Look for missing values and check data types.")
        st.dataframe(df.head(10), use_container_width=True)

        col_nulls, col_types = st.columns(2)
        with col_nulls:
            st.markdown("**Missing Values per Column**")
            null_counts = df.isnull().sum()
            if null_counts[null_counts > 0].empty:
                st.success("✅ No missing values found!")
            else:
                fig_null = px.bar(
                    x=null_counts[null_counts > 0].index,
                    y=null_counts[null_counts > 0].values,
                    labels={"x": "Column", "y": "Missing Count"},
                    color_discrete_sequence=["#a78bfa"],
                )
                fig_null.update_xaxes(tickangle=-30)
                st.plotly_chart(T(fig_null, h=280), use_container_width=True)

        with col_types:
            st.markdown("**Data Types**")
            tip("Numbers = numeric. Object = text. Datetime = dates.")
            st.dataframe(df.dtypes.to_frame(name='Type').astype(str), use_container_width=True)

    # ════════════════════════════════════════
    #  TAB 2 — PROCESSING
    # ════════════════════════════════════════
    with tab2:
        st.subheader("Data Cleaning Engine")
        tip("Choose what to fix and click Execute. The app will clean your data automatically.")

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
                    df_p = df.copy()
                    logs = []

                    # 1. Date Handling
                    if handle_dates:
                        date_cols = [c for c in df_p.columns if 'date' in c.lower()]
                        for col in date_cols:
                            df_p[col] = pd.to_datetime(df_p[col], dayfirst=True, errors='coerce').ffill()
                            df_p[col] = df_p[col].dt.date
                            logs.append(f"Fixed timestamps in: `{col}`")

                    # 2. Numeric Interpolation
                    if handle_numeric:
                        num_cols = df_p.select_dtypes(include=[np.number]).columns.tolist()
                        df_p[num_cols] = df_p[num_cols].interpolate().ffill().bfill()
                        logs.append("Interpolated numeric gaps (Linear Method)")

                    # 3. Categorical
                    if handle_cat:
                        cat_cols = df_p.select_dtypes(include=['object']).columns
                        for col in cat_cols:
                            df_p[col] = df_p[col].fillna("Unknown")
                        logs.append("Filled missing categories with 'Unknown'")

                    st.session_state.processed_df = df_p
                    st.success("✅ Preprocessing Complete!")
                    for l in logs:
                        st.markdown(f'<div class="log-box">✅ {l}</div>', unsafe_allow_html=True)

            if st.session_state.processed_df is not None:
                st.markdown("---")
                st.subheader("Cleaned Result Preview")
                st.dataframe(st.session_state.processed_df.head(5), use_container_width=True)

                # Before vs After metrics
                b1, b2, b3 = st.columns(3)
                b1.metric("Rows",          f"{len(st.session_state.processed_df):,}")
                b2.metric("Missing Before", int(df.isnull().sum().sum()))
                b3.metric("Missing After",  int(st.session_state.processed_df.isnull().sum().sum()),
                           delta=f"-{int(df.isnull().sum().sum()) - int(st.session_state.processed_df.isnull().sum().sum())} fixed",
                           delta_color="inverse")

                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=convert_df(st.session_state.processed_df),
                    file_name='fitness_cleaned.csv',
                    mime='text/csv'
                )

    # ════════════════════════════════════════
    #  TAB 3 — VISUALIZATION / OUTLIER DETECTOR
    # ════════════════════════════════════════
    with tab3:
        if st.session_state.processed_df is not None:
            processed_df = st.session_state.processed_df
            st.subheader("🕵️ Outlier Detector")
            tip("Select any numeric column to check if it has unusual values. Outliers are values that are much higher or lower than the rest.")

            num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()

            if num_cols:
                col_to_plot = st.selectbox("Select a metric to check for unusual values:", num_cols)

                # ── Stats ─────────────────────────────────────
                col_data = processed_df[col_to_plot].dropna()
                Q1       = col_data.quantile(0.25)
                Q3       = col_data.quantile(0.75)
                IQR      = Q3 - Q1
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

                # ── Box Plot ───────────────────────────────────
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

                # ── Histogram ─────────────────────────────────
                with v2:
                    st.markdown(f"**📊 Histogram — {col_to_plot}**")
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
                    # Outlier zones
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
                    # Mean line
                    fig_hist.add_vline(x=col_data.mean(), line_dash="dash",
                                       line_color="#f59e0b", line_width=2)
                    fig_hist.add_annotation(
                        x=col_data.mean(), y=1, yref="paper",
                        text=f"Mean: {col_data.mean():,.1f}",
                        showarrow=False, font=dict(color="#f59e0b", size=11),
                        xanchor="left", yanchor="top",
                        bgcolor="rgba(13,27,42,0.7)", bordercolor="#f59e0b",
                    )
                    # Median line
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
                        title=dict(text=f"Frequency Distribution — {col_to_plot}",
                                   font=dict(color="#a78bfa", size=13)),
                        xaxis_title=col_to_plot,
                        yaxis_title="Number of Records",
                        bargap=0.03,
                    )
                    fig_hist.update_xaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
                    fig_hist.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
                    st.plotly_chart(fig_hist, use_container_width=True)

                # ── Outlier rows table ─────────────────────────
                if len(outliers) > 0:
                    st.markdown("---")
                    st.markdown(f"### 🚨 Outlier Rows — {len(outliers)} found")
                    tip(f"These are the actual rows where **{col_to_plot}** has an unusual value.")
                    outlier_rows = processed_df[
                        (processed_df[col_to_plot] < Q1 - 1.5*IQR) |
                        (processed_df[col_to_plot] > Q3 + 1.5*IQR)
                    ].copy()
                    st.dataframe(outlier_rows, use_container_width=True)
                    st.markdown(f"""
                    <div class="info-box">
                        ⚠️ <strong>Outlier range for {col_to_plot}:</strong><br>
                        Any value below <strong>{lower_bound:,.1f}</strong> or
                        above <strong>{upper_bound:,.1f}</strong> is an outlier.<br>
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
        </div>
    """, unsafe_allow_html=True)