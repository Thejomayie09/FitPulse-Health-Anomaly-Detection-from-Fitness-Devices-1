import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fitness Data Pro", page_icon="💪", layout="wide")

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp { background: #0d1b2a; color: #dce8f5; }
    section[data-testid="stSidebar"] { background: #112236; border-right: 1px solid #1e3a5f; }
    section[data-testid="stSidebar"] * { color: #a8c8e8; }
    h1, h2, h3 { color: #a78bfa !important; font-family: 'Inter', sans-serif; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white; font-weight: 600; border: none;
        border-radius: 8px; padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(168, 85, 247, 0.4); }

    /* Metric Cards (The Boxes) */
    div[data-testid="metric-container"] {
        background: #162a41;
        border: 1px solid #1e4a6e;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Logs */
    .log-box {
        background: rgba(74, 144, 196, 0.1);
        border-left: 4px solid #a78bfa;
        border-radius: 4px;
        padding: 10px 15px;
        margin: 5px 0;
        font-family: 'Source Code Pro', monospace;
    }
    
    .stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80)
    st.markdown("## Fitness Data Pro")
    st.markdown("---")
    st.subheader("Data Pipeline")
    st.info("1. Upload 📂\n2. Profile 🔍\n3. Clean ⚙️\n4. Export ⬇️")
    st.markdown("---")
    st.caption("v2.0 | Outlier & Preprocessing Tool")

# ---------------- MAIN CONTENT ----------------
st.title("📊 Data Collection & Preprocessing")
st.markdown("Clean, normalize, and visualize your fitness tracking data easily.")

uploaded_file = st.file_uploader("Drop your fitness CSV here", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Session State to hold processed data
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None

    # ---------------- OVERVIEW METRICS (The Boxes) ----------------
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Total Rows", f"{len(df):,}")
    with m2: st.metric("Dimensions", f"{df.shape[1]} Cols")
    with m3: st.metric("Missing Cells", int(df.isnull().sum().sum()))
    with m4: 
        comp = 100 - (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
        st.metric("Completeness", f"{comp:.1f}%")

    # ---------------- TABBED VIEW ----------------
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
            handle_dates = st.checkbox("Normalize Dates", value=True)
            handle_numeric = st.checkbox("Interpolate Numbers", value=True)
            handle_cat = st.checkbox("Fill Categories", value=True)
            run_btn = st.button("🚀 Execute Pipeline")

        with c2:
            if run_btn:
                with st.spinner("Refining your data..."):
                    df_p = df.copy()
                    logs = []

                    # 1. Date Handling
                    if handle_dates:
                        date_cols = [c for c in df_p.columns if 'date' in c.lower()]
                        for col in date_cols:
                            df_p[col] = pd.to_datetime(df_p[col], dayfirst=True, errors='coerce').ffill()
                            df_p[col] = df_p[col].dt.date  # This removes the "00:00:00" for a cleaner look
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
                        logs.append(f"Filled missing categories with 'Unknown'")

                    st.session_state.processed_df = df_p
                    st.success("Preprocessing Complete!")
                    for l in logs:
                        st.markdown(f'<div class="log-box">✅ {l}</div>', unsafe_allow_html=True)

        if st.session_state.processed_df is not None:
            st.markdown("---")
            st.subheader("Cleaned Result")
            st.dataframe(st.session_state.processed_df.head(5), use_container_width=True)
            csv_data = convert_df(st.session_state.processed_df)
            st.download_button(label="📥 Download Cleaned CSV", data=csv_data, file_name='fitness_cleaned.csv', mime='text/csv')

    with tab3:
        if st.session_state.processed_df is not None:
            processed_df = st.session_state.processed_df
            st.subheader("📈 Simple Data Insights")
            
            # # Row 1: Quality Check
            # before_nulls = df.isnull().sum().sum()
            # after_nulls = processed_df.isnull().sum().sum()
            # fig_compare = px.bar(x=["Original", "Cleaned"], y=[before_nulls, after_nulls],
            #                     title="Missing Data Reduction", labels={'x': 'Status', 'y': 'Null Count'},
            #                     color=["Before", "After"], color_discrete_map={"Before": "#ef4444", "After": "#10b981"})
            # st.plotly_chart(fig_compare, use_container_width=True)

            # st.markdown("---")

            # Row 2: Outliers (Inside vs Outside)
            st.subheader("🕵️ Outlier Detector")
            num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if num_cols:
                col_to_plot = st.selectbox("Select a metric to check for unusual values:", num_cols)
                v1, v2 = st.columns(2)
                
                with v1:
                    fig_box = px.box(processed_df, y=col_to_plot, title=f"Range View ({col_to_plot})", points="all", color_discrete_sequence=['#a78bfa'])
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    st.info("**Box Plot Guide:** The 'Box' is your normal data. Dots outside the whiskers are **Outliers** (unexpectedly high or low values).")

                with v2:
                    fig_hist = px.histogram(processed_df, x=col_to_plot, title="Frequency View", color_discrete_sequence=['#6366f1'])
                    st.plotly_chart(fig_hist, use_container_width=True)
                    st.info(f"This shows how often different values for **{col_to_plot}** appear. High bars = common values.")
            else:
                st.warning("No numeric columns found.")
        else:
            st.warning("⚠️ Run the **Processing** tab first to see visualizations!")

else:
    st.markdown("""
        <div style="text-align:center; padding: 100px 20px; border: 2px dashed #1e3a5f; border-radius: 20px;">
            <h2 style="color: #4a90c4 !important;">Awaiting Dataset...</h2>
            <p>Upload a CSV file to begin the fitness data transformation.</p>
        </div>
    """, unsafe_allow_html=True)