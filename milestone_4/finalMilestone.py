import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse – M4",
    page_icon="📊",
    layout="wide",
)

# ── Shared CSS  (exact M1/M2/M3 palette) ──────────────────────
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
    border-radius: 8px; padding: 0.5rem 1.4rem; transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(168,85,247,0.4); }
.stButton > button:disabled { opacity: 0.4; transform: none !important; }

div[data-testid="metric-container"] {
    background: #162a41; border: 1px solid #1e4a6e;
    border-radius: 12px; padding: 1rem;
}
.stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }

details { background: #111f30; border: 1px solid #1e3a5f !important; border-radius: 10px; margin-bottom: 10px; }
details summary {
    background: linear-gradient(90deg, #162a41, #1a1f4a);
    border-radius: 10px; padding: 14px 18px;
    color: #a78bfa; font-weight: 700; font-size: 1.05rem;
    cursor: pointer; list-style: none;
}
details[open] summary { border-bottom: 1px solid #1e3a5f; border-radius: 10px 10px 0 0; }

.info-box  { background:rgba(99,102,241,0.12); border-left:4px solid #6366f1; border-radius:6px; padding:10px 14px; margin:8px 0; font-size:0.88rem; color:#a8c8e8; }
.ok-box    { background:rgba(16,185,129,0.10); border-left:4px solid #10b981; border-radius:6px; padding:10px 14px; margin:8px 0; font-size:0.88rem; color:#6ee7b7; }
.warn-box  { background:rgba(245,158,11,0.10); border-left:4px solid #f59e0b; border-radius:6px; padding:10px 14px; margin:8px 0; font-size:0.88rem; color:#fde68a; }
.alert-box { background:rgba(239,68,68,0.12);  border-left:4px solid #ef4444; border-radius:6px; padding:10px 14px; margin:8px 0; font-size:0.88rem; color:#fca5a5; }

.kpi-card { background:#162a41; border:1px solid #1e4a6e; border-radius:14px; padding:1.2rem 0.8rem; text-align:center; }
.kpi-num  { font-size:2.4rem; font-weight:800; line-height:1; margin-bottom:0.25rem; }
.kpi-lbl  { font-size:0.68rem; color:#5a7a9a; text-transform:uppercase; letter-spacing:0.08em; }
.kpi-sub  { font-size:0.65rem; color:#4a6a8a; margin-top:0.2rem; }

.stat-card {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 16px 18px;
    font-size: 0.85rem; line-height: 2;
}
.stat-title {
    font-size: 0.7rem; font-weight: 700; color: #4a6a8a;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 10px;
}
.anom-row {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(30,58,95,0.5);
    font-size: 0.82rem;
}

.prog-wrap { background:#162a41; border-radius:99px; height:7px; margin:8px 0; overflow:hidden; }
.prog-fill { background:linear-gradient(90deg,#6366f1,#a78bfa); height:100%; border-radius:99px;
             animation:pbar 1.4s ease-in-out infinite; }
@keyframes pbar { 0%,100%{opacity:1;width:60%} 50%{opacity:.5;width:90%} }

.step-done { background:#162a41; border:1px solid #1e4a6e; border-radius:8px;
             padding:6px 12px; font-size:0.82rem; color:#a8c8e8; margin:3px 0; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,27,42,0.6)",
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
def ok(t):    st.markdown(f'<div class="ok-box">✅ {t}</div>',    unsafe_allow_html=True)
def warn(t):  st.markdown(f'<div class="warn-box">⚠️ {t}</div>',  unsafe_allow_html=True)
def alert(t): st.markdown(f'<div class="alert-box">🚨 {t}</div>', unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
for k, v in [
    ("m4_pipeline_done", False),
    ("m4_master",        None),
    ("m4_anom_hr",       None),
    ("m4_anom_steps",    None),
    ("m4_anom_sleep",    None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── File registry ──────────────────────────────────────────────
M4_FILES = {
    "dailyActivity_merged.csv":     {"key_cols":["ActivityDate","TotalSteps","Calories"],  "label":"Daily Activity",    "icon":"🏃"},
    "hourlySteps_merged.csv":       {"key_cols":["ActivityHour","StepTotal"],              "label":"Hourly Steps",      "icon":"👟"},
    "hourlyIntensities_merged.csv": {"key_cols":["ActivityHour","TotalIntensity"],         "label":"Hourly Intensities","icon":"⚡"},
    "minuteSleep_merged.csv":       {"key_cols":["date","value","logId"],                  "label":"Minute Sleep",      "icon":"😴"},
    "heartrate_seconds_merged.csv": {"key_cols":["Time","Value"],                          "label":"Heart Rate",        "icon":"💓"},
}
def score_match(df, info):
    return sum(1 for c in info["key_cols"] if c in df.columns)

# ══════════════════════════════════════════════════════════════
#  DETECTION FUNCTIONS
# ══════════════════════════════════════════════════════════════
def detect_hr(master, hr_high=100, hr_low=50, sigma=2.0):
    df = master[["Id","Date","AvgHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["AvgHR"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["AvgHR"].rolling(3, center=True, min_periods=1).median()
    d["residual"]    = d["AvgHR"] - d["rolling_med"]
    std = d["residual"].std() if d["residual"].std() > 0 else 1.0
    d["thresh_high"] = d["AvgHR"] > hr_high
    d["thresh_low"]  = d["AvgHR"] < hr_low
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

def detect_steps(master, st_low=500, st_high=25000, sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    d["residual"]    = d["TotalSteps"] - d["rolling_med"]
    std = d["residual"].std() if d["residual"].std() > 0 else 1.0
    d["thresh_low"]  = d["TotalSteps"] < st_low
    d["thresh_high"] = d["TotalSteps"] > st_high
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

def detect_sleep(master, sl_low=60, sl_high=600, sigma=2.0):
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

# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS — Timeline only (stats/records shown as tables)
# ══════════════════════════════════════════════════════════════
def chart_timeline(anom_hr_f, anom_steps_f, anom_sleep_f, h=300):
    all_anoms = []
    for df_, sig in [(anom_hr_f,"Heart Rate"),(anom_steps_f,"Steps"),(anom_sleep_f,"Sleep")]:
        a = df_[df_["is_anomaly"]].copy()
        a["signal"] = sig
        val_col = {"Heart Rate":"AvgHR","Steps":"TotalSteps","Sleep":"TotalSleepMinutes"}[sig]
        if val_col in a.columns:
            a["value"] = a[val_col]
        else:
            a["value"] = 0
        all_anoms.append(a[["Date","signal","reason","value"]])
    if not all_anoms:
        return None
    combined = pd.concat(all_anoms, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    color_map = {"Heart Rate":"#38bdf8","Steps":"#10b981","Sleep":"#a78bfa"}
    fig = go.Figure()
    for sig, col in color_map.items():
        sub = combined[combined["signal"]==sig]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["Date"], y=sub["signal"], mode="markers",
                name=sig, marker=dict(color=col, size=14, symbol="diamond",
                                      line=dict(color="white",width=2)),
                hovertemplate=f"<b>{sig}</b><br>📅 %{{x|%d %b %Y}}<br>%{{customdata}}<extra>⚠️ ANOMALY</extra>",
                customdata=sub["reason"].values))
    T(fig, h)
    fig.update_layout(
        title="📅 Combined Anomaly Timeline — All Signals",
        xaxis_title="Date", yaxis_title="Signal",
        yaxis=dict(categoryorder="array",
                   categoryarray=["Sleep","Steps","Heart Rate"],
                   gridcolor="#1e3a5f"),
        hovermode="closest",
    )
    fig.update_xaxes(tickformat="%d %b %Y", tickangle=-30)
    return fig

# ══════════════════════════════════════════════════════════════
#  PDF GENERATION
# ══════════════════════════════════════════════════════════════
def generate_pdf(master, anom_hr, anom_steps, anom_sleep,
                 hr_high, hr_low, st_low, sl_low, sl_high, sigma):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buf = io.BytesIO()
    C_BG     = colors.HexColor("#0d1b2a")
    C_CARD   = colors.HexColor("#162a41")
    C_ACCENT = colors.HexColor("#a78bfa")
    C_BLUE   = colors.HexColor("#38bdf8")
    C_GREEN  = colors.HexColor("#10b981")
    C_RED    = colors.HexColor("#ef4444")
    C_AMBER  = colors.HexColor("#f59e0b")
    C_TEXT   = colors.HexColor("#dce8f5")
    C_MUTED  = colors.HexColor("#5a7a9a")
    C_BORDER = colors.HexColor("#1e3a5f")
    C_ROW1   = colors.HexColor("#0d1b2a")
    C_ROW2   = colors.HexColor("#111f30")
    PAGE_W, PAGE_H = A4

    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=18*mm, bottomMargin=18*mm)

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    s_title = S("t", fontName="Helvetica-Bold", fontSize=18, textColor=C_ACCENT,
                spaceAfter=4, leading=22, alignment=TA_CENTER)
    s_sub   = S("s", fontName="Helvetica", fontSize=9, textColor=C_MUTED,
                spaceAfter=8, alignment=TA_CENTER)
    s_h1    = S("h1", fontName="Helvetica-Bold", fontSize=12, textColor=C_ACCENT,
                spaceBefore=10, spaceAfter=4)
    s_body  = S("b", fontName="Helvetica", fontSize=8.5, textColor=C_TEXT,
                leading=13, spaceAfter=4)
    s_small = S("sm", fontName="Helvetica", fontSize=7.5, textColor=C_MUTED, leading=11)

    def hr_line():
        return HRFlowable(width="100%", thickness=0.5, color=C_ACCENT,
                          spaceAfter=6, spaceBefore=6)

    def section_hdr(text, color=C_ACCENT):
        return [Spacer(1,4*mm),
                Paragraph(text, S("sh", fontName="Helvetica-Bold", fontSize=11,
                                  textColor=color, spaceBefore=0, spaceAfter=2)),
                hr_line()]

    def kv_table(rows):
        data = [[Paragraph(k, S("kk", fontName="Helvetica-Bold", fontSize=9, textColor=C_MUTED)),
                 Paragraph(str(v), S("kv", fontName="Helvetica-Bold", fontSize=9, textColor=C_TEXT))]
                for k,v in rows]
        t = Table(data, colWidths=[55*mm,110*mm])
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[C_ROW1,C_ROW2]),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("GRID",(0,0),(-1,-1),0.25,C_BORDER),
        ]))
        return t

    def anom_table(df, val_col, val_label):
        sub = df[df["is_anomaly"]][[
            "Date",val_col,"rolling_med","residual","reason"]].copy().round(2)
        if sub.empty:
            return Paragraph("No anomalies detected.", s_body)
        sub["Date"] = sub["Date"].astype(str).str[:10]
        header = ["Date", val_label, "Expected", "Deviation", "Reason"]
        col_w  = [25*mm, 22*mm, 22*mm, 22*mm, 89*mm]
        td = [header] + [
            [str(r["Date"]), f"{r[val_col]:.1f}",
             f"{r['rolling_med']:.1f}", f"{r['residual']:.1f}", str(r["reason"])[:55]]
            for _,r in sub.head(25).iterrows()
        ]
        t = Table(td, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), C_BORDER),
            ("TEXTCOLOR",(0,0),(-1,0), C_ACCENT),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),7.5),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_ROW1,C_ROW2]),
            ("TEXTCOLOR",(0,1),(-1,-1), C_TEXT),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("GRID",(0,0),(-1,-1),0.3,C_BORDER),
            ("ALIGN",(1,0),(3,-1),"CENTER"),
        ]))
        elems = [t]
        if len(sub) > 25:
            elems.append(Paragraph(f"... and {len(sub)-25} more records.", s_small))
        return elems

    n_hr    = int(anom_hr["is_anomaly"].sum())
    n_steps = int(anom_steps["is_anomaly"].sum())
    n_sleep = int(anom_sleep["is_anomaly"].sum())
    n_total = n_hr + n_steps + n_sleep
    n_users = master["Id"].nunique()
    n_days  = master["Date"].nunique()
    d_min   = pd.to_datetime(master["Date"]).min().strftime("%d %b %Y")
    d_max   = pd.to_datetime(master["Date"]).max().strftime("%d %b %Y")

    kpi_data = [
        ["Metric","Count","% of Days"],
        ["Heart Rate Anomalies", str(n_hr),  f"{n_hr/max(len(anom_hr),1)*100:.1f}%"],
        ["Steps Anomalies",      str(n_steps),f"{n_steps/max(len(anom_steps),1)*100:.1f}%"],
        ["Sleep Anomalies",      str(n_sleep),f"{n_sleep/max(len(anom_sleep),1)*100:.1f}%"],
        ["TOTAL FLAGS",          str(n_total),"—"],
    ]
    kpi_t = Table(kpi_data, colWidths=[70*mm,55*mm,55*mm])
    kpi_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),C_BORDER),
        ("TEXTCOLOR",(0,0),(-1,0),C_ACCENT),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_ROW1,C_ROW2]),
        ("TEXTCOLOR",(0,1),(-1,-1),C_TEXT),
        ("FONTNAME",(0,4),(-1,4),"Helvetica-Bold"),
        ("TEXTCOLOR",(1,4),(-1,4),C_RED),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.3,C_BORDER),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
    ]))

    profile_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
    avail_p = [c for c in profile_cols if c in master.columns]
    user_p  = master.groupby("Id")[avail_p].mean().round(0).reset_index()
    user_p["Id"] = user_p["Id"].astype(str).str[-6:]
    ph = ["User (last 6)"] + [c[:12] for c in avail_p]
    pw = [180*mm/(len(avail_p)+1)] * (len(avail_p)+1)
    pd_data = [ph] + [[row["Id"]]+[f"{row[c]:,.0f}" for c in avail_p] for _,row in user_p.iterrows()]
    prof_t = Table(pd_data, colWidths=pw, repeatRows=1)
    prof_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),C_BORDER),
        ("TEXTCOLOR",(0,0),(-1,0),C_ACCENT),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_ROW1,C_ROW2]),
        ("TEXTCOLOR",(0,1),(-1,-1),C_TEXT),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("GRID",(0,0),(-1,-1),0.3,C_BORDER),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
    ]))

    story = []
    story.append(Spacer(1,8*mm))
    story.append(Paragraph("FitPulse — Anomaly Detection Report", s_title))
    story.append(Paragraph("Milestone 4  ·  Insights & Export Dashboard", s_sub))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}", s_sub))
    story.append(hr_line())
    story.append(Spacer(1,4*mm))

    story += section_hdr("1.  DATASET OVERVIEW", C_ACCENT)
    story.append(kv_table([
        ("Dataset Source",   "Real Fitbit Device Data — Kaggle (arashnic/fitbit)"),
        ("Total Users",      f"{n_users} participants"),
        ("Date Range",       f"{d_min}  →  {d_max}"),
        ("Total Days",       f"{n_days} observation days"),
        ("Pipeline",         "Milestone 4 — Anomaly Detection Dashboard"),
    ]))
    story.append(Spacer(1,5*mm))

    story += section_hdr("2.  ANOMALY SUMMARY", C_RED)
    story.append(kpi_t)
    story.append(Spacer(1,5*mm))

    story += section_hdr("3.  DETECTION THRESHOLDS USED", C_GREEN)
    story.append(kv_table([
        ("Heart Rate High",  f"> {hr_high} bpm"),
        ("Heart Rate Low",   f"< {hr_low} bpm"),
        ("Steps Low Alert",  f"< {st_low:,} steps/day"),
        ("Sleep Low",        f"< {sl_low} minutes/night"),
        ("Sleep High",       f"> {sl_high} minutes/night"),
        ("Residual Sigma",   f"+/- {sigma:.1f}sigma from 3-day rolling median"),
    ]))
    story.append(Spacer(1,5*mm))

    story += section_hdr("4.  DETECTION METHODOLOGY", C_BLUE)
    story.append(Paragraph(
        "<b>Three complementary methods were applied:</b><br/><br/>"
        "<b>① Threshold Violations</b> — Hard upper/lower bounds on each metric. Flags absolute out-of-range values immediately.<br/><br/>"
        "<b>② Residual-Based Detection</b> — A 3-day rolling median as the expected baseline. "
        f"Days deviating by more than +/- {sigma:.1f}sigma are flagged.<br/><br/>"
        "<b>③ DBSCAN Structural Outliers</b> — Users profiled on activity features and clustered. "
        "Users labelled -1 are structural outliers.",
        s_body))
    story.append(Spacer(1,5*mm))

    story += section_hdr("5.  HEART RATE ANOMALY RECORDS", C_RED)
    hr_e = anom_table(anom_hr,"AvgHR","Avg HR (bpm)")
    story += hr_e if isinstance(hr_e,list) else [hr_e]
    story.append(Spacer(1,5*mm))

    story += section_hdr("6.  STEP COUNT ANOMALY RECORDS", C_GREEN)
    st_e = anom_table(anom_steps,"TotalSteps","Steps")
    story += st_e if isinstance(st_e,list) else [st_e]
    story.append(Spacer(1,5*mm))

    story += section_hdr("7.  SLEEP ANOMALY RECORDS", C_ACCENT)
    sl_e = anom_table(anom_sleep,"TotalSleepMinutes","Sleep (min)")
    story += sl_e if isinstance(sl_e,list) else [sl_e]
    story.append(Spacer(1,5*mm))

    story += section_hdr("8.  USER ACTIVITY PROFILES", C_BLUE)
    story.append(Paragraph("Average daily metrics per user across the observation period:", s_body))
    story.append(prof_t)
    story.append(Spacer(1,5*mm))

    story += section_hdr("9.  CONCLUSION", C_GREEN)
    story.append(Paragraph(
        f"FitPulse Milestone 4 successfully processed <b>{n_users} users</b> over <b>{n_days} days</b> "
        f"of real Fitbit data. A total of <b>{n_total} anomalous events</b> were identified — "
        f"{n_hr} HR, {n_steps} steps, and {n_sleep} sleep anomalies.",
        s_body))

    def page_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#0a0e1a"))
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=True, stroke=False)
        canvas.setFillColor(colors.HexColor("#0d1b2a"))
        canvas.rect(0, PAGE_H-12*mm, PAGE_W, 12*mm, fill=True, stroke=False)
        canvas.setFillColor(colors.HexColor("#a78bfa"))
        canvas.rect(0, PAGE_H-12*mm, PAGE_W, 0.8, fill=True, stroke=False)
        canvas.setFillColor(colors.HexColor("#5a7a9a"))
        canvas.setFont("Helvetica", 7)
        canvas.drawCentredString(PAGE_W/2, 8*mm,
            f"FitPulse ML Pipeline  ·  Anomaly Detection Report  ·  Page {doc.page}")
        canvas.setFillColor(colors.HexColor("#1e3a5f"))
        canvas.rect(0, 6*mm, PAGE_W, 0.5, fill=True, stroke=False)
        canvas.restoreState()

    doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════
def generate_csv(anom_hr, anom_steps, anom_sleep):
    parts = []
    for df_, sig, vc in [
        (anom_hr,    "Heart Rate","AvgHR"),
        (anom_steps, "Steps",     "TotalSteps"),
        (anom_sleep, "Sleep",     "TotalSleepMinutes"),
    ]:
        sub = df_[df_["is_anomaly"]][["Date",vc,"rolling_med","residual","reason"]].copy()
        sub["signal"] = sig
        sub = sub.rename(columns={vc:"value","rolling_med":"expected"})
        parts.append(sub)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined[["signal","Date","value","expected","residual","reason"]]\
               .sort_values(["signal","Date"]).round(2)
    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=60)
    st.markdown("## Fitness Data Pro\n**Milestone 4**")
    st.markdown("---")

    pct = 100 if st.session_state.m4_pipeline_done else 0
    st.markdown(f"""
    <div style="margin-bottom:14px">
        <div style="font-size:0.72rem;color:#5a7a9a;margin-bottom:4px">PIPELINE · {pct}% COMPLETE</div>
        <div class="prog-wrap">
            <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#6366f1,#a78bfa);border-radius:99px;"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#5a7a9a;letter-spacing:0.1em">DETECTION THRESHOLDS</div>',
                unsafe_allow_html=True)
    hr_high = int(st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="m4_hr_high"))
    hr_low  = int(st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="m4_hr_low"))
    st_low  = int(st.number_input("Steps Low/day",    value=500, min_value=0,   max_value=2000,key="m4_st_low"))
    sl_low  = int(st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120, key="m4_sl_low"))
    sl_high = int(st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900, key="m4_sl_high"))
    sigma   = float(st.slider("Residual sigma", 1.0, 4.0, 2.0, 0.5, key="m4_sigma"))

    st.markdown("---")

    # Date + User filter — only after pipeline runs
    date_range = None
    sel_user   = None
    if st.session_state.m4_pipeline_done and st.session_state.m4_master is not None:
        _m = st.session_state.m4_master
        all_dates  = pd.to_datetime(_m["Date"])
        d_min_g    = all_dates.min().date()
        d_max_g    = all_dates.max().date()
        st.markdown('<div style="font-size:0.72rem;color:#5a7a9a;letter-spacing:0.1em">DATE FILTER</div>',
                    unsafe_allow_html=True)
        date_range = st.date_input("Date range", value=(d_min_g,d_max_g),
                                   min_value=d_min_g, max_value=d_max_g,
                                   key="m4_daterange", label_visibility="collapsed")
        st.markdown('<div style="font-size:0.72rem;color:#5a7a9a;letter-spacing:0.1em;margin-top:8px">USER FILTER</div>',
                    unsafe_allow_html=True)
        all_users    = sorted(_m["Id"].unique())
        user_options = ["All Users"] + [f"...{str(u)[-6:]}" for u in all_users]
        sel_lbl      = st.selectbox("User", user_options, key="m4_user", label_visibility="collapsed")
        sel_user     = None if sel_lbl=="All Users" else all_users[user_options.index(sel_lbl)-1]

    st.markdown("---")
    st.caption("v4.0 | Dashboard & Export")

# ══════════════════════════════════════════════════════════════
#  MAIN PAGE TITLE
# ══════════════════════════════════════════════════════════════
st.title("📊 FitPulse — Insights Dashboard")
st.markdown("Upload your 5 Fitbit CSV files, run detection, then filter by date & user and export your report.")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════
with st.expander("📂  Step 1 — Upload Your 5 Fitbit CSV Files", expanded=True):
    tip("Upload all 5 CSV files at once — hold <b>Ctrl</b> (Windows) or <b>Cmd</b> (Mac) while clicking.")

    m4_uploaded = st.file_uploader(
        "Upload all 5 CSV files",
        type=["csv"], accept_multiple_files=True,
        key="m4_uploader", label_visibility="collapsed",
    )

    m4_detected = {}
    m4_raw = []
    if m4_uploaded:
        for uf in m4_uploaded:
            try:
                uf.seek(0)
                m4_raw.append((uf.name, pd.read_csv(uf)))
            except Exception:
                pass
        for req_name, finfo in M4_FILES.items():
            best_s, best_d = 0, None
            for uname, udf in m4_raw:
                s = score_match(udf, finfo)
                if s > best_s:
                    best_s, best_d = s, udf
            if best_s >= 2:
                m4_detected[req_name] = best_d

    n_up = len(m4_detected)

    # File status grid
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    g = st.columns(5)
    for col_ui, (req_name, finfo) in zip(g, M4_FILES.items()):
        found = req_name in m4_detected
        bor   = "#10b981" if found else "#1e4a6e"
        tc    = "#10b981" if found else "#6b7280"
        col_ui.markdown(f"""
        <div style="background:#162a41;border:1px solid {bor};border-radius:12px;
                    padding:14px 8px;text-align:center;min-height:100px">
            <div style="font-size:1.8rem">{finfo['icon']}</div>
            <div style="font-size:0.78rem;font-weight:700;color:{tc};margin-top:6px">{finfo['label']}</div>
            <div style="font-size:0.62rem;color:#5a7a9a;margin-top:5px">
                {'✅ Detected' if found else '❌ Not found'}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    if n_up == 0:
        st.info("👆 Upload all 5 CSV files above to enable the pipeline.")
    elif n_up < 5:
        warn(f"{n_up}/5 files detected. Please also upload the remaining files.")
    else:
        ok(f"All 5 files detected and ready.")

    # Run button
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    run_disabled = (n_up < 5)
    run_clicked  = st.button("⚡  Load & Run Full Detection Pipeline",
                             disabled=run_disabled, key="m4_run")

    # Pipeline execution
    if run_clicked and n_up == 5:
        prog_ph = st.empty()

        steps_progress = [
            ("📂 Loading CSV files...",             15),
            ("🔗 Merging daily activity data...",   30),
            ("💓 Processing heart rate data...",    50),
            ("😴 Processing sleep data...",         65),
            ("🔗 Building master dataset...",       80),
            ("🚨 Running anomaly detection...",     95),
            ("✅ Complete!",                       100),
        ]

        for step_txt, pct_val in steps_progress:
            prog_ph.markdown(f"""
            <div style="background:linear-gradient(90deg,#1a2744,#1a1f4a);
                        border:1px solid #a78bfa;border-radius:12px;
                        padding:20px 24px;margin:12px 0">
                <div style="font-size:1.0rem;color:#a78bfa;font-weight:700;margin-bottom:10px">
                    ⏳ Running Pipeline...
                </div>
                <div style="font-size:0.85rem;color:#dce8f5;margin-bottom:10px">{step_txt}</div>
                <div class="prog-wrap">
                    <div style="width:{pct_val}%;height:100%;
                                background:linear-gradient(90deg,#6366f1,#a78bfa);
                                border-radius:99px;transition:width 0.3s"></div>
                </div>
                <div style="font-size:0.72rem;color:#4a6a8a;margin-top:6px">{pct_val}% complete</div>
            </div>""", unsafe_allow_html=True)

            if pct_val == 100:
                break

        try:
            daily    = m4_detected["dailyActivity_merged.csv"].copy()
            hourly_s = m4_detected.get("hourlySteps_merged.csv", pd.DataFrame()).copy()
            sleep    = m4_detected.get("minuteSleep_merged.csv", pd.DataFrame()).copy()
            hr       = m4_detected.get("heartrate_seconds_merged.csv", pd.DataFrame()).copy()

            def safe_dt(s, fmt):
                try:    return pd.to_datetime(s, format=fmt)
                except: return pd.to_datetime(s, infer_datetime_format=True, errors="coerce")

            daily["ActivityDate"] = safe_dt(daily["ActivityDate"], "%m/%d/%Y")

            # Heart rate processing
            hr_daily = pd.DataFrame()
            if not hr.empty and "Time" in hr.columns and "Value" in hr.columns:
                hr["Time"] = safe_dt(hr["Time"], "%m/%d/%Y %I:%M:%S %p")
                hr_min = (hr.set_index("Time").groupby("Id")["Value"]
                          .resample("1min").mean().reset_index())
                hr_min.columns = ["Id","Time","HeartRate"]
                hr_min = hr_min.dropna()
                hr_min["Date"] = hr_min["Time"].dt.date
                hr_daily = (hr_min.groupby(["Id","Date"])["HeartRate"]
                            .agg(["mean","max","min","std"]).reset_index()
                            .rename(columns={"mean":"AvgHR","max":"MaxHR",
                                             "min":"MinHR","std":"StdHR"}))

            # Sleep processing
            sleep_daily = pd.DataFrame()
            if not sleep.empty and "date" in sleep.columns and "value" in sleep.columns:
                sleep["date"] = safe_dt(sleep["date"], "%m/%d/%Y %I:%M:%S %p")
                sleep["Date"] = sleep["date"].dt.date
                sleep_daily   = (sleep.groupby(["Id","Date"])
                                 .agg(TotalSleepMinutes=("value","count"))
                                 .reset_index())

            # Build master
            master = daily.copy().rename(columns={"ActivityDate":"Date"})
            master["Date"] = master["Date"].dt.date
            if not hr_daily.empty:
                master = master.merge(hr_daily, on=["Id","Date"], how="left")
                for c in ["AvgHR","MaxHR","MinHR","StdHR"]:
                    master[c] = master.groupby("Id")[c].transform(
                        lambda x: x.fillna(x.median()))
            if not sleep_daily.empty:
                master = master.merge(sleep_daily, on=["Id","Date"], how="left")
                master["TotalSleepMinutes"] = master["TotalSleepMinutes"].fillna(0)

            anom_hr_r    = detect_hr(master,    hr_high, hr_low,  sigma)
            anom_steps_r = detect_steps(master, st_low,  25000,   sigma)
            anom_sleep_r = detect_sleep(master, sl_low,  sl_high, sigma)

            st.session_state.m4_master     = master
            st.session_state.m4_anom_hr    = anom_hr_r
            st.session_state.m4_anom_steps = anom_steps_r
            st.session_state.m4_anom_sleep = anom_sleep_r
            st.session_state.m4_pipeline_done = True
            prog_ph.empty()
            st.rerun()

        except Exception as e:
            prog_ph.empty()
            st.error(f"Pipeline error: {e}")
            st.exception(e)

    # Show success summary after pipeline
    if st.session_state.m4_pipeline_done:
        _m = st.session_state.m4_master
        ok(f"Pipeline complete — **{_m.shape[0]:,} rows** · **{_m['Id'].nunique()} users** · **{_m.shape[1]} columns**")
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Total Rows",  f"{_m.shape[0]:,}")
        mm2.metric("Users",       _m["Id"].nunique())
        mm3.metric("Date Range",  f"{pd.to_datetime(_m['Date']).min().strftime('%d %b')} → "
                                  f"{pd.to_datetime(_m['Date']).max().strftime('%d %b %y')}")
        mm4.metric("Columns",     _m.shape[1])

# ── Block rest of page until pipeline is done ─────────────────
if not st.session_state.m4_pipeline_done:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;
                border:2px dashed #1e3a5f;border-radius:20px;margin-top:1rem">
        <div style="font-size:2.5rem;margin-bottom:16px">📂</div>
        <h2 style="color:#4a90c4 !important;">Awaiting Data...</h2>
        <p style="color:#5a7a9a;font-size:0.95rem">
            Upload your 5 CSV files above, then click<br>
            <b style="color:#a78bfa">⚡ Load & Run Full Detection Pipeline</b>
        </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load filtered data ─────────────────────────────────────────
master     = st.session_state.m4_master
anom_hr    = st.session_state.m4_anom_hr
anom_steps = st.session_state.m4_anom_steps
anom_sleep = st.session_state.m4_anom_sleep

try:
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        d_from = pd.Timestamp(date_range[0])
        d_to   = pd.Timestamp(date_range[1])
    else:
        all_d  = pd.to_datetime(master["Date"])
        d_from, d_to = all_d.min(), all_d.max()
except Exception:
    all_d  = pd.to_datetime(master["Date"])
    d_from, d_to = all_d.min(), all_d.max()

def filt(df, dc="Date"):
    df2 = df.copy()
    df2[dc] = pd.to_datetime(df2[dc])
    return df2[(df2[dc] >= d_from) & (df2[dc] <= d_to)]

anom_hr_f    = filt(anom_hr)
anom_steps_f = filt(anom_steps)
anom_sleep_f = filt(anom_sleep)
master_f     = filt(master)
if sel_user:
    master_f = master_f[master_f["Id"] == sel_user]

# ══════════════════════════════════════════════════════════════
#  KPI STRIP
# ══════════════════════════════════════════════════════════════
n_hr_f    = int(anom_hr_f["is_anomaly"].sum())
n_steps_f = int(anom_steps_f["is_anomaly"].sum())
n_sleep_f = int(anom_sleep_f["is_anomaly"].sum())
n_total_f = n_hr_f + n_steps_f + n_sleep_f
n_users_f = master_f["Id"].nunique()
n_days_f  = master_f["Date"].nunique()

worst_hr_row = anom_hr_f[anom_hr_f["is_anomaly"]]
if not worst_hr_row.empty:
    worst_hr_day = worst_hr_row.iloc[worst_hr_row["residual"].abs().values.argmax()]["Date"]
    worst_hr_str = pd.Timestamp(worst_hr_day).strftime("%d %b")
else:
    worst_hr_str = "—"

st.markdown("---")
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpi_items = [
    (k1, str(n_total_f), "TOTAL FLAGS",      "across all signals", "#ef4444",  "rgba(239,68,68,0.25)"),
    (k2, str(n_hr_f),    "HR FLAGS",          "heart rate anomalies","#f472b6", "rgba(244,114,182,0.2)"),
    (k3, str(n_steps_f), "STEPS ALERTS",      "step count anomalies","#10b981", "rgba(16,185,129,0.2)"),
    (k4, str(n_sleep_f), "SLEEP FLAGS",       "sleep anomalies",    "#a78bfa",  "rgba(167,139,250,0.2)"),
    (k5, str(n_users_f), "USERS",             "in selected range",  "#38bdf8",  "rgba(56,189,248,0.2)"),
    (k6, worst_hr_str,   "PEAK HR ANOMALY",   "highest deviation day","#f59e0b","rgba(245,158,11,0.2)"),
]
for col, num, lbl, sub, color, bg in kpi_items:
    col.markdown(f"""
    <div class="kpi-card" style="border-color:{color.replace('1)','0.35)')};">
        <div class="kpi-num" style="color:{color}">{num}</div>
        <div class="kpi-lbl">{lbl}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
ok(f"Pipeline complete · {n_users_f} users · {n_days_f} days · {n_total_f} total anomalies flagged")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab_overview, tab_hr, tab_steps, tab_sleep, tab_export = st.tabs([
    "📊 Overview", "💓 Heart Rate", "👟 Steps", "😴 Sleep", "📥 Export Report"
])

# ──────────────────────────────────────────────────────────────
#  OVERVIEW TAB
# ──────────────────────────────────────────────────────────────
with tab_overview:
    with st.expander("📅  Combined Anomaly Timeline", expanded=True):
        tip("Each diamond = one anomaly event. Hover to see the date and reason. Filter by date and user in the sidebar.")
        fig_tl = chart_timeline(anom_hr_f, anom_steps_f, anom_sleep_f)
        if fig_tl:
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            ok("No anomalies in the selected date range.")

    with st.expander("🗂️  Recent Anomaly Log", expanded=True):
        tip("Last 15 anomaly events across all signals, sorted newest first.")
        all_log = []
        for df_, sig, clr in [
            (anom_hr_f,    "Heart Rate","#38bdf8"),
            (anom_steps_f, "Steps",     "#10b981"),
            (anom_sleep_f, "Sleep",     "#a78bfa"),
        ]:
            a = df_[df_["is_anomaly"]].copy()
            a["signal"] = sig; a["color"] = clr
            all_log.append(a[["Date","signal","color","reason"]])
        if all_log:
            log_df = pd.concat(all_log, ignore_index=True)
            log_df["Date"] = pd.to_datetime(log_df["Date"])
            log_df = log_df.sort_values("Date", ascending=False).head(15)
            # Header
            st.markdown("""
            <div style="display:flex;gap:14px;padding:7px 12px;
                        background:#0d1b2a;border-radius:6px 6px 0 0;
                        border:1px solid #1e3a5f;border-bottom:2px solid #1e3a5f;
                        font-size:0.72rem;font-weight:700;color:#4a6a8a;
                        text-transform:uppercase;letter-spacing:0.07em;">
                <span style="width:24px"></span>
                <span style="min-width:95px">Signal</span>
                <span style="min-width:100px">Date</span>
                <span>Reason</span>
            </div>""", unsafe_allow_html=True)
            for _, row in log_df.iterrows():
                st.markdown(f"""
                <div class="anom-row" style="padding-left:12px;border:1px solid #1a2f44;
                     border-top:none;background:#111f30;">
                    <span>🚨</span>
                    <span style="color:{row['color']};font-weight:700;font-size:0.82rem;min-width:95px">{row['signal']}</span>
                    <span style="color:#a8c8e8;font-size:0.8rem;min-width:100px">{row['Date'].strftime('%d %b %Y')}</span>
                    <span style="color:#64748b;font-size:0.78rem;font-style:italic">{row['reason']}</span>
                </div>""", unsafe_allow_html=True)
        else:
            ok("No anomalies detected in selected range.")

# ──────────────────────────────────────────────────────────────
#  HEART RATE TAB  — Statistics + Records (no chart)
# ──────────────────────────────────────────────────────────────
with tab_hr:
    badge_hr = f'<span style="margin-left:auto;background:rgba(239,68,68,0.15);border:1px solid #ef4444;border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#ef4444;font-weight:700">{n_hr_f} anomalies</span>'
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1rem 0 0.8rem">
        <div style="background:rgba(239,68,68,0.15);border:1px solid #ef4444;border-radius:8px;
                    width:38px;height:38px;display:flex;align-items:center;justify-content:center;font-size:1.3rem">💓</div>
        <span style="font-family:Inter,sans-serif;font-size:1.15rem;font-weight:700;color:#dce8f5">Heart Rate — Deep Dive</span>
        {badge_hr}
    </div>""", unsafe_allow_html=True)

    col_stat, col_rec = st.columns(2)

    with col_stat:
        st.markdown('<div class="stat-title">HR STATISTICS</div>', unsafe_allow_html=True)
        mean_hr = anom_hr_f["AvgHR"].mean()
        max_hr  = anom_hr_f["AvgHR"].max()
        min_hr  = anom_hr_f["AvgHR"].min()
        st.markdown(f"""
        <div class="stat-card">
            <div>Mean HR: <b style="color:#38bdf8">{mean_hr:.1f} bpm</b></div>
            <div>Max HR:  <b style="color:#ef4444">{max_hr:.1f} bpm</b></div>
            <div>Min HR:  <b style="color:#f9a8d4">{min_hr:.1f} bpm</b></div>
            <div>Anomaly days: <b style="color:#ef4444">{n_hr_f}</b> of {len(anom_hr_f)} total</div>
            <div>Anomaly rate: <b style="color:#ef4444">{n_hr_f/max(len(anom_hr_f),1)*100:.1f}%</b></div>
            <div>Normal range: <b style="color:#a8c8e8">{hr_low} – {hr_high} bpm</b></div>
        </div>""", unsafe_allow_html=True)
        if n_hr_f > 0:
            st.markdown("")
            alert(f"{n_hr_f} HR anomaly days detected")
        else:
            st.markdown("")
            ok("No HR anomalies in selected range")

    with col_rec:
        st.markdown('<div class="stat-title">HR ANOMALY RECORDS</div>', unsafe_allow_html=True)
        hr_disp = anom_hr_f[anom_hr_f["is_anomaly"]][
            ["Date","AvgHR","rolling_med","residual","reason"]].round(2)
        if not hr_disp.empty:
            st.dataframe(hr_disp.rename(columns={
                "AvgHR":"Avg HR","rolling_med":"Expected",
                "residual":"Deviation","reason":"Reason"}),
                use_container_width=True, height=260)
        else:
            ok("No HR anomalies in selected range.")

# ──────────────────────────────────────────────────────────────
#  STEPS TAB  — Statistics + Records (no chart)
# ──────────────────────────────────────────────────────────────
with tab_steps:
    badge_st = f'<span style="margin-left:auto;background:rgba(16,185,129,0.15);border:1px solid #10b981;border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#10b981;font-weight:700">{n_steps_f} alerts</span>'
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1rem 0 0.8rem">
        <div style="background:rgba(16,185,129,0.15);border:1px solid #10b981;border-radius:8px;
                    width:38px;height:38px;display:flex;align-items:center;justify-content:center;font-size:1.3rem">👟</div>
        <span style="font-family:Inter,sans-serif;font-size:1.15rem;font-weight:700;color:#dce8f5">Steps — Deep Dive</span>
        {badge_st}
    </div>""", unsafe_allow_html=True)

    col_stat2, col_rec2 = st.columns(2)

    with col_stat2:
        st.markdown('<div class="stat-title">STEPS STATISTICS</div>', unsafe_allow_html=True)
        mean_st = anom_steps_f["TotalSteps"].mean()
        max_st  = anom_steps_f["TotalSteps"].max()
        min_st  = anom_steps_f["TotalSteps"].min()
        days_lt = int((anom_steps_f["TotalSteps"] < 500).sum())
        st.markdown(f"""
        <div class="stat-card">
            <div>Mean steps/day: <b style="color:#10b981">{mean_st:,.0f}</b></div>
            <div>Max steps/day:  <b style="color:#38bdf8">{max_st:,.0f}</b></div>
            <div>Min steps/day:  <b style="color:#ef4444">{min_st:,.0f}</b></div>
            <div>Anomaly days:   <b style="color:#ef4444">{n_steps_f}</b> of {len(anom_steps_f)} total</div>
            <div>Anomaly rate:   <b style="color:#ef4444">{n_steps_f/max(len(anom_steps_f),1)*100:.1f}%</b></div>
            <div>Days &lt; 500 steps: <b style="color:#ef4444">{days_lt}</b></div>
            <div>Alert range: <b style="color:#a8c8e8">{st_low:,} – 25,000 steps/day</b></div>
        </div>""", unsafe_allow_html=True)
        if n_steps_f > 0:
            st.markdown("")
            alert(f"{n_steps_f} step alert days detected")
        else:
            st.markdown("")
            ok("No step anomalies in selected range")

    with col_rec2:
        st.markdown('<div class="stat-title">STEPS ANOMALY RECORDS</div>', unsafe_allow_html=True)
        st_disp = anom_steps_f[anom_steps_f["is_anomaly"]][
            ["Date","TotalSteps","rolling_med","residual","reason"]].round(2)
        if not st_disp.empty:
            st.dataframe(st_disp.rename(columns={
                "TotalSteps":"Steps","rolling_med":"Expected",
                "residual":"Deviation","reason":"Reason"}),
                use_container_width=True, height=260)
        else:
            ok("No step anomalies in selected range.")

# ──────────────────────────────────────────────────────────────
#  SLEEP TAB  — Statistics + Records (no chart)
# ──────────────────────────────────────────────────────────────
with tab_sleep:
    badge_sl = f'<span style="margin-left:auto;background:rgba(167,139,250,0.15);border:1px solid #a78bfa;border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#a78bfa;font-weight:700">{n_sleep_f} anomalies</span>'
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1rem 0 0.8rem">
        <div style="background:rgba(167,139,250,0.15);border:1px solid #a78bfa;border-radius:8px;
                    width:38px;height:38px;display:flex;align-items:center;justify-content:center;font-size:1.3rem">😴</div>
        <span style="font-family:Inter,sans-serif;font-size:1.15rem;font-weight:700;color:#dce8f5">Sleep — Deep Dive</span>
        {badge_sl}
    </div>""", unsafe_allow_html=True)

    col_stat3, col_rec3 = st.columns(2)

    with col_stat3:
        st.markdown('<div class="stat-title">SLEEP STATISTICS</div>', unsafe_allow_html=True)
        mean_sl = anom_sleep_f["TotalSleepMinutes"].mean()
        max_sl  = anom_sleep_f["TotalSleepMinutes"].max()
        nonzero = anom_sleep_f[anom_sleep_f["TotalSleepMinutes"] > 0]["TotalSleepMinutes"]
        min_sl  = nonzero.min() if not nonzero.empty else 0
        days_lt_sl = int((anom_sleep_f["TotalSleepMinutes"] < sl_low).sum())
        st.markdown(f"""
        <div class="stat-card">
            <div>Mean sleep/night: <b style="color:#a78bfa">{mean_sl:.0f} min</b></div>
            <div>Max sleep/night:  <b style="color:#38bdf8">{max_sl:.0f} min</b></div>
            <div>Min (non-zero):   <b style="color:#ef4444">{min_sl:.0f} min</b></div>
            <div>Anomaly days:     <b style="color:#ef4444">{n_sleep_f}</b> of {len(anom_sleep_f)} total</div>
            <div>Anomaly rate:     <b style="color:#ef4444">{n_sleep_f/max(len(anom_sleep_f),1)*100:.1f}%</b></div>
            <div>Days &lt; {sl_low} min: <b style="color:#ef4444">{days_lt_sl}</b></div>
            <div>Healthy range: <b style="color:#a8c8e8">{sl_low} – {sl_high} min/night</b></div>
        </div>""", unsafe_allow_html=True)
        if n_sleep_f > 0:
            st.markdown("")
            alert(f"{n_sleep_f} sleep anomaly days detected")
        else:
            st.markdown("")
            ok("No sleep anomalies in selected range")

    with col_rec3:
        st.markdown('<div class="stat-title">SLEEP ANOMALY RECORDS</div>', unsafe_allow_html=True)
        sl_disp = anom_sleep_f[anom_sleep_f["is_anomaly"]][
            ["Date","TotalSleepMinutes","rolling_med","residual","reason"]].round(2)
        if not sl_disp.empty:
            st.dataframe(sl_disp.rename(columns={
                "TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected",
                "residual":"Deviation","reason":"Reason"}),
                use_container_width=True, height=260)
        else:
            ok("No sleep anomalies in selected range.")

# ──────────────────────────────────────────────────────────────
#  EXPORT TAB
# ──────────────────────────────────────────────────────────────
with tab_export:
    with st.expander("📄  PDF Report", expanded=True):
        tip("Full multi-page PDF with executive summary, anomaly tables, user profiles, and methodology.")
        st.markdown("""
        <div style="background:#162a41;border:1px solid #1e4a6e;border-radius:12px;
                    padding:16px 18px;margin-bottom:1rem">
            <div style="font-weight:700;color:#a78bfa;margin-bottom:8px">📄 What's in the PDF (9 sections):</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.82rem;color:#a8c8e8">
                <div>✅ Dataset overview</div>
                <div>✅ Anomaly summary table</div>
                <div>✅ Detection thresholds used</div>
                <div>✅ Methodology explanation</div>
                <div>✅ HR anomaly records</div>
                <div>✅ Steps anomaly records</div>
                <div>✅ Sleep anomaly records</div>
                <div>✅ User activity profiles</div>
                <div style="grid-column:1/-1">✅ Conclusion & findings</div>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button("📄  Generate & Download PDF Report", key="m4_gen_pdf"):
            with st.spinner("Generating PDF…"):
                try:
                    pdf_buf = generate_pdf(
                        master_f, anom_hr_f, anom_steps_f, anom_sleep_f,
                        hr_high, hr_low, st_low, sl_low, sl_high, sigma)
                    fname_pdf = f"FitPulse_M4_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        label="⬇️  Download PDF Report",
                        data=pdf_buf,
                        file_name=fname_pdf,
                        mime="application/pdf",
                        key="m4_dl_pdf")
                    ok(f"PDF ready — click the button above to download.")
                except Exception as e:
                    st.error(f"PDF error: {e}")
                    st.exception(e)

    with st.expander("📊  CSV Export", expanded=True):
        tip("All anomaly records from all three signals combined into one CSV.")
        csv_data  = generate_csv(anom_hr_f, anom_steps_f, anom_sleep_f)
        fname_csv = f"FitPulse_M4_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        st.download_button(
            label="⬇️  Download Anomaly CSV",
            data=csv_data, file_name=fname_csv,
            mime="text/csv", key="m4_dl_csv")
        with st.expander("👁️  Preview CSV (first 20 rows)"):
            try:
                st.dataframe(pd.read_csv(io.StringIO(csv_data.decode())).head(20),
                             use_container_width=True)
            except Exception:
                st.info("Preview unavailable.")

    with st.expander("🏁  Milestone 4 Completion Checklist", expanded=False):
        ok("🎉 Milestone 4 pipeline complete!")
        for icon, label in [
            ("📂","5 Fitbit CSV files loaded and merged"),
            ("💓","Heart rate processed (1-min resampling)"),
            ("😴","Sleep data processed from minute-level logs"),
            ("🔗","Master dataset built"),
            ("①", "Threshold violations detected"),
            ("②", "Residual-based detection (±sigma) applied"),
            ("📅","Combined anomaly timeline"),
            ("💓","HR statistics + records table"),
            ("👟","Steps statistics + records table"),
            ("😴","Sleep statistics + records table"),
            ("🔍","Date & user filter in sidebar"),
            ("📄","PDF report export (ReportLab)"),
            ("📊","CSV anomaly export"),
        ]:
            st.markdown(f"✅ {icon} {label}")