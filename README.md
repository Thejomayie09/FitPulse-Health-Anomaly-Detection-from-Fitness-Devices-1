# 💪 FitPulse — Health Anomaly Detection from Fitness Devices

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?style=flat-square&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-3F4F75?style=flat-square&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**An end-to-end Fitbit data analytics pipeline — from raw CSV to anomaly detection and interactive reporting.**

[Overview](#-overview) • [Features](#-features) • [Project Structure](#-project-structure) • [Installation](#-installation) • [Usage](#-usage) • [Milestones](#-milestones) • [Dataset](#-dataset)

</div>

---

## 📌 Overview

FitPulse is a multi-milestone Streamlit web application that processes real Fitbit wearable device data to detect health anomalies in **heart rate**, **step count**, and **sleep duration**. The project covers the complete ML pipeline from raw data ingestion to interactive dashboard and PDF report export.

Built as part of an academic project (Batch 13), the app is structured around 4 progressive milestones, each building on the previous one. Files uploaded in Milestone 2 are **automatically shared** with Milestones 3 and 4 — no re-uploading needed.

---

## ✨ Features

- 🏠 **Home Dashboard** — Pipeline overview with milestone cards and quick navigation
- 📊 **Milestone 1** — CSV upload, data inspection, cleaning engine, outlier detection with box plots & histograms
- 🧬 **Milestone 2** — TSFresh feature extraction, Prophet forecasting (HR/Steps/Sleep), KMeans + DBSCAN clustering, PCA + t-SNE visualisation
- 🚨 **Milestone 3** — Three anomaly detection methods: Threshold Violations, Residual-Based (±σ), DBSCAN Structural Outliers; accuracy simulation (90%+ target)
- 📈 **Milestone 4** — Interactive insights dashboard with date/user filters, KPI strip, statistics cards, anomaly records tables, PDF report export (ReportLab), and CSV export
- 🔄 **Shared file state** — Upload once in M2, automatically available in M3 and M4

---

## 🗂 Project Structure

```
ALL MILESTONES/
│
├── fitpulse_app.py                  ← 🚀 Main app (all 4 milestones combined)
├── preprocessing.py                 ← Shared backend functions
│
├── milestone_1/
│   ├── milestone1App.py             ← Standalone Milestone 1
│   ├── preprocessing.py
│   └── notebook/
│
├── milestone_2/
│   ├── milestone2App.py             ← Standalone Milestone 2
│   ├── preprocessing.py
│   └── notebook/
│
├── milestone_3/
│   └── milestone3App.py             ← Standalone Milestone 3
│
├── milestone_4/
│   └── finalMilestone.py            ← Standalone Milestone 4
│
├── Documentation/
│   └── Fitpulse_Documentation_final.pptx
│
├── Screenshots/
│   ├── ss_m1/
│   ├── ss_m2/
│   ├── ss_m3/
│   └── ss_m4/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Thejomayie09/FitPulse-Health-Anomaly-Detection-from-Fitness-Devices-1.git
cd FitPulse-Health-Anomaly-Detection-from-Fitness-Devices-1
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run fitpulse_app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📦 Requirements

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.0.0
scikit-learn>=1.3.0
reportlab>=4.0.0
prophet>=1.1.4
tsfresh>=0.20.0
```

> **Note:** `tsfresh` and `prophet` are optional — the app works without them but TSFresh extraction and Prophet forecasting in Milestone 2 will be disabled until installed.

---

## 🚀 Usage

### Running the Combined App

```bash
streamlit run fitpulse_app.py
```

Use the **sidebar dropdown** to navigate between milestones:

| Option | Description |
|--------|-------------|
| 🏠 Home | Landing page with pipeline overview |
| 📊 Milestone 1 | Data Preprocessing |
| 🧬 Milestone 2 | Feature Extraction & Modelling |
| 🚨 Milestone 3 | Anomaly Detection |
| 📈 Milestone 4 | Insights Dashboard |

### Running Individual Milestones

Each milestone can also be run standalone:

```bash
streamlit run milestone_1/milestone1App.py
streamlit run milestone_2/milestone2App.py
streamlit run milestone_3/milestone3App.py
streamlit run milestone_4/finalMilestone.py
```

> **Important:** Standalone milestone apps require `preprocessing.py` to be in the **same directory** as the app file.

---

## 📚 Milestones

### 📊 Milestone 1 — Data Preprocessing

Upload any fitness CSV file and:
- Inspect raw data: shape, missing values, column data types
- Clean data: normalise dates, interpolate numeric gaps, fill categorical nulls
- Detect outliers using the IQR method with interactive box plots and histograms
- Export the cleaned dataset as CSV

---

### 🧬 Milestone 2 — Feature Extraction & Modelling

Upload 5 Fitbit CSV files and:

- **TSFresh** — automatically extract 10+ statistical features from heart rate time series per user
- **Prophet** — forecast HR, steps, and sleep trends with 30-day projections and 80% confidence intervals
- **KMeans** — cluster users into activity groups (active / moderate / sedentary)
- **DBSCAN** — identify structural outliers whose behaviour doesn't match any cluster
- **PCA + t-SNE** — visualise high-dimensional user profiles in 2D scatter maps

> Files uploaded in Milestone 2 are **automatically shared** with Milestones 3 and 4 via session state. No re-uploading needed when switching milestones.

---

### 🚨 Milestone 3 — Anomaly Detection

Detects unusual health patterns using three complementary methods:

| Method | Description |
|--------|-------------|
| ① Threshold Violations | Hard upper/lower limits on HR, steps, and sleep |
| ② Residual-Based (±σ) | 3-day rolling median baseline; flags days deviating by ±2σ |
| ③ DBSCAN Structural Outliers | Clusters users by activity profile; label −1 = structural outlier |

**Five output charts:**
1. Heart Rate Anomaly Detection Chart (line + band + anomaly markers)
2. Sleep Pattern Visualization (dual subplot with residual bars)
3. Step Count Trend with Alert Bands (vertical red highlights)
4. DBSCAN Outlier PCA Projection
5. Simulated Detection Accuracy (injection testing — 90%+ target)

---

### 📈 Milestone 4 — Insights Dashboard

Interactive dashboard with full export capability:

**KPI Strip (6 cards):**
Total Flags · HR Flags · Steps Alerts · Sleep Flags · Users · Peak HR Anomaly Day

**5 Tabs:**
| Tab | Content |
|-----|---------|
| 📊 Overview | Combined anomaly timeline + recent anomaly log |
| 💓 Heart Rate | Statistics card (mean/max/min/rate) + anomaly records table |
| 👟 Steps | Statistics card + anomaly records table |
| 😴 Sleep | Statistics card + anomaly records table |
| 📥 Export | PDF report (9 sections) + CSV export + completion checklist |

**Sidebar filters** (appear after pipeline runs):
- Date range picker
- User selector (all users or individual)

---

## 📁 Dataset

This project uses the **FitBit Fitness Tracker Data** from Kaggle:

> **Source:** [arashnic/fitbit](https://www.kaggle.com/datasets/arashnic/fitbit)
> **Description:** Personal fitness tracker data from 35 Fitbit users (April–May 2016)
> **License:** CC0 — Public Domain

### Required Files (for Milestones 2, 3, and 4)

| File | Key Columns |
|------|-------------|
| `dailyActivity_merged.csv` | `ActivityDate`, `TotalSteps`, `Calories`, `VeryActiveMinutes`, `SedentaryMinutes` |
| `heartrate_seconds_merged.csv` | `Id`, `Time`, `Value` |
| `hourlyIntensities_merged.csv` | `Id`, `ActivityHour`, `TotalIntensity` |
| `hourlySteps_merged.csv` | `Id`, `ActivityHour`, `StepTotal` |
| `minuteSleep_merged.csv` | `Id`, `date`, `value`, `logId` |

> Files are **auto-detected by column structure** — any file name works as long as the required columns are present.

---

## 🧠 Technical Details

### Anomaly Detection — How It Works

**① Threshold Violations** — Hard-coded health limits:
```
Heart Rate  :  < 50 bpm  or  > 100 bpm
Steps       :  < 500/day  or  > 25,000/day
Sleep       :  < 60 min/night  or  > 600 min/night
```

**② Residual-Based Detection** — Statistical baseline approach:
```
rolling_median  =  3-day centered rolling median
residual        =  actual_value − rolling_median
anomaly         =  |residual| > σ × std(residual)
```

**③ DBSCAN Structural Outliers** — User-level profiling:
```
Features used  :  Steps, Calories, ActiveMinutes, SedentaryMinutes, SleepMinutes
Preprocessing  :  StandardScaler
Parameters     :  eps=2.2, min_samples=2
Outlier label  :  −1
```

### Preprocessing Pipeline (`preprocessing.py`)

| Function | Description |
|----------|-------------|
| `load_all_files()` | Reads and validates all 5 CSV files |
| `parse_timestamps()` | Normalises date formats across files |
| `resample_heartrate()` | Resamples HR data from seconds → 1-minute intervals |
| `build_master_df()` | Merges all files into one daily master table per user |
| `prepare_tsfresh_input()` | Formats HR data for TSFresh extraction |
| `build_clustering_features()` | Prepares feature matrix for KMeans and DBSCAN |

---

## 📸 Screenshots

Screenshots are organised in the `Screenshots/` folder:

```
Screenshots/
├── ss_m1/   ← Milestone 1: Preprocessing & outlier detection
├── ss_m2/   ← Milestone 2: TSFresh heatmap, Prophet forecast, clustering
├── ss_m3/   ← Milestone 3: Anomaly charts, DBSCAN PCA, accuracy simulation
└── ss_m4/   ← Milestone 4: KPI strip, dashboard tabs, export
```

---

## 👩‍💻 Author

**Thejomayie K** — Batch 13

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Fitbit Dataset — arashnic (Kaggle)](https://www.kaggle.com/datasets/arashnic/fitbit)
- [Streamlit](https://streamlit.io/) — Web app framework
- [TSFresh](https://tsfresh.readthedocs.io/) — Automated time series feature extraction
- [Prophet (Meta)](https://facebook.github.io/prophet/) — Time series forecasting
- [ReportLab](https://www.reportlab.com/) — PDF generation without kaleido
- [Plotly](https://plotly.com/) — Interactive visualisations
- [scikit-learn](https://scikit-learn.org/) — KMeans, DBSCAN, PCA, t-SNE