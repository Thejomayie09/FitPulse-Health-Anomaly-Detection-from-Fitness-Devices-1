# FitPulse-Health-Anomaly-Detection-from-Fitness-Devices-1

# 🧬 FitPulse — Health Anomaly Detection from Fitness Devices

> A complete data science pipeline that takes raw Fitbit fitness data, cleans it, extracts patterns, forecasts trends, and groups users by behaviour — built with Streamlit.

---

## 📌 Project Overview

FitPulse is a multi-milestone data analytics platform built on real Fitbit dataset containing data from **30 users** across **5 sensor files**. The project simulates what a real health analytics platform like Google Fit or Apple Health does behind the scenes — processing messy sensor data into meaningful insights.

---

## 🗂️ Project Structure

```
FitPulse/
│
├── app.py                          ← Main application (Milestone 1 + 2 combined)
├── preprocessing.py                ← All data processing functions
├── requirements.txt                ← Python dependencies
│
├── milestone_1/
│   ├── milestone1App.py            ← Standalone Milestone 1 app
│   ├── preprocessing.py
│   └── notebook/
│       └── Fitness_Health_Tracking_Dataset.csv
│
├── milestone_2/
│   ├── milestone2App.py            ← Standalone Milestone 2 app
│   ├── preprocessing.py
│   └── notebook/
│       ├── dailyActivity_merged.csv
│       ├── heartrate_seconds_merged.csv
│       ├── hourlyIntensities_merged.csv
│       ├── hourlySteps_merged.csv
│       └── minuteSleep_merged.csv
│
└── Screenshots/
    ├── ss_m1/
    └── ss_m2/
```

---

## 🏁 Milestone 1 — Data Collection & Preprocessing

### What it does
Takes a single fitness CSV file and cleans it completely before any analysis.

### Steps
| Step | What happens |
|------|-------------|
| Upload | User uploads any fitness CSV file |
| Inspection | Preview raw data, check missing values, view data types |
| Processing | Fix timestamps, interpolate missing numbers, fill empty text |
| Visualization | Detect outliers using Box Plot and Histogram |

### Key Features
- **Duplicate removal** — exact duplicate rows are deleted
- **Interpolation** — missing numeric values estimated from surrounding values
- **Date normalisation** — all date columns converted to proper datetime format
- **Outlier detection** — IQR method to find and highlight unusual values
- **Download** — cleaned CSV available for download

---

## 🧬 Milestone 2 — Feature Extraction & Modelling

### What it does
Takes 5 Fitbit sensor files, merges them into a master dataset, extracts features, forecasts future trends, and groups users by fitness behaviour.

### The 5 Input Files
| File | Contains |
|------|----------|
| Daily Activity | Steps, calories, distance per day |
| Heart Rate | Heart rate every few seconds |
| Hourly Intensities | Activity intensity each hour |
| Hourly Steps | Step count each hour |
| Minute Sleep | Sleep state every minute |

### Pipeline Steps

#### 📥 Data Loading & Cleaning
- Auto-detects which file is which from the filename
- Drops duplicate rows
- Drops rows with invalid timestamps
- Interpolates missing numeric values per user
- Fills missing text values with 'Unknown'

#### 🗂️ Master Dataset
- Merges all 5 files into one table
- One row = one user, one day
- Heart rate aggregated to daily average
- Sleep aggregated to daily total minutes

#### 🔬 TSFresh Feature Extraction
- Reads minute-by-minute heart rate data
- Automatically calculates ~50 features per user
- Features include: mean, standard deviation, maximum, skewness, etc.
- Visualised as an interactive heatmap

#### 📈 Prophet Forecasting
- Forecasts next 30 days for 3 metrics:
  - 💓 Heart Rate (bpm)
  - 👟 Daily Steps
  - 😴 Sleep Duration (minutes)
- Shows 80% confidence interval
- Interactive hover showing CI Upper, CI Lower, and Trend values

#### 🔵 Clustering
- **KMeans** — groups users into k clusters (you choose k using Elbow Chart)
- **DBSCAN** — auto-detects groups and flags outlier users
- **PCA** — visualises groups in 2D (one dot = one user)
- **t-SNE** — advanced 2D visualisation with better group separation
- Plain-English group descriptions (Highly Active / Moderate / Sedentary)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Thejomayie09/FitPulse-Health-Anomaly-Detection-from-Fitness-Devices-1.git
cd FitPulse-Health-Anomaly-Detection-from-Fitness-Devices-1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
scikit-learn
tsfresh
prophet
matplotlib
```

Or install all at once:
```bash
pip install streamlit pandas numpy plotly scikit-learn tsfresh prophet matplotlib
```

---

## 📊 Dataset

This project uses the **FitBit Fitness Tracker Dataset** — publicly available data from 30 Fitbit users who consented to share their personal tracker data.

- **Users:** 30 participants
- **Duration:** ~30 days (March–May 2016)
- **Metrics:** Steps, heart rate, sleep, calories, activity intensity

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web application framework |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Plotly | Interactive visualisations |
| Scikit-learn | KMeans, DBSCAN, PCA, t-SNE |
| TSFresh | Automated time series feature extraction |
| Prophet | Time series forecasting |

---


## 👩‍💻 Author

**Thejomayie**  
GitHub: [@Thejomayie09](https://github.com/Thejomayie09)

---

## 📄 License

This project is licensed under the MIT License.