import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  MILESTONE 1  –  Single-file preprocessing
# ─────────────────────────────────────────────
def preprocess_data(file):
    logs = []

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".json"):
        df = pd.read_json(file)
    else:
        raise ValueError("Unsupported file format")

    logs.append("File loaded successfully.")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["Date"] = df["Date"].dt.tz_localize(
            "UTC", ambiguous="NaT", nonexistent="shift_forward"
        )
        logs.append("Date column converted to datetime and normalised to UTC.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logs.append("Numeric columns identified.")

    null_summary = df.isnull().sum()
    logs.append("Null value summary generated.")

    if "User_ID" in df.columns:
        df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
            lambda x: x.interpolate(method="linear")
        )
        logs.append("Numeric columns interpolated per User_ID.")
        df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )
        logs.append("Remaining numeric nulls forward/backward filled.")

    if "Workout_Type" in df.columns:
        df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")
        logs.append("Workout_Type nulls filled with 'No Workout'.")

    logs.append("Preprocessing completed successfully.")
    return df, logs, null_summary


# ─────────────────────────────────────────────
#  MILESTONE 2  –  Multi-file pipeline helpers
# ─────────────────────────────────────────────

TIMESTAMP_COLUMNS = {
    "daily_activity":     ["ActivityDate"],
    "heartrate":          ["Time"],
    "hourly_intensities": ["ActivityHour"],
    "hourly_steps":       ["ActivityHour"],
    "minute_sleep":       ["date"],
}


# ---------- Step 1-2  Load, clean & preview ----------
def load_all_files(uploaded_files: dict) -> dict:
    """
    Load CSVs and immediately apply per-file cleaning:
      - Drop fully duplicate rows
      - Drop rows where the key timestamp column is null/unparseable
      - Forward/backward fill numeric nulls per user (if Id present), else median fill
      - Fill categorical nulls with 'Unknown'
    Returns {label: cleaned DataFrame}
    """
    dfs = {}
    for label, f in uploaded_files.items():
        if f is None:
            continue
        df = pd.read_csv(f)

        # 1. Drop exact duplicate rows
        df = df.drop_duplicates()

        # 2. Parse & drop rows with invalid timestamps
        ts_col = TIMESTAMP_COLUMNS.get(label, [None])[0]
        if ts_col and ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True, errors="coerce")
            df = df.dropna(subset=[ts_col])

        # 3. Numeric null handling
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            if "Id" in df.columns:
                # Per-user: interpolate then ffill/bfill
                df[num_cols] = df.groupby("Id")[num_cols].transform(
                    lambda x: x.interpolate(method="linear").ffill().bfill()
                )
            # Any still-remaining nulls -> median fill
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # 4. Categorical null handling
        cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != ts_col]
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")

        dfs[label] = df
    return dfs


# ---------- Step 3  Timestamp parsing ----------
def parse_timestamps(dfs: dict):
    logs = []
    for label, df in dfs.items():
        for col in TIMESTAMP_COLUMNS.get(label, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                logs.append(f"[{label}] `{col}` → datetime64")
        dfs[label] = df
    return dfs, logs


# ---------- Step 4  Null check ----------
def null_summary_all(dfs: dict) -> dict:
    return {label: df.isnull().sum() for label, df in dfs.items()}


# ---------- Step 5  Stats ----------
def dataset_stats(dfs: dict) -> dict:
    return {label: df.describe(include="all") for label, df in dfs.items()}


# ---------- Step 6  Heart-rate resampling ----------
def resample_heartrate(df_hr: pd.DataFrame):
    logs = []
    id_col, ts_col, val_col = "Id", "Time", "Value"

    if ts_col not in df_hr.columns or val_col not in df_hr.columns:
        logs.append("⚠️  Heart-rate columns missing – skipped resampling.")
        return df_hr, logs

    df = df_hr.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    if id_col in df.columns:
        parts = []
        for uid, grp in df.groupby(id_col):
            g = grp.set_index(ts_col).sort_index()
            g = g[[val_col]].resample("1min").mean().interpolate()
            g[id_col] = uid
            parts.append(g.reset_index())
        result = pd.concat(parts, ignore_index=True)
        logs.append(f"Resampled to 1-min across {df[id_col].nunique()} users.")
    else:
        result = (
            df.set_index(ts_col)
            .sort_index()[[val_col]]
            .resample("1min").mean()
            .interpolate()
            .reset_index()
        )
        logs.append("Heart-rate resampled to 1-min granularity.")

    return result, logs


# ---------- Step 8  Master DataFrame ----------
def build_master_df(dfs: dict):
    logs = []
    da  = dfs.get("daily_activity")
    hr  = dfs.get("heartrate")
    hint = dfs.get("hourly_intensities")
    hst  = dfs.get("hourly_steps")
    ms   = dfs.get("minute_sleep")

    if da is None:
        logs.append("⚠️  daily_activity is required.")
        return pd.DataFrame(), logs

    master = da.copy()
    date_col = "ActivityDate"
    if date_col not in master.columns:
        logs.append(f"⚠️  '{date_col}' not found.")
        return master, logs

    master[date_col] = pd.to_datetime(master[date_col], errors="coerce")
    master["date_only"] = master[date_col].dt.date

    def _grp(df):
        return ["Id", "date_only"] if "Id" in df.columns else ["date_only"]

    # heart-rate → daily mean
    if hr is not None and "Time" in hr.columns and "Value" in hr.columns:
        h = hr.copy()
        h["Time"] = pd.to_datetime(h["Time"], errors="coerce")
        h["date_only"] = h["Time"].dt.date
        g = _grp(h)
        hd = h.groupby(g)["Value"].mean().reset_index().rename(columns={"Value": "AvgHeartRate"})
        master = master.merge(hd, on=g, how="left")
        logs.append("Daily mean heart-rate merged.")

    # hourly intensities → daily sum
    if hint is not None and "ActivityHour" in hint.columns:
        h = hint.copy()
        h["ActivityHour"] = pd.to_datetime(h["ActivityHour"], errors="coerce")
        h["date_only"] = h["ActivityHour"].dt.date
        g = _grp(h)
        num_c = [c for c in h.select_dtypes(include=[np.number]).columns if c not in ["Id"]]
        hd = h.groupby(g)[num_c].sum().reset_index()
        master = master.merge(hd, on=g, how="left", suffixes=("", "_hint"))
        logs.append("Hourly intensities aggregated & merged.")

    # hourly steps → daily sum
    if hst is not None and "ActivityHour" in hst.columns:
        h = hst.copy()
        h["ActivityHour"] = pd.to_datetime(h["ActivityHour"], errors="coerce")
        h["date_only"] = h["ActivityHour"].dt.date
        g = _grp(h)
        num_c = [c for c in h.select_dtypes(include=[np.number]).columns if c not in ["Id"]]
        hd = h.groupby(g)[num_c].sum().reset_index()
        master = master.merge(hd, on=g, how="left", suffixes=("", "_hst"))
        logs.append("Hourly steps aggregated & merged.")

    # minute sleep → daily sum
    if ms is not None and "date" in ms.columns:
        h = ms.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        h["date_only"] = h["date"].dt.date
        g = _grp(h)
        num_c = [c for c in h.select_dtypes(include=[np.number]).columns if c not in ["Id", "logId"]]
        hd = h.groupby(g)[num_c].sum().reset_index()
        if num_c:
            hd = hd.rename(columns={num_c[0]: "TotalSleepMinutes"})
        master = master.merge(hd, on=g, how="left", suffixes=("", "_sleep"))
        logs.append("Minute sleep aggregated & merged.")

    # Drop duplicate rows in master (can appear after merge on shared columns)
    before = len(master)
    master = master.drop_duplicates()
    if len(master) < before:
        logs.append(f"Dropped {before - len(master)} duplicate rows from master.")

    # Fill remaining numeric nulls: interpolate first, then median fallback
    num = master.select_dtypes(include=[np.number]).columns.tolist()
    if num:
        if "Id" in master.columns:
            master[num] = master.groupby("Id")[num].transform(
                lambda x: x.interpolate(method="linear").ffill().bfill()
            )
        # Final safety net: any still-null cells -> median
        master[num] = master[num].fillna(master[num].median())
    logs.append("Numeric nulls filled (interpolation + median fallback).")

    # Fill remaining categorical nulls
    cat = master.select_dtypes(include=["object"]).columns.tolist()
    for col in cat:
        master[col] = master[col].fillna("Unknown")
    logs.append("Categorical nulls filled with 'Unknown'.")

    logs.append(f"Master DataFrame: {master.shape[0]} rows × {master.shape[1]} cols.")
    return master, logs


# ---------- Step 10  TSFresh prep ----------
def prepare_tsfresh_input(hr_df: pd.DataFrame):
    needed = {"Id", "Time", "Value"}
    if not needed.issubset(hr_df.columns):
        return None
    ts = hr_df[["Id", "Time", "Value"]].copy()
    ts["Time"] = pd.to_datetime(ts["Time"], errors="coerce")
    ts = ts.dropna(subset=["Time", "Value"]).sort_values(["Id", "Time"])
    ts["t"] = ts.groupby("Id")["Time"].transform(
        lambda s: (s - s.min()).dt.total_seconds().astype(int)
    )
    return ts[["Id", "t", "Value"]].rename(columns={"Id": "id", "t": "time", "Value": "value"})


# ---------- Step 18  Clustering features ----------
def build_clustering_features(master: pd.DataFrame) -> pd.DataFrame:
    num_cols = master.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in ["Id"]]
    feat = master[feat_cols].copy().fillna(master[feat_cols].median())
    return feat