import pandas as pd
import numpy as np

def preprocess_data(file):

    logs = []

    # Load file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".json"):
        df = pd.read_json(file)
    else:
        raise ValueError("Unsupported file format")

    logs.append("File loaded successfully.")

    # ---------------- DATE NORMALIZATION ----------------
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["Date"] = df["Date"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward")
        logs.append("Date column converted to datetime and normalized to UTC.")

    # ---------------- NUMERIC COLUMN IDENTIFICATION ----------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logs.append("Numeric columns identified.")

    # ---------------- NULL SUMMARY ----------------
    null_summary = df.isnull().sum()
    logs.append("Null value summary generated.")

    # ---------------- GROUP-WISE INTERPOLATION ----------------
    if "User_ID" in df.columns:
        df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
            lambda x: x.interpolate(method="linear")
        )
        logs.append("Numeric columns interpolated per User_ID.")

        df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )
        logs.append("Remaining numeric nulls forward/backward filled.")

    # ---------------- CATEGORICAL HANDLING ----------------
    if "Workout_Type" in df.columns:
        df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")
        logs.append("Workout_Type nulls filled with 'No Workout'.")

    logs.append("Preprocessing completed successfully.")

    return df, logs, null_summary