import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Path configuration
USER_DIR = Path("user_data")
OUTPUT_DIR = Path("output/hrv_eeg_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

user_id = "user_12"
eeg_file = USER_DIR / user_id / "eeg.csv"
hrv_file = USER_DIR / user_id / "hrv.csv"

if eeg_file.exists() and hrv_file.exists():
    # === Load EEG ===
    eeg_df = pd.read_csv(eeg_file)
    eeg_df["timestamp"] = pd.to_datetime(eeg_df["timestamp"], errors="coerce").dt.tz_localize(None)

    # === Load HRV ===
    hrv_df = pd.read_csv(hrv_file)
    hrv_df["timestamp"] = pd.to_datetime(hrv_df["timestamp"], errors="coerce").dt.tz_localize(None)

    # HRV smoothing (rolling mean)
    if "rmssd" in hrv_df.columns:
        hrv_df["rmssd_smooth"] = hrv_df["rmssd"].rolling(window=5, min_periods=1).mean()
    else:
        hrv_df["rmssd_smooth"] = np.nan

    # === Plot dual-axis time series ===
    fig, ax1 = plt.subplots(figsize=(12,6))

    # EEG Attention (left axis)
    ax1.plot(eeg_df["timestamp"], eeg_df["attention"],
             color="blue", alpha=0.7, label="EEG Attention")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("EEG Attention", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # HRV RMSSD (right axis)
    ax2 = ax1.twinx()
    ax2.plot(hrv_df["timestamp"], hrv_df["rmssd_smooth"],
             color="red", alpha=0.7, label="HRV (smoothed)")
    ax2.set_ylabel("HRV (RMSSD)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(f"{user_id} - EEG vs HRV Time Series (Overlayed)")
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{user_id}_timeseries_eeg_hrv.png", dpi=300)
    plt.close()

    print(f"✅ Time series alignment plot saved: {OUTPUT_DIR}/{user_id}_timeseries_eeg_hrv.png")

else:
    print(f"⚠️ Missing eeg.csv or hrv.csv file for {user_id}")
