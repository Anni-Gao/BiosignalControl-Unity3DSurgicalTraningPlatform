import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import zscore

# === Path Config ===
SUMMARY_DIR = Path("output/summary")
OUTPUT_DIR = Path("output/hrv_eeg_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load Data ===
summary_file = SUMMARY_DIR / "summary_compare_all_users.csv"
trajectory_file = SUMMARY_DIR / "summary_trajectory_status.csv"

df = pd.read_csv(summary_file)
traj_df = pd.read_csv(trajectory_file)

# Keep only condition A and B
df = df[df['condition'].isin(['A', 'B'])]
traj_df = traj_df[traj_df['condition'].isin(['A', 'B'])]

# === Outlier Removal Function ===
def remove_outliers_iqr(df, column, k=1.5, non_negative=True):
    if column not in df.columns or df[column].isna().all():
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    if non_negative:
        lower = max(0, lower)
    return df[(df[column] >= lower) & (df[column] <= upper)]

# === HRV Outlier Removal ===
df = remove_outliers_iqr(df, "hrv_rmssd", k=3, non_negative=True)

# === Plot Style ===
sns.set(style="whitegrid", font_scale=1.2)

# === Grouped Correlation Scatter Plot (EEG vs HRV, colored by Condition) ===
plt.figure(figsize=(8,6))

# Scatter plot grouped by condition
sns.scatterplot(
    data=df,
    x="attention",
    y="hrv_rmssd",
    hue="condition",
    palette={"A":"blue", "B":"orange"},
    alpha=0.6,
    s=60
)

# Overall regression line (with 95% CI)
sns.regplot(
    data=df,
    x="attention",
    y="hrv_rmssd",
    scatter=False,
    line_kws={'color':'red'},
    ci=95
)

plt.title("EEG Attention vs HRV (RMSSD) by Condition\nwith Overall Regression Line (95% CI)")
plt.xlabel("EEG Attention")
plt.ylabel("HRV (RMSSD)")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_attention_vs_hrv_by_condition.png", dpi=300)
plt.close()


# === Bland–Altman Plot ===
# Standardize (z-score)
df["EEG_z"] = zscore(df["attention"].dropna())
df["HRV_z"] = zscore(df["hrv_rmssd"].dropna())

# Calculate mean & difference
mean_vals = (df["EEG_z"] + df["HRV_z"]) / 2
diff_vals = df["EEG_z"] - df["HRV_z"]

# Agreement analysis parameters
bias = np.mean(diff_vals)
sd = np.std(diff_vals)
loa_upper = bias + 1.96 * sd
loa_lower = bias - 1.96 * sd

plt.figure(figsize=(8,6))
plt.scatter(mean_vals, diff_vals, alpha=0.6, color="teal")
plt.axhline(bias, color="red", linestyle="--", label=f"Mean diff = {bias:.2f}")
plt.axhline(loa_upper, color="orange", linestyle="--", label=f"+1.96 SD = {loa_upper:.2f}")
plt.axhline(loa_lower, color="orange", linestyle="--", label=f"-1.96 SD = {loa_lower:.2f}")

plt.title("Bland–Altman Plot: EEG vs HRV (Standardized)")
plt.xlabel("Mean of EEG & HRV (z-score)")
plt.ylabel("EEG - HRV (z-score)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bland_altman_eeg_hrv.png", dpi=300)
plt.close()
