import os
import glob
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==== Paths ====
script_dir = os.path.dirname(__file__)
base_dir = os.path.join(script_dir, "range")
eeg_dir = os.path.join(base_dir, "eeg")
hrv_dir = os.path.join(base_dir, "hrv")
output_dir = os.path.join(script_dir, "output", "summary")
os.makedirs(output_dir, exist_ok=True)


# ==== Load JSON or CSV ====
def load_json_or_csv(path):
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        df = pd.read_csv(path)
        return df.iloc[0].to_dict()


# ==== Outlier removal (IQR) ====
def remove_outliers(df, columns, k=1.5):
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df or df[col].isna().all():
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - k * IQR, Q3 + k * IQR
        mask &= df[col].between(lower, upper)
    return df[mask]


# ==== Collect EEG & HRV files ====
eeg_files = sorted(glob.glob(os.path.join(eeg_dir, "*.[cC][sS][vV]")) +
                   glob.glob(os.path.join(eeg_dir, "*.json")))

records = []

for eeg_file in eeg_files:
    fname = os.path.basename(eeg_file)
    user_part = fname.split("_")[0]  # e.g. user1, user17
    user_num = int(user_part.replace("user", ""))

    # Determine state (relax vs device)
    if "_relax" in fname:
        state = "relax"
    else:
        state = "relax" if user_num <= 16 else "device"

    # --- Load EEG ---
    try:
        eeg_data = load_json_or_csv(eeg_file)
        att = eeg_data.get("attention", [None, None])
        if isinstance(att, (list, tuple)) and len(att) >= 2:
            attention_low, attention_high = att[0], att[1]
        else:
            attention_low, attention_high = att, None
        stress_max = eeg_data.get("stress_max", None)
    except Exception as e:
        print(f"Error reading EEG file {eeg_file}: {e}")
        continue

    # --- Match HRV file ---
    if state == "device":
        hrv_candidates = [
            os.path.join(hrv_dir, f"{user_part}_range.csv"),
            os.path.join(hrv_dir, f"{user_part}_range.json"),
        ]
    else:
        hrv_candidates = [
            os.path.join(hrv_dir, f"{user_part}_relax_range.csv"),
            os.path.join(hrv_dir, f"{user_part}_relax_range.json"),
            os.path.join(hrv_dir, f"{user_part}_range.csv"),
            os.path.join(hrv_dir, f"{user_part}_range.json"),
        ]

    hrv_file = next((f for f in hrv_candidates if os.path.exists(f)), None)
    if not hrv_file:
        print(f"No HRV file found for {user_part}")
        continue

    # --- Load HRV ---
    try:
        hrv_data = load_json_or_csv(hrv_file)
        hrv_low = hrv_data.get("hrv_low", None)
        hrv_high = hrv_data.get("hrv_high", None)
    except Exception as e:
        print(f"Error reading HRV file {hrv_file}: {e}")
        continue

    # --- Record ---
    records.append({
        "user_id": user_part,
        "state": state,
        "hrv_low": hrv_low,
        "hrv_high": hrv_high,
        "attention_low": attention_low,
        "attention_high": attention_high,
        "stress_max": stress_max
    })


# ==== Save combined dataset ====
all_users_file = os.path.join(output_dir, "all_users_status.csv")
if records:
    df = pd.DataFrame(records)
    df.to_csv(all_users_file, index=False)
    print(f"Combined DataFrame saved to {all_users_file}")
else:
    raise SystemExit("❌ No data was loaded. Please check file paths.")


# ==== Summary (by state) ====
summary = df.groupby('state').agg(['mean', 'std']).reset_index()
summary_file = os.path.join(output_dir, "status_summary.csv")
summary.to_csv(summary_file, index=False)
print(f"Overall summary saved to {summary_file}")
print(summary)


# ==== Select users 1–24 only ====
df_selected = df[df['user_id'].str.extract('user(\\d+)')[0].astype(int).between(1, 24)]
summary_stats = df_selected[['hrv_low', 'hrv_high', 'attention_low', 'attention_high']].agg(['mean', 'std'])
print("\nSummary stats for user1–24:")
print(summary_stats)


# ==== Boxplots (Relax vs Device) ====
sns.set(style="whitegrid")

# HRV
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='state', y='hrv_low', data=df_selected)
plt.title("HRV Low: Relax vs Device")
plt.subplot(1, 2, 2)
sns.boxplot(x='state', y='hrv_high', data=df_selected)
plt.title("HRV High: Relax vs Device")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hrv_low_high_boxplot.png"))
plt.close()

# Attention
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='state', y='attention_low', data=df_selected)
plt.title("Attention Low: Relax vs Device")
plt.subplot(1, 2, 2)
sns.boxplot(x='state', y='attention_high', data=df_selected)
plt.title("Attention High: Relax vs Device")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "attention_low_high_boxplot.png"))
plt.close()


# ==== Combined (user1–16 relax, user17–24 device) ====
users_1_16 = df[(df['user_id'].str.extract('user(\\d+)')[0].astype(int).between(1, 16)) & (df['state'] == 'relax')]
users_17_24 = df[(df['user_id'].str.extract('user(\\d+)')[0].astype(int).between(17, 24)) & (df['state'] == 'device')]
combined_users = pd.concat([users_1_16, users_17_24], ignore_index=True)

# Remove outliers
cols_to_check = ['hrv_low', 'hrv_high', 'attention_low', 'attention_high', 'stress_max']
combined_users_clean = remove_outliers(combined_users, cols_to_check)

# Stats after cleaning
overall_stats_clean = combined_users_clean[cols_to_check].agg(['mean', 'std'])
print("\nStats after removing outliers:")
print(overall_stats_clean)

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=combined_users_clean[['hrv_low', 'hrv_high']])
plt.title("HRV Low & High (cleaned)")
plt.subplot(1, 2, 2)
sns.boxplot(data=combined_users_clean[['attention_low', 'attention_high']])
plt.title("Attention Low & High (cleaned)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "overall_boxplot_calibration.png"))
plt.close()

print("\n✅ All plots saved to output/summary/")
