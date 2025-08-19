import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import LineString
import seaborn as sns
from scipy.stats import ttest_ind

# === Path configuration ===
USER_DIR = Path("user_data")
SUMMARY_DIR = Path("output/summary")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ['condition_A', 'condition_B']

# === Function: Calculate area difference ===
def calculate_region_area(draw_points, traj_points, width_draw=0.2, width_traj=0.2):
    """ Calculate the symmetric difference area between drawn trajectory and reference trajectory """
    if len(draw_points) < 2 or len(traj_points) < 2:
        return np.nan
    draw_line = LineString(draw_points)
    traj_line = LineString(traj_points)
    draw_buffer = draw_line.buffer(width_draw, cap_style=2, join_style=2)
    traj_buffer = traj_line.buffer(width_traj, cap_style=2, join_style=2)
    return draw_buffer.symmetric_difference(traj_buffer).area

# === Function: Analyze per-user area difference and duration ===
def analyze_user_area_duration(user_path: Path):
    user_results = []

    for cond in CONDITIONS:
        cond_path = user_path / cond
        if not cond_path.exists():
            continue

        for file_idx in range(1, 11):  # Only use 1–10
            draw_file = cond_path / f"draw_{file_idx:02d}.csv"
            traj_file = cond_path / f"traj_{file_idx:02d}.csv"

            if not draw_file.exists() or not traj_file.exists():
                continue

            draw_df = pd.read_csv(draw_file, skiprows=2)
            traj_df = pd.read_csv(traj_file)

            # Completion time
            if 'timestamp' in draw_df.columns:
                draw_df['timestamp'] = pd.to_datetime(draw_df['timestamp'], errors='coerce').dt.tz_localize(None)
                duration = (draw_df['timestamp'].max() - draw_df['timestamp'].min()).total_seconds()
            else:
                duration = np.nan

            # Area difference
            draw_points = draw_df[['x', 'z']].values
            traj_points = traj_df[['x', 'z']].values
            area_diff = calculate_region_area(draw_points, traj_points)

            # Trajectory type (Type A: 1–5, Type B: 6–10)
            traj_type = 'A' if 1 <= file_idx <= 5 else 'B'

            user_results.append({
                'user_id': user_path.name,
                'condition': cond[-1],   # A or B
                'file_id': file_idx,
                'traj_type': traj_type,
                'area_diff': area_diff,
                'duration': duration
            })
    return pd.DataFrame(user_results)

# === Batch analysis for all users ===
all_users_results = []
for user_folder in sorted(USER_DIR.iterdir()):
    if user_folder.is_dir():
        df_user = analyze_user_area_duration(user_folder)
        if not df_user.empty:
            all_users_results.append(df_user)

if all_users_results:
    all_users_df = pd.concat(all_users_results, ignore_index=True)
    all_users_df.to_csv(SUMMARY_DIR / "summary_all_users_area_duration_1to10.csv", index=False)

# === Outlier removal (IQR method) ===
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

clean_df = remove_outliers_iqr(all_users_df, "area_diff")
clean_df = remove_outliers_iqr(clean_df, "duration")

# === Line plot ===
fig, ax1 = plt.subplots(figsize=(12,6))

color_map = {'A': 'blue', 'B': 'green'}
duration_color_map = {'A': 'red', 'B': 'orange'}

# Background shading for Type A / B
ax1.axvspan(0.5, 5.5, color='#4DD0E1', alpha=0.3, label="Type A (1–5)")
ax1.axvspan(5.5, 10.5, color='#FFB74D', alpha=0.3, label="Type B (6–10)")

for traj_type, style in zip(['A','B'], ['-','--']):
    subset = all_users_df[all_users_df['traj_type']==traj_type]
    if subset.empty:
        continue
    subset = subset.groupby('file_id')['area_diff'].mean().reset_index()
    ax1.plot(subset['file_id'], subset['area_diff'], marker='o',
             linestyle=style, color=color_map[traj_type], label=f"AreaDiff Type {traj_type}")

ax1.set_xlabel("Trial ID")
ax1.set_ylabel("Area Difference", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.set_xticks(range(1,11))
ax1.set_xlim(0.5, 10.5)

# Duration curve
ax2 = ax1.twinx()
for traj_type, style in zip(['A','B'], ['-','--']):
    subset = all_users_df[all_users_df['traj_type']==traj_type]
    if subset.empty:
        continue
    subset = subset.groupby('file_id')['duration'].mean().reset_index()
    ax2.plot(subset['file_id'], subset['duration'], marker='s',
             linestyle=style, color=duration_color_map[traj_type], label=f"Duration Type {traj_type}")

ax2.set_ylabel("Duration (s)", color='black')
ax2.tick_params(axis='y', labelcolor='black')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left')

plt.title("Average Area Difference and Completion Time Across All Users\nType A (1–5) vs Type B (6–10)")
plt.tight_layout()
plt.savefig(SUMMARY_DIR / "average_area_duration_typeA_typeB.png", dpi=300)
plt.close()

print("✅ Line plot of AreaDiff & Duration has been generated")

# === Boxplots + individual data points ===
fig, axes = plt.subplots(1, 2, figsize=(12,6))

# --- AreaDiff ---
sns.boxplot(data=clean_df, x="traj_type", y="area_diff", ax=axes[0],
            palette={"A":"#64B5F6", "B":"#81C784"}, width=0.5, showfliers=False)
sns.stripplot(data=clean_df, x="traj_type", y="area_diff", ax=axes[0],
              color="red", size=4, jitter=True, alpha=0.7)
axes[0].set_title("Area Difference by Trajectory Type")
axes[0].set_xlabel("Trajectory Type")
axes[0].set_ylabel("Area Difference")

# --- Duration ---
sns.boxplot(data=clean_df, x="traj_type", y="duration", ax=axes[1],
            palette={"A":"#FFB74D", "B":"#4DB6AC"}, width=0.5, showfliers=False)
sns.stripplot(data=clean_df, x="traj_type", y="duration", ax=axes[1],
              color="red", size=4, jitter=True, alpha=0.7)
axes[1].set_title("Completion Time by Trajectory Type")
axes[1].set_xlabel("Trajectory Type")
axes[1].set_ylabel("Duration (s)")

plt.suptitle("Comparison of Type A (1–5) vs Type B (6–10)\n(outliers removed, with data points)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(SUMMARY_DIR / "boxplot_with_points_typeA_typeB.png", dpi=300)
plt.close()

print("✅ Boxplots with data points for Type A / B have been generated (outliers removed)")

# === Compute mean ± SD + p-values ===
typeA = clean_df[clean_df['traj_type'] == 'A']
typeB = clean_df[clean_df['traj_type'] == 'B']

# t-test
p_area = ttest_ind(typeA['area_diff'], typeB['area_diff'], nan_policy='omit').pvalue
p_duration = ttest_ind(typeA['duration'], typeB['duration'], nan_policy='omit').pvalue

print("=== Mean ± Standard Deviation (after removing outliers) ===")
for t, df in zip(['Type A (1–5)', 'Type B (6–10)'], [typeA, typeB]):
    area_mean = df['area_diff'].mean()
    area_std = df['area_diff'].std()
    dur_mean = df['duration'].mean()
    dur_std = df['duration'].std()
    print(f"{t}:")
    print(f"  Area Difference = {area_mean:.2f} ± {area_std:.2f}")
    print(f"  Duration = {dur_mean:.2f} ± {dur_std:.2f} s")
    print()

print("=== Between-group p-values ===")
print(f"Area Difference p-value: {p_area:.4f}")
print(f"Duration p-value: {p_duration:.4f}")
