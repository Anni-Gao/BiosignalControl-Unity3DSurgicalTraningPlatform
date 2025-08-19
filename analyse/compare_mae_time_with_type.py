import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import LineString

# === Path Config ===
USER_DIR = Path("user_data")
SUMMARY_DIR = Path("output/summary")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ['condition_A', 'condition_B']

# === Function: Area Difference Calculation ===
def calculate_region_area(draw_points, traj_points, width_draw=0.2, width_traj=0.2):
    """Calculate the symmetric difference area between drawn trajectory and reference trajectory."""
    if len(draw_points) < 2 or len(traj_points) < 2:
        return np.nan
    draw_line = LineString(draw_points)
    traj_line = LineString(traj_points)
    draw_buffer = draw_line.buffer(width_draw, cap_style=2, join_style=2)
    traj_buffer = traj_line.buffer(width_traj, cap_style=2, join_style=2)
    return draw_buffer.symmetric_difference(traj_buffer).area

# === Function: Analyze Single User (Trajectory Area & Duration) ===
def analyze_user_area_duration(user_path: Path):
    user_results = []

    for cond in CONDITIONS:
        cond_path = user_path / cond
        if not cond_path.exists():
            continue

        for file_idx in range(1, 14):  # traj_01 to traj_13
            draw_file = cond_path / f"draw_{file_idx:02d}.csv"
            traj_file = cond_path / f"traj_{file_idx:02d}.csv"

            if not draw_file.exists() or not traj_file.exists():
                continue

            draw_df = pd.read_csv(draw_file, skiprows=2)
            traj_df = pd.read_csv(traj_file)

            # Completion Time
            if 'timestamp' in draw_df.columns:
                draw_df['timestamp'] = pd.to_datetime(draw_df['timestamp'], errors='coerce').dt.tz_localize(None)
                duration = (draw_df['timestamp'].max() - draw_df['timestamp'].min()).total_seconds()
            else:
                duration = np.nan

            # Area Difference
            draw_points = draw_df[['x', 'z']].values
            traj_points = traj_df[['x', 'z']].values
            area_diff = calculate_region_area(draw_points, traj_points)

            # Phase Mapping
            if 1 <= file_idx <= 5:
                traj_type = 'A'
                phase = 'Train'
            elif file_idx == 13:
                traj_type = 'A'
                phase = 'Test'
            elif 6 <= file_idx <= 10:
                traj_type = 'B'
                phase = 'Train'
            elif file_idx == 12:
                traj_type = 'B'
                phase = 'Test'
            elif file_idx == 11:
                traj_type = 'C'
                phase = 'Test'

            user_results.append({
                'user_id': user_path.name,
                'condition': cond[-1],   # A / B
                'file_id': file_idx,
                'traj_type': traj_type,
                'phase': phase,
                'area_diff': area_diff,
                'duration': duration
            })
    return pd.DataFrame(user_results)

# === Batch Analysis for All Users ===
all_users_results = []
for user_folder in sorted(USER_DIR.iterdir()):
    if user_folder.is_dir():
        df_user = analyze_user_area_duration(user_folder)
        if not df_user.empty:
            all_users_results.append(df_user)

if all_users_results:
    all_users_df = pd.concat(all_users_results, ignore_index=True)
else:
    all_users_df = pd.DataFrame()
    print("⚠️ No user data found. Please check file paths and formats!")

# === Compute Mean ± SD ===
metrics = ['area_diff', 'duration']
summary_list = []

if not all_users_df.empty:
    for metric in metrics:
        for ttype in ['A', 'B', 'C']:
            for phase in ['Train', 'Test']:
                subset = all_users_df[(all_users_df['traj_type']==ttype) & (all_users_df['phase']==phase)]
                mean_val = subset[metric].mean()
                std_val = subset[metric].std()
                summary_list.append({
                    'Metric': 'Trajectory MAE (mm)' if metric=='area_diff' else 'Completion Time (s)',
                    'Trajectory Type': f"Type {ttype}",
                    'Phase': phase,
                    'Mean ± SD': f"{mean_val:.2f} ± {std_val:.2f}"
                })

    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(SUMMARY_DIR / "summary_by_type.csv", index=False)
    print(summary_df)
else:
    print("⚠️ DataFrame is empty, unable to compute Mean ± SD")
