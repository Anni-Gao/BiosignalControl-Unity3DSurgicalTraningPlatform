import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from pathlib import Path
from dateutil import parser

# === Path configuration ===
USER_DIR = Path("user_data")
OUTPUT_DIR = Path("output")
(OUTPUT_DIR / "users").mkdir(parents=True, exist_ok=True)
CONDITIONS = ['condition_A', 'condition_B']


def load_csv_file(csv_path, columns):
    """ Load csv file with fault-tolerant timestamp parsing """
    df = pd.read_csv(csv_path)
    df.columns = columns

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception:
        df['timestamp'] = pd.NaT

    if df['timestamp'].isna().any():
        parsed_times = []
        for ts in df['timestamp'].astype(str):
            try:
                parsed_times.append(parser.parse(ts))
            except Exception:
                parsed_times.append(pd.NaT)
        df['timestamp'] = pd.to_datetime(parsed_times, errors='coerce')

    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df


def calculate_region_area(draw_points, traj_points, width_draw=0.2, width_traj=0.2):
    """ Compute symmetric difference area between drawn trajectory and reference trajectory (trajectory-level) """
    if len(draw_points) < 2 or len(traj_points) < 2:
        return np.nan
    draw_line = LineString(draw_points)
    traj_line = LineString(traj_points)
    draw_buffer = draw_line.buffer(width_draw, cap_style=2, join_style=2)
    traj_buffer = traj_line.buffer(width_traj, cap_style=2, join_style=2)
    return draw_buffer.symmetric_difference(traj_buffer).area


def count_pauses(action_series):
    """ Count the number of move → stop transitions """
    if action_series.empty:
        return 0
    return ((action_series.shift(1) == 'move') & (action_series == 'stop')).sum()


def remove_outliers(df, column, k=1.5):
    """ Remove outliers using IQR method """
    if df[column].isna().all():
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


def analyze_user(user_path: Path):
    print(f"\n=== Analyzing user: {user_path.name} ===")

    user_output_dir = OUTPUT_DIR / "users" / user_path.name
    user_output_dir.mkdir(parents=True, exist_ok=True)

    eeg_df = load_csv_file(user_path / "eeg.csv", ["timestamp", "attention", "stress", "action"])
    hrv_df = load_csv_file(user_path / "hrv.csv", ["timestamp", "rmssd", "average", "action"])

    # Remove HRV outliers
    if "rmssd" in hrv_df.columns:
        Q1 = hrv_df['rmssd'].quantile(0.25)
        Q3 = hrv_df['rmssd'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        lower_bound = max(0, Q1 - 3 * IQR)
        hrv_df = hrv_df[(hrv_df['rmssd'] >= lower_bound) & (hrv_df['rmssd'] <= upper_bound)]

    all_strokes = []       # stroke-level data (keep EEG/HRV/pause)
    trajectory_stats = []  # trajectory-level data (area_diff + completion time)

    for cond in CONDITIONS:
        cond_path = user_path / cond
        if not cond_path.exists():
            continue

        for file_idx in range(1, 14):  # 13 trajectories per group
            group = "train" if file_idx <= 10 else "test"
            draw_file = cond_path / f"draw_{file_idx:02d}.csv"
            traj_file = cond_path / f"traj_{file_idx:02d}.csv"

            if not draw_file.exists() or not traj_file.exists():
                continue

            draw_df = pd.read_csv(draw_file, skiprows=2)
            draw_df['timestamp'] = pd.to_datetime(draw_df['timestamp'], errors='coerce').dt.tz_localize(None)
            traj_df = pd.read_csv(traj_file)
            traj_points = traj_df[['x', 'z']].values

            traj_pause_count = 0

            # Iterate over strokes (count pause + EEG/HRV, no separate area_diff here)
            for stroke_id, stroke_data in draw_df.groupby('stroke'):
                stroke_points = stroke_data[['x', 'z']].values
                if len(stroke_points) < 2:
                    continue

                # time window
                t_start = stroke_data['timestamp'].min()
                t_end = stroke_data['timestamp'].max()

                eeg_slice = eeg_df[(eeg_df['timestamp'] >= t_start) & (eeg_df['timestamp'] <= t_end)]
                hrv_slice = hrv_df[(hrv_df['timestamp'] >= t_start) & (hrv_df['timestamp'] <= t_end)]

                # pause
                if cond in ['condition_A', 'condition_C']:
                    pause_slice = hrv_slice['action'] if not hrv_slice.empty else pd.Series([])
                else:
                    pause_slice = eeg_slice['action'] if not eeg_slice.empty else pd.Series([])

                stroke_pause_count = count_pauses(pause_slice)
                traj_pause_count += stroke_pause_count

                all_strokes.append({
                    'user_id': user_path.name,
                    'condition': cond[-1],
                    'group': group,
                    'file_id': file_idx,
                    'stroke': stroke_id,
                    'attention': eeg_slice['attention'].mean() if not eeg_slice.empty else np.nan,
                    'hrv_rmssd': hrv_slice['rmssd'].mean() if not hrv_slice.empty else np.nan,
                    'pause_count': stroke_pause_count
                })

            # Trajectory-level metrics: whole trajectory area_diff + total pause + completion time
            if not draw_df.empty:
                completion_time = (draw_df['timestamp'].max() - draw_df['timestamp'].min()).total_seconds()
            else:
                completion_time = np.nan

            draw_points = draw_df[['x', 'z']].values
            area_diff = calculate_region_area(draw_points, traj_points, width_draw=0.2, width_traj=0.2)

            trajectory_stats.append({
                'user_id': user_path.name,
                'condition': cond[-1],
                'group': group,
                'file_id': file_idx,
                'pause_count': traj_pause_count,
                'area_diff': area_diff,
                'completion_time': completion_time
            })

    # Save stroke-level data
    user_stats = pd.DataFrame(all_strokes)
    user_stats.to_csv(user_output_dir / f"summary_{user_path.name}.csv", index=False)

    # Save trajectory-level data
    traj_df = pd.DataFrame(trajectory_stats)
    traj_df.to_csv(user_output_dir / f"trajectory_summary_{user_path.name}.csv", index=False)

    return traj_df


# === Batch analysis ===
all_users_traj = []
for user_folder in sorted(USER_DIR.iterdir()):
    if user_folder.is_dir():
        user_result = analyze_user(user_folder)
        all_users_traj.append(user_result)

summary_dir = OUTPUT_DIR / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

if all_users_traj:
    all_users_df = pd.concat(all_users_traj, ignore_index=True)
    all_users_df.to_csv(summary_dir / "summary_all_users.csv", index=False)

    # Condition A / B comparison
    results = []
    for cond in ['A', 'B']:
        df_cond = all_users_df[all_users_df['condition'] == cond].copy()
        df_cond = remove_outliers(df_cond, 'area_diff')
        df_cond = remove_outliers(df_cond, 'pause_count')
        df_cond = remove_outliers(df_cond, 'completion_time')

        results.append({
            'condition': cond,
            'Average Trajectory MAE (mm)': round(df_cond['area_diff'].mean(), 2),
            'Average Completion Time (s)': round(df_cond['completion_time'].mean(), 2),
            'Average Pause Count': round(df_cond['pause_count'].mean(), 2)
        })

    summary_results = pd.DataFrame(results)
    summary_results.to_csv(summary_dir / "summary_condition_A_B.csv", index=False)
    print("\n=== Condition A vs B Summary ===")
    print(summary_results)

print("\nAnalysis complete ✅ Data exported to output folder")
