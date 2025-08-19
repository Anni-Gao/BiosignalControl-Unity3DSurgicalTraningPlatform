import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import LineString
from dateutil import parser

# === Path Configuration ===
USER_DIR = Path("user_data")
OUTPUT_DIR = Path("output")
(OUTPUT_DIR / "users").mkdir(parents=True, exist_ok=True)


def load_csv_file(csv_path, columns):
    """Load CSV file and robustly parse timestamps"""
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
    """Calculate the symmetric difference area between drawn trajectory and reference trajectory"""
    if len(draw_points) < 2 or len(traj_points) < 2:
        return np.nan
    draw_line = LineString(draw_points)
    traj_line = LineString(traj_points)
    draw_buffer = draw_line.buffer(width_draw, cap_style=2, join_style=2)
    traj_buffer = traj_line.buffer(width_traj, cap_style=2, join_style=2)
    return draw_buffer.symmetric_difference(traj_buffer).area


def count_pauses(action_series):
    """Count the number of move → stop transitions"""
    if action_series.empty:
        return 0
    return ((action_series.shift(1) == 'move') & (action_series == 'stop')).sum()


def remove_outliers(df, column, k=1.5):
    """Remove outliers using the IQR method"""
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

    # Read EEG and HRV
    eeg_df = load_csv_file(user_path / "eeg.csv", ["timestamp", "attention", "stress", "action"])
    hrv_df = load_csv_file(user_path / "hrv.csv", ["timestamp", "rmssd", "average", "action"])

    # HRV outlier removal
    if "rmssd" in hrv_df.columns:
        Q1 = hrv_df['rmssd'].quantile(0.25)
        Q3 = hrv_df['rmssd'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        lower_bound = max(0, Q1 - 3 * IQR)
        hrv_df = hrv_df[(hrv_df['rmssd'] >= lower_bound) & (hrv_df['rmssd'] <= upper_bound)]

    all_strokes = []

    for cond in ['condition_A', 'condition_B']:
        cond_path = user_path / cond
        if not cond_path.exists():
            continue

        for file_idx in range(1, 14):
            # Determine Type and Phase
            if 1 <= file_idx <= 5:
                ttype = 'A'
                phase = 'Train'
            elif file_idx == 13:
                ttype = 'A'
                phase = 'Test'
            elif 6 <= file_idx <= 10:
                ttype = 'B'
                phase = 'Train'
            elif file_idx == 12:
                ttype = 'B'
                phase = 'Test'
            elif file_idx == 11:
                ttype = 'C'
                phase = 'Test'

            draw_file = cond_path / f"draw_{file_idx:02d}.csv"
            traj_file = cond_path / f"traj_{file_idx:02d}.csv"
            if not draw_file.exists() or not traj_file.exists():
                continue

            draw_df = pd.read_csv(draw_file, skiprows=2)
            draw_df['timestamp'] = pd.to_datetime(draw_df['timestamp'], errors='coerce').dt.tz_localize(None)
            traj_df = pd.read_csv(traj_file)
            traj_points = traj_df[['x', 'z']].values

            traj_pause_count = 0
            traj_area_diffs = []

            # Iterate over strokes
            for stroke_id, stroke_data in draw_df.groupby('stroke'):
                stroke_points = stroke_data[['x', 'z']].values
                if len(stroke_points) < 2:
                    continue

                # Area difference
                area_diff = calculate_region_area(stroke_points, traj_points, width_draw=0.2, width_traj=0.5)
                traj_area_diffs.append(area_diff)

                t_start = stroke_data['timestamp'].min()
                t_end = stroke_data['timestamp'].max()

                eeg_slice = eeg_df[(eeg_df['timestamp'] >= t_start) & (eeg_df['timestamp'] <= t_end)]
                hrv_slice = hrv_df[(hrv_df['timestamp'] >= t_start) & (hrv_df['timestamp'] <= t_end)]

                # Pause count
                pause_slice = hrv_slice['action'] if not hrv_slice.empty else pd.Series([])

                stroke_pause_count = count_pauses(pause_slice)
                traj_pause_count += stroke_pause_count

            # Aggregate trajectory-level metrics
            completion_time = (draw_df['timestamp'].max() - draw_df['timestamp'].min()).total_seconds() if not draw_df.empty else np.nan
            all_strokes.append({
                'user_id': user_path.name,
                'type': ttype,
                'phase': phase,
                'file_id': file_idx,
                'Trajectory MAE': np.nanmean(traj_area_diffs) if traj_area_diffs else np.nan,
                'Completion Time': completion_time,
                'Scalpel Retraction Count': traj_pause_count,
                'HRV RMSSD': hrv_slice['rmssd'].mean() if not hrv_slice.empty else np.nan,
                'EEG Attention': eeg_slice['attention'].mean() if not eeg_slice.empty else np.nan
            })

    # Save user data
    user_stats = pd.DataFrame(all_strokes)
    user_stats.to_csv(user_output_dir / f"summary_{user_path.name}.csv", index=False)
    return user_stats


# === Batch Analysis ===
all_users_data = []
for user_folder in sorted(USER_DIR.iterdir()):
    if user_folder.is_dir():
        user_result = analyze_user(user_folder)
        all_users_data.append(user_result)

summary_dir = OUTPUT_DIR / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

if all_users_data:
    all_users_df = pd.concat(all_users_data, ignore_index=True)
    all_users_df.to_csv(summary_dir / "summary_all_users.csv", index=False)

    # Aggregate statistics by Type + Phase (Mean ± SD)
    metrics = ['Trajectory MAE', 'Completion Time', 'Scalpel Retraction Count', 'HRV RMSSD', 'EEG Attention']
    results = []
    for ttype in ['A', 'B', 'C']:
        for phase in ['Train', 'Test']:
            subset = all_users_df[(all_users_df['type'] == ttype) & (all_users_df['phase'] == phase)]
            row = {'Type': ttype, 'Phase': phase}
            for metric in metrics:
                mean_val = subset[metric].mean()
                std_val = subset[metric].std()
                row[metric] = f"{mean_val:.2f} ± {std_val:.2f}" if pd.notna(mean_val) else "nan ± nan"
            results.append(row)

    summary_metrics_df = pd.DataFrame(results)
    summary_metrics_df.to_csv(summary_dir / "summary_by_type_phase.csv", index=False)
    print("\n=== Summary by Type + Phase ===")
    print(summary_metrics_df)

print("\nAnalysis complete ✅ Data exported to output folder")
