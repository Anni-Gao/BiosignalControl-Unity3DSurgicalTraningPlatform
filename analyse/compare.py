import pandas as pd
import numpy as np
from pathlib import Path
from dateutil import parser
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns

# === Path Config ===
USER_DIR = Path("user_data")
OUTPUT_DIR = Path("output")
(OUTPUT_DIR / "users").mkdir(parents=True, exist_ok=True)

CONDITIONS = ['condition_A', 'condition_B']


def load_csv_file(csv_path, columns):
    """Load a CSV file with robust timestamp parsing"""
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


# === Trajectory comparison functions ===
def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _tangents_2d(P):
    N = len(P)
    T = np.zeros_like(P)
    if N == 1:
        return T
    if N == 2:
        d = P[1] - P[0]
        d = _normalize(d[None, :])[0]
        T[0] = d
        T[1] = d
        return T
    T[1:-1] = _normalize(P[2:] - P[:-2])
    T[0] = _normalize((P[1] - P[0])[None, :])[0]
    T[-1] = _normalize((P[-1] - P[-2])[None, :])[0]
    return T


def _normals_from_tangents_2d(T):
    Nn = np.stack([-T[:, 1], T[:, 0]], axis=1)
    return _normalize(Nn)


def ribbon_from_polyline_2d(P, width, miter_limit=4.0):
    P = np.asarray(P, float)
    Np = P.shape[0]
    w = np.full(Np, float(width)) if np.isscalar(width) else np.asarray(width, float)
    hw = 0.5 * w
    T = _tangents_2d(P)
    Nn = _normals_from_tangents_2d(T)
    Nm = Nn.copy()
    if Np >= 3:
        for i in range(1, Np - 1):
            n0, n1 = Nn[i - 1], Nn[i + 1]
            n_avg = _normalize((n0 + n1)[None, :])[0]
            if np.dot(n_avg, Nn[i]) < 0:
                n_avg = -n_avg
            denom = max(abs(np.dot(n_avg, Nn[i])), 1e-6)
            miter_scale = min(1.0 / denom, miter_limit)
            Nm[i] = n_avg * miter_scale
    L = P + Nm * hw[:, None]
    R = P - Nm * hw[:, None]
    return L, R


def ribbon_polygon(L, R):
    return np.vstack([L, R[::-1]])


def to_valid_polygon(coords):
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def ribbon_overlap_stats(P1, P2, width1, width2=None, miter_limit=4.0):
    if width2 is None:
        width2 = width1

    L1, R1 = ribbon_from_polyline_2d(P1, width1, miter_limit=miter_limit)
    L2, R2 = ribbon_from_polyline_2d(P2, width2, miter_limit=miter_limit)

    A = to_valid_polygon(ribbon_polygon(L1, R1))
    B = to_valid_polygon(ribbon_polygon(L2, R2))

    I = A.intersection(B)
    U = A.union(B)
    A_u = A.difference(B)
    B_u = B.difference(A)
    S = A.symmetric_difference(B)

    stats = {
        "area_A": A.area,
        "area_B": B.area,
        "overlap_area": I.area,
        "unique_A_area": A_u.area,
        "unique_B_area": B_u.area,
        "symmetric_diff_area": S.area,
        "union_area": U.area,
        "IoU": (I.area / U.area) if U.area > 0 else np.nan,
    }
    return stats, A, B, I, A_u, B_u, U, S


def calculate_region_mse(P_draw, P_traj, width_draw=0.20, width_traj=0.20, miter_limit=4.0):
    if len(P_draw) == 0 or len(P_traj) == 0:
        return {
            "area_A": 0,
            "area_B": 0,
            "overlap_area": 0,
            "unique_A_area": 0,
            "unique_B_area": 0,
            "symmetric_diff_area": 0,
            "union_area": 0,
            "IoU": np.nan,
        }
    stats, *_ = ribbon_overlap_stats(P_draw, P_traj, width_draw, width_traj, miter_limit)
    return stats


# === User Analysis ===
def analyze_user(user_path: Path):
    print(f"\n=== Analyzing user: {user_path.name} ===")

    user_output_dir = OUTPUT_DIR / "users" / user_path.name
    user_output_dir.mkdir(parents=True, exist_ok=True)

    eeg_df = load_csv_file(user_path / "eeg.csv", ["timestamp", "attention", "stress", "action"])
    hrv_df = load_csv_file(user_path / "hrv.csv", ["timestamp", "rmssd", "average", "action"])

    # HRV outlier removal
    if "rmssd" in hrv_df.columns:
        Q1 = hrv_df["rmssd"].quantile(0.25)
        Q3 = hrv_df["rmssd"].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        lower_bound = max(0, Q1 - 3 * IQR)
        hrv_df = hrv_df[(hrv_df["rmssd"] >= lower_bound) & (hrv_df["rmssd"] <= upper_bound)]

    all_strokes = []       # EEG / HRV per stroke
    trajectory_status = [] # trajectory-level stats

    for cond in CONDITIONS:
        cond_path = user_path / cond
        if not cond_path.exists():
            print(f"⚠️ Missing folder: {cond_path}")
            continue

        for file_idx in range(1, 14):
            draw_file = cond_path / f"draw_{file_idx:02d}.csv"
            traj_file = cond_path / f"traj_{file_idx:02d}.csv"

            if not draw_file.exists() or not traj_file.exists():
                print(f"⚠️ Missing file: {draw_file} or {traj_file}")
                continue

            draw_df = pd.read_csv(draw_file, skiprows=2)
            draw_df["timestamp"] = pd.to_datetime(draw_df["timestamp"], errors="coerce").dt.tz_localize(None)

            traj_df = pd.read_csv(traj_file)
            traj_points = traj_df[["x", "z"]].values
            traj_tree = cKDTree(traj_points)

            if draw_df.empty:
                traj_start_time, traj_end_time = np.nan, np.nan
            else:
                traj_start_time = draw_df["timestamp"].min()
                traj_end_time = draw_df["timestamp"].max()

            completion_time = (
                (traj_end_time - traj_start_time).total_seconds()
                if pd.notna(traj_end_time) and pd.notna(traj_start_time)
                else np.nan
            )
            if completion_time > 50:
                completion_time = 50.0

            stroke_num = draw_df["stroke"].nunique()

            if draw_df.empty:
                end_dist = np.inf
            else:
                draw_end_point = draw_df[["x", "z"]].values[-1]
                traj_end_point = traj_points[-1]
                end_dist = np.linalg.norm(draw_end_point - traj_end_point)

            draw_points = draw_df[["x", "z"]].values if not draw_df.empty else np.empty((0, 2))

            # === Trajectory-level area difference ===
            region_stats = calculate_region_mse(draw_points, traj_points,
                                                width_draw=0.2, width_traj=0.2)
            mse_out_area = region_stats["symmetric_diff_area"]
            out_area = region_stats["unique_A_area"]

            trajectory_status.append({
                "user_id": user_path.name,
                "condition": cond[-1],
                "file_id": file_idx,
                "stroke_count": stroke_num,
                "completion_time_s": completion_time,
                "out_region_mse": mse_out_area,
                "out_region_area": out_area,
                "end_point_dist": end_dist,
                "group": "Train" if file_idx <= 10 else "Test",
            })

            # === EEG / HRV per stroke ===
            for stroke_id, stroke_data in draw_df.groupby("stroke"):
                stroke_points = stroke_data[["x", "z"]].values
                if len(stroke_points) < 2:
                    continue

                t_start = stroke_data["timestamp"].min()
                t_end = stroke_data["timestamp"].max()

                eeg_slice = eeg_df[(eeg_df["timestamp"] >= t_start) & (eeg_df["timestamp"] <= t_end)]
                hrv_slice = hrv_df[(hrv_df["timestamp"] >= t_start) & (hrv_df["timestamp"] <= t_end)]

                all_strokes.append({
                    "user_id": user_path.name,
                    "condition": cond[-1],
                    "file_id": file_idx,
                    "stroke": stroke_id,
                    "attention": eeg_slice["attention"].mean() if not eeg_slice.empty else np.nan,
                    "hrv_rmssd": hrv_slice["rmssd"].mean() if not hrv_slice.empty else np.nan,
                    "group": "Train" if file_idx <= 10 else "Test",
                })

    # === Save results ===
    user_stats = pd.DataFrame(all_strokes)
    stroke_count_df = pd.DataFrame(trajectory_status)

    user_stats.to_csv(user_output_dir / f"summary_{user_path.name}.csv", index=False)
    stroke_count_df.to_csv(user_output_dir / f"trajectory_status_{user_path.name}.csv", index=False)

    if not user_stats.empty:
        plot_trajectory_stats(stroke_count_df, user_stats, user_path.name, user_output_dir)

    return user_stats


# === Visualization ===
def plot_trajectory_stats(stroke_count_df, user_stats, user_id, save_dir):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True, constrained_layout=True)

    # ---- Stroke Count ----
    ax = axes[0]
    sns.lineplot(data=stroke_count_df, x="file_id", y="stroke_count",
                 hue="condition", marker="o", ax=ax)
    ax.set_title(f"User {user_id} - Stroke Count by Condition and Trajectory")
    ax.set_ylabel("Stroke Count")
    ax.set_xlim(1, 13)

    # ---- Completion Time vs out_region_mse ----
    ax = axes[1]
    sns.lineplot(data=stroke_count_df, x="file_id", y="out_region_mse",
                 hue="condition", marker="o", ax=ax)
    ax.set_title("Out Region MSE (Trajectory-level)")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Trajectory File ID")
    ax.set_xlim(1, 13)

    plt.savefig(save_dir / f"{user_id}_trajectory_stats.png", dpi=300)
    plt.close()


# === Multi-user batch analysis ===
all_users_stats = []
for user_folder in sorted(USER_DIR.iterdir()):
    if user_folder.is_dir():
        user_result = analyze_user(user_folder)
        all_users_stats.append(user_result)

summary_dir = OUTPUT_DIR / "summary"
summary_dir.mkdir(exist_ok=True)

if all_users_stats:
    all_users_df = pd.concat(all_users_stats, ignore_index=True)
    all_users_df.to_csv(summary_dir / "summary_compare_all_users.csv", index=False)

trajectory_status_files = list((OUTPUT_DIR / "users").rglob("trajectory_status_*.csv"))
if trajectory_status_files:
    all_status_dfs = []
    for f in trajectory_status_files:
        df = pd.read_csv(f)
        all_status_dfs.append(df)
    if all_status_dfs:
        all_status_df = pd.concat(all_status_dfs, ignore_index=True)
        all_status_df.to_csv(summary_dir / "summary_trajectory_status.csv", index=False)

print("\nAnalysis complete ✅ Data exported to output/users/{user_id} and output/summary")
