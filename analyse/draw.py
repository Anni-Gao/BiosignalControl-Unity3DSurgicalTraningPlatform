import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# ========== 1. Load data ==========
draw_df = pd.read_csv('draw_12.csv', skiprows=2)  # skip first two comment rows
traj_df = pd.read_csv('traj_12.csv')
draw_df['timestamp'] = pd.to_datetime(draw_df['timestamp'])

# ========== 2. Build KDTree for nearest distance matching ==========
traj_points = traj_df[['x', 'z']].values
traj_tree = cKDTree(traj_points)

stroke_mse = {}
stroke_stats = []

# Iterate over each stroke (works even if there’s only one stroke)
for stroke_id, stroke_data in draw_df.groupby('stroke'):
    x = stroke_data['x'].values
    y = stroke_data['z'].values
    stroke_points = np.column_stack((x, y))

    # nearest-point distances
    dists, _ = traj_tree.query(stroke_points)
    mse = np.mean(dists ** 2)
    stroke_mse[stroke_id] = mse

    # time stats
    t_start = stroke_data['timestamp'].min()
    t_end = stroke_data['timestamp'].max()
    duration = (t_end - t_start).total_seconds()
    point_count = len(stroke_data)

    stroke_stats.append({
        'stroke': stroke_id,
        'mse': mse,
        'start_time': t_start,
        'end_time': t_end,
        'duration_sec': duration,
        'draw_speed': point_count / duration if duration > 0 else np.nan
    })

# Convert to DataFrame
stroke_stats_df = pd.DataFrame(stroke_stats)

# ===== 3. Visualization (normal offset, fixed width) =====

def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def _tangents_2d(P):
    """P: (N,2) polyline -> unit tangents (N,2)"""
    N = len(P)
    T = np.zeros_like(P)
    if N == 1:
        return T
    if N == 2:
        d = P[1]-P[0]
        d = _normalize(d[None,:])[0]
        T[0] = d; T[1] = d
        return T
    T[1:-1] = _normalize(P[2:] - P[:-2])
    T[0]  = _normalize((P[1]-P[0])[None,:])[0]
    T[-1] = _normalize((P[-1]-P[-2])[None,:])[0]
    return T

def _normals_from_tangents_2d(T):
    """Rotate +90° to get left normals"""
    N = np.stack([-T[:,1], T[:,0]], axis=1)
    return _normalize(N)

def ribbon_from_polyline_2d(P, width, miter_limit=4.0):
    """
    P: (N,2) centerline points; width: scalar or (N,)
    Returns left/right border vertices L, R (each (N,2))
    """
    P = np.asarray(P, float)
    Np = P.shape[0]
    w = np.full(Np, float(width)) if np.isscalar(width) else np.asarray(width, float)
    hw = 0.5 * w

    T = _tangents_2d(P)
    N = _normals_from_tangents_2d(T)

    # compute miter normals, clamp with miter_limit
    Nm = N.copy()
    if Np >= 3:
        for i in range(1, Np-1):
            n0, n1 = N[i-1], N[i+1]
            n_avg = _normalize((n0 + n1)[None,:])[0]
            if np.dot(n_avg, N[i]) < 0:  # avoid flip
                n_avg = -n_avg
            denom = max(abs(np.dot(n_avg, N[i])), 1e-6)
            miter_scale = min(1.0/denom, miter_limit)
            Nm[i] = n_avg * miter_scale

    L = P + Nm * hw[:,None]
    R = P - Nm * hw[:,None]
    return L, R

def ribbon_polygon(L, R):
    """Join left/right edges into a closed polygon"""
    return np.vstack([L, R[::-1]])

# ---- Plot ----
plt.figure(figsize=(8, 6))

# each stroke with fixed width
fixed_width = 0.20  # adjust as needed (same unit as x/z)
for stroke_id, stroke_data in draw_df.groupby('stroke'):
    P = stroke_data[['x','z']].to_numpy()
    L, R = ribbon_from_polyline_2d(P, fixed_width, miter_limit=4.0)
    poly = ribbon_polygon(L, R)
    # draw filled ribbon and centerline
    plt.fill(poly[:,0], poly[:,1], alpha=0.30, linewidth=0, label=None)
    plt.plot(P[:,0], P[:,1], linewidth=2, label=f'Stroke {stroke_id}')

# reference trajectory with larger width
ref_width = 0.5
Pref = traj_df[['x','z']].to_numpy()
Lr, Rr = ribbon_from_polyline_2d(Pref, ref_width, miter_limit=4.0)
poly_ref = ribbon_polygon(Lr, Rr)
plt.fill(poly_ref[:,0], poly_ref[:,1], alpha=0.20, label=None)
plt.plot(Pref[:,0], Pref[:,1], 'k--', linewidth=2, label='Reference Trajectory')

plt.xlabel('X')
plt.ylabel('Z (Y)')
plt.title('Trajectory Comparison (constant width, normal-offset)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
