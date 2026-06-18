#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PATH_CSV = "/home/dxr-labtop/pyroki/for_tips/logs/path_log.csv"
OBS_CSV = "/home/dxr-labtop/pyroki/for_tips/logs/obstacle_log.csv"
OUT_DIR = "/home/dxr-labtop/pyroki/for_tips/logs/plots"

os.makedirs(OUT_DIR, exist_ok=True)

SWAP_XY = False
NEGATE_X = True
NEGATE_Y = True

def map_xy(x, y):
    xp, yp = (y, x) if SWAP_XY else (x, y)
    if NEGATE_X:
        xp = -xp
    if NEGATE_Y:
        yp = -yp
    return xp, yp

df = pd.read_csv(PATH_CSV)
obs_df = pd.read_csv(OBS_CSV) if os.path.exists(OBS_CSV) else pd.DataFrame()

if df.empty:
    raise RuntimeError("path_log.csv is empty")

plan_ids = sorted(
    {c.split("_")[1] for c in df.columns if c.startswith("plan_") and c.endswith("_x")},
    key=lambda s: int(s)
)

def get_last_plan(df):
    if not plan_ids:
        return None
    last = df.iloc[-1]
    pts = []
    for i in plan_ids:
        xcol = f"plan_{i}_x"
        ycol = f"plan_{i}_y"
        zcol = f"plan_{i}_z"
        if all(c in df.columns for c in [xcol, ycol, zcol]):
            if pd.notna(last[xcol]) and pd.notna(last[ycol]) and pd.notna(last[zcol]):
                pts.append([float(last[xcol]), float(last[ycol]), float(last[zcol])])
    return np.array(pts, dtype=float) if pts else None

last_plan = get_last_plan(df)

def nearest_mean_distance(df):
    req = {"wp_x", "wp_y", "wp_z", "actual_x", "actual_y", "actual_z"}
    if not req.issubset(df.columns):
        return np.nan, None

    wp = df[["wp_x", "wp_y", "wp_z"]].dropna().drop_duplicates().to_numpy(dtype=float)
    actual = df[["actual_x", "actual_y", "actual_z"]].dropna().to_numpy(dtype=float)

    actual[:, 0] *= -1.0
    actual[:, 1] *= -1.0

    if len(wp) == 0 or len(actual) == 0:
        return np.nan, None

    nearest_rows = []
    dists = []
    for i, w in enumerate(wp):
        dist_vec = np.linalg.norm(actual - w[None, :], axis=1)
        j = int(np.argmin(dist_vec))
        d = float(dist_vec[j])
        dists.append(d)
        nearest_rows.append({
            "wp_idx": i,
            "wp_x": w[0], "wp_y": w[1], "wp_z": w[2],
            "closest_actual_x": actual[j, 0],
            "closest_actual_y": actual[j, 1],
            "closest_actual_z": actual[j, 2],
            "nearest_dist": d,
        })

    nearest_df = pd.DataFrame(nearest_rows)
    return float(np.mean(dists)), nearest_df

mean_nn_dist, nearest_df = nearest_mean_distance(df)

if nearest_df is not None:
    nearest_df.to_csv(os.path.join(OUT_DIR, "nearest_waypoint_actual.csv"), index=False)

metrics = pd.DataFrame([{
    "mean_nearest_wp_to_actual_dist_m": mean_nn_dist,
    "num_waypoint_samples": int(df[["wp_x", "wp_y", "wp_z"]].dropna().drop_duplicates().shape[0]) if {"wp_x","wp_y","wp_z"}.issubset(df.columns) else 0,
    "num_actual_samples": int(df[["actual_x", "actual_y", "actual_z"]].dropna().shape[0]) if {"actual_x","actual_y","actual_z"}.issubset(df.columns) else 0,
}])
metrics.to_csv(os.path.join(OUT_DIR, "path_metrics.csv"), index=False)

def latest_obs_table(obs_df):
    if obs_df.empty:
        return pd.DataFrame()
    needed = {"object_id", "class_name", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "t"}
    if not needed.issubset(obs_df.columns):
        return pd.DataFrame()
    return obs_df.sort_values("t").groupby("object_id", as_index=False).last()

latest_obs = latest_obs_table(obs_df)

def add_obstacles_xy(fig, latest_obs):
    if latest_obs.empty:
        return
    x, y = latest_obs["center_x"], latest_obs["center_y"]
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        name="Obstacle centers",
        text=latest_obs["class_name"],
        textposition="top center",
        marker=dict(size=10, symbol="x")
    ))
    for _, r in latest_obs.iterrows():
        cx, cy = r["center_x"], r["center_y"]
        sx = float(r["size_y"] if SWAP_XY else r["size_x"])
        sy = float(r["size_x"] if SWAP_XY else r["size_y"])
        x0 = cx - sx / 2.0
        x1 = cx + sx / 2.0
        y0 = cy - sy / 2.0
        y1 = cy + sy / 2.0
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(width=2), fillcolor="rgba(200,0,0,0.15)")

def add_obstacles_xz(fig, latest_obs):
    if latest_obs.empty:
        return
    fig.add_trace(go.Scatter(
        x=latest_obs["center_x"], y=latest_obs["center_z"],
        mode="markers+text",
        name="Obstacle centers",
        text=latest_obs["class_name"],
        textposition="top center",
        marker=dict(size=10, symbol="x")
    ))
    for _, r in latest_obs.iterrows():
        cx, cz = r["center_x"], r["center_z"]
        sx = float(r["size_x"])
        sz = float(r["size_z"])
        fig.add_shape(type="rect",
                      x0=cx - sx/2.0, y0=cz - sz/2.0,
                      x1=cx + sx/2.0, y1=cz + sz/2.0,
                      line=dict(width=2), fillcolor="rgba(200,0,0,0.15)")

def add_obstacles_yz(fig, latest_obs):
    if latest_obs.empty:
        return
    fig.add_trace(go.Scatter(
        x=latest_obs["center_y"], y=latest_obs["center_z"],
        mode="markers+text",
        name="Obstacle centers",
        text=latest_obs["class_name"],
        textposition="top center",
        marker=dict(size=10, symbol="x")
    ))
    for _, r in latest_obs.iterrows():
        cy, cz = r["center_y"], r["center_z"]
        sy = float(r["size_y"])
        sz = float(r["size_z"])
        fig.add_shape(type="rect",
                      x0=cy - sy/2.0, y0=cz - sz/2.0,
                      x1=cy + sy/2.0, y1=cz + sz/2.0,
                      line=dict(width=2), fillcolor="rgba(200,0,0,0.15)")

# XY
fig_xy = go.Figure()
if {"actual_x", "actual_y"}.issubset(df.columns):
    x, y = map_xy(df["actual_x"], df["actual_y"])
    fig_xy.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Actual path"))
if {"wp_x", "wp_y"}.issubset(df.columns):
    fig_xy.add_trace(go.Scatter(
        x=df["wp_x"], y=df["wp_y"], mode="lines+markers",
        name="Nominal path", line=dict(dash="dash")
    ))
if last_plan is not None:
    fig_xy.add_trace(go.Scatter(
        x=last_plan[:, 0], y=last_plan[:, 1],
        mode="lines+markers", name="Planned path (last cycle)"
    ))
add_obstacles_xy(fig_xy, latest_obs)
fig_xy.update_layout(title=f"XY path with obstacles | mean nearest dist={mean_nn_dist:.4f} m")
fig_xy.update_xaxes(title_text="X (m)")
fig_xy.update_yaxes(title_text="Y (m)", scaleanchor="x", scaleratio=1)
fig_xy.write_image(os.path.join(OUT_DIR, "path_xy.png"))

# XZ
fig_xz = go.Figure()
if {"actual_x", "actual_z"}.issubset(df.columns):
    x, y = map_xy(df["actual_x"], df["actual_y"])
    fig_xz.add_trace(go.Scatter(
        x=x, y=df["actual_z"], mode="lines+markers", name="Actual path"
    ))
if {"wp_x", "wp_z"}.issubset(df.columns):
    fig_xz.add_trace(go.Scatter(
        x=df["wp_x"], y=df["wp_z"], mode="lines+markers",
        name="Nominal path", line=dict(dash="dash")
    ))
if last_plan is not None:
    fig_xz.add_trace(go.Scatter(
        x=last_plan[:, 0], y=last_plan[:, 2],
        mode="lines+markers", name="Planned path (last cycle)"
    ))
add_obstacles_xz(fig_xz, latest_obs)
fig_xz.update_layout(title=f"XZ path with obstacles | mean nearest dist={mean_nn_dist:.4f} m")
fig_xz.update_xaxes(title_text="X (m)")
fig_xz.update_yaxes(title_text="Z (m)", scaleanchor="x", scaleratio=1)
fig_xz.write_image(os.path.join(OUT_DIR, "path_xz.png"))

# YZ
fig_yz = go.Figure()
if {"actual_y", "actual_z"}.issubset(df.columns):
    x, y = map_xy(df["actual_x"], df["actual_y"])
    fig_yz.add_trace(go.Scatter(
        x=y, y=df["actual_z"], mode="lines+markers", name="Actual path"
    ))
if {"wp_y", "wp_z"}.issubset(df.columns):
    fig_yz.add_trace(go.Scatter(
        x=df["wp_y"], y=df["wp_z"], mode="lines+markers",
        name="Nominal path", line=dict(dash="dash")
    ))
if last_plan is not None:
    fig_yz.add_trace(go.Scatter(
        x=last_plan[:, 1], y=last_plan[:, 2],
        mode="lines+markers", name="Planned path (last cycle)"
    ))
add_obstacles_yz(fig_yz, latest_obs)
fig_yz.update_layout(title=f"YZ path with obstacles | mean nearest dist={mean_nn_dist:.4f} m")
fig_yz.update_xaxes(title_text="Y (m)")
fig_yz.update_yaxes(title_text="Z (m)", scaleanchor="x", scaleratio=1)
fig_yz.write_image(os.path.join(OUT_DIR, "path_yz.png"))

# 3D
fig_3d = go.Figure()
if {"actual_x", "actual_y", "actual_z"}.issubset(df.columns):
    x, y = map_xy(df["actual_x"], df["actual_y"])
    fig_3d.add_trace(go.Scatter3d(
        x=x, y=y, z=df["actual_z"],
        mode="lines+markers", name="Actual path"
    ))
if {"wp_x", "wp_y", "wp_z"}.issubset(df.columns):
    fig_3d.add_trace(go.Scatter3d(
        x=df["wp_x"], y=df["wp_y"], z=df["wp_z"],
        mode="lines+markers", name="Nominal path"
    ))
if last_plan is not None:
    fig_3d.add_trace(go.Scatter3d(
        x=last_plan[:, 0], y=last_plan[:, 1], z=last_plan[:, 2],
        mode="lines+markers", name="Planned path (last cycle)"
    ))
if not latest_obs.empty:
    fig_3d.add_trace(go.Scatter3d(
        x=latest_obs["center_x"], y=latest_obs["center_y"], z=latest_obs["center_z"],
        mode="markers+text", name="Obstacle centers",
        text=latest_obs["class_name"], textposition="top center"
    ))
fig_3d.update_layout(
    title=f"3D path with obstacles | mean nearest dist={mean_nn_dist:.4f} m",
    scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)")
)
fig_3d.write_image(os.path.join(OUT_DIR, "path_3d.png"))

summary = {
    "mean_nearest_wp_to_actual_dist_m": None if np.isnan(mean_nn_dist) else mean_nn_dist,
    "plots": [
        os.path.join(OUT_DIR, "path_xy.png"),
        os.path.join(OUT_DIR, "path_xz.png"),
        os.path.join(OUT_DIR, "path_yz.png"),
        os.path.join(OUT_DIR, "path_3d.png"),
    ],
}
with open(os.path.join(OUT_DIR, "plot_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Saved plots to:", OUT_DIR)
print("Mean nearest waypoint-to-actual distance [m]:", mean_nn_dist)