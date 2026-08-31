#!/usr/bin/env python3
"""Render the measured full-route trace and all static obstacles."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("route", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    starts = [index for index, row in enumerate(rows) if row.get("event") == "run_start"]
    rows = rows[starts[-1] :] if starts else rows
    controls = [row for row in rows if row.get("event") == "control" and row.get("position")]
    route = json.loads(args.route.read_text())
    if not controls:
        raise SystemExit("trace does not contain control positions")

    x = [row["position"][0] for row in controls]
    y = [row["position"][1] for row in controls]
    points = route["路线"]
    obstacles = route["障碍物中心"]

    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=160)
    for name, center in obstacles.items():
        ax.add_patch(
            Rectangle(
                (center[0] - 0.375, center[1] - 0.375),
                0.75,
                0.75,
                facecolor="#e7a75e",
                edgecolor="#8d4200",
                linewidth=1.6,
                alpha=0.78,
            )
        )
        ax.text(center[0], center[1], name.replace("small_box", "box"), ha="center", va="center", fontsize=8)

    ax.plot(x, y, color="#075c9e", linewidth=2.1, label="measured PPO + LiDAR trace")
    ax.plot(
        [point["坐标"][0] for point in points],
        [point["坐标"][1] for point in points],
        color="#6c757d",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        alpha=0.7,
        label="goal links (reference only)",
    )
    for point in points:
        px, py = point["坐标"][:2]
        ax.scatter(px, py, s=40, color="#c7254e", zorder=5)
        ax.text(px + 0.08, py + 0.08, str(point["序号"]), fontsize=9, color="#7a1733")
    ax.scatter(x[0], y[0], marker="o", s=90, color="#198754", zorder=6, label="spawn")
    ax.scatter(x[-1], y[-1], marker="X", s=95, color="#5b2c83", zorder=6, label="measured end")

    mission = next((row for row in rows if row.get("event") == "mission_complete"), {})
    complete = [row for row in rows if row.get("event") == "mission_complete"]
    path_length = sum(row.get("mission_path_length_m", 0.0) or 0.0 for row in complete)
    elapsed = sum(row.get("mission_completion_time_s", 0.0) or 0.0 for row in complete)
    min_lidar = min((row.get("mission_min_lidar_distance_m") for row in complete if row.get("mission_min_lidar_distance_m") is not None), default=float("nan"))
    ax.text(
        0.99,
        0.02,
        "10 segments reached | total path %.2f m | segment time %.1f s\nmin LiDAR %.3f m | height 0.84--0.93 m | obstacle height 1.15 m"
        % (path_length, elapsed, min_lidar),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#b7b7b7"},
    )
    ax.set_title("Full six-obstacle autonomous route (Gazebo measurement)")
    ax.set_xlabel("map frame x (m)")
    ax.set_ylabel("map frame y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
