#!/usr/bin/env python3
"""Render a compact, reproducible acceptance plot from a navigation JSONL trace."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def load_trace(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    controls = [row for row in rows if row.get("event") == "control" and row.get("position")]
    mission = next((row for row in rows if row.get("event") == "mission_complete"), None)
    return controls, mission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    controls, mission = load_trace(args.trace)
    if not controls or mission is None:
        raise SystemExit("trace does not contain a position trace and mission_complete record")

    x = [row["position"][0] for row in controls]
    y = [row["position"][1] for row in controls]
    start = (x[0], y[0])
    final_pose = (x[-1], y[-1])
    goal = (mission.get("goal_position") or [3.5, 0.855335, 1.0])[:2]

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=150)
    ax.add_patch(Rectangle((1.56144, 0.555335), 0.6, 0.6,
                           facecolor="#efb366", edgecolor="#9c4e00",
                           linewidth=2.0, alpha=0.75, label="static obstacle (1.15 m high)"))
    ax.plot(x, y, color="#0b5fa5", linewidth=2.2, label="closed-loop flight trace")
    ax.scatter(*start, s=80, color="#1b8a3e", zorder=5, label="spawn (0, 0)")
    ax.scatter(*goal, s=90, marker="*", color="#c32828", zorder=5, label="goal (3.5, 0.855)")
    ax.scatter(*final_pose, s=58, marker="D", color="#63358c", zorder=5,
               label="final pose (within 0.30 m tolerance)")
    ax.add_patch(Circle(goal, mission.get("mission_goal_tolerance_m", 0.3),
                        fill=False, linestyle="--", linewidth=1.2,
                        edgecolor="#c32828", alpha=0.65))

    # Mark every tenth recorded control point to make the direction obvious.
    step = max(1, len(x) // 18)
    ax.scatter(x[::step], y[::step], s=10, color="#4aa3df", alpha=0.65, zorder=3)
    ax.annotate("start", start, xytext=(6, 8), textcoords="offset points")
    ax.annotate("goal", goal, xytext=(6, 8), textcoords="offset points")
    ax.annotate("final", final_pose, xytext=(6, -14), textcoords="offset points")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x in map frame (m)")
    ax.set_ylabel("y in map frame (m)")
    ax.set_title("Low-altitude PPO + LiDAR autonomous obstacle avoidance")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.99,
        0.02,
        "height command: 1.0 m | obstacle top: 1.15 m | "
        "min LiDAR: %.3f m\npath: %.3f m | observed speed: %.3f m/s | error: %.3f m"
        % (
            mission.get("mission_min_lidar_distance_m") or float("nan"),
            mission.get("mission_path_length_m") or float("nan"),
            mission.get("mission_observed_speed_mps") or float("nan"),
            mission.get("mission_waypoint_error_m") or float("nan"),
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#b7b7b7"},
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
