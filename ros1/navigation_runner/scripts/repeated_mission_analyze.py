#!/usr/bin/env python3
"""汇总多次 Gazebo 任务的均值和标准差。"""

import argparse
import json
import math
from pathlib import Path


def finite(values):
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]


def stats(values):
    values = finite(values)
    if not values:
        return {"mean": None, "std": None, "count": 0}
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return {"mean": avg, "std": math.sqrt(variance), "count": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="per-run mission JSON files")
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = []
    for filename in args.input:
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        per_run = payload.get("per_run", [])
        for index, raw_item in enumerate(per_run):
            item = dict(raw_item)
            item["source"] = filename
            item["episode_index"] = index + 1
            runs.append(item)
    fields = ("success_rate", "collision_rate", "average_speed_mps", "waypoint_error_m", "completion_time_s", "path_length_m")
    summary = {
        "schema_version": 1,
        "kind": "gazebo_repeated_mission_metrics",
        "group": args.group,
        "runs": len(runs),
        "success_rate": stats([1.0 if r.get("outcome") == "success" else 0.0 for r in runs]),
        "collision_rate": {"mean": None, "std": None, "count": 0, "status": "raycast_geometry_proxy"},
        "metrics": {},
        "per_run": runs,
    }
    for field in fields:
        if field == "collision_rate":
            continue
        values = [r.get(field) for r in runs]
        # Arrival time is meaningful only for successful episodes.
        if field == "completion_time_s":
            values = [r.get(field) for r in runs if r.get("outcome") == "success"]
        summary["metrics"][field] = stats(values)
    collision_values = [
        1.0 if isinstance(r.get("min_lidar_distance_m"), (int, float))
        and math.isfinite(float(r["min_lidar_distance_m"]))
        and float(r["min_lidar_distance_m"]) < 0.30 else 0.0
        for r in runs
    ]
    if collision_values:
        summary["collision_rate"] = {**stats(collision_values), "status": "raycast_geometry_proxy"}
    speed_target = 0.35 if args.group == "clean_sim" else 0.70
    waypoint_target = 1.0 if args.group == "clean_sim" else 1.5
    success_stats = summary["success_rate"]
    summary["acceptance"] = {
        "minimum_runs": len(runs) >= 5,
        "success_rate_ge_0_80": success_stats["mean"] is not None and success_stats["mean"] >= 0.80,
        "success_rate_std_lt_0_05": success_stats["std"] is not None and success_stats["std"] < 0.05,
        "collision_rate_lt_0_15": summary["collision_rate"]["mean"] is not None and summary["collision_rate"]["mean"] < 0.15,
        "average_speed_mps_ge_target": summary["metrics"]["average_speed_mps"]["mean"] is not None and summary["metrics"]["average_speed_mps"]["mean"] >= speed_target,
        "waypoint_error_m_le_target": summary["metrics"]["waypoint_error_m"]["mean"] is not None and summary["metrics"]["waypoint_error_m"]["mean"] <= waypoint_target,
        "speed_target_mps": speed_target,
        "waypoint_target_m": waypoint_target,
    }
    summary["result"] = "passed" if all(summary["acceptance"].get(k, False) for k in (
        "minimum_runs", "success_rate_ge_0_80", "success_rate_std_lt_0_05",
        "collision_rate_lt_0_15", "average_speed_mps_ge_target", "waypoint_error_m_le_target"
    )) else "failed"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
