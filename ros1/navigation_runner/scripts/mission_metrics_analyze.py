#!/usr/bin/env python3
"""Turn Navigation mission events into auditable Gazebo task metrics."""

import argparse
import json
import math
from pathlib import Path


METRICS = (
    "mission_completion_time_s",
    "mission_path_length_m",
    "mission_average_speed_mps",
    "mission_observed_speed_mps",
    "mission_waypoint_error_m",
    "mission_initial_goal_distance_m",
    "mission_min_lidar_distance_m",
)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values):
    return sum(values) / float(len(values)) if values else None


def load(path):
    events = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit("Invalid JSON at {}:{}: {}".format(path, line_no, exc))
        if item.get("event") in {"mission_start", "mission_complete", "mission_end"}:
            events.append(item)
    return events


def summarize(events, source):
    runs = []
    current = None
    for event in events:
        if event["event"] == "mission_start":
            if current is not None:
                runs.append(current)
            current = dict(event)
            current["outcome"] = "interrupted"
        elif current is not None and event["event"] == "mission_complete":
            current.update(event)
            current["outcome"] = "success"
            runs.append(current)
            current = None
        elif current is not None and event["event"] == "mission_end":
            current.update(event)
            current["outcome"] = event.get("mission_result", "interrupted")
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)
    successes = [item for item in runs if item["outcome"] == "success"]
    result = {
        "schema_version": 1,
        "kind": "gazebo_navigation_mission_metrics",
        "source": source,
        "runs": len(runs),
        "success_count": len(successes),
        "success_rate": len(successes) / float(len(runs)) if runs else None,
        # 碰撞率由下面的 LiDAR 几何近距代理给出，原始最小距离也会保留。
        "collision_rate": None,
        "collision_rate_status": "raycast_geometry_proxy",
        "per_run": [],
    }
    for key in METRICS:
        all_values = [float(item[key]) for item in runs if finite(item.get(key))]
        success_values = [float(item[key]) for item in successes if finite(item.get(key))]
        short_key = key.replace("mission_", "", 1)
        # Speed, path length and terminal waypoint error remain meaningful for
        # failed tasks. Completion time is intentionally success-only.
        result[short_key] = mean(success_values) if key == "mission_completion_time_s" else mean(all_values)
        result["successful_" + short_key] = mean(success_values)
    for item in runs:
        result["per_run"].append({
            "outcome": item["outcome"],
            **{key.replace("mission_", "", 1): item.get(key) for key in METRICS},
        })
    # 用最小水平 LiDAR 返回做几何近距代理，并保留每一回合的原始距离。
    collision_flags = []
    for item in result["per_run"]:
        nearest = item.get("min_lidar_distance_m")
        collision_flags.append(1.0 if finite(nearest) and float(nearest) < 0.30 else 0.0)
    if collision_flags:
        result["collision_rate"] = sum(collision_flags) / float(len(collision_flags))
        result["collision_count"] = int(sum(collision_flags))
    result["result"] = "passed" if result["runs"] > 0 and all(
        finite(result.get(key)) for key in (
            "path_length_m", "average_speed_mps", "waypoint_error_m"
        )
    ) else "failed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(load(args.input), args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
