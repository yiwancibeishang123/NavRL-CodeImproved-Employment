#!/usr/bin/env python3
"""Summarize, tabulate and plot the Gazebo ROS/YOLO/PPO soak-test evidence."""

import argparse
import csv
import json
import math
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    point = (len(values) - 1) * q
    low, high = int(math.floor(point)), int(math.ceil(point))
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (point - low)


def finite_values(records, path):
    values = []
    for record in records:
        value = record
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def line_points(records, value_path, width, x0, y0, height):
    values = finite_values(records, value_path)
    if not values or len(records) < 2:
        return ""
    maximum = max(values)
    minimum = min(values)
    spread = max(maximum - minimum, 1e-6)
    first = records[0].get("wall_offset_s", 0.0)
    last = max(records[-1].get("wall_offset_s", first), first + 1e-6)
    points = []
    for record in records:
        value = record
        for key in value_path:
            value = value.get(key) if isinstance(value, dict) else None
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        x = x0 + (float(record.get("wall_offset_s", first)) - first) / (last - first) * width
        y = y0 + height - (float(value) - minimum) / spread * height
        points.append("{:.2f},{:.2f}".format(x, y))
    return " ".join(points)


def write_svg(records, output):
    width, height = 960, 440
    memory = line_points(records, ("rss_mib", "stack"), 820, 90, 50, 130)
    # Keep the plot consistent with the summary: sub-millisecond timer ticks
    # mean that no new image arrived, rather than a detector forward pass.
    latency_records = [
        record for record in records
        if isinstance(record.get("yolo_latency_ms"), (int, float))
        and math.isfinite(float(record["yolo_latency_ms"]))
        and float(record["yolo_latency_ms"]) >= 1.0
    ]
    latency = line_points(latency_records, ("yolo_latency_ms",), 820, 90, 270, 130)
    svg = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"440\" viewBox=\"0 0 960 440\">
<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>
<text x=\"48\" y=\"30\" font-family=\"sans-serif\" font-size=\"18\">Gazebo ROS + YOLO + PPO 20 min soak test</text>
<text x=\"48\" y=\"68\" font-family=\"sans-serif\" font-size=\"13\">Stack RSS (MiB)</text>
<line x1=\"90\" y1=\"50\" x2=\"90\" y2=\"180\" stroke=\"#64748b\"/><line x1=\"90\" y1=\"180\" x2=\"910\" y2=\"180\" stroke=\"#64748b\"/>
<polyline points=\"{memory}\" fill=\"none\" stroke=\"#0f766e\" stroke-width=\"2\"/>
<text x=\"48\" y=\"288\" font-family=\"sans-serif\" font-size=\"13\">YOLO forward latency (ms)</text>
<line x1=\"90\" y1=\"270\" x2=\"90\" y2=\"400\" stroke=\"#64748b\"/><line x1=\"90\" y1=\"400\" x2=\"910\" y2=\"400\" stroke=\"#64748b\"/>
<polyline points=\"{latency}\" fill=\"none\" stroke=\"#b45309\" stroke-width=\"2\"/>
<text x=\"90\" y=\"425\" font-family=\"sans-serif\" font-size=\"12\">wall-clock time (s)</text>
</svg>""".format(memory=memory, latency=latency)
    Path(output).write_text(svg + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navigation", required=True)
    parser.add_argument("--health", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--minimum-duration", type=float, default=1200.0)
    args = parser.parse_args()
    navigation_entries = load_jsonl(args.navigation)
    navigation = [item for item in navigation_entries if item.get("event") == "control"]
    mission_starts = [item for item in navigation_entries if item.get("event") == "mission_start"]
    mission_completes = [item for item in navigation_entries if item.get("event") == "mission_complete"]
    health_payload = json.loads(Path(args.health).read_text(encoding="utf-8"))
    health = health_payload.get("records", [])
    duration = health[-1].get("wall_offset_s", 0.0) if health else 0.0
    yolo_all = finite_values(health, ("yolo_latency_ms",))
    # ``yolo_time`` is emitted every timer tick. A sub-millisecond sample is
    # the no-new-image bookkeeping path, not a detector forward pass.
    yolo = [value for value in yolo_all if value >= 1.0]
    stack = finite_values(health, ("rss_mib", "stack"))
    ppo = finite_values(navigation, ("inference_wall_ms",))
    summary = {
        "schema_version": 1,
        "kind": "gazebo_ros_yolo_ppo_long_run_summary",
        "wall_duration_s": duration,
        "monitor_samples": len(health),
        "navigation_control_records": len(navigation),
        "mission_starts": len(mission_starts),
        "mission_completions": len(mission_completes),
        "mission_completion_rate": len(mission_completes) / float(len(mission_starts)) if mission_starts else None,
        "yolo_forward_ms_p50": percentile(yolo, 0.50),
        "yolo_forward_ms_p95": percentile(yolo, 0.95),
        "yolo_forward_samples": len(yolo),
        "ppo_inference_ms_p50": percentile(ppo, 0.50),
        "ppo_inference_ms_p95": percentile(ppo, 0.95),
        "control_overrun_rate": (
            sum(bool(item.get("control_overrun")) for item in navigation) / float(len(navigation))
            if navigation else None
        ),
        "timing_degrade_rate": (
            sum(bool(item.get("timing_degrade_active")) for item in navigation) / float(len(navigation))
            if navigation else None
        ),
        "watchdog_degraded_rate": (
            sum(bool(item.get("watchdog_degraded")) for item in health if item.get("watchdog_degraded") is not None)
            / float(sum(item.get("watchdog_degraded") is not None for item in health))
            if any(item.get("watchdog_degraded") is not None for item in health) else None
        ),
        "stack_rss_mib_initial": stack[0] if stack else None,
        "stack_rss_mib_final": stack[-1] if stack else None,
        "stack_rss_mib_max": max(stack) if stack else None,
        "stack_rss_growth_mib": stack[-1] - stack[0] if len(stack) >= 2 else None,
    }
    summary["result"] = "passed" if (
        summary["wall_duration_s"] >= args.minimum_duration - 1.0
        and summary["monitor_samples"] >= max(2, int(args.minimum_duration * 0.8))
        and summary["navigation_control_records"] > 0
        and summary["yolo_forward_ms_p95"] is not None
        and summary["ppo_inference_ms_p95"] is not None
    ) else "failed"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "long_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "long_run_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wall_offset_s", "stack_rss_mib", "navigation_rss_mib", "yolo_rss_mib", "yolo_latency_ms", "watchdog_degraded"])
        for item in health:
            rss = item.get("rss_mib", {})
            writer.writerow([item.get("wall_offset_s"), rss.get("stack"), rss.get("navigation"), rss.get("yolo"), item.get("yolo_latency_ms"), item.get("watchdog_degraded")])
    write_svg(health, out_dir / "long_run_stability.svg")
    lines = ["# 长时稳定性验收", "", "| 指标 | 数值 |", "|---|---:|"]
    for key, value in summary.items():
        if key in {"schema_version", "kind"}:
            continue
        lines.append("| {} | {} |".format(key, "N/A" if value is None else value))
    (out_dir / "long_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
