#!/usr/bin/env python3
"""Summarize clean-sim, perturbed-sim, and optional physical JSONL logs.

Only values recorded in the supplied logs are reported. In particular,
omitting ``--physical`` does not estimate a physical-world success rate.
"""

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = (
    "inference_wall_ms",
    "inference_device_ms",
    "control_ms",
    "control_period_jitter_ms",
    "scheduler_lag_ms",
    "odom_age_s",
    "raycast_age_s",
    "obstacle_age_s",
    "action_delta_norm",
    "process_rss_mib",
    "cuda_allocated_mib",
    "cuda_reserved_mib",
)
TASK_METRICS = (
    "success_rate", "collision_rate", "completion_time_s", "path_length_m",
    "average_speed_mps", "waypoint_error_m",
)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def metric_value(record, key):
    # Logs created before the profiling update used only ``inference_ms``.
    if key == "inference_wall_ms" and key not in record:
        return record.get("inference_ms")
    return record.get(key)


def numeric_values(records, key):
    values = []
    for record in records:
        value = metric_value(record, key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def load_jsonl(path):
    entries = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit("Invalid JSON at {}:{}: {}".format(path, line_no, exc))
        if not isinstance(entry, dict):
            raise SystemExit("Expected JSON object at {}:{}".format(path, line_no))
        entries.append(entry)
    records = [entry for entry in entries if entry.get("event") == "control"]
    if not records:
        raise SystemExit("No control records found in {}".format(path))
    metadata = [entry for entry in entries if entry.get("event") == "run_start"]
    return records, (metadata[-1] if metadata else {})


def load_task_summary(path):
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Task summary must be a JSON object: {}".format(path))
    result = {}
    for key in TASK_METRICS:
        value = payload.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[key] = float(value)
    return result


def _rate(records, predicate):
    return sum(bool(predicate(record)) for record in records) / float(len(records)) if records else None


def _rate_for_key(records, key):
    eligible = [record for record in records if key in record and isinstance(record[key], bool)]
    return sum(record[key] for record in eligible) / float(len(eligible)) if eligible else None


def _time_span(records):
    timestamps = numeric_values(records, "timestamp")
    if len(timestamps) < 2:
        return None
    duration = max(timestamps) - min(timestamps)
    return duration if duration > 0.0 else None


def summarize(records, task_summary, control_period_s=None):
    commands = [record for record in records if record.get("status") == "command"]
    duration_s = _time_span(records)
    command_duration_s = _time_span(commands)
    policy_records = [
        record for record in records
        if record.get("inference_mode") in ("local_policy", "policy_service_round_trip")
    ]
    if not policy_records:
        # Compatibility with legacy logs that do not carry inference_mode.
        policy_records = [record for record in records if (metric_value(record, "inference_wall_ms") or 0.0) > 0.0]

    summary = {
        "control_records": len(records),
        "command_records": len(commands),
        "policy_inference_records": len(policy_records),
        "hold_rate": _rate(records, lambda record: record.get("status") == "hold"),
        "stale_observation_hold_rate": _rate(
            records, lambda record: record.get("reason") == "stale_or_incomplete_observation"
        ),
        "safety_intervention_rate": _rate(commands, lambda record: record.get("safety_intervention")),
        "control_overrun_rate": _rate_for_key(records, "control_overrun"),
        "timing_degrade_rate": _rate_for_key(records, "timing_degrade_active"),
        "timing_degrade_hold_rate": _rate(
            records, lambda record: record.get("reason") == "control_timing_degraded"
        ),
        "control_rate_hz": None if duration_s is None else (len(records) - 1) / duration_s,
        "command_rate_hz": None if command_duration_s is None else (len(commands) - 1) / command_duration_s,
    }
    if control_period_s and control_period_s > 0.0:
        summary["configured_control_rate_hz"] = 1.0 / control_period_s
    for key in METRICS:
        values = numeric_values(records, key)
        summary[key + "_p50"] = percentile(values, 0.50)
        summary[key + "_p95"] = percentile(values, 0.95)
        summary[key + "_max"] = max(values) if values else None

    inference_values = numeric_values(policy_records, "inference_wall_ms")
    summary["serial_policy_throughput_hz"] = (
        None if not inference_values else 1000.0 / (sum(inference_values) / len(inference_values))
    )
    summary.update(task_summary)
    return summary


def comparison_to_clean(groups):
    clean = groups["clean_sim"]["summary"]
    comparisons = {}
    for label, group in groups.items():
        if label == "clean_sim":
            continue
        result = {}
        for key, value in group["summary"].items():
            baseline = clean.get(key)
            if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
                continue
            if not math.isfinite(value) or not math.isfinite(baseline):
                continue
            result[key] = {
                "absolute": value - baseline,
                "relative_percent": None if baseline == 0.0 else (value - baseline) / abs(baseline) * 100.0,
            }
        comparisons[label] = result
    return comparisons


def format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return str(value)


def report_metadata(metadata):
    runtime = metadata.get("runtime", {}) if isinstance(metadata, dict) else {}
    if not isinstance(runtime, dict):
        runtime = {}
    keys = ("device", "cuda_device", "torch", "cuda_runtime", "platform", "python")
    return {key: runtime.get(key) for key in keys if runtime.get(key) is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True, help="clean simulation metrics JSONL")
    parser.add_argument("--perturbed", required=True, help="perturbed simulation metrics JSONL")
    parser.add_argument("--physical", help="physical deployment metrics JSONL, available after field test")
    parser.add_argument("--clean-task-summary", help="optional JSON with success_rate/collision_rate/etc.")
    parser.add_argument("--perturbed-task-summary", help="optional JSON with success_rate/collision_rate/etc.")
    parser.add_argument("--physical-task-summary", help="optional JSON with success_rate/collision_rate/etc.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    input_groups = {
        "clean_sim": (args.clean, args.clean_task_summary),
        "perturbed_sim": (args.perturbed, args.perturbed_task_summary),
    }
    if args.physical:
        input_groups["physical"] = (args.physical, args.physical_task_summary)

    groups = {}
    for label, (log_path, task_path) in input_groups.items():
        records, metadata = load_jsonl(log_path)
        period = metadata.get("control_period_s") if isinstance(metadata.get("control_period_s"), (int, float)) else None
        groups[label] = {
            "summary": summarize(records, load_task_summary(task_path), period),
            "run_metadata": report_metadata(metadata),
        }
    report = {
        "schema_version": 2,
        "groups": groups,
        "comparison_to_clean": comparison_to_clean(groups),
        "physical_log_supplied": bool(args.physical),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sim_real_gap_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    keys = sorted({key for group in groups.values() for key in group["summary"]})
    with (args.out_dir / "sim_real_gap_metrics.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["metric"] + list(groups))
        for key in keys:
            writer.writerow([key] + [format_value(groups[label]["summary"].get(key)) for label in groups])

    lines = [
        "# 仿真与实际应用差距汇总",
        "",
        "本文件由 `sim_real_gap_analyze.py` 自动生成。`N/A` 表示日志未提供该指标，不能据此推断性能。",
        "",
        "| 指标 | " + " | ".join(groups) + " |",
        "|---|" + "|".join(["---:"] * len(groups)) + "|",
    ]
    for key in keys:
        values = " | ".join(format_value(groups[label]["summary"].get(key)) for label in groups)
        lines.append("| {} | {} |".format(key, values))
    lines.extend(["", "## 运行环境", ""])
    for label, group in groups.items():
        metadata = group["run_metadata"]
        rendered = "; ".join("{}={}".format(key, value) for key, value in metadata.items()) or "未记录"
        lines.append("- {}: {}".format(label, rendered))
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- `inference_wall_ms` 是控制节点观察到的前向延迟；CUDA 下 `inference_device_ms` 仅是 GPU kernel 时间，两者不可互换。",
        "- `control_ms` 包含观测构造、安全服务和指令发布路径；`control_overrun_rate` 依据记录的控制周期判断。",
        "- `timing_degrade_rate`/`timing_degrade_hold_rate` 表示控制节点检测到连续超周期或 scheduler lag 超阈值后主动保持的记录比例；保持动作不复用迟到的 PPO 动作。",
        "- clean_sim 与 perturbed_sim 的差异量化可控噪声、丢包和时延敏感性。未传入 physical 日志时，报告不包含也不推断真机性能。",
        "- 成功率、碰撞率、完成时间和路径长度只能来自独立任务汇总 JSON，不能由控制日志替代。",
    ])
    (args.out_dir / "sim_real_gap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", args.out_dir / "sim_real_gap_summary.md")


if __name__ == "__main__":
    main()
