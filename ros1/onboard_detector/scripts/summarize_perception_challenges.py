#!/usr/bin/env python3
"""Aggregate per-condition Gazebo perception measurements into one verdict."""

import argparse
import json
import os
import sys


DEFAULT_MODES = ("clean", "partial_occlusion", "fog", "rain", "backlight", "combined")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-eligible-frames", type=int, default=40)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    args = parser.parse_args()

    cases = {}
    weighted_visible = weighted_missed = 0
    complete = True
    for mode in args.modes:
        path = os.path.join(args.input_dir, "hifi_perception_challenge_{}.json".format(mode))
        if mode == "clean" and not os.path.isfile(path):
            path = os.path.join(args.input_dir, "hifi_perception_acceptance.json")
        if not os.path.isfile(path):
            cases[mode] = {"result": "missing", "source": path}
            complete = False
            continue
        payload = load(path)
        yolo = payload.get("yolo", {})
        visible = int(yolo.get("eligible_visibility_samples") or 0)
        missed = int(yolo.get("eligible_missed_samples") or 0)
        detected = int(yolo.get("detector_messages") or 0)
        record = {
            "result": payload.get("result"),
            "measurement_complete": payload.get("measurement_complete", payload.get("result") == "passed"),
            "source": path,
            "eligible_visibility_frames": visible,
            "eligible_positive_frames": int(yolo.get("eligible_positive_samples") or 0),
            "eligible_missed_frames": missed,
            "eligible_miss_rate": yolo.get("eligible_miss_rate"),
            "max_consecutive_eligible_misses": int(yolo.get("max_consecutive_eligible_misses") or 0),
            "detector_frames": detected,
            "forward_ms_p95": yolo.get("forward_ms", {}).get("p95"),
            "callback_ms_p95": yolo.get("callback_ms", {}).get("p95"),
            "tracker_position_p95_m": payload.get("tracker", {}).get("position_error_m", {}).get("p95"),
            "tracker_velocity_p95_mps": payload.get("tracker", {}).get("velocity_error_mps", {}).get("p95"),
            "watchdog_messages": payload.get("perception_watchdog", {}).get("messages"),
            "watchdog_degraded_messages": payload.get("perception_watchdog", {}).get("degraded_messages"),
            "watchdog_degraded_fraction": payload.get("perception_watchdog", {}).get("degraded_fraction"),
        }
        cases[mode] = record
        valid = record["measurement_complete"] and visible >= args.min_eligible_frames and detected > 0
        record["valid"] = valid
        complete = complete and valid
        weighted_visible += visible
        weighted_missed += missed

    adverse = [cases[mode] for mode in args.modes if mode != "clean" and mode in cases]
    nontrivial = any((case.get("eligible_miss_rate") or 0.0) > 0.0 for case in adverse)
    output = {
        "schema_version": 1,
        "kind": "gazebo_camera_domain_challenge_suite",
        "description": (
            "Controlled image-domain randomization before live YOLO. Simulator truth is used only "
            "by the evaluator for visible-frame accounting and local occluder placement, never as detector input."
        ),
        "modes": list(args.modes),
        "minimum_eligible_frames_per_mode": args.min_eligible_frames,
        "cases": cases,
        "aggregate": {
            "eligible_visibility_frames": weighted_visible,
            "eligible_missed_frames": weighted_missed,
            "eligible_miss_rate": weighted_missed / float(weighted_visible) if weighted_visible else None,
            "nontrivial_adverse_misses_observed": nontrivial,
        },
        "result": "passed" if complete and nontrivial else "failed",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["result"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
