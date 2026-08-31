#!/usr/bin/env python3
"""Recompute the recovery verdict from an existing watchdog trace."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--combined-duration", type=float, default=8.0)
    args = parser.parse_args()
    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    transitions = result.get("watchdog_transitions", [])
    degraded = any(item.get("degraded") and float(item.get("offset_s", 0.0)) <= args.combined_duration for item in transitions)
    recovered = bool(result.get("automatic_recovery_after_clean"))
    positives = int(result.get("positive_detection_events_after_clean", 0))
    result["degraded_during_combined"] = degraded
    result["result"] = "passed" if degraded and recovered and positives > 0 else "failed"
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
