#!/usr/bin/env python3
"""Sample end-to-end long-run health without relying on simulated time."""

import argparse
import json
import os
import subprocess
import time

import rospy
from std_msgs.msg import Bool, Float64


class LongRunMonitor:
    def __init__(self, args):
        self.args = args
        self.latest_yolo_ms = None
        self.latest_degraded = None
        self.records = []
        rospy.Subscriber("/yolo_detector/yolo_time", Float64, self.on_yolo, queue_size=5)
        rospy.Subscriber("/yolo_detector/perception_degraded", Bool, self.on_degraded, queue_size=5)

    def on_yolo(self, message):
        # yolo_detector publishes wall time in seconds; the monitor stores
        # latency in milliseconds to match the live evaluator.
        value = float(message.data) * 1000.0
        self.latest_yolo_ms = value if value >= 0.0 else None

    def on_degraded(self, message):
        self.latest_degraded = bool(message.data)

    @staticmethod
    def rss_by_process():
        try:
            result = subprocess.run(
                ["ps", "-eo", "rss=,args="],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        groups = {
            "navigation": ("navigation_node.py",),
            "yolo": ("yolo_detector_node.py",),
            "gazebo": ("gzserver", "gzclient"),
            "detector": ("detector_node",),
        }
        totals = {key: 0.0 for key in groups}
        for line in result.stdout.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2 or not fields[0].isdigit():
                continue
            rss_mib = int(fields[0]) / 1024.0
            command = fields[1]
            for key, markers in groups.items():
                if any(marker in command for marker in markers):
                    totals[key] += rss_mib
        totals["stack"] = sum(totals.values())
        return {key: round(value, 4) for key, value in totals.items()}

    def run(self):
        started = time.monotonic()
        next_sample = started
        while not rospy.is_shutdown() and time.monotonic() - started < self.args.duration:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(0.05, next_sample - now))
                continue
            record = {
                "event": "long_run_health",
                "wall_offset_s": round(now - started, 4),
                "wall_epoch_s": round(time.time(), 6),
                "yolo_latency_ms": self.latest_yolo_ms,
                "watchdog_degraded": self.latest_degraded,
                "rss_mib": self.rss_by_process(),
            }
            self.records.append(record)
            next_sample += self.args.period
        output = {
            "schema_version": 1,
            "kind": "gazebo_ros_yolo_ppo_long_run_raw",
            "duration_target_s": self.args.duration,
            "sample_period_s": self.args.period,
            "records": self.records,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.args.output)), exist_ok=True)
        with open(self.args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print("wrote", self.args.output, "samples", len(self.records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--period", type=float, default=1.0)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("long_run_monitor")
    LongRunMonitor(args).run()


if __name__ == "__main__":
    main()
