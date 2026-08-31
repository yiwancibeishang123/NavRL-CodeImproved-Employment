#!/usr/bin/env python3
"""Verify watchdog entry and automatic recovery on one live Gazebo chain."""

import argparse
import json
import os
import time

import rospy
from std_msgs.msg import Bool, String
from vision_msgs.msg import Detection2DArray


class RecoveryEvaluator:
    def __init__(self, args):
        self.args = args
        self.mode_pub = rospy.Publisher("/camera/challenge/set_mode", String, queue_size=2, latch=True)
        self.degraded = None
        self.transitions = []
        self.box_events = []
        rospy.Subscriber("/yolo_detector/perception_degraded", Bool, self.on_degraded, queue_size=10)
        rospy.Subscriber("/yolo_detector/detected_bounding_boxes", Detection2DArray, self.on_boxes, queue_size=10)

    def on_degraded(self, message):
        value = bool(message.data)
        if value != self.degraded:
            self.degraded = value
            self.transitions.append({"wall_time_s": time.monotonic(), "degraded": value})

    def on_boxes(self, message):
        self.box_events.append({
            "wall_time_s": time.monotonic(),
            "count": len(message.detections),
        })

    def publish_mode_for(self, mode, duration_s):
        started = time.monotonic()
        deadline = started + duration_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.mode_pub.publish(String(data=mode))
            rospy.sleep(0.25)
        return started, time.monotonic()

    def run(self):
        ready_deadline = time.monotonic() + self.args.ready_timeout
        while not rospy.is_shutdown() and self.degraded is None and time.monotonic() < ready_deadline:
            rospy.sleep(0.05)
        combined_start, combined_end = self.publish_mode_for("combined", self.args.combined_duration)
        clean_start, clean_end = self.publish_mode_for("clean", self.args.clean_duration)
        # A watchdog alarm may already be latched at the instant the
        # composite stream starts. Treat that as protection active during the
        # interval; only a clean transition after the mode switch counts as
        # recovery.
        degraded_during_combined = self.degraded is True or any(
            item["degraded"] and item["wall_time_s"] <= combined_end
            for item in self.transitions
        )
        recovery_times = [
            item["wall_time_s"] for item in self.transitions
            if not item["degraded"] and item["wall_time_s"] >= clean_start
        ]
        positive_after_recovery = [
            item for item in self.box_events
            if item["wall_time_s"] >= clean_start and item["count"] > 0
        ]
        recovered = bool(recovery_times) and self.degraded is False
        result = {
            "schema_version": 1,
            "kind": "gazebo_perception_watchdog_recovery",
            "challenge": {
                "combined_duration_s": round(combined_end - combined_start, 3),
                "clean_duration_s": round(clean_end - clean_start, 3),
                "require_detection": True,
            },
            "degraded_during_combined": degraded_during_combined,
            "automatic_recovery_after_clean": recovered,
            "recovery_latency_s": None if not recovery_times else round(recovery_times[0] - clean_start, 4),
            "positive_detection_events_after_clean": len(positive_after_recovery),
            "watchdog_transitions": [
                {"offset_s": round(item["wall_time_s"] - combined_start, 4), "degraded": item["degraded"]}
                for item in self.transitions
            ],
        }
        result["result"] = "passed" if (
            result["degraded_during_combined"]
            and result["automatic_recovery_after_clean"]
            and result["positive_detection_events_after_clean"] > 0
        ) else "failed"
        os.makedirs(os.path.dirname(os.path.abspath(self.args.output)), exist_ok=True)
        with open(self.args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["result"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--combined-duration", type=float, default=8.0)
    parser.add_argument("--clean-duration", type=float, default=12.0)
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("perception_watchdog_recovery_eval")
    return RecoveryEvaluator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
