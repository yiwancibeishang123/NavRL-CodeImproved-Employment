#!/usr/bin/env python3
"""Guard the YOLO stream and request a conservative navigation fallback.

The watchdog is deliberately independent of the detector implementation.  It
only observes published timing/heartbeat topics and can optionally treat a
run of empty detections as a fault when the mission declares that a target is
expected in the current corridor.  With the default ``require_detection``
false, an empty scene is not a fault.
"""

import json
import time

import rospy
from std_msgs.msg import Bool, Float64, String
from vision_msgs.msg import Detection2DArray


class PerceptionWatchdog:
    def __init__(self):
        self.latency_budget_s = max(0.001, float(rospy.get_param("~latency_budget_s", 0.045)))
        self.image_age_budget_s = max(0.01, float(rospy.get_param("~image_age_budget_s", 0.15)))
        self.heartbeat_timeout_s = max(0.05, float(rospy.get_param("~heartbeat_timeout_s", 0.35)))
        self.max_bad_cycles = max(1, int(rospy.get_param("~max_bad_cycles", 2)))
        self.clear_good_cycles = max(1, int(rospy.get_param("~clear_good_cycles", 3)))
        self.require_detection = bool(rospy.get_param("~require_detection", False))
        self.max_empty_cycles = max(1, int(rospy.get_param("~max_empty_cycles", 4)))
        self.last_yolo_time = rospy.Time(0)
        self.last_image_age = rospy.Time(0)
        self.last_boxes = rospy.Time(0)
        self.last_box_count = None
        self.bad_streak = 0
        self.good_streak = 0
        self.empty_streak = 0
        self.degraded = False
        self.reason = "startup"
        self.transition_count = 0
        self.last_publish = 0.0

        self.degraded_pub = rospy.Publisher("/yolo_detector/perception_degraded", Bool, queue_size=2, latch=True)
        self.status_pub = rospy.Publisher("/yolo_detector/perception_watchdog", String, queue_size=2, latch=True)
        rospy.Subscriber("/yolo_detector/yolo_time", Float64, self.on_yolo_time, queue_size=5)
        rospy.Subscriber("/yolo_detector/yolo_image_age", Float64, self.on_image_age, queue_size=5)
        rospy.Subscriber("/yolo_detector/detected_bounding_boxes", Detection2DArray, self.on_boxes, queue_size=5)
        rospy.Timer(rospy.Duration(0.05), self.evaluate)

    def on_yolo_time(self, message):
        self.last_yolo_time = rospy.Time.now()
        # yolo_time is published in seconds; keep a finite, non-negative value.
        value = float(message.data)
        self.yolo_latency_s = value if value >= 0.0 else float("inf")

    def on_image_age(self, message):
        self.last_image_age = rospy.Time.now()
        value = float(message.data)
        self.image_age_s = value if value >= 0.0 else float("inf")

    def on_boxes(self, message):
        self.last_boxes = rospy.Time.now()
        self.last_box_count = len(message.detections)
        if self.require_detection:
            self.empty_streak = self.empty_streak + 1 if self.last_box_count == 0 else 0

    @staticmethod
    def age(stamp):
        if stamp == rospy.Time(0):
            return float("inf")
        return max(0.0, (rospy.Time.now() - stamp).to_sec())

    def evaluate(self, _event):
        now = rospy.Time.now()
        yolo_age = self.age(self.last_yolo_time)
        image_age_msg_age = self.age(self.last_image_age)
        box_age = self.age(self.last_boxes)
        reasons = []
        if yolo_age > self.heartbeat_timeout_s or image_age_msg_age > self.heartbeat_timeout_s or box_age > self.heartbeat_timeout_s:
            reasons.append("detector_heartbeat_timeout")
        if getattr(self, "yolo_latency_s", float("inf")) > self.latency_budget_s:
            reasons.append("yolo_latency_budget")
        if getattr(self, "image_age_s", float("inf")) > self.image_age_budget_s:
            reasons.append("image_age_budget")
        if self.require_detection and self.empty_streak >= self.max_empty_cycles:
            reasons.append("consecutive_empty_detections")
        bad = bool(reasons)
        self.bad_streak = self.bad_streak + 1 if bad else 0
        self.good_streak = self.good_streak + 1 if not bad else 0
        if not self.degraded and self.bad_streak >= self.max_bad_cycles:
            self.degraded = True
            self.transition_count += 1
        elif self.degraded and self.good_streak >= self.clear_good_cycles:
            self.degraded = False
            self.transition_count += 1
        if reasons:
            self.reason = reasons[0]
        elif not self.degraded:
            self.reason = "healthy"
        payload = {
            "stamp": round(now.to_sec(), 6),
            "degraded": self.degraded,
            "reason": self.reason,
            "bad_streak": self.bad_streak,
            "empty_streak": self.empty_streak,
            "transition_count": self.transition_count,
            "require_detection": self.require_detection,
            "yolo_latency_s": getattr(self, "yolo_latency_s", None),
            "image_age_s": getattr(self, "image_age_s", None),
            "last_box_count": self.last_box_count,
            "yolo_heartbeat_age_s": None if yolo_age == float("inf") else round(yolo_age, 6),
        }
        self.degraded_pub.publish(Bool(data=self.degraded))
        # Keep status latched and human/machine-readable without flooding logs.
        if time.monotonic() - self.last_publish >= 0.1 or reasons:
            self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
            self.last_publish = time.monotonic()


def main():
    rospy.init_node("perception_watchdog")
    PerceptionWatchdog()
    rospy.spin()


if __name__ == "__main__":
    main()
