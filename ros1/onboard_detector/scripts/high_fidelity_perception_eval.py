#!/usr/bin/env python3
"""Evaluate the live Gazebo camera -> YOLO -> depth tracker chain against truth.

This evaluator subscribes to the actual simulated camera and model-state
streams.  It never publishes a detection, pose, or velocity.  The target
velocity truth is derived from successive Gazebo poses because scripted Gazebo
models may be teleported along a path and consequently report zero twist.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import rospy
import tf
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, PointStamped
from onboard_detector.srv import GetDynamicObstacles
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64
from vision_msgs.msg import Detection2DArray


def percentile(values, value):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * value
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summary(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


class Evaluator:
    def __init__(self, target_model, detector_only=False, color_topic="/camera/color/image_raw"):
        self.target_model = target_model
        self.detector_only = detector_only
        self.color_messages = 0
        self.depth_messages = 0
        self.positive_box_messages = 0
        self.detector_messages = 0
        self.out_of_view_positive_samples = 0
        self.eligible_visibility_samples = 0
        self.eligible_positive_samples = 0
        self.eligible_missed_samples = 0
        self.max_consecutive_eligible_misses = 0
        self.current_consecutive_eligible_misses = 0
        self.camera_info = None
        self.tf_listener = tf.TransformListener()
        self.forward_ms = []
        self.callback_ms = []
        self.current_truth = None
        self.last_truth = None
        self.samples = []
        self.last_service_error = None
        self.watchdog_messages = 0
        self.watchdog_degraded_messages = 0
        self.watchdog_last_degraded = None

        self.color_topic = color_topic
        rospy.Subscriber(color_topic, Image, self.on_color, queue_size=1)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.on_depth, queue_size=1)
        rospy.Subscriber("/yolo_detector/yolo_forward_time", Float64, self.on_forward, queue_size=10)
        rospy.Subscriber("/yolo_detector/yolo_time", Float64, self.on_callback, queue_size=10)
        rospy.Subscriber("/yolo_detector/perception_degraded", Bool, self.on_watchdog, queue_size=10)
        rospy.Subscriber("/yolo_detector/detected_bounding_boxes", Detection2DArray, self.on_boxes, queue_size=10)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.on_camera_info, queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.on_model_states, queue_size=10)

    def on_color(self, _message):
        self.color_messages += 1

    def on_depth(self, _message):
        self.depth_messages += 1

    def on_camera_info(self, message):
        self.camera_info = message

    def on_forward(self, message):
        if math.isfinite(message.data) and message.data > 0.0:
            self.forward_ms.append(message.data * 1000.0)

    def on_callback(self, message):
        if math.isfinite(message.data) and message.data >= 0.0:
            self.callback_ms.append(message.data * 1000.0)

    def on_watchdog(self, message):
        self.watchdog_messages += 1
        self.watchdog_last_degraded = bool(message.data)
        if self.watchdog_last_degraded:
            self.watchdog_degraded_messages += 1

    def on_boxes(self, message):
        self.detector_messages += 1
        positive = bool(message.detections)
        if positive:
            self.positive_box_messages += 1
        visible = self.target_is_visible()
        if visible:
            self.eligible_visibility_samples += 1
            if positive:
                self.eligible_positive_samples += 1
                self.current_consecutive_eligible_misses = 0
            else:
                self.eligible_missed_samples += 1
                self.current_consecutive_eligible_misses += 1
                self.max_consecutive_eligible_misses = max(
                    self.max_consecutive_eligible_misses,
                    self.current_consecutive_eligible_misses,
                )
        elif positive:
            self.out_of_view_positive_samples += 1

    def target_is_visible(self):
        """Define eligible frames from simulated truth, CameraInfo and TF."""
        if self.current_truth is None or self.camera_info is None:
            return False
        point = PointStamped()
        point.header.frame_id = "map"
        point.header.stamp = rospy.Time(0)
        point.point.x = self.current_truth["x"]
        point.point.y = self.current_truth["y"]
        point.point.z = self.current_truth["z"] + 0.9
        try:
            camera_point = self.tf_listener.transformPoint(
                self.camera_info.header.frame_id or "camera_link", point
            )
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return False
        z = float(camera_point.point.z)
        if not math.isfinite(z) or z <= 0.05:
            return False
        u = self.camera_info.K[0] * camera_point.point.x / z + self.camera_info.K[2]
        v = self.camera_info.K[4] * camera_point.point.y / z + self.camera_info.K[5]
        return (
            math.isfinite(u) and math.isfinite(v)
            and 0.0 <= u < self.camera_info.width
            and 0.0 <= v < self.camera_info.height
        )

    def on_model_states(self, message):
        try:
            index = message.name.index(self.target_model)
        except ValueError:
            return
        pose = message.pose[index].position
        now = time.monotonic()
        truth = {"x": pose.x, "y": pose.y, "z": pose.z, "time": now}
        if self.last_truth is not None:
            dt = now - self.last_truth["time"]
            if dt > 1e-3:
                truth["vx"] = (truth["x"] - self.last_truth["x"]) / dt
                truth["vy"] = (truth["y"] - self.last_truth["y"]) / dt
        self.last_truth = truth
        self.current_truth = truth

    def sample_tracker(self, service):
        if self.current_truth is None:
            return
        try:
            response = service(Point(0.0, 0.0, 0.0), 20.0)
        except rospy.ServiceException as exc:
            self.last_service_error = str(exc)
            return
        if not response.position:
            return
        target = self.current_truth
        candidates = []
        for position, velocity, size in zip(response.position, response.velocity, response.size):
            if not all(math.isfinite(item) for item in (
                position.x, position.y, position.z, velocity.x, velocity.y, velocity.z,
                size.x, size.y, size.z,
            )):
                continue
            position_error = math.sqrt((position.x - target["x"]) ** 2 + (position.y - target["y"]) ** 2)
            candidates.append((position_error, position, velocity))
        if not candidates:
            return
        position_error, position, velocity = min(candidates, key=lambda item: item[0])
        record = {
            "position_error_m": position_error,
            "truth": {key: target.get(key) for key in ("x", "y", "z", "vx", "vy")},
            "estimate": {"x": position.x, "y": position.y, "z": position.z, "vx": velocity.x, "vy": velocity.y},
        }
        if "vx" in target and "vy" in target:
            record["velocity_error_mps"] = math.sqrt((velocity.x - target["vx"]) ** 2 + (velocity.y - target["vy"]) ** 2)
        self.samples.append(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model", default="person1_0.5_0.5_1.8")
    parser.add_argument("--detector-only", action="store_true", help="run a no-target negative control")
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--challenge-mode", default="clean", help="label for a camera-domain challenge run")
    parser.add_argument("--allow-zero-positives", action="store_true", help="record a valid all-missed challenge without failing the harness")
    parser.add_argument("--duration", type=float, default=35.0)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--max-position-p95-m", type=float, default=1.50)
    parser.add_argument("--max-velocity-p95-mps", type=float, default=1.50)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("high_fidelity_perception_eval", anonymous=True)
    evaluator = Evaluator(args.target_model, detector_only=args.detector_only, color_topic=args.color_topic)
    result = {
        "schema_version": 1,
        "kind": "gazebo_high_fidelity_perception_truth_evaluation",
        "target_model": args.target_model,
        "evaluation_mode": "detector_only_no_target" if args.detector_only else "target_truth_tracking",
        "challenge_mode": args.challenge_mode,
        "color_topic": args.color_topic,
        "duration_s": args.duration,
        "limits": {
            "minimum_tracker_samples": args.min_samples,
            "position_p95_m": args.max_position_p95_m,
            "velocity_p95_mps": args.max_velocity_p95_mps,
            "truth_source": "/gazebo/model_states; velocity differentiated from pose",
        },
        "result": "failed",
    }

    try:
        rospy.wait_for_service("/onboard_detector/get_dynamic_obstacles", timeout=30.0)
        service = rospy.ServiceProxy("/onboard_detector/get_dynamic_obstacles", GetDynamicObstacles)
        deadline = time.monotonic() + args.duration
        rate = rospy.Rate(8.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            evaluator.sample_tracker(service)
            rate.sleep()

        position_errors = [record["position_error_m"] for record in evaluator.samples]
        velocity_errors = [record["velocity_error_mps"] for record in evaluator.samples if "velocity_error_mps" in record]
        result.update({
            "camera": {
                "color_messages": evaluator.color_messages,
                "depth_messages": evaluator.depth_messages,
            },
            "yolo": {
                "positive_box_messages": evaluator.positive_box_messages,
                "detector_messages": evaluator.detector_messages,
                "negative_box_messages": evaluator.detector_messages - evaluator.positive_box_messages,
                "out_of_view_positive_samples": evaluator.out_of_view_positive_samples,
                "eligible_visibility_samples": evaluator.eligible_visibility_samples,
                "eligible_positive_samples": evaluator.eligible_positive_samples,
                "eligible_missed_samples": evaluator.eligible_missed_samples,
                "eligible_miss_rate": (
                    evaluator.eligible_missed_samples / float(evaluator.eligible_visibility_samples)
                    if evaluator.eligible_visibility_samples else None
                ),
                "max_consecutive_eligible_misses": evaluator.max_consecutive_eligible_misses,
                "false_positive_assessment": "not_identifiable_single_target_scene",
                "false_positive_proxy_out_of_view_positive_samples": evaluator.out_of_view_positive_samples,
                "false_positive_frame_rate": (
                    evaluator.positive_box_messages / float(evaluator.detector_messages)
                    if evaluator.detector_messages else None
                ),
                "forward_ms": summary(evaluator.forward_ms),
                "callback_ms": summary(evaluator.callback_ms),
            },
            "tracker": {
                "matched_samples": len(evaluator.samples),
                "position_error_m": summary(position_errors),
                "velocity_error_mps": summary(velocity_errors),
                "last_service_error": evaluator.last_service_error,
            },
            "perception_watchdog": {
                "messages": evaluator.watchdog_messages,
                "degraded_messages": evaluator.watchdog_degraded_messages,
                "degraded_fraction": (
                    evaluator.watchdog_degraded_messages / float(evaluator.watchdog_messages)
                    if evaluator.watchdog_messages else None
                ),
                "last_degraded": evaluator.watchdog_last_degraded,
                "contract": "degraded=true requests position hold/zero velocity in navigation.py",
            },
            "sample_preview": evaluator.samples[:10],
        })
        if args.detector_only:
            result["yolo"]["false_positive_assessment"] = "estimated_no_target_control"
            result["yolo"]["false_positive_proxy_out_of_view_positive_samples"] = None
            passed = (
                evaluator.color_messages > 0
                and evaluator.depth_messages > 0
                and evaluator.detector_messages > 0
                and result["yolo"]["false_positive_frame_rate"] is not None
                and result["yolo"]["false_positive_frame_rate"] <= 0.05
            )
        else:
            passed = (
                evaluator.color_messages > 0
                and evaluator.depth_messages > 0
                and evaluator.positive_box_messages > 0
                and evaluator.detector_messages > 0
                and len(evaluator.samples) >= args.min_samples
                and position_errors and percentile(position_errors, 0.95) <= args.max_position_p95_m
                and velocity_errors and percentile(velocity_errors, 0.95) <= args.max_velocity_p95_mps
            )
            if args.allow_zero_positives:
                # A deliberately degraded image can produce zero detections;
                # that is a measured miss rate, not a broken ROS chain.
                passed = (
                    evaluator.color_messages > 0
                    and evaluator.depth_messages > 0
                    and evaluator.detector_messages > 0
                    and evaluator.eligible_visibility_samples > 0
                )
                result["measurement_complete"] = bool(passed)
        result["result"] = "passed" if passed else "failed"
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
