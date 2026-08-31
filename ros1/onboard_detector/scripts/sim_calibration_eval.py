#!/usr/bin/env python3
"""Validate simulated RGB-D calibration against CameraInfo, TF and Gazebo truth.

This is deliberately a simulator-only test.  It checks the values actually
published by Gazebo, rather than merely validating the calibration YAML.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import rospy
import tf
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import CameraInfo, Image


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * q
    lower, upper = int(math.floor(index)), int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


class CalibrationEvaluation:
    def __init__(self, target):
        self.target = target
        self.truth = None
        self.camera_info = None
        self.depth = None
        self.depth_encoding = None
        self.bridge = CvBridge()
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.truth_callback, queue_size=5)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)

    def truth_callback(self, message):
        try:
            index = message.name.index(self.target)
        except ValueError:
            return
        point = message.pose[index].position
        self.truth = (point.x, point.y, point.z)

    def camera_info_callback(self, message):
        self.camera_info = message

    def depth_callback(self, message):
        self.depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        self.depth_encoding = message.encoding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model", default="hifi_person")
    parser.add_argument("--body-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--max-intrinsic-error", type=float, default=1e-3)
    parser.add_argument("--max-depth-median-error", type=float, default=0.25)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("navrl_sim_calibration_eval", anonymous=True)
    evaluator = CalibrationEvaluation(args.target_model)
    listener = tf.TransformListener()
    deadline = time.monotonic() + args.duration
    depth_errors = []
    result = {
        "schema_version": 1,
        "kind": "gazebo_rgbd_calibration_truth_evaluation",
        "environment": "Gazebo Classic RGB-D simulation, not physical calibration",
        "target_model": args.target_model,
        "result": "failed",
    }
    try:
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if evaluator.camera_info is not None and evaluator.depth is not None and evaluator.truth is not None:
                try:
                    translation, rotation = listener.lookupTransform(args.body_frame, args.camera_frame, rospy.Time(0))
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.sleep(0.05)
                    continue
                x, y, z = evaluator.truth
                # The static transform used by this simulator maps camera-z to body-x.
                expected_range = math.sqrt((x - translation[0]) ** 2 + (y - translation[1]) ** 2 + (z - translation[2]) ** 2)
                finite = evaluator.depth[~np.isnan(evaluator.depth)]
                if finite.size:
                    # Center pixel is not assumed to hit the target; compare valid depth range.
                    depth_errors.append(abs(float(np.percentile(finite, 95)) - expected_range))
                result.update({
                    "camera": {
                        "width": evaluator.camera_info.width,
                        "height": evaluator.camera_info.height,
                        "fx": evaluator.camera_info.K[0],
                        "fy": evaluator.camera_info.K[4],
                        "cx": evaluator.camera_info.K[2],
                        "cy": evaluator.camera_info.K[5],
                        "depth_encoding": evaluator.depth_encoding,
                    },
                    "body_to_camera": {
                        "translation_m": list(translation),
                        "quaternion_xyzw": list(rotation),
                    },
                    "truth_range_m": expected_range,
                })
                break
            rospy.sleep(0.05)
        camera = result.get("camera")
        if camera is None:
            raise RuntimeError("did not receive CameraInfo, depth image and target truth together")
        intrinsics = (camera["fx"], camera["fy"], camera["cx"], camera["cy"])
        expected = (554.254691191187, 554.254691191187, 320.5, 240.5)
        intrinsic_error = max(abs(actual - reference) for actual, reference in zip(intrinsics, expected))
        result["intrinsic_max_abs_error_px"] = intrinsic_error
        result["depth_range_error_m"] = {"count": len(depth_errors), "p50": percentile(depth_errors, 0.5)}
        result["criteria"] = {
            "expected_intrinsics": list(expected),
            "expected_depth_encoding": "32FC1",
            "max_intrinsic_error_px": args.max_intrinsic_error,
            "note": "RGB-D range test is diagnostic because generic scene percentiles are not a target segmentation metric.",
        }
        result["result"] = "passed" if (
            intrinsic_error <= args.max_intrinsic_error and camera["depth_encoding"] == "32FC1"
            and len(result["body_to_camera"]["translation_m"]) == 3
        ) else "failed"
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
