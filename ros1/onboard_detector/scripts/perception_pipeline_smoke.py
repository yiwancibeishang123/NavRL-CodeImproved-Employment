#!/usr/bin/env python3
"""Record a runtime smoke test for the simulated camera-to-tracker pipeline.

The probe deliberately does not publish synthetic bounding boxes.  A zero
``person_detections`` result is therefore evidence of a live pipeline with no
recognised target in the current scene, rather than a successful tracking
accuracy test.
"""

import argparse
import json
import math
import os
import sys
import time

import rospy
from geometry_msgs.msg import Point
from onboard_detector.srv import GetDynamicObstacles
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from vision_msgs.msg import Detection2DArray


def wait_message(topic, msg_type, timeout):
    try:
        return rospy.wait_for_message(topic, msg_type, timeout=timeout)
    except rospy.ROSException as exc:
        raise RuntimeError("Timed out waiting for {}: {}".format(topic, exc))


def finite_vector(vector):
    return all(math.isfinite(value) for value in (vector.x, vector.y, vector.z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("perception_pipeline_smoke", anonymous=True)
    started = time.time()
    result = {
        "schema_version": 1,
        "test": "gazebo_camera_depth_yolo_dynamic_tracker_smoke",
        "started_at_unix_s": started,
        "result": "failed",
        "limits": {
            "note": "No synthetic bounding boxes are injected by this probe.",
            "accuracy_note": "Target detection/tracking accuracy needs labelled simulated or physical data.",
        },
    }

    try:
        color = wait_message("/camera/color/image_raw", Image, args.timeout)
        depth = wait_message("/camera/depth/image_raw", Image, args.timeout)
        forward = wait_message("/yolo_detector/yolo_forward_time", Float64, args.timeout)
        callback = wait_message("/yolo_detector/yolo_time", Float64, args.timeout)
        boxes = wait_message(
            "/yolo_detector/detected_bounding_boxes",
            Detection2DArray,
            args.timeout,
        )
        rospy.wait_for_service("/onboard_detector/get_dynamic_obstacles", timeout=args.timeout)
        service = rospy.ServiceProxy("/onboard_detector/get_dynamic_obstacles", GetDynamicObstacles)
        response = service(Point(0.0, 0.0, 0.0), 100.0)

        timing_values = (forward.data, callback.data)
        if not all(math.isfinite(value) and value >= 0.0 for value in timing_values):
            raise RuntimeError("YOLO timing topic contained invalid values: {}".format(timing_values))
        if forward.data <= 0.0:
            raise RuntimeError("YOLO forward time is zero; no image was processed")
        if not (len(response.position) == len(response.velocity) == len(response.size)):
            raise RuntimeError("Dynamic-obstacle service returned mismatched arrays")
        if not all(finite_vector(v) for v in response.position + response.velocity + response.size):
            raise RuntimeError("Dynamic-obstacle service returned non-finite values")

        result.update(
            {
                "result": "passed",
                "elapsed_s": time.time() - started,
                "color_image": {"width": color.width, "height": color.height, "encoding": color.encoding},
                "depth_image": {"width": depth.width, "height": depth.height, "encoding": depth.encoding},
                "yolo": {
                    "forward_time_ms": forward.data * 1000.0,
                    "callback_time_ms": callback.data * 1000.0,
                    "person_detections": len(boxes.detections),
                },
                "tracker": {
                    "service": "/onboard_detector/get_dynamic_obstacles",
                    "reported_targets": len(response.position),
                    "velocity_vectors_finite": True,
                },
            }
        )
    except Exception as exc:  # Keep a machine-readable diagnostic artifact.
        result["error"] = str(exc)
        result["elapsed_s"] = time.time() - started

    directory = os.path.dirname(os.path.abspath(args.output))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
