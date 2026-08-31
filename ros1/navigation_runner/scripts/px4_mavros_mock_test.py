#!/usr/bin/env python3
"""ROS-contract test for the read-only MAVROS preflight and hold setpoint.

This uses ROS publishers as an FCU stand-in.  It proves topic/message
compatibility only; it never connects to, arms, or commands a real vehicle.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import rospy
from mavros_msgs.msg import PositionTarget, State
from nav_msgs.msg import Odometry

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from navigation import Navigation


def publish_mock(state_pub, odom_pub, odom):
    state = State()
    state.connected = True
    state.armed = False
    state.mode = "MANUAL"
    state_pub.publish(state)
    odom.header.stamp = rospy.Time.now()
    odom_pub.publish(odom)


def wait_until(predicate, timeout_s, state_pub, odom_pub, odom):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not rospy.is_shutdown():
        publish_mock(state_pub, odom_pub, odom)
        if predicate():
            return True
        time.sleep(0.05)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rospy.init_node("navrl_px4_mavros_mock_test", anonymous=True)
    state_pub = rospy.Publisher("/mavros/state", State, queue_size=1)
    odom_pub = rospy.Publisher("/mavros/local_position/odom", Odometry, queue_size=1)
    captured = {"msg": None}
    rospy.Subscriber(
        "/mavros/setpoint_raw/local",
        PositionTarget,
        lambda message: captured.__setitem__("msg", message),
        queue_size=1,
    )

    odom = Odometry()
    odom.pose.pose.position.x = 1.25
    odom.pose.pose.position.y = -0.50
    odom.pose.pose.position.z = 0.90
    odom.pose.pose.orientation.w = 1.0

    preflight = subprocess.Popen(
        [sys.executable, str(SCRIPT_DIR / "px4_preflight_check.py"), "_timeout:=3.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    completed = wait_until(lambda: preflight.poll() is not None, 5.0, state_pub, odom_pub, odom)
    if not completed:
        preflight.terminate()
        preflight.wait(timeout=2.0)
        raise RuntimeError("mock preflight did not finish")
    stdout, stderr = preflight.communicate(timeout=1.0)
    if preflight.returncode != 0:
        raise RuntimeError("mock preflight failed: {}{}".format(stdout, stderr))

    nav = Navigation.__new__(Navigation)
    nav.odom_received = True
    nav.px4_control = True
    nav.odom = odom
    nav.auto_takeoff = False
    nav.action_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
    if not wait_until(lambda: nav.action_pub.get_num_connections() > 0, 2.0, state_pub, odom_pub, odom):
        raise RuntimeError("mock setpoint subscriber did not connect")
    nav.publish_stop()
    if not wait_until(lambda: captured["msg"] is not None, 2.0, state_pub, odom_pub, odom):
        raise RuntimeError("hold setpoint was not published")

    message = captured["msg"]
    ignored_velocity = message.IGNORE_VX | message.IGNORE_VY | message.IGNORE_VZ
    ignored_acceleration = message.IGNORE_AFX | message.IGNORE_AFY | message.IGNORE_AFZ
    ignored_heading = message.IGNORE_YAW | message.IGNORE_YAW_RATE
    if message.coordinate_frame != message.FRAME_LOCAL_NED:
        raise RuntimeError("hold command must use FRAME_LOCAL_NED")
    if (message.type_mask & (ignored_velocity | ignored_acceleration | ignored_heading)) != (
        ignored_velocity | ignored_acceleration | ignored_heading
    ):
        raise RuntimeError("hold command does not mask velocity/acceleration/yaw fields")
    position = message.position
    if any(abs(actual - expected) > 1e-6 for actual, expected in zip(
        (position.x, position.y, position.z),
        (odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z),
    )):
        raise RuntimeError("hold command position differs from current odometry")

    report = {
        "kind": "px4_mavros_mock_contract",
        "result": "passed",
        "preflight_stdout": stdout.strip(),
        "hold_setpoint": {
            "coordinate_frame": int(message.coordinate_frame),
            "type_mask": int(message.type_mask),
            "position": [position.x, position.y, position.z],
        },
        "boundary": "Mock ROS transport test only; no FCU was connected, armed, or commanded.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
