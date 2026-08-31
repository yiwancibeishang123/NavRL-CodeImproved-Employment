#!/usr/bin/env python3
"""Read-only PX4/MAVROS preflight checks; never arms or changes flight mode."""
import time

import rospy
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry


def main():
    rospy.init_node("navrl_px4_preflight_check", anonymous=True)
    state = {"msg": None}
    odom = {"msg": None}
    rospy.Subscriber("/mavros/state", State, lambda msg: state.__setitem__("msg", msg), queue_size=1)
    rospy.Subscriber("/mavros/local_position/odom", Odometry, lambda msg: odom.__setitem__("msg", msg), queue_size=1)
    timeout = rospy.get_param("~timeout", 5.0)
    # Wall time is deliberate: a MAVROS startup can set /use_sim_time while
    # /clock is absent or paused, and ROS time would then never reach timeout.
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if state["msg"] is not None and odom["msg"] is not None:
            break
        time.sleep(0.05)
    if state["msg"] is None:
        raise SystemExit("PX4 preflight failed: /mavros/state has no message")
    if odom["msg"] is None:
        raise SystemExit("PX4 preflight failed: /mavros/local_position/odom has no message")
    msg = state["msg"]
    pose = odom["msg"].pose.pose.position
    print("connected:", msg.connected)
    print("armed:", msg.armed)
    print("mode:", msg.mode)
    print("local_position:", pose.x, pose.y, pose.z)
    if not msg.connected:
        raise SystemExit("PX4 preflight failed: FCU is not connected")
    print("PX4/MAVROS read-only preflight passed")


if __name__ == "__main__":
    main()
