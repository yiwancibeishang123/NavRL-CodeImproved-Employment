#!/usr/bin/env python3
"""Publish a repeatable wall-clock patrol route for soak testing."""

import argparse
import time

import rospy
from geometry_msgs.msg import PoseStamped


def parse_route(value):
    points = []
    for item in value.split(";"):
        x, y = item.split(",")
        points.append((float(x), float(y)))
    if not points:
        raise ValueError("route must contain at least one x,y waypoint")
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, required=True, help="wall-clock soak duration in seconds")
    parser.add_argument("--period", type=float, default=30.0, help="wall-clock seconds per waypoint")
    parser.add_argument("--route", default="2,0;0,2;-2,0;0,-2")
    parser.add_argument("--height", type=float, default=0.1)
    args = parser.parse_args(rospy.myargv()[1:])
    route = parse_route(args.route)
    rospy.init_node("mission_patrol")
    publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=2, latch=True)
    started = time.monotonic()
    index = 0
    while not rospy.is_shutdown() and time.monotonic() - started < args.duration:
        x, y = route[index % len(route)]
        for _ in range(3):
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = x
            goal.pose.position.y = y
            goal.pose.position.z = args.height
            goal.pose.orientation.w = 1.0
            publisher.publish(goal)
            time.sleep(0.1)
        index += 1
        deadline = time.monotonic() + args.period
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
