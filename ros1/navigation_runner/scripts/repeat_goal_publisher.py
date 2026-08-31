#!/usr/bin/env python3
"""按固定顺序发布多个航点，给重复任务测试使用。"""

import argparse
import time

import rospy
from geometry_msgs.msg import PoseStamped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=float, default=8.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--route", default="2,0;2,0.35;2,0.70;2,1.05;2,1.40")
    args = parser.parse_args(rospy.myargv()[1:])
    route = [tuple(float(v) for v in item.split(",")) for item in args.route.split(";")]
    rospy.init_node("repeat_goal_publisher")
    publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=2, latch=False)
    rospy.sleep(1.0)
    for x, y in route:
        if rospy.is_shutdown():
            break
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = args.height
        goal.pose.orientation.w = 1.0
        for _ in range(5):
            publisher.publish(goal)
            rospy.sleep(0.1)
        time.sleep(max(0.0, args.period - 0.5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
