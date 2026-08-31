#!/usr/bin/env python3
"""Publish the goal used by the low-altitude obstacle-avoidance demo."""

import rospy
from geometry_msgs.msg import PoseStamped


def main():
    rospy.init_node("obstacle_demo_goal_publisher", anonymous=True)
    publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)

    goal_x = float(rospy.get_param("~goal_x", 3.5))
    goal_y = float(rospy.get_param("~goal_y", 0.855335))
    goal_z = float(rospy.get_param("~goal_z", 1.0))

    # The goal lies beyond simple_box_1.  Reaching it requires a horizontal
    # detour because the demo altitude remains below the obstacle top.
    message = PoseStamped()
    message.header.frame_id = "map"
    message.header.stamp = rospy.Time.now()
    message.pose.position.x = goal_x
    message.pose.position.y = goal_y
    message.pose.position.z = goal_z
    message.pose.orientation.w = 1.0

    deadline = rospy.Time.now() + rospy.Duration.from_sec(20.0)
    while publisher.get_num_connections() == 0 and rospy.Time.now() < deadline:
        rospy.sleep(0.1)

    if publisher.get_num_connections() == 0:
        rospy.logerr("navigation node did not subscribe to /move_base_simple/goal")
        return

    publisher.publish(message)
    rospy.loginfo(
        "Published obstacle demo goal: (%.3f, %.3f, %.3f) in map frame",
        goal_x,
        goal_y,
        goal_z,
    )


if __name__ == "__main__":
    main()
