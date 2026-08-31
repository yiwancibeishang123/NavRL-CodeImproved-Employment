#!/usr/bin/env python3
"""Run the full six-obstacle route through the deployed navigation node."""

import argparse
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


# Main inspection points cover all six boxes. The extra points keep the return
# leg in a clear lower corridor; they are still reached by the same controller.
DEFAULT_ROUTE = [
    (2.8, -0.9),
    (4.9, -1.1),
    (7.0, 0.0),
    (6.6, -2.3),
    (7.9, -2.3),
    (9.7, -0.3),
    (9.7, -2.3),
    (0.8, -2.3),
    (0.8, 0.0),
    (0.0, 0.0),
]


def parse_route(value):
    return [tuple(float(item) for item in point.split(",")) for point in value.split(";") if point]


class RouteRunner:
    def __init__(self, route, height, tolerance, timeout, publish_period):
        self.route = route
        self.height = height
        self.tolerance = tolerance
        self.timeout = timeout
        self.publish_period = publish_period
        self.position = None
        self.odom_sub = rospy.Subscriber(
            rospy.get_param("~odom_topic", "/CERLAB/quadcopter/odom"),
            Odometry,
            self.odom_callback,
            queue_size=1,
        )
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=2, latch=True)

    def odom_callback(self, message):
        p = message.pose.pose.position
        self.position = (p.x, p.y, p.z)

    def distance(self, goal):
        if self.position is None:
            return float("inf")
        return math.sqrt(
            (self.position[0] - goal[0]) ** 2
            + (self.position[1] - goal[1]) ** 2
            + (self.position[2] - self.height) ** 2
        )

    def make_goal(self, point):
        message = PoseStamped()
        message.header.frame_id = "map"
        message.header.stamp = rospy.Time.now()
        message.pose.position.x = point[0]
        message.pose.position.y = point[1]
        message.pose.position.z = self.height
        message.pose.orientation.w = 1.0
        return message

    def run(self):
        rospy.loginfo("Full route has %d points, including return to spawn", len(self.route))
        while not rospy.is_shutdown() and self.position is None:
            rospy.sleep(0.1)

        for index, point in enumerate(self.route, start=1):
            goal = self.make_goal(point)
            started = time.monotonic()
            rospy.loginfo("Waypoint %d/%d -> (%.2f, %.2f)", index, len(self.route), point[0], point[1])
            # Publish exactly once. The publisher is latched, and repeatedly
            # sending the same goal would reset the navigation stabilizer.
            deadline = time.monotonic() + 30.0
            while not rospy.is_shutdown() and self.goal_pub.get_num_connections() == 0:
                if time.monotonic() > deadline:
                    rospy.logerr("navigation node did not subscribe to goal topic")
                    return 2
                rospy.sleep(0.1)
            self.goal_pub.publish(goal)
            while not rospy.is_shutdown():
                now = time.monotonic()
                current_distance = self.distance(point)
                if current_distance <= self.tolerance:
                    rospy.loginfo("Waypoint %d reached, error %.3f m", index, current_distance)
                    rospy.sleep(0.5)
                    break
                if now - started > self.timeout:
                    rospy.logerr(
                        "Waypoint %d timed out at error %.3f m; route stopped",
                        index,
                        current_distance,
                    )
                    return 2
                rospy.sleep(0.05)
        rospy.loginfo("Full six-obstacle route completed and returned to spawn")
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        default=";".join("{:.6f},{:.6f}".format(x, y) for x, y in DEFAULT_ROUTE),
        help="semicolon-separated x,y map-frame points",
    )
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.30)
    parser.add_argument("--timeout", type=float, default=55.0)
    parser.add_argument("--publish-period", type=float, default=0.25)
    args = parser.parse_args(rospy.myargv()[1:])
    route = parse_route(args.route)
    if len(route) < 2:
        raise SystemExit("route must contain at least two points")
    rospy.init_node("full_obstacle_route_runner")
    runner = RouteRunner(route, args.height, args.tolerance, args.timeout, args.publish_period)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
