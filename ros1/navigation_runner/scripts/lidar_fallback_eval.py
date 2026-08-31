#!/usr/bin/env python3
"""Black-box Gazebo check for the LiDAR-only visual-failure fallback."""

import argparse
import json
import math
import os
import time

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool


class LidarFallbackEvaluator:
    def __init__(self, args):
        self.args = args
        self.degraded_seen = False
        self.placement_time = None
        self.commands_after_placement = []
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=2, latch=True)
        rospy.Subscriber("/yolo_detector/perception_degraded", Bool, self.on_degraded, queue_size=5)
        rospy.Subscriber("/CERLAB/quadcopter/cmd_vel", TwistStamped, self.on_command, queue_size=10)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=args.ready_timeout)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    def on_degraded(self, message):
        self.degraded_seen = self.degraded_seen or bool(message.data)

    def on_command(self, message):
        if self.placement_time is None:
            return
        velocity = message.twist.linear
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        if speed > self.args.min_escape_speed:
            self.commands_after_placement.append({
                "wall_time_s": time.monotonic(),
                "local_velocity_mps": [velocity.x, velocity.y, velocity.z],
                "speed_mps": speed,
            })

    def publish_goal(self):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = self.args.goal_x
        goal.pose.position.y = self.args.goal_y
        goal.pose.position.z = self.args.goal_z
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

    def place_obstacle(self):
        state = ModelState()
        state.model_name = self.args.target_model
        state.reference_frame = "world"
        state.pose.position.x = self.args.obstacle_x
        state.pose.position.y = self.args.obstacle_y
        state.pose.position.z = self.args.obstacle_z
        state.pose.orientation.w = 1.0
        response = self.set_model_state(state)
        if not response.success:
            raise RuntimeError(response.status_message)

    def run(self):
        deadline = time.monotonic() + self.args.ready_timeout
        while not rospy.is_shutdown() and not self.degraded_seen and time.monotonic() < deadline:
            self.publish_goal()
            time.sleep(0.2)
        if self.degraded_seen:
            self.place_obstacle()
            self.placement_time = time.monotonic()
            observe_deadline = self.placement_time + self.args.observe_duration
            while not rospy.is_shutdown() and time.monotonic() < observe_deadline:
                self.publish_goal()
                time.sleep(0.2)
        result = {
            "schema_version": 1,
            "kind": "gazebo_lidar_only_visual_failure_fallback",
            "visual_watchdog_degraded": self.degraded_seen,
            "target_model": self.args.target_model,
            "obstacle_position_m": [self.args.obstacle_x, self.args.obstacle_y, self.args.obstacle_z],
            "independent_sensor": "Gazebo depth-derived occupancy-map raycast; no YOLO/tracker obstacle state used",
            "escape_commands_after_obstacle": len(self.commands_after_placement),
            "first_escape_latency_s": None if not self.commands_after_placement else round(
                self.commands_after_placement[0]["wall_time_s"] - self.placement_time, 4
            ),
            "first_local_velocity_mps": None if not self.commands_after_placement else self.commands_after_placement[0]["local_velocity_mps"],
        }
        result["result"] = "passed" if (
            result["visual_watchdog_degraded"] and result["escape_commands_after_obstacle"] > 0
        ) else "failed"
        os.makedirs(os.path.dirname(os.path.abspath(self.args.output)), exist_ok=True)
        with open(self.args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["result"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model", default="person1_0.5_0.5_1.8")
    parser.add_argument("--ready-timeout", type=float, default=25.0)
    parser.add_argument("--observe-duration", type=float, default=10.0)
    parser.add_argument("--min-escape-speed", type=float, default=0.05)
    parser.add_argument("--goal-x", type=float, default=2.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-z", type=float, default=0.1)
    parser.add_argument("--obstacle-x", type=float, default=0.9)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument("--obstacle-z", type=float, default=0.1)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("lidar_fallback_eval")
    return LidarFallbackEvaluator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
