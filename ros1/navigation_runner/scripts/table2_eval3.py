#!/usr/bin/env python3

import math
import os
import signal
import subprocess
import time

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist

from table2_eval import Table2Evaluator


class Table2Evaluator3(Table2Evaluator):
    def __init__(self):
        super().__init__()

        # Default route for the Excel-generated indoor benchmark.
        self.scene_name = rospy.get_param("~scene_name", "xlsx_map_benchmark.world")
        self.runs = rospy.get_param("~runs", 20)
        self.timeout_sec = rospy.get_param("~timeout_sec", 120.0)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.1)
        self.goal_x = rospy.get_param("~goal_x", 4.04)
        self.goal_y = rospy.get_param("~goal_y", 17.5)
        self.goal_z = rospy.get_param("~goal_z", 0.1)
        self.leg_goal_tolerance = rospy.get_param("~leg_goal_tolerance", self.goal_tolerance)
        self.leg_timeout_sec = rospy.get_param("~leg_timeout_sec", 45.0)
        self.route_points = rospy.get_param(
            "~route_points",
            [
                [4.04, 17.5],
            ],
        )

        self.expected_start_x = rospy.get_param("~expected_start_x", -4.04)
        self.expected_start_y = rospy.get_param("~expected_start_y", -18.5)
        self.expected_start_z = rospy.get_param("~expected_start_z", 0.1)
        self.spawn_xy_tolerance = rospy.get_param("~spawn_xy_tolerance", 0.3)
        self.spawn_z_tolerance = rospy.get_param("~spawn_z_tolerance", 0.2)
        self.spawn_settle_sec = rospy.get_param("~spawn_settle_sec", 0.5)
        self.navigation_startup_sec = rospy.get_param("~navigation_startup_sec", 1.0)
        self.nav_process = None

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospy.loginfo(
            "[table2_eval3] Excel-map evaluator ready. "
            "scene=%s, expected_start=(%.2f, %.2f, %.2f), goal=(%.2f, %.2f, %.2f), runs=%d",
            self.scene_name,
            self.expected_start_x,
            self.expected_start_y,
            self.expected_start_z,
            self.goal_x,
            self.goal_y,
            self.goal_z,
            self.runs,
        )
        rospy.loginfo("[table2_eval3] Route waypoints=%s", self.route_points)
        rospy.loginfo(
            "[table2_eval3] This evaluator assumes the UAV spawn point comes from start.launch. "
            "It does not relocate the robot between runs."
        )
        self.disable_internal_navigation()

    def disable_internal_navigation(self):
        if hasattr(self.nav, "odom_sub"):
            self.nav.odom_sub.unregister()
        if hasattr(self.nav, "goal_sub"):
            self.nav.goal_sub.unregister()
        self.nav.odom_received = False
        self.nav.goal_received = False
        self.nav.goal = None
        self.nav.target_dir = None
        self.nav.raypoints = []
        self.nav.dynamic_obstacles = []

    def start_navigation_process(self):
        self.stop_navigation_process()
        self.nav_process = subprocess.Popen(
            ["rosrun", "navigation_runner", "navigation_node.py"],
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        time.sleep(self.navigation_startup_sec)

    def stop_navigation_process(self):
        if self.nav_process is None or self.nav_process.poll() is not None:
            self.nav_process = None
            return
        os.killpg(os.getpgid(self.nav_process.pid), signal.SIGINT)
        try:
            self.nav_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.nav_process.pid), signal.SIGKILL)
            self.nav_process.wait(timeout=5)
        self.nav_process = None

    def reset_navigation_state(self):
        super().reset_navigation_state()
        self.disable_internal_navigation()

    def set_start_pose(self, start_x, start_y):
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = "world"
        state.pose.position.x = start_x
        state.pose.position.y = start_y
        state.pose.position.z = self.expected_start_z
        state.pose.orientation.w = 1.0
        state.twist = Twist()

        response = self.set_model_state_srv(state)
        if not response.success:
            rospy.logerr("[table2_eval3] Failed to set model state: %s", response.status_message)
            return False

        self.current_odom = None
        time.sleep(self.spawn_settle_sec)
        return True

    def wait_for_spawn_pose(self, start_x, start_y, timeout_sec=5.0):
        deadline = time.monotonic() + timeout_sec
        while not rospy.is_shutdown():
            if self.current_odom is not None:
                pos = self.current_odom.pose.pose.position
                xy_ok = (
                    abs(pos.x - start_x) <= self.spawn_xy_tolerance
                    and abs(pos.y - start_y) <= self.spawn_xy_tolerance
                )
                z_ok = (
                    abs(pos.z - self.expected_start_z) <= self.spawn_z_tolerance
                    or abs(pos.z - self.takeoff_height) <= self.takeoff_tolerance
                )
                if xy_ok and z_ok:
                    return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return False

    def publish_goal_xy(self, goal_x, goal_y):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.position.z = self.goal_z
        goal_msg.pose.orientation.w = 1.0

        rate = rospy.Rate(self.goal_publish_hz)
        for _ in range(self.goal_publish_count):
            goal_msg.header.stamp = rospy.Time.now()
            self.goal_pub.publish(goal_msg)
            rate.sleep()

    def leg_goal_distance(self):
        if self.current_odom is None:
            return float("inf")
        pos = self.current_odom.pose.pose.position
        return math.hypot(pos.x - self.goal_x, pos.y - self.goal_y)

    def run_once(self, run_idx):
        result = {
            "run_idx": run_idx,
            "collision_count": 0,
            "success": False,
            "reason": "timeout",
            "elapsed_sec": 0.0,
            "completed_legs": 0,
            "route": "START->" + "->".join(f"({pt[0]:.1f},{pt[1]:.1f})" for pt in self.route_points),
        }

        try:
            self.current_odom = None
            self.stop_navigation_process()
            self.reset_navigation_state()
            self.reset_world()

            if not self.set_start_pose(self.expected_start_x, self.expected_start_y):
                result["reason"] = "set_pose_failed"
                return result

            if not self.wait_for_odom():
                result["reason"] = "odom_timeout"
                return result

            if not self.wait_for_spawn_pose(self.expected_start_x, self.expected_start_y):
                result["reason"] = "spawn_timeout"
                return result

            self.start_navigation_process()

            if not self.wait_for_takeoff():
                result["reason"] = "takeoff_timeout"
                return result

            time.sleep(self.post_takeoff_settle_sec)

            start_time = time.monotonic()
            prev_collision = False
            poll_interval = 1.0 / max(self.poll_hz, 1e-6)
            leg_idx = 0

            if not self.route_points:
                result["success"] = True
                result["reason"] = "route_trivial"
                return result

            self.goal_x, self.goal_y = self.route_points[leg_idx]
            self.publish_goal_xy(self.goal_x, self.goal_y)
            leg_deadline = time.monotonic() + self.leg_timeout_sec

            while not rospy.is_shutdown():
                elapsed = time.monotonic() - start_time
                result["elapsed_sec"] = elapsed
                if time.monotonic() > leg_deadline:
                    result["reason"] = f"leg_timeout_{leg_idx + 1}"
                    break

                current_collision = self.is_static_collision() or self.is_dynamic_collision()
                if current_collision and not prev_collision:
                    result["collision_count"] += 1
                    rospy.loginfo(
                        "[table2_eval3] Run %d collision #%d detected.",
                        run_idx,
                        result["collision_count"],
                    )
                prev_collision = current_collision

                if self.leg_goal_distance() <= self.leg_goal_tolerance:
                    result["completed_legs"] += 1
                    if leg_idx == len(self.route_points) - 1:
                        result["success"] = True
                        result["reason"] = "route_completed"
                        break

                    leg_idx += 1
                    self.goal_x, self.goal_y = self.route_points[leg_idx]
                    rospy.loginfo(
                        "[table2_eval3] Run %d advancing to waypoint %d/%d: (%.2f, %.2f)",
                        run_idx,
                        leg_idx + 1,
                        len(self.route_points),
                        self.goal_x,
                        self.goal_y,
                    )
                    time.sleep(self.post_takeoff_settle_sec)
                    self.publish_goal_xy(self.goal_x, self.goal_y)
                    leg_deadline = time.monotonic() + self.leg_timeout_sec

                time.sleep(poll_interval)

            return result
        finally:
            self.stop_navigation_process()
            self.reset_navigation_state()

    def summarize(self, results):
        total_collisions = sum(item["collision_count"] for item in results)
        avg_collisions = total_collisions / float(len(results)) if results else 0.0
        success_count = sum(1 for item in results if item["success"])
        avg_elapsed = sum(item["elapsed_sec"] for item in results) / float(len(results)) if results else 0.0

        print("")
        print("======== Table II Generated-Env Evaluation ========")
        print(f"Scene: {self.scene_name}")
        print(
            "Expected start from start.launch: "
            f"({self.expected_start_x:.2f}, {self.expected_start_y:.2f}, {self.expected_start_z:.2f})"
        )
        print(f"Route: START -> {self.route_points}")
        for item in results:
            print(
                f"Run {item['run_idx']:02d}: "
                f"collisions={item['collision_count']}, "
                f"completed_legs={item['completed_legs']}, "
                f"success={item['success']}, "
                f"reason={item['reason']}, "
                f"elapsed={item['elapsed_sec']:.2f}s"
            )
        print("---------------------------------------------------")
        print(f"Runs: {len(results)}")
        print(f"Total collisions: {total_collisions}")
        print(f"Average collisions per run: {avg_collisions:.3f}")
        print(f"Success rate: {success_count}/{len(results)} = {success_count / float(len(results)):.3f}")
        print(f"Average elapsed time per run: {avg_elapsed:.2f}s")
        print("===================================================")


if __name__ == "__main__":
    evaluator = Table2Evaluator3()
    evaluator.run()
