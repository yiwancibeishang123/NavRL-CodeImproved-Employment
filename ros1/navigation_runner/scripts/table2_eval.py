#!/usr/bin/env python3

import math
import os
import signal
import subprocess
import time
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rospy
from hydra import compose, initialize_config_dir
from geometry_msgs.msg import Point, PoseStamped
from map_manager.srv import CheckPosCollision
from nav_msgs.msg import Odometry
from navigation import Navigation
from onboard_detector.srv import GetDynamicObstacles
from std_srvs.srv import Empty

CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg")


class Table2Evaluator:
    def __init__(self):
        rospy.init_node("table2_eval", anonymous=True)

        self.model_name = rospy.get_param("~model_name", "quadcopter")
        self.runs = rospy.get_param("~runs", 20)
        self.timeout_sec = rospy.get_param("~timeout_sec", 90.0)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.1)
        self.takeoff_height = rospy.get_param("~takeoff_height", 1.0)
        self.takeoff_tolerance = rospy.get_param("~takeoff_tolerance", 0.2)
        self.poll_hz = rospy.get_param("~poll_hz", 20.0)
        self.dynamic_query_range = rospy.get_param("~dynamic_query_range", 5.0)
        self.robot_radius = rospy.get_param("~robot_radius", 0.3)
        self.robot_height_margin = rospy.get_param("~robot_height_margin", 0.3)
        self.goal_publish_count = rospy.get_param("~goal_publish_count", 10)
        self.goal_publish_hz = rospy.get_param("~goal_publish_hz", 10.0)
        self.reset_settle_sec = rospy.get_param("~reset_settle_sec", 2.0)
        self.post_takeoff_settle_sec = rospy.get_param("~post_takeoff_settle_sec", 1.0)
        self.restart_support_nodes_each_run = rospy.get_param("~restart_support_nodes_each_run", True)
        self.support_nodes_ready_timeout_sec = rospy.get_param("~support_nodes_ready_timeout_sec", 15.0)
        self.goal_x = rospy.get_param("~goal_x", 39.0)
        self.goal_y = rospy.get_param("~goal_y", 1.0)
        self.goal_z = rospy.get_param("~goal_z", 0.1)
        self.odom_topic = rospy.get_param("~odom_topic", "/CERLAB/quadcopter/odom")

        self.current_odom = None
        self.owned_support_processes = []
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)

        rospy.loginfo("Waiting for ROS services needed by table2_eval...")
        rospy.wait_for_service("/gazebo/reset_simulation")

        self.refresh_service_proxies()

        clean_local_map = rospy.get_param("/occupancy_map/clean_local_map", None)
        map_size = rospy.get_param("/occupancy_map/map_size", None)
        rospy.loginfo(
            "[table2_eval] occupancy_map clean_local_map=%s, map_size=%s",
            clean_local_map,
            map_size,
        )
        rospy.loginfo(
            "[table2_eval] Navigation model selection follows navigation.py. "
            "Change the checkpoint there, and table2_eval will use the same model via Navigation(cfg)."
        )
        if clean_local_map is False:
            rospy.logwarn(
                "[table2_eval] /occupancy_map/clean_local_map is false. "
                "Repeated runs may leave stale occupied voxels that behave like an invisible wall."
            )

        rospy.set_param("rl/enable_manual_safety_stop", False)
        with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
            self.cfg = compose(config_name="train")
        self.nav = Navigation(self.cfg)
        self.nav.run()

    def refresh_service_proxies(self, timeout=None):
        rospy.wait_for_service("/occupancy_map/check_pos_collision", timeout=timeout)
        rospy.wait_for_service("/onboard_detector/get_dynamic_obstacles", timeout=timeout)
        self.reset_simulation_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.check_pos_collision_srv = rospy.ServiceProxy("/occupancy_map/check_pos_collision", CheckPosCollision)
        self.get_dynamic_obstacles_srv = rospy.ServiceProxy(
            "/onboard_detector/get_dynamic_obstacles",
            GetDynamicObstacles,
        )

    def start_background_process(self, cmd):
        process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        self.owned_support_processes.append(process)
        return process

    def stop_process(self, process):
        if process is None or process.poll() is not None:
            return
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)

    def kill_ros_node(self, node_name):
        subprocess.run(
            ["rosnode", "kill", node_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def restart_support_nodes(self):
        rospy.loginfo("[table2_eval] Restarting occupancy_map and fake_detector to clear stale state.")
        for process in self.owned_support_processes:
            self.stop_process(process)
        self.owned_support_processes = []

        self.kill_ros_node("/occupancy_map_node")
        self.kill_ros_node("/fake_detector_node")
        self.kill_ros_node("/fake_detector")
        time.sleep(1.0)

        self.start_background_process(["rosrun", "map_manager", "occupancy_map_node"])
        self.start_background_process(
            ["rosrun", "onboard_detector", "fake_detector_node", "__name:=fake_detector_node"]
        )

        deadline = time.monotonic() + self.support_nodes_ready_timeout_sec
        while not rospy.is_shutdown():
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise rospy.ROSException("Timed out waiting for support nodes to restart.")
                self.refresh_service_proxies(timeout=remaining)
                return
            except rospy.ROSException:
                time.sleep(0.2)

    def odom_callback(self, odom_msg):
        self.current_odom = odom_msg

    def build_goal_msg(self):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = self.goal_x
        goal_msg.pose.position.y = self.goal_y
        goal_msg.pose.position.z = self.goal_z
        goal_msg.pose.orientation.w = 1.0
        return goal_msg

    def reset_world(self):
        self.reset_simulation_srv()
        if self.restart_support_nodes_each_run:
            self.restart_support_nodes()
        time.sleep(self.reset_settle_sec)

    def reset_navigation_state(self):
        self.nav.goal = None
        self.nav.goal_received = False
        self.nav.target_dir = None
        self.nav.stable_times = 0
        self.nav.has_action = False
        self.nav.safety_stop = False
        self.nav.raypoints = []
        self.nav.dynamic_obstacles = []

    def wait_for_odom(self, timeout_sec=10.0):
        deadline = time.monotonic() + timeout_sec
        while not rospy.is_shutdown() and self.current_odom is None:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def wait_for_takeoff(self, timeout_sec=20.0):
        deadline = time.monotonic() + timeout_sec
        while not rospy.is_shutdown():
            if self.current_odom is not None:
                curr_z = self.current_odom.pose.pose.position.z
                if abs(curr_z - self.takeoff_height) <= self.takeoff_tolerance:
                    return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        return False

    def publish_goal(self):
        goal_msg = self.build_goal_msg()
        rate = rospy.Rate(self.goal_publish_hz)
        for _ in range(self.goal_publish_count):
            goal_msg.header.stamp = rospy.Time.now()
            self.goal_pub.publish(goal_msg)
            rate.sleep()

    def is_static_collision(self):
        if self.current_odom is None:
            return False
        pos = self.current_odom.pose.pose.position
        response = self.check_pos_collision_srv(pos.x, pos.y, pos.z, True)
        return response.occupied

    def is_dynamic_collision(self):
        if self.current_odom is None:
            return False

        pos = self.current_odom.pose.pose.position
        current_position = Point(x=pos.x, y=pos.y, z=pos.z)
        response = self.get_dynamic_obstacles_srv(current_position, self.dynamic_query_range)

        for obs_pos, obs_size in zip(response.position, response.size):
            horizontal_distance = math.hypot(obs_pos.x - pos.x, obs_pos.y - pos.y)
            obstacle_radius = max(obs_size.x, obs_size.y) / 2.0
            vertical_distance = abs(obs_pos.z - pos.z)
            if (
                horizontal_distance <= obstacle_radius + self.robot_radius
                and vertical_distance <= obs_size.z / 2.0 + self.robot_height_margin
            ):
                return True
        return False

    def goal_distance(self):
        if self.current_odom is None:
            return float("inf")
        pos = self.current_odom.pose.pose.position
        return math.sqrt(
            (pos.x - self.goal_x) ** 2
            + (pos.y - self.goal_y) ** 2
            + (pos.z - self.takeoff_height) ** 2
        )

    def run_once(self, run_idx):
        result = {
            "run_idx": run_idx,
            "collision_count": 0,
            "success": False,
            "reason": "timeout",
            "elapsed_sec": 0.0,
        }

        self.current_odom = None
        self.reset_navigation_state()
        self.reset_world()

        if not self.wait_for_odom():
            result["reason"] = "odom_timeout"
            return result

        if not self.wait_for_takeoff():
            result["reason"] = "takeoff_timeout"
            return result

        time.sleep(self.post_takeoff_settle_sec)

        self.publish_goal()

        start_time = time.monotonic()
        prev_collision = False
        poll_interval = 1.0 / max(self.poll_hz, 1e-6)

        while not rospy.is_shutdown():
            elapsed = time.monotonic() - start_time
            result["elapsed_sec"] = elapsed
            if elapsed > self.timeout_sec:
                result["reason"] = "timeout"
                break

            current_collision = self.is_static_collision() or self.is_dynamic_collision()
            if current_collision and not prev_collision:
                result["collision_count"] += 1
                rospy.loginfo(
                    "[table2_eval] Run %d collision #%d detected.",
                    run_idx,
                    result["collision_count"],
                )
            prev_collision = current_collision

            if self.goal_distance() <= self.goal_tolerance:
                result["success"] = True
                result["reason"] = "goal_reached"
                break

            time.sleep(poll_interval)

        return result

    def summarize(self, results):
        total_collisions = sum(item["collision_count"] for item in results)
        avg_collisions = total_collisions / float(len(results)) if results else 0.0
        success_count = sum(1 for item in results if item["success"])
        avg_elapsed = sum(item["elapsed_sec"] for item in results) / float(len(results)) if results else 0.0

        print("")
        print("========== Table II Style Evaluation ==========")
        for item in results:
            print(
                f"Run {item['run_idx']:02d}: "
                f"collisions={item['collision_count']}, "
                f"success={item['success']}, "
                f"reason={item['reason']}, "
                f"elapsed={item['elapsed_sec']:.2f}s"
            )
        print("-----------------------------------------------")
        print(f"Runs: {len(results)}")
        print(f"Total collisions: {total_collisions}")
        print(f"Average collisions per run: {avg_collisions:.3f}")
        print(f"Success rate: {success_count}/{len(results)} = {success_count / float(len(results)):.3f}")
        print(f"Average elapsed time per run: {avg_elapsed:.2f}s")
        print("===============================================")

    def run(self):
        results = []
        for run_idx in range(1, self.runs + 1):
            rospy.loginfo("[table2_eval] Starting run %d/%d", run_idx, self.runs)
            result = self.run_once(run_idx)
            results.append(result)
            rospy.loginfo(
                "[table2_eval] Run %d finished: collisions=%d, success=%s, reason=%s",
                run_idx,
                result["collision_count"],
                result["success"],
                result["reason"],
            )
        self.summarize(results)


if __name__ == "__main__":
    evaluator = Table2Evaluator()
    evaluator.run()
