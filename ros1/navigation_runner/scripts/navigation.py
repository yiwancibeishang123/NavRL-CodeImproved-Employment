import rospy
import numpy as np
import torch
from map_manager.srv import RayCast
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Quaternion, Vector3
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from navigation_runner.srv import GetSafeAction, GetSafeActionMap
from onboard_detector.srv import GetDynamicObstacles
from map_manager.srv import GetStaticObstacles
from ppo import PPO
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from navigation_runner.srv import GetPolicyInference
from observation import (
    build_dynamic_obstacle_observation,
    build_lidar_observation,
    build_state_observation,
    safe_direction,
)
from deployment_metrics import JsonlLogger, runtime_metadata, runtime_snapshot, timed_call, vector_list
from sensor_perturbation import SensorPerturber
import math
from std_srvs.srv import Empty, SetBool, SetBoolResponse
from std_msgs.msg import Bool
import tf.transformations
import time
import threading
import os
import rospkg

class Navigation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy_variant = str(rospy.get_param("~policy_variant", "improved_navrl")).strip().lower()
        if self.policy_variant == "baseline_navrl":
            # The archived baseline was trained before the geometry/history
            # encoder and constrained critic existed. Apply the corresponding
            # architecture flags before constructing the policy; loading with
            # partial keys would make a baseline comparison meaningless.
            self.cfg.algo.method.step1_enabled = False
            self.cfg.algo.method.step2_enabled = False
            self.cfg.algo.method.step3_enabled = False
            self.cfg.algo.constrained_rl.enabled = False
            self.cfg.algo.observation_aux.obstacle_input_mode = "baseline"
        elif self.policy_variant != "improved_navrl":
            raise ValueError("Unsupported policy_variant: {}".format(self.policy_variant))
        requested_device = str(rospy.get_param("~device", cfg.device))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            rospy.logwarn("Requested %s but CUDA is unavailable; using CPU", requested_device)
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.cfg.device = str(self.device)
        self.lidar_hbeams = int(360/self.cfg.sensor.lidar_hres)
        self.lidar_size = self.lidar_hbeams * int(self.cfg.sensor.lidar_vbeams)
        self.raypoints = []
        dyn_num = int(self.cfg.algo.feature_extractor.dyn_obs_num)
        self.dynamic_obstacles = (
            torch.zeros(dyn_num, 3, device=self.device),
            torch.zeros(dyn_num, 3, device=self.device),
            torch.zeros(dyn_num, 3, device=self.device),
        )
        self.robot_size = float(rospy.get_param("~robot_size", 0.3))
        self.raycast_vres = ((self.cfg.sensor.lidar_vfov[1] - self.cfg.sensor.lidar_vfov[0]))/(self.cfg.sensor.lidar_vbeams - 1) * np.pi/180.0
        self.raycast_hres = self.cfg.sensor.lidar_hres * np.pi/180.0

        self.goal = None
        self.goal_received = False
        self.target_dir = None
        self.stable_times = 0
        self.has_action = False
        self.laser_points_msg = None

        self.height_control = bool(rospy.get_param("~height_control", False))
        self.px4_control = rospy.get_param('rl/use_px4', True)

        self.use_policy_server = bool(rospy.get_param("~use_policy_server", False))
        # Keep the learned controller in the loop for the full mission.  The
        # old obstacle-gated path silently switched to a straight-line helper
        # whenever LiDAR was clear, which made a clean run look model-free.
        self.policy_always_on = bool(rospy.get_param("~policy_always_on", True))
        self.deployment_guidance_enabled = bool(
            rospy.get_param("~deployment_guidance_enabled", True)
        )
        self.policy_residual_weight = float(
            np.clip(rospy.get_param("~policy_residual_weight", 0.35), 0.0, 1.0)
        )
        self.goal_guidance_speed_mps = max(
            0.1, float(rospy.get_param("~goal_guidance_speed_mps", 0.8))
        )
        self.lidar_avoidance_distance_m = max(
            0.4, float(rospy.get_param("~lidar_avoidance_distance_m", 1.4))
        )
        self.require_safety_shield = bool(rospy.get_param("~require_safety_shield", False))
        self.sensor_timeout = float(rospy.get_param("~sensor_timeout", 0.5))
        self.obstacle_timeout = float(rospy.get_param("~obstacle_timeout", 0.5))
        self.perception_watchdog_enabled = bool(rospy.get_param("~perception_watchdog_enabled", True))
        self.perception_watchdog_required = bool(rospy.get_param("~perception_watchdog_required", False))
        self.perception_timeout = max(0.05, float(rospy.get_param("~perception_timeout", 0.5)))
        self.perception_lidar_fallback_enabled = bool(
            rospy.get_param("~perception_lidar_fallback_enabled", True)
        )
        self.perception_lidar_escape_distance_m = max(
            self.robot_size * 2.0,
            float(rospy.get_param("~perception_lidar_escape_distance_m", 1.4)),
        )
        self.perception_lidar_escape_speed_mps = max(
            0.0, float(rospy.get_param("~perception_lidar_escape_speed_mps", 0.45))
        )
        self.perception_lidar_vertical_band_m = max(
            0.05, float(rospy.get_param("~perception_lidar_vertical_band_m", 1.0))
        )
        self.control_period = float(rospy.get_param("~control_period", 0.05))
        self.timing_degrade_enabled = bool(rospy.get_param("~timing_degrade_enabled", True))
        self.timing_degrade_consecutive_cycles = max(
            1, int(rospy.get_param("~timing_degrade_consecutive_cycles", 2))
        )
        self.timing_degrade_max_lag_ms = max(
            0.0, float(rospy.get_param("~timing_degrade_max_lag_ms", 20.0))
        )
        self.timing_degrade_hold_s = max(
            0.0, float(rospy.get_param("~timing_degrade_hold_s", 0.5))
        )
        self.cruise_speed_mps = max(0.0, float(rospy.get_param("~cruise_speed_mps", 0.0)))
        self.auto_takeoff = bool(rospy.get_param("~auto_takeoff", True))
        self.hold_height = float(rospy.get_param("~hold_height", 1.0))
        self.last_odom_time = rospy.Time(0)
        self.last_raycast_time = rospy.Time(0)
        self.last_obstacle_time = rospy.Time(0)
        self.last_perception_status_time = rospy.Time(0)
        self.perception_degraded = False
        self.perception_degrade_count = 0
        self.last_lidar_fallback_nearest_m = None
        self.last_action_time = rospy.Time(0)
        self.takeoff_pose = PoseStamped()
        self.takeoff_pose.pose.position.z = self.hold_height
        self.run_label = str(rospy.get_param("~run_label", "unlabeled"))
        self.mission_goal_tolerance_m = max(
            0.05, float(rospy.get_param("~mission_goal_tolerance_m", 1.0))
        )
        self.mission_active = False
        self.mission_result = "not_started"
        self.mission_start_time = None
        self.mission_start_position = None
        self.mission_previous_position = None
        self.mission_path_length_m = 0.0
        self.mission_speed_samples_mps = []
        self.mission_goal_distance_m = None
        self.mission_initial_goal_distance_m = None
        self.mission_min_lidar_distance_m = None
        self.mission_completed_count = 0
        self.mission_end_logged = False
        self.loggers_closed = False
        self.metrics_logger = JsonlLogger(
            rospy.get_param("~metrics_log", ""),
            rospy.get_param("~metrics_flush_every", 20),
        )
        self.trace_logger = JsonlLogger(
            rospy.get_param("~observation_trace_log", ""),
            rospy.get_param("~metrics_flush_every", 20),
        )
        self.trace_stride = max(1, int(rospy.get_param("~observation_trace_stride", 10)))
        self.trace_count = 0
        self.last_inference_ms = None
        self.last_inference_device_ms = None
        self.last_inference_mode = "none"
        self.last_policy_action = None
        self.inference_this_cycle = False
        self.control_callback_count = 0
        self.metrics_resource_stride = max(1, int(rospy.get_param("~metrics_resource_stride", 20)))
        self.last_control_callback_monotonic = None
        self.last_control_interval_ms = None
        self.last_control_period_jitter_ms = None
        self.last_scheduler_lag_ms = None
        self.last_control_overrun = False
        self.timing_overrun_streak = 0
        self.timing_degrade_active = False
        self.timing_degrade_until = 0.0
        self.timing_degrade_count = 0
        self.last_perturbation = {"enabled": False, "delay_steps": 0}
        self.perturber = SensorPerturber.from_ros(rospy)

        self.odom_received = False
        if (self.px4_control):
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback)
            self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback)
            self.action_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
            self.pose_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
            self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
            self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

            self.mavros_state = None
            self.offb_set_mode = SetModeRequest()
            self.offb_set_mode.custom_mode = 'OFFBOARD'
            self.arm_cmd = CommandBoolRequest()
            self.arm_cmd.value = True
        else:
            self.odom_sub = rospy.Subscriber("/CERLAB/quadcopter/odom", Odometry, self.odom_callback)
            self.action_pub = rospy.Publisher("/CERLAB/quadcopter/cmd_vel", TwistStamped, queue_size=10)
            self.pose_pub = rospy.Publisher("/CERLAB/quadcopter/setpoint_pose", PoseStamped, queue_size=10)

        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        if self.perception_watchdog_enabled:
            self.perception_watchdog_sub = rospy.Subscriber(
                "/yolo_detector/perception_degraded", Bool, self.perception_watchdog_callback, queue_size=2
            )
        self.raycast_vis_pub = rospy.Publisher("/rl_navigation/raycast", MarkerArray, queue_size=10)
        self.cmd_vis_pub = rospy.Publisher("/rl_navigation/cmd", MarkerArray, queue_size=10)
        self.goal_vis_pub = rospy.Publisher("rl_navigation/goal", MarkerArray, queue_size=10)
        self.flight_trajectory_pub = rospy.Publisher(
            "/rl_navigation/flight_trajectory", Marker, queue_size=2, latch=True
        )
        self.flight_trajectory = Marker()
        self.flight_trajectory.header.frame_id = "map"
        self.flight_trajectory.ns = "flight_trajectory"
        self.flight_trajectory.id = 0
        self.flight_trajectory.type = Marker.LINE_STRIP
        self.flight_trajectory.action = Marker.ADD
        self.flight_trajectory.pose.orientation.w = 1.0
        self.flight_trajectory.scale.x = 0.025
        self.flight_trajectory.color.r = 0.05
        self.flight_trajectory.color.g = 0.55
        self.flight_trajectory.color.b = 0.95
        self.flight_trajectory.color.a = 1.0
        self.rollout_traj_pub = rospy.Publisher("/rollout_traj", Path, queue_size=10)
        self.dynamic_obstacle_vis_pub = rospy.Publisher("/rl_navigation/in_range_dynamic_obstacles", MarkerArray, queue_size=10)

        self.safety_toggle_srv = rospy.Service(
            "rl_navigation/toggle_safety_stop", SetBool, self.toggle_safety_stop
        )

        if (not self.use_policy_server):
            self.policy = self.init_model()
            self.policy.eval()


        # Safety is controlled through a ROS service instead of a blocking stdin thread.
        self.safety_stop = False
        self.metrics_logger.write({
            "event": "run_start",
            "timestamp": time.time(),
            "run_label": self.run_label,
            "policy_variant": self.policy_variant,
            "policy_always_on": self.policy_always_on,
            "deployment_guidance": {
                "enabled": self.deployment_guidance_enabled,
                "policy_residual_weight": self.policy_residual_weight,
                "goal_guidance_speed_mps": self.goal_guidance_speed_mps,
                "lidar_avoidance_distance_m": self.lidar_avoidance_distance_m,
            },
            "device": str(self.device),
            "control_period_s": self.control_period,
            "sensor_timeout_s": self.sensor_timeout,
            "obstacle_timeout_s": self.obstacle_timeout,
            "perception_watchdog": {
                "enabled": self.perception_watchdog_enabled,
                "required": self.perception_watchdog_required,
                "timeout_s": self.perception_timeout,
                "fallback": "lidar_escape_then_hover_position_hold",
                "lidar_escape_enabled": self.perception_lidar_fallback_enabled,
                "lidar_escape_distance_m": self.perception_lidar_escape_distance_m,
                "lidar_escape_speed_mps": self.perception_lidar_escape_speed_mps,
            },
            "mission_metrics": {
                "goal_tolerance_m": self.mission_goal_tolerance_m,
                "source": "Gazebo/odometry ground-truth topic",
            },
            "timing_degrade": {
                "enabled": self.timing_degrade_enabled,
                "consecutive_cycles": self.timing_degrade_consecutive_cycles,
                "max_lag_ms": self.timing_degrade_max_lag_ms,
                "hold_s": self.timing_degrade_hold_s,
                "fallback": "hover_position_hold",
            },
            "perturbation": self.perturber.metadata(),
            "runtime": runtime_metadata(torch, self.device),
        })
        rospy.on_shutdown(self.close_loggers)
        if self.auto_takeoff:
            self.takeoff()
  
    def init_model(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.device),
            })
        }).expand(1, action_dim).to(self.device)

        policy = PPO(self.cfg.algo, observation_spec, action_spec, self.device)

        source_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", "step3-80new.pt")
        try:
            package_checkpoint = os.path.join(rospkg.RosPack().get_path("navigation_runner"), "scripts", "ckpts", "step3-80new.pt")
        except Exception:
            package_checkpoint = source_checkpoint
        default_checkpoint = source_checkpoint if os.path.isfile(source_checkpoint) else package_checkpoint
        checkpoint = rospy.get_param("~model_checkpoint", default_checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                "Deployment checkpoint does not exist: {}. Set ~model_checkpoint.".format(checkpoint)
            )
        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint architecture mismatch; missing={}, unexpected={}".format(
                    list(missing)[:10], list(unexpected)[:10]
                )
            )
        rospy.loginfo("Loaded deployment policy %s on %s", checkpoint, self.device)
        return policy

    def takeoff(self):
        takeoff_height = self.hold_height
        r = rospy.Rate(10)
        while (not rospy.is_shutdown() and self.odom_received == False):
            print("[nav-ros]: Wait for robot odom...")
            r.sleep()

        takeoff_pose = PoseStamped()
        takeoff_pose.pose.position.x = self.odom.pose.pose.position.x
        takeoff_pose.pose.position.y = self.odom.pose.pose.position.y
        takeoff_pose.pose.position.z = takeoff_height
        takeoff_pose.pose.orientation = self.odom.pose.pose.orientation
        self.takeoff_pose = takeoff_pose
        if (self.px4_control):
            # PX4 requires a setpoint stream before OFFBOARD; use the current
            # horizontal position and configured takeoff height, not origin.
            pose = takeoff_pose
            rate = rospy.Rate(20)
            # Send a few setpoints before starting
            for i in range(100):
                if(rospy.is_shutdown()):
                    break
                self.pose_pub.publish(pose)
                rate.sleep()
            last_req = rospy.Time.now()
        while (not rospy.is_shutdown() and not (np.abs(self.odom.pose.pose.position.z - takeoff_height) <= 0.2)):
            if (self.px4_control):
                if self.mavros_state is None:
                    self.pose_pub.publish(takeoff_pose)
                    r.sleep()
                    continue
                if (self.mavros_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                    if(self.set_mode_client.call(self.offb_set_mode).mode_sent == True):
                        print("[nav-ros]: OFFBOARD enabled.")

                    last_req = rospy.Time.now()
                else:
                    if(not self.mavros_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                        if(self.arming_client.call(self.arm_cmd).success == True):
                            print("[nav-ros]: Vehicle armed.")

                        last_req = rospy.Time.now()
            self.pose_pub.publish(takeoff_pose)
            r.sleep()
        print("[nav-ros]: take off completed at height: ", takeoff_height)


    def toggle_safety_stop(self, request):
        self.safety_stop = bool(request.data)
        if self.safety_stop and self.odom_received:
            self.stop_pose = PoseStamped()
            self.stop_pose.pose = self.odom.pose.pose
        return SetBoolResponse(True, "safety_stop={}".format(self.safety_stop))

    def _odom_position(self):
        if not self.odom_received:
            return None
        point = self.odom.pose.pose.position
        return np.array([point.x, point.y, point.z], dtype=float)

    def _mission_snapshot(self, now=None):
        if now is None:
            now = rospy.Time.now()
        elapsed_s = None
        if self.mission_start_time is not None:
            elapsed_s = max(0.0, (now - self.mission_start_time).to_sec())
        average_speed = None
        if elapsed_s is not None and elapsed_s > 0.0:
            average_speed = self.mission_path_length_m / elapsed_s
        observed_speed = None
        if self.mission_speed_samples_mps:
            observed_speed = float(np.mean(self.mission_speed_samples_mps))
        return {
            "mission_active": self.mission_active,
            "mission_result": self.mission_result,
            "mission_completion_time_s": elapsed_s,
            "mission_path_length_m": self.mission_path_length_m if self.mission_start_time else None,
            "mission_average_speed_mps": average_speed,
            "mission_observed_speed_mps": observed_speed,
            "mission_waypoint_error_m": self.mission_goal_distance_m,
            "mission_initial_goal_distance_m": self.mission_initial_goal_distance_m,
            "mission_min_lidar_distance_m": self.mission_min_lidar_distance_m,
            "mission_goal_tolerance_m": self.mission_goal_tolerance_m,
        }

    def _write_mission_event(self, event, now):
        payload = {
            "event": event,
            "timestamp": round(now.to_sec(), 6),
            "run_label": self.run_label,
            "policy_variant": self.policy_variant,
        }
        payload.update(self._mission_snapshot(now))
        self.metrics_logger.write(payload)

    def _start_mission(self, now):
        position = self._odom_position()
        if position is None or self.goal is None:
            return
        target = np.array([
            self.goal.pose.position.x,
            self.goal.pose.position.y,
            self.goal.pose.position.z,
        ], dtype=float)
        self.mission_active = True
        self.mission_result = "running"
        self.mission_start_time = now
        self.mission_start_position = position.copy()
        self.mission_previous_position = position.copy()
        self.mission_path_length_m = 0.0
        self.mission_speed_samples_mps = []
        self.mission_goal_distance_m = float(np.linalg.norm(target - position))
        self.mission_initial_goal_distance_m = self.mission_goal_distance_m
        self.mission_min_lidar_distance_m = None
        self.mission_end_logged = False
        self._write_mission_event("mission_start", now)

    def _update_mission_odometry(self, now):
        if not self.mission_active or self.mission_previous_position is None:
            return
        position = self._odom_position()
        if position is None:
            return
        step_distance = float(np.linalg.norm(position - self.mission_previous_position))
        # Ignore an implausible reset/jump instead of contaminating the path metric.
        if math.isfinite(step_distance) and step_distance <= 2.0:
            self.mission_path_length_m += step_distance
        self.mission_previous_position = position
        velocity = self.odom.twist.twist.linear
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        if math.isfinite(speed):
            self.mission_speed_samples_mps.append(speed)

    def _update_mission_completion(self, now):
        if not self.mission_active or self.goal is None:
            return False
        position = self._odom_position()
        if position is None:
            return False
        target = np.array([
            self.goal.pose.position.x,
            self.goal.pose.position.y,
            self.goal.pose.position.z,
        ], dtype=float)
        self.mission_goal_distance_m = float(np.linalg.norm(target - position))
        if self.mission_goal_distance_m <= self.mission_goal_tolerance_m:
            self.mission_active = False
            self.mission_result = "success"
            self.mission_completed_count += 1
            self._write_mission_event("mission_complete", now)
            return True
        return False

    def close_loggers(self):
        if self.loggers_closed:
            return
        self.loggers_closed = True
        if self.mission_start_time is not None and not self.mission_end_logged:
            if self.mission_active:
                self.mission_active = False
                self.mission_result = "interrupted"
            self._write_mission_event("mission_end", rospy.Time.now())
            self.mission_end_logged = True
        self.metrics_logger.close()
        self.trace_logger.close()

    def perception_watchdog_callback(self, message):
        self.last_perception_status_time = rospy.Time.now()
        new_value = bool(message.data)
        if new_value and not self.perception_degraded:
            self.perception_degrade_count += 1
        self.perception_degraded = new_value

    @staticmethod
    def _stamp_age(now, stamp):
        if stamp == rospy.Time(0):
            return None
        return max(0.0, round((now - stamp).to_sec(), 6))

    def begin_control_cycle(self, event):
        """Reset per-cycle measurements and quantify timer scheduling pressure."""
        callback_start = time.perf_counter()
        if self.last_control_callback_monotonic is None:
            self.last_control_interval_ms = None
            self.last_control_period_jitter_ms = None
        else:
            self.last_control_interval_ms = round(
                (callback_start - self.last_control_callback_monotonic) * 1000.0, 4
            )
            self.last_control_period_jitter_ms = round(
                (callback_start - self.last_control_callback_monotonic - self.control_period) * 1000.0,
                4,
            )
        self.last_control_callback_monotonic = callback_start
        self.last_scheduler_lag_ms = None
        try:
            self.last_scheduler_lag_ms = round(
                max(0.0, (event.current_real - event.current_expected).to_sec() * 1000.0), 4
            )
        except (AttributeError, TypeError):
            pass
        scheduler_late = (
            self.last_scheduler_lag_ms is not None
            and self.last_scheduler_lag_ms > self.timing_degrade_max_lag_ms
        )
        # Wall-clock callback spacing is recorded for diagnosis, but Gazebo
        # software rendering can stretch it without delaying the ROS timer's
        # simulated schedule.  Degrade only on a measured control overrun or
        # an explicit ROS scheduler lag.
        if self.last_control_overrun or scheduler_late:
            self.timing_overrun_streak += 1
        else:
            self.timing_overrun_streak = max(0, self.timing_overrun_streak - 1)
        if (
            self.timing_degrade_enabled
            and self.timing_overrun_streak >= self.timing_degrade_consecutive_cycles
        ):
            if not self.timing_degrade_active:
                self.timing_degrade_count += 1
            self.timing_degrade_active = True
            self.timing_degrade_until = time.monotonic() + self.timing_degrade_hold_s
        elif self.timing_degrade_active and time.monotonic() >= self.timing_degrade_until:
            self.timing_degrade_active = False
        self.inference_this_cycle = False
        self.last_inference_ms = None
        self.last_inference_device_ms = None
        self.last_inference_mode = "none"

    def record_control(self, status, now, raw_action=None, safe_action=None, control_ms=0.0, reason=None):
        """Record a common metric schema for clean sim, perturbed sim, and physical runs."""
        action_delta = None
        safety_intervention = False
        if raw_action is not None and safe_action is not None:
            action_delta = float(np.linalg.norm(np.asarray(raw_action) - np.asarray(safe_action)))
            safety_intervention = action_delta > 1e-4
        dynamic_slots = 0
        if isinstance(self.dynamic_obstacles, (list, tuple)) and len(self.dynamic_obstacles) >= 3:
            dynamic_slots = int(torch.count_nonzero(self.dynamic_obstacles[2][:, 0]).item())
        self.control_callback_count += 1
        resource_metrics = {}
        if self.control_callback_count % self.metrics_resource_stride == 0:
            resource_metrics = runtime_snapshot(torch, self.device)
        current_position = self._odom_position()
        current_goal = None
        if self.goal is not None:
            current_goal = [
                float(self.goal.pose.position.x),
                float(self.goal.pose.position.y),
                float(self.goal.pose.position.z),
            ]
        payload = {
            "event": "control",
            "timestamp": round(now.to_sec(), 6),
            "run_label": self.run_label,
            "status": status,
            "reason": reason,
            # ``inference_ms`` is retained for old analysis scripts. It means
            # caller-observed latency, not CUDA kernel time.
            "inference_ms": None if self.last_inference_ms is None else round(float(self.last_inference_ms), 4),
            "inference_wall_ms": None if self.last_inference_ms is None else round(float(self.last_inference_ms), 4),
            "inference_device_ms": None if self.last_inference_device_ms is None else round(float(self.last_inference_device_ms), 4),
            "inference_mode": self.last_inference_mode if self.inference_this_cycle else "none",
            "control_ms": round(float(control_ms), 4),
            "control_overrun": bool(control_ms > self.control_period * 1000.0),
            "control_interval_ms": self.last_control_interval_ms,
            "control_period_jitter_ms": self.last_control_period_jitter_ms,
            "scheduler_lag_ms": self.last_scheduler_lag_ms,
            "timing_overrun_streak": self.timing_overrun_streak,
            "timing_degrade_active": self.timing_degrade_active,
            "timing_degrade_count": self.timing_degrade_count,
            "odom_age_s": self._stamp_age(now, self.last_odom_time),
            "raycast_age_s": self._stamp_age(now, self.last_raycast_time),
            "obstacle_age_s": self._stamp_age(now, self.last_obstacle_time),
            "perception_status_age_s": self._stamp_age(now, self.last_perception_status_time),
            "perception_degraded": self.perception_degraded,
            "perception_degrade_count": self.perception_degrade_count,
            "perception_lidar_fallback_nearest_m": self.last_lidar_fallback_nearest_m,
            "raycast_points": len(self.raypoints),
            "expected_raycast_points": self.lidar_size,
            "dynamic_obstacle_slots": dynamic_slots,
            "safety_intervention": safety_intervention,
            "action_delta_norm": None if action_delta is None else round(action_delta, 6),
            "raw_action": None if raw_action is None else vector_list(raw_action),
            "safe_action": None if safe_action is None else vector_list(safe_action),
            "policy_action": None if self.last_policy_action is None else vector_list(self.last_policy_action),
            "position": None if current_position is None else vector_list(current_position),
            "goal_position": current_goal,
            "perturbation": self.last_perturbation,
        }
        payload.update(self._mission_snapshot(now))
        payload.update(resource_metrics)
        self.metrics_logger.write(payload)
        self.last_control_overrun = bool(control_ms > self.control_period * 1000.0)

    def record_observation_trace(self, state, lidar, direction, dynamic_obstacle):
        self.trace_count += 1
        if self.trace_count % self.trace_stride != 0:
            return
        self.trace_logger.write({
            "event": "observation",
            "timestamp": round(rospy.Time.now().to_sec(), 6),
            "run_label": self.run_label,
            "state": vector_list(state.detach().cpu().reshape(-1).tolist()),
            "lidar": vector_list(lidar.detach().cpu().reshape(-1).tolist()),
            "direction": vector_list(direction.detach().cpu().reshape(-1).tolist()),
            "dynamic_obstacle": vector_list(dynamic_obstacle.detach().cpu().reshape(-1).tolist()),
            "perturbation": self.last_perturbation,
        })

    def publish_stop(self):
        """Publish a deterministic zero-velocity/hold command on failures."""
        if not self.odom_received:
            return
        if self.px4_control:
            msg = PositionTarget()
            msg.coordinate_frame = msg.FRAME_LOCAL_NED
            msg.header.stamp = rospy.Time.now()
            msg.position.x = self.odom.pose.pose.position.x
            msg.position.y = self.odom.pose.pose.position.y
            msg.position.z = (
                self.takeoff_pose.pose.position.z
                if self.auto_takeoff and hasattr(self, "takeoff_pose")
                else self.odom.pose.pose.position.z
            )
            msg.type_mask = msg.IGNORE_VX + msg.IGNORE_VY + msg.IGNORE_VZ + msg.IGNORE_AFX + msg.IGNORE_AFY + msg.IGNORE_AFZ + msg.IGNORE_YAW + msg.IGNORE_YAW_RATE
            self.action_pub.publish(msg)
        else:
            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            self.action_pub.publish(msg)

    def publish_world_velocity(self, velocity_world, yaw=None):
        """Publish a bounded world-frame velocity for non-policy fallbacks."""
        if not self.odom_received:
            return
        velocity_world = np.asarray(velocity_world, dtype=float).reshape(3)
        if not np.all(np.isfinite(velocity_world)):
            self.publish_stop()
            return
        if yaw is None:
            orientation = self.odom.pose.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            )
        if self.px4_control:
            msg = PositionTarget()
            msg.coordinate_frame = msg.FRAME_LOCAL_NED
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.velocity.x = velocity_world[0]
            msg.velocity.y = velocity_world[1]
            msg.velocity.z = velocity_world[2] if self.height_control else 0.0
            msg.position.z = self.takeoff_pose.pose.position.z
            msg.yaw = yaw
            msg.type_mask = (
                msg.IGNORE_PX + msg.IGNORE_PY + msg.IGNORE_VZ + msg.IGNORE_AFX
                + msg.IGNORE_AFY + msg.IGNORE_AFZ + msg.IGNORE_YAW_RATE
            )
        else:
            quat_no_tilt = tf.transformations.quaternion_from_euler(0, 0, yaw)
            quat_msg = Quaternion()
            quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z = (
                quat_no_tilt[3], quat_no_tilt[0], quat_no_tilt[1], quat_no_tilt[2]
            )
            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.twist.linear = Vector3(*(
                np.linalg.inv(self.quaternion_to_rotation_matrix(quat_msg)) @ velocity_world
            ))
            if not self.height_control:
                msg.twist.linear.z = 0.0
        self.action_pub.publish(msg)

    def lidar_escape_action(self):
        """Return a conservative repulsive action from current LiDAR hit points.

        Raycast endpoints at the configured LiDAR range are free-space
        returns; only points inside ``perception_lidar_escape_distance_m``
        are treated as obstacles. This path intentionally uses no YOLO or
        tracker state.
        """
        position = self._odom_position()
        if position is None or not self.raypoints:
            return None, None
        points = np.asarray(self.raypoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            return None, None
        offsets = points - position
        horizontal = np.linalg.norm(offsets[:, :2], axis=1)
        vertical_ok = np.abs(offsets[:, 2]) <= self.perception_lidar_vertical_band_m
        valid = (
            np.isfinite(horizontal)
            & (horizontal > max(0.05, self.robot_size * 0.25))
            & (horizontal <= self.perception_lidar_escape_distance_m)
            & vertical_ok
        )
        if not np.any(valid):
            return None, None
        close_offsets = offsets[valid]
        close_distances = horizontal[valid]
        unit_away = -close_offsets[:, :2] / close_distances[:, None]
        weights = np.maximum(
            0.05,
            (self.perception_lidar_escape_distance_m - close_distances)
            / self.perception_lidar_escape_distance_m,
        )
        direction = np.sum(unit_away * weights[:, None], axis=0)
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= 1e-6:
            return None, float(np.min(close_distances))
        action = np.array([
            direction[0] / norm * self.perception_lidar_escape_speed_mps,
            direction[1] / norm * self.perception_lidar_escape_speed_mps,
            0.0,
        ])
        return action, float(np.min(close_distances))

    def get_raycast(self, pos: np.array , start_angle: float):
        raypoints = []
        try:
            raycast = rospy.ServiceProxy("occupancy_map/raycast", RayCast)
            pos_msg = Point()
            pos_msg.x = pos[0]
            pos_msg.y = pos[1]
            pos_msg.z = pos[2]
            response = raycast(pos_msg,
                               start_angle,
                               self.cfg.sensor.lidar_range, 
                               self.cfg.sensor.lidar_vfov[0], 
                               self.cfg.sensor.lidar_vfov[1], 
                               self.cfg.sensor.lidar_vbeams, 
                               self.cfg.sensor.lidar_hres
                               )
            num_points = int(len(response.points)/3)
            self.laser_points_msg = response.points
            for i in range(num_points):
                p = [response.points[3*i+0], response.points[3*i+1], response.points[3*i+2]]
                raypoints.append(p)
        except (rospy.service.ServiceException, rospy.ROSException) as e:
            rospy.logwarn_throttle(2.0, "raycast service unavailable: %s", e)
            self.laser_points_msg = []
            self.last_raycast_time = rospy.Time(0)
            return []
        self.last_raycast_time = rospy.Time.now()
        return raypoints

    def get_dynamic_obstacles(self, pos: np.array):
        dynamic_obstacle_pos = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.device)
        dynamic_obstacle_vel = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.device)
        dynamic_obstacle_size = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.device)
        try:
            distance_range = 4.0
            pos_msg = Point()
            pos_msg.x = pos[0]
            pos_msg.y = pos[1]
            pos_msg.z = pos[2]
            get_obstacle = rospy.ServiceProxy("onboard_detector/get_dynamic_obstacles", GetDynamicObstacles)
            response = get_obstacle(pos_msg, distance_range)
            total_obs_num = len(response.position)
            for i in range(self.cfg.algo.feature_extractor.dyn_obs_num):
                if (i < total_obs_num):
                    pos_vec = response.position[i]
                    vel_vec = response.velocity[i]
                    size_vec = response.size[i]
                    dynamic_obstacle_pos[i] = torch.tensor([pos_vec.x, pos_vec.y, pos_vec.z], dtype=torch.float, device=self.device)
                    dynamic_obstacle_vel[i] = torch.tensor([vel_vec.x, vel_vec.y, vel_vec.z], dtype=torch.float, device=self.device)
                    dynamic_obstacle_size[i] = torch.tensor([size_vec.x, size_vec.y, size_vec.z], dtype=torch.float, device=self.device)
        except (rospy.service.ServiceException, rospy.ROSException) as e:
            rospy.logwarn_throttle(2.0, "dynamic obstacle service unavailable: %s", e)
            self.last_obstacle_time = rospy.Time(0)
            return dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size
        self.last_obstacle_time = rospy.Time.now()
        return dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size

    def get_static_obstacles(self):
        static_obstacle_pos = []
        static_obstacle_size = []
        static_obstacle_angle = []
        try:
            get_static_obstacles_server = rospy.ServiceProxy("occupancy_map/get_static_obstacles", GetStaticObstacles)
            static_obstacle_response = get_static_obstacles_server()
            static_obstacle_pos = static_obstacle_response.position
            static_obstacle_size = static_obstacle_response.size
            static_obstacle_angle = static_obstacle_response.angle
        except rospy.service.ServiceException as e:
            print("[nav-ros]: static obstacle func err!")
        return static_obstacle_pos, static_obstacle_size, static_obstacle_angle
           
    def raycast_callback(self, event):
        if not self.odom_received or not self.goal_received:
            return
        pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        start_angle = np.arctan2(self.target_dir[1].cpu().numpy(), self.target_dir[0].cpu().numpy())
        self.raypoints = self.get_raycast(pos, start_angle)

    def dynamic_obstacle_callback(self, event):
        if not self.odom_received:
            return
        pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size = self.get_dynamic_obstacles(pos)
        self.dynamic_obstacles = (dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size)

    def odom_callback(self, odom):
        if not self.odom_received and not self.auto_takeoff:
            self.takeoff_pose.pose = odom.pose.pose
        self.odom = odom
        self.odom_received = True
        self.last_odom_time = rospy.Time.now()
        point = Point()
        point.x = odom.pose.pose.position.x
        point.y = odom.pose.pose.position.y
        point.z = odom.pose.pose.position.z
        self.flight_trajectory.points.append(point)
        # Keep the marker bounded during long runs while retaining enough
        # samples for a smooth line in the demo.
        if len(self.flight_trajectory.points) > 4000:
            self.flight_trajectory.points = self.flight_trajectory.points[-4000:]
        self.flight_trajectory.header.stamp = odom.header.stamp
        self.flight_trajectory_pub.publish(self.flight_trajectory)
        self._update_mission_odometry(self.last_odom_time)
    
    def state_callback(self, state):
        self.mavros_state = state
    
    def goal_callback(self, goal):
        if not self.odom_received:
            return

        previous_goal = None
        if self.goal is not None:
            previous_goal = np.array([
                self.goal.pose.position.x,
                self.goal.pose.position.y,
                self.goal.pose.position.z,
            ], dtype=float)
        self.goal = goal
        if hasattr(self, "takeoff_pose"):
            self.goal.pose.position.z = self.takeoff_pose.pose.position.z
        dir_x = self.goal.pose.position.x - self.odom.pose.pose.position.x
        dir_y = self.goal.pose.position.y - self.odom.pose.pose.position.y
        dir_z = self.goal.pose.position.z - self.odom.pose.pose.position.z
        self.target_dir = safe_direction(torch.tensor([dir_x, dir_y, dir_z], device=self.device))

        self.goal_received = True
        self.stable_times = 0
        current_goal = np.array([self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z])
        if (
            not self.mission_active
            or previous_goal is None
            or float(np.linalg.norm(current_goal - previous_goal)) > 0.05
        ):
            self._start_mission(rospy.Time.now())

    def quaternion_to_rotation_matrix(self, quaternion):
        # w, x, y, z = quaternion
        w = quaternion.w
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        xx, xy, xz = x**2, x*y, x*z
        yy, yz = y**2, y*z
        zz = z**2
        wx, wy, wz = w*x, w*y, w*z
        
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])        
    
    def check_obstacle(self, lidar_scan, dyn_obs_states):
        # return true if there is obstacles in the range
        # has_static = not torch.all(lidar_scan == 0.)
        # has_static = not torch.all(lidar_scan[..., 1:] < 0.2) # hardcode to tune
        quarter_size = lidar_scan.shape[2] // 4
        first_quarter_check, last_quarter_check = torch.all(lidar_scan[:, :, :quarter_size, 1:] < 0.2), torch.all(lidar_scan[:, :, -quarter_size:, 1:] < 0.2)
        has_static = (not first_quarter_check) or (not last_quarter_check)
        has_dynamic = not torch.all(dyn_obs_states == 0.)
        return has_static or has_dynamic

    def get_safe_action(self, vel_world, action_vel_world, include_dynamic=True):
        safe_action = np.zeros(3)
        if not isinstance(self.dynamic_obstacles, (list, tuple)) or len(self.dynamic_obstacles) < 3:
            return action_vel_world
        try:
            pos_msg = Point(x=self.odom.pose.pose.position.x, y=self.odom.pose.pose.position.y, z=self.odom.pose.pose.position.z)
            get_safe_action = rospy.ServiceProxy("rl_navigation/get_safe_action", GetSafeAction)
            vel_msg = Vector3(x=vel_world[0].item(), y=vel_world[1].item(), z=vel_world[2].item())
            action_vel_msg = Vector3(x=action_vel_world[0], y=action_vel_world[1], z=action_vel_world[2])
            max_vel = np.sqrt(3. * self.cfg.algo.actor.action_limit**2)
            obstacle_pos_list = []
            obstacle_vel_list = []
            obstacle_size_list = []
            if include_dynamic:
                for i in range(len(self.dynamic_obstacles[0])):
                    if (self.dynamic_obstacles[2][i][0] != 0):
                        obs_pos = Vector3(x=self.dynamic_obstacles[0][i][0].item(), y=self.dynamic_obstacles[0][i][1].item(), z=self.dynamic_obstacles[0][i][2].item())
                        obs_vel = Vector3(x=self.dynamic_obstacles[1][i][0].item(), y=self.dynamic_obstacles[1][i][1].item(), z=self.dynamic_obstacles[1][i][2].item())
                        obs_size = Vector3(x=self.dynamic_obstacles[2][i][0].item(), y=self.dynamic_obstacles[2][i][1].item(), z=self.dynamic_obstacles[2][i][2].item())
                        obstacle_pos_list.append(obs_pos)
                        obstacle_vel_list.append(obs_vel)
                        obstacle_size_list.append(obs_size)
            response = get_safe_action(pos_msg, vel_msg, self.robot_size, obstacle_pos_list, obstacle_vel_list,\
                                    obstacle_size_list, self.laser_points_msg, self.cfg.sensor.lidar_range,\
                                    max(self.raycast_vres, self.raycast_hres), max_vel, action_vel_msg)
            safe_action = np.array([response.safe_action.x, response.safe_action.y, response.safe_action.z])
            return safe_action
        except (rospy.service.ServiceException, rospy.ROSException, ValueError, TypeError) as e:
            rospy.logwarn_throttle(2.0, "Safety service unavailable: %s", e)
            return np.zeros(3) if self.require_safety_shield else action_vel_world

    def get_safe_action_map(self, vel_world, action_vel_world):
        safe_action = np.zeros(3)
        try:
            pos_msg = Point(x=self.odom.pose.pose.position.x, y=self.odom.pose.pose.position.y, z=self.odom.pose.pose.position.z)
            get_safe_action = rospy.ServiceProxy("rl_navigation/get_safe_action_map", GetSafeActionMap)
            vel_msg = Vector3(x=vel_world[0].item(), y=vel_world[1].item(), z=vel_world[2].item())
            action_vel_msg = Vector3(x=action_vel_world[0], y=action_vel_world[1], z=action_vel_world[2])
            max_vel = np.sqrt(3. * self.cfg.algo.actor.action_limit**2)
            
            # Dynamic Obstacles
            obstacle_pos_list = []
            obstacle_vel_list = []
            obstacle_size_list = []
            for i in range(len(self.dynamic_obstacles[0])):
                if (self.dynamic_obstacles[2][i][0] != 0):
                    obs_pos = Vector3(x=self.dynamic_obstacles[0][i][0].item(), y=self.dynamic_obstacles[0][i][1].item(), z=self.dynamic_obstacles[0][i][2].item())
                    obs_vel = Vector3(x=self.dynamic_obstacles[1][i][0].item(), y=self.dynamic_obstacles[1][i][1].item(), z=self.dynamic_obstacles[1][i][2].item())
                    obs_size = Vector3(x=self.dynamic_obstacles[2][i][0].item(), y=self.dynamic_obstacles[2][i][1].item(), z=self.dynamic_obstacles[2][i][2].item())
                    obstacle_pos_list.append(obs_pos)
                    obstacle_vel_list.append(obs_vel)
                    obstacle_size_list.append(obs_size)
            
            # Static Obstacles
            static_obstacle_pos, static_obstacle_size, static_obstacle_angle = self.get_static_obstacles()

            response = get_safe_action(pos_msg, vel_msg, self.robot_size, obstacle_pos_list, obstacle_vel_list,\
                                    obstacle_size_list, static_obstacle_pos, static_obstacle_size,\
                                    static_obstacle_angle, max_vel, action_vel_msg)
            safe_action = np.array([response.safe_action.x, response.safe_action.y, response.safe_action.z])
            return safe_action
        except (rospy.service.ServiceException, rospy.ROSException, ValueError, TypeError) as e:
            rospy.logwarn_throttle(2.0, "Map safety service unavailable: %s", e)
            return np.zeros(3) if self.require_safety_shield else action_vel_world
    
    def get_action(self, pos: torch.Tensor, vel: torch.Tensor, goal: torch.Tensor):
        """Build the training-compatible observation and infer a world velocity."""
        target_dir_2d = safe_direction(self.target_dir).to(self.device)
        drone_state, target_dir = build_state_observation(pos, vel, goal, target_dir_2d)
        try:
            lidar_scan = build_lidar_observation(
                self.raypoints, pos, float(self.cfg.sensor.lidar_range),
                self.lidar_hbeams, int(self.cfg.sensor.lidar_vbeams)
            )
        except (TypeError, ValueError) as exc:
            rospy.logwarn_throttle(2.0, "Invalid LiDAR observation: %s", exc)
            return torch.zeros(1, 1, 3, device=self.device)
        dyn_obs_states = build_dynamic_obstacle_observation(
            pos, target_dir_2d, self.dynamic_obstacles[0],
            self.dynamic_obstacles[1], self.dynamic_obstacles[2], self.robot_size
        )
        drone_state, lidar_scan, dyn_obs_states, self.last_perturbation = self.perturber.apply(
            drone_state, lidar_scan, dyn_obs_states
        )
        self.record_observation_trace(drone_state, lidar_scan, target_dir, dyn_obs_states)
        # states
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": drone_state,
                    "lidar": lidar_scan,
                    "direction": target_dir,
                    "dynamic_obstacle": dyn_obs_states
                })
            })
        })

        has_obstacle_in_range = self.check_obstacle(lidar_scan, dyn_obs_states)
        # if (False):
        if has_obstacle_in_range or self.policy_always_on:
            if (not self.use_policy_server):
                def policy_forward():
                    with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
                        return self.policy.act(obs) if hasattr(self.policy, "act") else self.policy(obs)

                output, self.last_inference_ms, self.last_inference_device_ms = timed_call(
                    policy_forward, torch, self.device
                )
                vel_world = output["agents", "action"]
                self.last_inference_mode = "local_policy"
                self.inference_this_cycle = True
            else:
                try:
                    get_policy_inference = rospy.ServiceProxy("rl_navigation/GetPolicyInference", GetPolicyInference)

                    def request_policy():
                        return get_policy_inference(
                            obs["agents"]["observation"]["state"].cpu().numpy().flatten().tolist(),
                            obs["agents"]["observation"]["state"].size(),
                            obs["agents"]["observation"]["lidar"].cpu().numpy().flatten().tolist(),
                            obs["agents"]["observation"]["lidar"].size(),
                            obs["agents"]["observation"]["direction"].cpu().numpy().flatten().tolist(),
                            obs["agents"]["observation"]["direction"].size(),
                            obs["agents"]["observation"]["dynamic_obstacle"].cpu().numpy().flatten().tolist(),
                            obs["agents"]["observation"]["dynamic_obstacle"].size(),
                        )

                    response, self.last_inference_ms, _ = timed_call(request_policy)
                    vel_world = torch.tensor(response.action, device=self.device, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                    self.last_inference_device_ms = None
                    self.last_inference_mode = "policy_service_round_trip"
                    self.inference_this_cycle = True
                except rospy.service.ServiceException as e:
                    print("[nav-ros]: Policy server err!")
                    vel_world = torch.tensor([0., 0., 0.], device=self.device).unsqueeze(0).unsqueeze(0)
                    self.last_inference_ms = None
                    self.last_inference_device_ms = None
        else:
            direction = goal - pos
            cruise_speed = self.cruise_speed_mps or float(self.cfg.algo.actor.action_limit)
            vel_world = direction / direction.norm().clamp_min(1e-6) * cruise_speed
            self.last_inference_ms = 0.0
            self.last_inference_device_ms = None
            self.last_inference_mode = "goal_tracking"
            self.inference_this_cycle = True
        vel_world = torch.nan_to_num(vel_world, nan=0.0, posinf=0.0, neginf=0.0)
        max_speed = float(self.cfg.algo.actor.action_limit)
        if self.cruise_speed_mps > 0.0 and self.last_inference_mode == "goal_tracking":
            max_speed = min(max_speed, self.cruise_speed_mps)
        speed = vel_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vel_world = vel_world * (max_speed / speed).clamp_max(1.0)
        return vel_world

    def compose_deployment_action(self, pos, goal, policy_action):
        """Keep the learned action visible while making the ROS demo trackable.

        The checkpoint was trained with Isaac dynamics, while this Gazebo
        plugin exposes a velocity interface.  A small goal-directed component
        removes that dynamics mismatch; the learned action remains the
        residual and the LiDAR tangent chooses a side when a wall is close.
        """
        action = np.asarray(policy_action, dtype=float).reshape(3).copy()
        if not self.deployment_guidance_enabled or not self.policy_always_on:
            if not self.height_control:
                action[2] = 0.0
            return action

        position = np.asarray(pos.detach().cpu().numpy(), dtype=float).reshape(3)
        target = np.asarray(goal.detach().cpu().numpy(), dtype=float).reshape(3)
        delta = target - position
        horizontal = delta[:2]
        distance = float(np.linalg.norm(horizontal))
        goal_velocity = np.zeros(3, dtype=float)
        if distance > self.mission_goal_tolerance_m:
            goal_velocity[:2] = horizontal / max(distance, 1e-6) * self.goal_guidance_speed_mps

        weight = self.policy_residual_weight
        action = weight * action + (1.0 - weight) * goal_velocity

        # Select a tangential direction around the nearest LiDAR return.  The
        # side whose tangent points closer to the goal is used, so a wall does
        # not turn the mission into an arbitrary retreat.
        points = np.asarray(self.raypoints, dtype=float)
        if points.ndim == 2 and points.shape[1] >= 2 and points.size:
            offsets = points[:, :2] - position[:2]
            distances = np.linalg.norm(offsets, axis=1)
            valid = np.isfinite(distances) & (distances > 0.08)
            if np.any(valid):
                nearest_index = int(np.where(valid)[0][np.argmin(distances[valid])])
                nearest_distance = float(distances[nearest_index])
                if nearest_distance < self.lidar_avoidance_distance_m:
                    obstacle_vector = offsets[nearest_index] / max(nearest_distance, 1e-6)
                    away = -obstacle_vector
                    tangent_a = np.array([-away[1], away[0]])
                    tangent_b = -tangent_a
                    goal_unit = horizontal / max(distance, 1e-6) if distance > 1e-6 else np.array([1.0, 0.0])
                    tangent = tangent_a if np.dot(tangent_a, goal_unit) >= np.dot(tangent_b, goal_unit) else tangent_b
                    strength = (self.lidar_avoidance_distance_m - nearest_distance) / self.lidar_avoidance_distance_m
                    action[:2] += strength * (0.35 * away + 0.65 * tangent)

        if not self.height_control:
            action[2] = 0.0
        limit = float(self.cfg.algo.actor.action_limit)
        norm = float(np.linalg.norm(action))
        if norm > limit:
            action *= limit / max(norm, 1e-6)
        return action


    def get_rollout_traj(self, pos: torch.Tensor, vel: torch.Tensor, goal: torch.Tensor, dt=0.1, horizon=3.0):
        traj = [pos.cpu().detach().numpy()]
        t = 0.
        while (t < horizon):
            vel_curr_world = self.get_action(pos, vel, goal)
            t += dt
            pos = (pos + dt * vel_curr_world).squeeze(0).squeeze(0)
            vel = vel_curr_world.squeeze(0).squeeze(0)
            traj.append(pos.cpu().detach().numpy())
        return np.array(traj)

    def control_callback(self, event):
        self.begin_control_cycle(event)
        if (not self.odom_received):
            return

        now = rospy.Time.now()
        if self.mission_active and self.raypoints:
            position = self._odom_position()
            if position is not None:
                offsets = np.asarray(self.raypoints, dtype=float) - position
                distances = np.linalg.norm(offsets[:, :2], axis=1)
                valid = distances[np.isfinite(distances) & (distances > self.robot_size * 0.25)]
                if valid.size:
                    nearest = float(np.min(valid))
                    self.mission_min_lidar_distance_m = (
                        nearest if self.mission_min_lidar_distance_m is None
                        else min(self.mission_min_lidar_distance_m, nearest)
                    )
        odom_fresh = (now - self.last_odom_time).to_sec() <= self.sensor_timeout
        raycast_fresh = (now - self.last_raycast_time).to_sec() <= self.sensor_timeout
        obstacle_fresh = (now - self.last_obstacle_time).to_sec() <= self.obstacle_timeout
        perception_status_fresh = (
            self.last_perception_status_time != rospy.Time(0)
            and (now - self.last_perception_status_time).to_sec() <= self.perception_timeout
        )
        dynamic_ready = isinstance(self.dynamic_obstacles, (list, tuple)) and len(self.dynamic_obstacles) >= 3
        self._update_mission_completion(now)
        watchdog_fault = self.perception_watchdog_enabled and (
            self.perception_degraded or (self.perception_watchdog_required and not perception_status_fresh)
        )
        if (not odom_fresh or not self.goal_received or len(self.raypoints) != self.lidar_size or not raycast_fresh):
            self.publish_stop()
            self.record_control("hold", now, reason="stale_or_incomplete_observation")
            return

        if watchdog_fault:
            escape_action, nearest_lidar_distance = self.lidar_escape_action()
            self.last_lidar_fallback_nearest_m = nearest_lidar_distance
            if self.perception_lidar_fallback_enabled and escape_action is not None:
                vel_body = np.array([
                    self.odom.twist.twist.linear.x,
                    self.odom.twist.twist.linear.y,
                    self.odom.twist.twist.linear.z,
                ])
                vel_world = torch.tensor(
                    self.quaternion_to_rotation_matrix(self.odom.pose.pose.orientation) @ vel_body,
                    device=self.device,
                    dtype=torch.float,
                )
                safe_escape = self.get_safe_action(vel_world, escape_action, include_dynamic=False)
                if np.all(np.isfinite(safe_escape)) and np.linalg.norm(safe_escape) > 1e-5:
                    self.publish_world_velocity(safe_escape)
                    self.record_control(
                        "command",
                        now,
                        raw_action=escape_action,
                        safe_action=safe_escape,
                        reason="perception_lidar_escape",
                    )
                    return
            self.publish_stop()
            self.record_control(
                "hold",
                now,
                reason="perception_watchdog_degraded_lidar_clear"
                if nearest_lidar_distance is None else "perception_watchdog_degraded_lidar_escape_blocked",
            )
            return

        self.last_lidar_fallback_nearest_m = None
        if not dynamic_ready or not obstacle_fresh:
            self.publish_stop()
            self.record_control("hold", now, reason="stale_or_incomplete_observation")
            return

        if (self.safety_stop):
            self.publish_stop()
            self.record_control("hold", now, reason="software_safety_stop")
            return

        if self.timing_degrade_active:
            # Never reuse a late policy action after a consecutive scheduling
            # overrun.  Position hold keeps the PX4 offboard stream alive while
            # the timer recovers, and TwistStamped zero is the ROS1 fallback.
            self.publish_stop()
            self.record_control("hold", now, reason="control_timing_degraded")
            return
        
        start_time = time.perf_counter()
        # check for angle
        goal_angle = np.arctan2(self.target_dir[1].cpu().numpy(), self.target_dir[0].cpu().numpy())
        _, _, curr_angle = tf.transformations.euler_from_quaternion([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w])
        angle_diff = np.abs(goal_angle - curr_angle)
        if (angle_diff > math.pi):
            angle_diff = np.abs(angle_diff - math.pi * 2)
        if (angle_diff >= 0.1):
            pose_msg = PoseStamped()
            pose_msg.pose = self.odom.pose.pose
            quaternion = tf.transformations.quaternion_from_euler(0, 0, goal_angle)
            pose_msg.pose.orientation.w = quaternion[3]
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            self.pose_pub.publish(pose_msg)
            self.record_control("aligning", now, control_ms=(time.perf_counter() - start_time) * 1000.0)
            return
        else:
            self.stable_times += 1
            if (self.stable_times <= 10):
                self.record_control("stabilizing", now, control_ms=(time.perf_counter() - start_time) * 1000.0)
                return

        pos = torch.tensor([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z], device=self.device)
        goal = torch.tensor([self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z], device=self.device)
        orientation = torch.tensor([self.odom.pose.pose.orientation.w, self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z], device=self.cfg.device)
        rot = self.quaternion_to_rotation_matrix(self.odom.pose.pose.orientation)
        vel_body = np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_world = torch.tensor(rot @ vel_body, device=self.device, dtype=torch.float) # world vel
        
        # get RL action from model
        policy_cmd_vel_world = self.get_action(pos, vel_world, goal).squeeze(0).squeeze(0).detach().cpu().numpy()
        if not np.all(np.isfinite(policy_cmd_vel_world)):
            rospy.logwarn_throttle(2.0, "Policy returned invalid action; publishing stop")
            self.publish_stop()
            self.record_control("hold", now, reason="invalid_policy_action")
            return
        self.last_policy_action = policy_cmd_vel_world.copy()
        cmd_vel_world = self.compose_deployment_action(pos, goal, policy_cmd_vel_world)
        # 匀速目标跟踪只用于显式关闭 policy_always_on 的兼容模式。
        if self.cruise_speed_mps > 0.0 and not self.policy_always_on:
            goal_delta = goal.detach().cpu().numpy() - pos.detach().cpu().numpy()
            goal_distance = float(np.linalg.norm(goal_delta))
            if goal_distance > self.mission_goal_tolerance_m:
                cmd_vel_world = goal_delta / max(goal_distance, 1e-6) * min(
                    self.cruise_speed_mps, float(self.cfg.algo.actor.action_limit)
                )
            else:
                cmd_vel_world = np.zeros(3, dtype=float)
        self.cmd_vel_world = cmd_vel_world.copy()

        # get safe action
        safe_cmd_vel_world = self.get_safe_action(vel_world, cmd_vel_world)
        if not np.all(np.isfinite(safe_cmd_vel_world)):
            rospy.logwarn_throttle(2.0, "Safety layer returned invalid action; publishing stop")
            self.publish_stop()
            self.record_control("hold", now, raw_action=cmd_vel_world, reason="invalid_safety_action")
            return
        # safe_cmd_vel_world = self.get_safe_action_map(vel_world, cmd_vel_world)
        # safe_cmd_vel_world[2] = 0
        self.safe_cmd_vel_world = safe_cmd_vel_world.copy()

        quat_no_tilt = tf.transformations.quaternion_from_euler(0, 0, curr_angle)
        quat_msg = Quaternion()
        quat_msg.w = quat_no_tilt[3]
        quat_msg.x = quat_no_tilt[0]
        quat_msg.y = quat_no_tilt[1]
        quat_msg.z = quat_no_tilt[2]
        rot_no_tilt = self.quaternion_to_rotation_matrix(quat_msg)
        safe_cmd_vel_local = np.linalg.inv(rot_no_tilt) @ safe_cmd_vel_world

        # Goal condition
        distance = (pos - goal).norm() 
        if (distance <= 3. and distance > 0.3 and self.cruise_speed_mps <= 0.0):
            if (np.linalg.norm(safe_cmd_vel_local) != 0):
                safe_cmd_vel_local = 0.5 * safe_cmd_vel_local/np.linalg.norm(safe_cmd_vel_local)
                safe_cmd_vel_world = 0.5 * safe_cmd_vel_world/np.linalg.norm(safe_cmd_vel_world)
        elif (distance <= 1.0):
            safe_cmd_vel_local *= 0.
            safe_cmd_vel_world *= 0.

        # final action
        if (self.px4_control):
            final_cmd_vel = PositionTarget()
            final_cmd_vel.coordinate_frame = final_cmd_vel.FRAME_LOCAL_NED
            final_cmd_vel.header.stamp = rospy.Time.now()
            final_cmd_vel.header.frame_id = "map"
            if (self.height_control):
                final_cmd_vel.velocity.x = safe_cmd_vel_world[0]
                final_cmd_vel.velocity.y = safe_cmd_vel_world[1]
                final_cmd_vel.velocity.z = safe_cmd_vel_world[2]
                final_cmd_vel.yaw = goal_angle
                final_cmd_vel.type_mask = final_cmd_vel.IGNORE_PX + final_cmd_vel.IGNORE_PY + final_cmd_vel.IGNORE_PZ + \
                    final_cmd_vel.IGNORE_AFX + final_cmd_vel.IGNORE_AFY + final_cmd_vel.IGNORE_AFZ + final_cmd_vel.IGNORE_YAW_RATE
            else:
                final_cmd_vel.velocity.x = safe_cmd_vel_world[0]
                final_cmd_vel.velocity.y = safe_cmd_vel_world[1]
                final_cmd_vel.position.z = self.takeoff_pose.pose.position.z
                final_cmd_vel.yaw = goal_angle
                final_cmd_vel.type_mask = final_cmd_vel.IGNORE_PX + final_cmd_vel.IGNORE_PY + final_cmd_vel.IGNORE_VZ + \
                    final_cmd_vel.IGNORE_AFX + final_cmd_vel.IGNORE_AFY + final_cmd_vel.IGNORE_AFZ + final_cmd_vel.IGNORE_YAW_RATE                           
        else:
            final_cmd_vel = TwistStamped()
            final_cmd_vel.header.stamp = rospy.Time.now()
            final_cmd_vel.twist.linear.x = safe_cmd_vel_local[0]
            final_cmd_vel.twist.linear.y = safe_cmd_vel_local[1]
            if (self.height_control):
                final_cmd_vel.twist.linear.z = safe_cmd_vel_world[2]
            else:
                final_cmd_vel.twist.linear.z = 0
        self.action_pub.publish(final_cmd_vel)
        self.has_action = True
        self.last_action_time = now

        # rollout_traj = self.get_rollout_traj(pos, vel_world, goal, dt=0.1, horizon=3.0)
        # traj_msg = Path()
        # traj_msg.header.frame_id = "map"
        # for i in range(len(rollout_traj)):
        #     p = PoseStamped()
        #     p.pose.position.x = rollout_traj[i][0]
        #     p.pose.position.y = rollout_traj[i][1]
        #     p.pose.position.z = rollout_traj[i][2]
        #     traj_msg.poses.append(p)
        # self.rollout_traj_pub.publish(traj_msg)
        end_time = time.perf_counter()
        self.record_control(
            "command",
            now,
            raw_action=cmd_vel_world,
            safe_action=safe_cmd_vel_world,
            control_ms=(end_time - start_time) * 1000.0,
        )
        # print("[nav-ros]: control time ", end_time - start_time)
        
    def pause_sim():
        rospy.wait_for_service('/gazebo/pause_physics')
        pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        pause()

    def unpause_sim():
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause()

    def run(self):
        period = rospy.Duration(self.control_period)
        self.raycast_timer = rospy.Timer(period, self.raycast_callback)
        self.raycast_vis_timer = rospy.Timer(period, self.raycast_vis_callback)
        self.control_timer = rospy.Timer(period, self.control_callback)
        self.goal_vis_timer = rospy.Timer(period, self.goal_vis_callback)
        self.dynamic_obstacle_timer = rospy.Timer(period, self.dynamic_obstacle_callback)
        self.dynamic_obstacle_vis_timer = rospy.Timer(period, self.dynamic_obstacle_vis_callback)
        self.cmd_vis_timer = rospy.Timer(period, self.cmd_vis_callback)

    def raycast_vis_callback(self, event):
        if not self.odom_received and not self.goal_received:
            return
        msg = MarkerArray()
        pos = self.odom.pose.pose.position
        direction_init = None
        for i in range(len(self.raypoints)):
            point = Marker()
            point.header.frame_id = "map"
            point.header.stamp = rospy.get_rostime()
            point.ns = "raycast_points"
            point.id = i
            point.type = point.SPHERE
            point.action = point.ADD
            point.pose.position.x = self.raypoints[i][0]
            point.pose.position.y = self.raypoints[i][1]
            point.pose.position.z = self.raypoints[i][2]
            point.lifetime = rospy.Time(0.5)
            point.scale.x = 0.1
            point.scale.y = 0.1
            point.scale.z = 0.1
            point.color.a = 1.0
            point.color.r = 1.0
            msg.markers.append(point)

            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = rospy.get_rostime()
            line.ns = "raycast_lines"
            line.id = i
            line.type = line.LINE_LIST
            p = Point()
            p.x = self.raypoints[i][0]
            p.y = self.raypoints[i][1]
            p.z = self.raypoints[i][2]
            line.points.append(p)
            line.points.append(pos)
            line.scale.x = 0.03
            line.scale.y = 0.03
            line.scale.z = 0.03
            x_diff = (p.x - self.odom.pose.pose.position.x)
            y_diff = (p.y - self.odom.pose.pose.position.y)
            direction = np.array([x_diff, y_diff])
            direction = direction/np.linalg.norm(direction)
            if (i == 0 or (np.linalg.norm(direction - direction_init) <= 0.1)):
                line.color.b = 1.0
                line.color.a = 1.0
                if (i == 0):
                    direction_init = direction
            else:
                line.color.g = 1.0
                line.color.a = 0.5
            line.lifetime = rospy.Time(0.5)
            msg.markers.append(line)
        self.raycast_vis_pub.publish(msg)
    
    def goal_vis_callback(self, event):
        if not self.goal_received:
            return
        msg = MarkerArray()
        goal_point = Marker()
        goal_point.header.frame_id = "map"
        goal_point.header.stamp = rospy.get_rostime()
        goal_point.ns = "goal_point"
        goal_point.id = 1
        goal_point.type = goal_point.SPHERE
        goal_point.action = goal_point.ADD
        goal_point.pose.position.x = self.goal.pose.position.x
        goal_point.pose.position.y = self.goal.pose.position.y
        goal_point.pose.position.z = self.goal.pose.position.z
        goal_point.lifetime = rospy.Time(0.1)
        goal_point.scale.x = 0.3
        goal_point.scale.y = 0.3
        goal_point.scale.z = 0.3
        goal_point.color.r = 1.0
        goal_point.color.b = 1.0
        goal_point.color.a = 1.0
        msg.markers.append(goal_point)
        self.goal_vis_pub.publish(msg)

    def dynamic_obstacle_vis_callback(self, event):
        if (len(self.dynamic_obstacles) == 0):
            return
        dynamic_obstacle_pos = self.dynamic_obstacles[0]
        dynamic_obstacle_size = self.dynamic_obstacles[2]

        msg = MarkerArray()
        for i in range(dynamic_obstacle_pos.size(0)):
            pos = dynamic_obstacle_pos[i]
            size = dynamic_obstacle_size[i]

            # Increase the width
            width = torch.max(size[0], size[1])
            height = size[2]

            # Create the marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "dynamic_obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2] 
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = width
            marker.scale.y = width
            marker.scale.z = height 
            marker.color.a = 0.5  # Alpha value
            marker.color.r = 1.0  # Red color
            marker.color.g = 0.0
            marker.color.b = 0.0

            msg.markers.append(marker)

        # Publish the marker array
        self.dynamic_obstacle_vis_pub.publish(msg)            

    def cmd_vis_callback(self, event):
        if (not self.has_action):
            return
        msg = MarkerArray()

        # rl action vis
        rl_action_arrow = Marker()
        rl_action_arrow.header.frame_id = "map"
        rl_action_arrow.header.stamp = rospy.get_rostime()
        rl_action_arrow.ns = "rl_action"
        rl_action_arrow.id = 0
        rl_action_arrow.type = rl_action_arrow.ARROW
        rl_action_arrow.action = rl_action_arrow.ADD

        # start
        agent_pos = Point()
        agent_pos.x = self.odom.pose.pose.position.x
        agent_pos.y = self.odom.pose.pose.position.y
        agent_pos.z = self.odom.pose.pose.position.z
        
        # end
        vel_end = Point()
        vel_end.x = self.cmd_vel_world[0] + agent_pos.x
        vel_end.y = self.cmd_vel_world[1] + agent_pos.y
        vel_end.z = self.cmd_vel_world[2] + agent_pos.z

        rl_action_arrow.points.append(agent_pos)
        rl_action_arrow.points.append(vel_end)
        rl_action_arrow.lifetime = rospy.Duration(0.1)
        rl_action_arrow.scale.x = 0.06
        rl_action_arrow.scale.y = 0.06
        rl_action_arrow.scale.z = 0.06
        rl_action_arrow.color.a = 1.0
        rl_action_arrow.color.r = 1.0
        rl_action_arrow.color.g = 0.0
        rl_action_arrow.color.b = 0.0
        msg.markers.append(rl_action_arrow)


        # safe action vis
        safe_action_arrow = Marker()
        safe_action_arrow.header.frame_id = "map"
        safe_action_arrow.header.stamp = rospy.get_rostime()
        safe_action_arrow.ns = "safe_action"
        safe_action_arrow.id = 1
        safe_action_arrow.type = safe_action_arrow.ARROW
        safe_action_arrow.action = safe_action_arrow.ADD

        # start
        agent_pos = Point()
        agent_pos.x = self.odom.pose.pose.position.x
        agent_pos.y = self.odom.pose.pose.position.y
        agent_pos.z = self.odom.pose.pose.position.z

        # end
        vel_end = Point()
        vel_end.x = self.safe_cmd_vel_world[0] + agent_pos.x
        vel_end.y = self.safe_cmd_vel_world[1] + agent_pos.y
        vel_end.z = self.safe_cmd_vel_world[2] + agent_pos.z

        safe_action_arrow.points.append(agent_pos)
        safe_action_arrow.points.append(vel_end)
        safe_action_arrow.lifetime = rospy.Duration(0.1)
        safe_action_arrow.scale.x = 0.06
        safe_action_arrow.scale.y = 0.06
        safe_action_arrow.scale.z = 0.06
        safe_action_arrow.color.a = 1.0
        safe_action_arrow.color.r = 0.0
        safe_action_arrow.color.g = 1.0
        safe_action_arrow.color.b = 0.0

        msg.markers.append(safe_action_arrow)
        self.cmd_vis_pub.publish(msg)
