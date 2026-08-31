#!/usr/bin/env python3
"""ROS 2 MAVROS acceptance flight for an isolated PX4 SITL instance.

The test is intentionally scoped to simulation: wait for FCU telemetry, stream
setpoints, enter OFFBOARD, arm, hold one metre AGL, then request AUTO.LAND.
All service round-trip and pose observations are saved for audit.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import EstimatorStatus, State, StatusText
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = int(math.floor(index)), int(math.ceil(index))
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (index - low)


class SitlAcceptance(Node):
    def __init__(self, height):
        super().__init__("navrl_px4_mavros_sitl_acceptance")
        self.height = height
        self.state = None
        self.odom = None
        self.estimator = None
        self.status_text = []
        self.pose_samples = []
        self.setpoint_times = []
        self.publisher = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
        self.create_subscription(State, "/mavros/state", self.state_callback, 20)
        # MAVROS publishes local odometry as best-effort sensor data.  A ROS 2
        # subscriber with the default reliable profile is incompatible and gets
        # no samples, even when the FCU is already connected.
        self.create_subscription(Odometry, "/mavros/local_position/odom", self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(EstimatorStatus, "/mavros/estimator_status", self.estimator_callback, 10)
        self.create_subscription(StatusText, "/mavros/statustext/recv", self.status_text_callback, qos_profile_sensor_data)
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")

    def state_callback(self, message):
        self.state = message

    def odom_callback(self, message):
        self.odom = message
        self.pose_samples.append((time.monotonic(), message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z))

    def estimator_callback(self, message):
        self.estimator = message

    def status_text_callback(self, message):
        self.status_text.append(message.text)
        self.status_text = self.status_text[-20:]

    def publish_setpoint(self):
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.pose.position.z = self.height
        message.pose.orientation.w = 1.0
        self.publisher.publish(message)
        self.setpoint_times.append(time.monotonic())


def spin_until(node, predicate, timeout, keepalive=False):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        started = time.monotonic()
        if keepalive:
            node.publish_setpoint()
        rclpy.spin_once(node, timeout_sec=0.01 if keepalive else 0.1)
        if predicate():
            return True
        if keepalive:
            remaining = 0.05 - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
    return False


def stream_setpoints(node, duration, rate_hz=20.0):
    """Publish at a real wall-clock rate accepted by PX4 OFFBOARD checks."""
    period = 1.0 / rate_hz
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        started = time.monotonic()
        node.publish_setpoint()
        rclpy.spin_once(node, timeout_sec=0.0)
        remaining = period - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)


def estimator_snapshot(message):
    if message is None:
        return None
    fields = (
        "attitude_status_flag", "velocity_horiz_status_flag", "velocity_vert_status_flag",
        "pos_horiz_rel_status_flag", "pos_horiz_abs_status_flag", "pos_vert_abs_status_flag",
        "pos_vert_agl_status_flag", "const_pos_mode_status_flag", "gps_glitch_status_flag",
        "accel_error_status_flag",
    )
    return {field: bool(getattr(message, field)) for field in fields}


def reach_altitude(node, target_height, timeout, tolerance):
    samples = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        started = time.monotonic()
        node.publish_setpoint()
        rclpy.spin_once(node, timeout_sec=0.0)
        if node.odom is not None:
            altitude = node.odom.pose.pose.position.z
            samples.append(altitude)
            if abs(altitude - target_height) <= tolerance:
                return samples, True
        remaining = 0.05 - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)
    return samples, False


def invoke(node, client, request, timeout, keepalive=False):
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError("service unavailable: {}".format(client.srv_name))
    start = time.perf_counter()
    future = client.call_async(request)
    if not spin_until(node, future.done, timeout, keepalive=keepalive):
        raise RuntimeError("service timed out: {}".format(client.srv_name))
    response = future.result()
    if response is None:
        raise RuntimeError("service failed: {}".format(client.srv_name))
    return response, (time.perf_counter() - start) * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--telemetry-timeout", type=float, default=45.0)
    parser.add_argument("--settle-seconds", type=float, default=12.0)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--takeoff-timeout", type=float, default=15.0)
    parser.add_argument("--altitude-tolerance", type=float, default=0.20)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "kind": "px4_gazebo_mavros_sitl_offboard_acceptance",
        "environment": "PX4 SITL + Gazebo Garden + ROS 2 Humble MAVROS; no physical vehicle",
        "result": "failed",
        "target_height_m": args.height,
    }
    rclpy.init()
    node = SitlAcceptance(args.height)
    try:
        if not spin_until(node, lambda: node.state is not None and node.state.connected and node.odom is not None, args.telemetry_timeout):
            raise RuntimeError("timed out waiting for connected FCU and local odometry")
        result["initial_state"] = {"connected": node.state.connected, "armed": node.state.armed, "mode": node.state.mode}
        # Do not arm immediately on the first odometry sample.  PX4's EKF
        # health checks need a short stationary interval; streaming the same
        # setpoint here also preserves the OFFBOARD pre-stream requirement.
        stream_setpoints(node, args.settle_seconds)
        stream_setpoints(node, 2.0)
        result["pre_offboard_diagnostics"] = {
            "estimator": estimator_snapshot(node.estimator),
            "recent_status_text": node.status_text,
        }
        arm_request = CommandBool.Request()
        arm_request.value = True
        arm_response, arm_ms = invoke(node, node.arm_client, arm_request, 10.0, keepalive=True)
        if not arm_response.success:
            raise RuntimeError("PX4 rejected arm request")
        if not spin_until(node, lambda: node.state is not None and node.state.armed, 8.0, keepalive=True):
            raise RuntimeError("PX4 did not arm")
        # PX4 v1.15 dynamic modes can require a stricter local-position health
        # bit before entering OFFBOARD while disarmed.  Keep pre-streaming,
        # arm in the initial safe mode, then transition to OFFBOARD and verify
        # the transition explicitly.
        mode_request = SetMode.Request()
        mode_request.custom_mode = "OFFBOARD"
        mode_response, mode_ms = invoke(node, node.mode_client, mode_request, 10.0, keepalive=True)
        if not mode_response.mode_sent:
            raise RuntimeError("PX4 rejected OFFBOARD mode request")
        if not spin_until(node, lambda: node.state is not None and node.state.mode == "OFFBOARD", 8.0, keepalive=True):
            raise RuntimeError("PX4 did not enter OFFBOARD")
        result["offboard_state"] = {"armed": node.state.armed, "mode": node.state.mode}
        takeoff_samples, reached_height = reach_altitude(node, args.height, args.takeoff_timeout, args.altitude_tolerance)
        result["takeoff"] = {
            "samples": len(takeoff_samples),
            "last_altitude_m": takeoff_samples[-1] if takeoff_samples else None,
            "reached_target": reached_height,
            "tolerance_m": args.altitude_tolerance,
        }
        if not reached_height:
            raise RuntimeError("did not reach target altitude before hold window")
        hold_until = time.monotonic() + args.hold_seconds
        hold_samples = []
        while time.monotonic() < hold_until:
            started = time.monotonic()
            node.publish_setpoint()
            rclpy.spin_once(node, timeout_sec=0.0)
            if node.odom is not None:
                hold_samples.append(node.odom.pose.pose.position.z)
            remaining = 0.05 - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        land_request = SetMode.Request()
        land_request.custom_mode = "AUTO.LAND"
        land_response, land_ms = invoke(node, node.mode_client, land_request, 10.0)
        if not land_response.mode_sent:
            raise RuntimeError("PX4 rejected AUTO.LAND mode request")
        landed = spin_until(node, lambda: node.state is not None and node.state.mode == "AUTO.LAND", 8.0)
        if not landed:
            raise RuntimeError("PX4 did not enter AUTO.LAND")
        intervals = [1000.0 * (b - a) for a, b in zip(node.setpoint_times, node.setpoint_times[1:])]
        altitude_error = [abs(z - args.height) for z in hold_samples]
        result.update({
            "services": {"offboard_set_mode_ms": mode_ms, "arm_ms": arm_ms, "land_set_mode_ms": land_ms},
            "setpoint_stream": {"count": len(node.setpoint_times), "interval_ms_p50": percentile(intervals, 0.5), "interval_ms_p95": percentile(intervals, 0.95)},
            "hold_altitude": {"samples": len(hold_samples), "mean_m": statistics.mean(hold_samples) if hold_samples else None, "error_p95_m": percentile(altitude_error, 0.95)},
            "final_state": {"connected": node.state.connected, "armed": node.state.armed, "mode": node.state.mode},
        })
        result["result"] = "passed" if (len(hold_samples) >= 10 and percentile(altitude_error, 0.95) <= 0.35) else "failed"
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
        result["failure_diagnostics"] = {
            "last_mode": node.state.mode if node.state is not None else None,
            "estimator": estimator_snapshot(node.estimator),
            "recent_status_text": node.status_text,
        }
    finally:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        node.destroy_node()
        rclpy.shutdown()
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
