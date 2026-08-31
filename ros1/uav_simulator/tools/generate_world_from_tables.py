#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[3]
UAV_SIM_ROOT = REPO_ROOT / "ros1" / "uav_simulator"
TABLE_ROOT = UAV_SIM_ROOT / "scenario_tables"
WORLD_OUT_ROOT = UAV_SIM_ROOT / "worlds" / "generated_tables"
LAUNCH_OUT_ROOT = UAV_SIM_ROOT / "launch" / "generated_tables"


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: str, default: float = 0.0) -> float:
    text = str(value).strip()
    return default if text == "" else float(text)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_config(path: Path) -> Dict[str, str]:
    rows = _load_csv(path)
    config = {}
    for row in rows:
        key = (row.get("key") or "").strip()
        if not key:
            continue
        config[key] = (row.get("value") or "").strip()
    return config


def _color(row: Dict[str, str], prefix: str = "color_") -> str:
    r = _as_float(row.get(f"{prefix}r", 0.7))
    g = _as_float(row.get(f"{prefix}g", 0.7))
    b = _as_float(row.get(f"{prefix}b", 0.7))
    a = _as_float(row.get(f"{prefix}a", 1.0))
    return f"{r} {g} {b} {a}"


def _pose(x: float, y: float, z: float, yaw: float = 0.0) -> str:
    return f"{x} {y} {z} 0 0 {yaw}"


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _static_box_model(row: Dict[str, str]) -> str:
    name = escape((row.get("name") or "static_box").strip())
    x = _as_float(row.get("x", 0.0))
    y = _as_float(row.get("y", 0.0))
    z = _as_float(row.get("z", 0.0))
    yaw = _as_float(row.get("yaw", 0.0))
    sx = _as_float(row.get("size_x", 1.0))
    sy = _as_float(row.get("size_y", 1.0))
    sz = _as_float(row.get("size_z", 1.0))
    color = _color(row)
    return f"""<model name="{name}">
  <static>true</static>
  <pose>{_pose(x, y, z, yaw)}</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>{sx} {sy} {sz}</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>{sx} {sy} {sz}</size>
        </box>
      </geometry>
      <material>
        <ambient>{color}</ambient>
        <diffuse>{color}</diffuse>
        <specular>0.2 0.2 0.2 1</specular>
      </material>
    </visual>
  </link>
</model>"""


def _dynamic_person_model(row: Dict[str, str], waypoints: List[Dict[str, str]]) -> str:
    name = escape((row.get("name") or "person").strip())
    vel = _as_float(row.get("velocity", 0.5))
    ang_vel = _as_float(row.get("angular_velocity", 0.8))
    loop = 1 if _as_bool(row.get("loop", "false")) else 0
    wp_xml = "\n".join(
        f"      <waypoint>{_as_float(wp['x'])} {_as_float(wp['y'])} {_as_float(wp.get('z', 0.0))}</waypoint>"
        for wp in waypoints
    )
    return f"""<model name="{name}">
  <pose>0 0 0 0 0 0</pose>
  <link name="link">
    <collision name="bottom">
      <pose>0 0 0.01 0 0 1.57</pose>
      <geometry>
        <box>
          <size>0.35 0.75 0.02</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <pose>0 0 -0.02 0 0 1.57</pose>
      <geometry>
        <mesh>
          <uri>model://person/meshes/walking.dae</uri>
        </mesh>
      </geometry>
    </visual>
  </link>
  <plugin name="obstacle_motion" filename="libobstaclePathPlugin.so">
    <loop>{loop}</loop>
    <velocity>{vel}</velocity>
    <angular_velocity>{ang_vel}</angular_velocity>
    <path>
{wp_xml}
    </path>
  </plugin>
</model>"""


def _dynamic_box_or_cylinder_model(
    row: Dict[str, str], waypoints: List[Dict[str, str]], model_type: str
) -> str:
    name = escape((row.get("name") or model_type).strip())
    x = _as_float(row.get("x", 0.0))
    y = _as_float(row.get("y", 0.0))
    z = _as_float(row.get("z", 0.0))
    yaw = _as_float(row.get("yaw", 0.0))
    vel = _as_float(row.get("velocity", 0.5))
    ang_vel = _as_float(row.get("angular_velocity", 0.0))
    loop = 1 if _as_bool(row.get("loop", "false")) else 0
    color = _color(row)
    wp_xml = "\n".join(
        f"      <waypoint>{_as_float(wp['x'])} {_as_float(wp['y'])} {_as_float(wp.get('z', 0.0))}</waypoint>"
        for wp in waypoints
    )
    if model_type == "box":
        sx = _as_float(row.get("size_x", 0.5))
        sy = _as_float(row.get("size_y", 0.5))
        sz = _as_float(row.get("size_z", 1.8))
        geometry = f"""<box>
          <size>{sx} {sy} {sz}</size>
        </box>"""
    else:
        radius = _as_float(row.get("radius", 0.25))
        length = _as_float(row.get("size_z", 1.8))
        geometry = f"""<cylinder>
          <radius>{radius}</radius>
          <length>{length}</length>
        </cylinder>"""
    return f"""<model name="{name}">
  <static>true</static>
  <pose>{_pose(x, y, z, yaw)}</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        {geometry}
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        {geometry}
      </geometry>
      <material>
        <ambient>{color}</ambient>
        <diffuse>{color}</diffuse>
        <specular>0.2 0.2 0.2 1</specular>
      </material>
    </visual>
  </link>
  <plugin name="obstacle_motion" filename="libobstaclePathPlugin.so">
    <orientation>false</orientation>
    <loop>{loop}</loop>
    <velocity>{vel}</velocity>
    <angular_velocity>{ang_vel}</angular_velocity>
    <path>
{wp_xml}
    </path>
  </plugin>
</model>"""


def _dynamic_model(row: Dict[str, str], waypoints: List[Dict[str, str]]) -> str:
    model_type = (row.get("model_type") or "person").strip().lower()
    if not waypoints:
        raise ValueError(f"Dynamic obstacle '{row.get('name', '')}' has no waypoints.")
    if model_type == "person":
        return _dynamic_person_model(row, waypoints)
    if model_type in {"box", "cylinder"}:
        return _dynamic_box_or_cylinder_model(row, waypoints, model_type)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _world_xml(
    config: Dict[str, str],
    static_rows: List[Dict[str, str]],
    dynamic_rows: List[Dict[str, str]],
    waypoint_rows: List[Dict[str, str]],
) -> str:
    ground_size_x = _as_float(config.get("ground_size_x", 120.0))
    ground_size_y = _as_float(config.get("ground_size_y", 60.0))
    ground_color = config.get("ground_color", "0.3 0.3 0.3 1")

    waypoints_by_id: Dict[str, List[Dict[str, str]]] = {}
    for row in waypoint_rows:
        obs_id = (row.get("obstacle_id") or "").strip()
        if not obs_id:
            continue
        waypoints_by_id.setdefault(obs_id, []).append(row)
    for obs_id, rows in waypoints_by_id.items():
        rows.sort(key=lambda r: int((r.get("order") or "0").strip() or 0))

    static_models = "\n\n".join(_static_box_model(row) for row in static_rows if (row.get("name") or "").strip())
    dynamic_models = "\n\n".join(
        _dynamic_model(row, waypoints_by_id.get((row.get("id") or "").strip(), []))
        for row in dynamic_rows
        if (row.get("id") or "").strip()
    )

    return f"""<sdf version='1.7'>
  <world name='default'>
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>{ground_size_x} {ground_size_y}</size>
            </plane>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>{ground_size_x} {ground_size_y}</size>
            </plane>
          </geometry>
          <material>
            <ambient>{ground_color}</ambient>
            <diffuse>{ground_color}</diffuse>
          </material>
        </visual>
      </link>
    </model>

{_indent(static_models, 4) if static_models else ''}

{_indent(dynamic_models, 4) if dynamic_models else ''}

    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>1</shadows>
    </scene>
    <wind/>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>0</latitude_deg>
      <longitude_deg>0</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
  </world>
</sdf>
"""


def _launch_xml(config: Dict[str, str], world_rel_path: str) -> str:
    spawn_x = _as_float(config.get("spawn_x", 0.0))
    spawn_y = _as_float(config.get("spawn_y", 0.0))
    spawn_z = _as_float(config.get("spawn_z", 0.1))
    spawn_yaw = _as_float(config.get("spawn_yaw", 0.0))
    return f"""<launch>
  <arg name="paused" default="false"/>
  <arg name="verbose" default="false"/>
  <arg name="use_sim_time" default="true"/>
  <arg name="debug" default="false"/>
  <arg name="world_name" default="$(find uav_simulator)/{world_rel_path}" />

  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="paused" value="$(arg paused)"/>
    <arg name="world_name" value="$(arg world_name)"/>
    <arg name="verbose" value="$(arg verbose)"/>
    <arg name="use_sim_time" value="$(arg use_sim_time)"/>
    <arg name="debug" value="$(arg debug)"/>
  </include>

  <param name="robot_description" command="cat '$(find uav_simulator)/urdf/quadcopter.urdf'" />

  <node pkg="topic_tools" type="throttle" name="gt_pose_throttle" args="messages /CERLAB/quadcopter/pose_raw 30 /CERLAB/quadcopter/pose"/>
  <node pkg="topic_tools" type="throttle" name="gt_vel_throttle" args="messages /CERLAB/quadcopter/vel_raw 30 /CERLAB/quadcopter/vel"/>
  <node pkg="topic_tools" type="throttle" name="gt_acc_throttle" args="messages /CERLAB/quadcopter/acc_raw 30 /CERLAB/quadcopter/acc"/>
  <node pkg="topic_tools" type="throttle" name="gt_odom_throttle" args="messages /CERLAB/quadcopter/odom_raw 30 /CERLAB/quadcopter/odom"/>
  <node name="spawn_gazebo_model" pkg="gazebo_ros" type="spawn_model"
        args="-urdf -param robot_description -model quadcopter -x {spawn_x} -y {spawn_y} -z {spawn_z} -Y {spawn_yaw}"
        respawn="false" />

  <include file="$(find uav_simulator)/launch/setupTF.launch">
    <arg name="pose_topic" value="/CERLAB/quadcopter/pose" />
  </include>

  <node name="keyboard_control" pkg="uav_simulator" type="keyboard_control" />
</launch>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate Gazebo world/launch files from CSV tables.")
    parser.add_argument("scenario", help="Scenario folder name under ros1/uav_simulator/scenario_tables")
    args = parser.parse_args()

    scenario_dir = TABLE_ROOT / args.scenario
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    config = _load_config(scenario_dir / "scenario_config.csv")
    static_rows = _load_csv(scenario_dir / "static_boxes.csv")
    dynamic_rows = _load_csv(scenario_dir / "dynamic_obstacles.csv")
    waypoint_rows = _load_csv(scenario_dir / "dynamic_waypoints.csv")

    world_name = config.get("world_name", args.scenario).strip() or args.scenario
    launch_name = config.get("launch_name", args.scenario).strip() or args.scenario

    WORLD_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LAUNCH_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    world_path = WORLD_OUT_ROOT / f"{world_name}.world"
    launch_path = LAUNCH_OUT_ROOT / f"{launch_name}.launch"

    world_path.write_text(_world_xml(config, static_rows, dynamic_rows, waypoint_rows), encoding="utf-8")
    rel_world = world_path.relative_to(UAV_SIM_ROOT).as_posix()
    launch_path.write_text(_launch_xml(config, rel_world), encoding="utf-8")

    print(f"[ok] world:  {world_path}")
    print(f"[ok] launch: {launch_path}")


if __name__ == "__main__":
    main()
