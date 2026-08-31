#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[3]
UAV_SIM_ROOT = REPO_ROOT / "ros1" / "uav_simulator"
WORLD_OUT_ROOT = UAV_SIM_ROOT / "worlds" / "generated_maps"
LAUNCH_OUT_ROOT = UAV_SIM_ROOT / "launch" / "generated_maps"
DEFAULT_MAP_PATH = REPO_ROOT / "map"
DEFAULT_CATKIN_UAV_SIM = Path("/home/lzx/catkin_ws/src/ros1/uav_simulator")

CELL_X = 1.01
CELL_Y = 1.0
OUTER_WALL_MARGIN = 1.2
STATIC_CHARS = {"0", "1"}
DYNAMIC_FORWARD_CHAR = "2"
DYNAMIC_REVERSE_CHAR = "3"
MARKER_CHARS = {"x", "X", "y", "Y"}
EXCEL_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
WOOD_MATERIAL = "Gazebo/Wood"
STATIC_OBSTACLE_MATERIAL = "Gazebo/CeilingTiled"
GROUND_MATERIAL = "Gazebo/Grey"
WALL_THICKNESS = 0.15
WALL_HEIGHT = 2.5
OBSTACLE_THICKNESS = 1.0
OBSTACLE_HEIGHT = 2.0


def _read_txt_map(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Map is empty: {path}")
    return lines


def _excel_col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx


def _read_xlsx_map(path: Path) -> List[str]:
    with ZipFile(path) as zf:
        names = set(zf.namelist())
        shared: List[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", EXCEL_NS):
                shared.append("".join(node.text or "" for node in item.findall(".//x:t", EXCEL_NS)))

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in names:
            sheets = sorted(name for name in names if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
            if not sheets:
                raise ValueError(f"No worksheet found in {path}")
            sheet_name = sheets[0]

        root = ET.fromstring(zf.read(sheet_name))
        cells: Dict[Tuple[int, int], str] = {}
        max_row = 0
        max_col = 0
        for cell in root.findall(".//x:c", EXCEL_NS):
            ref = cell.attrib.get("r", "")
            row_digits = "".join(ch for ch in ref if ch.isdigit())
            if not row_digits:
                continue
            row = int(row_digits)
            col = _excel_col_to_idx(ref)
            value = ""
            value_node = cell.find("x:v", EXCEL_NS)
            if value_node is not None:
                value = value_node.text or ""
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
            else:
                inline_node = cell.find("x:is", EXCEL_NS)
                if inline_node is not None:
                    value = "".join(node.text or "" for node in inline_node.findall(".//x:t", EXCEL_NS))

            normalized = value.strip()[:1]
            if normalized in STATIC_CHARS.union({DYNAMIC_FORWARD_CHAR, DYNAMIC_REVERSE_CHAR}).union(MARKER_CHARS):
                cells[(row, col)] = normalized
                max_row = max(max_row, row)
                max_col = max(max_col, col)

        if not cells:
            raise ValueError(f"No recognized cells found in {path}: expected 0/1/2/3/x/y")
        return ["".join(cells.get((row, col), " ") for col in range(1, max_col + 1)).rstrip() for row in range(1, max_row + 1)]


def _read_map(path: Path) -> List[str]:
    if path.suffix.lower() == ".xlsx":
        return _read_xlsx_map(path)
    return _read_txt_map(path)


def _grid_cols(lines: Sequence[str]) -> int:
    cols = [
        idx
        for line in lines
        for idx, ch in enumerate(line)
        if ch in STATIC_CHARS.union({DYNAMIC_FORWARD_CHAR, DYNAMIC_REVERSE_CHAR}).union(MARKER_CHARS)
    ]
    if not cols:
        raise ValueError("Map has no recognized cells: 0/1/2/3")
    return max(cols) + 1


def _active_bounds(lines: Sequence[str]) -> Tuple[int, int, int, int]:
    cells = [
        (row, col)
        for row, line in enumerate(lines)
        for col, ch in enumerate(line)
        if ch in STATIC_CHARS.union({DYNAMIC_FORWARD_CHAR, DYNAMIC_REVERSE_CHAR}).union(MARKER_CHARS)
    ]
    if not cells:
        raise ValueError("Map has no recognized active cells")
    min_row = min(row for row, _ in cells)
    max_row = max(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    max_col = max(col for _, col in cells)
    return min_row, max_row, min_col, max_col


def _cell_to_xy(row: int, col: int, lines: Sequence[str], grid_cols: int) -> Tuple[float, float]:
    min_row, max_row, min_col, max_col = _active_bounds(lines)
    center_col = (min_col + max_col) / 2.0
    center_row = (min_row + max_row) / 2.0
    return (col - center_col) * CELL_X, (center_row - row) * CELL_Y


def _static_runs(lines: Sequence[str], grid_cols: int) -> List[Dict[str, float]]:
    runs = []
    for row_idx, line in enumerate(lines):
        row = line.ljust(grid_cols)
        col = 0
        while col < grid_cols:
            ch = row[col]
            if ch in STATIC_CHARS:
                start = col
                while col + 1 < grid_cols and row[col + 1] == ch:
                    col += 1
                end = col
                x_start, y = _cell_to_xy(row_idx, start, lines, grid_cols)
                x_end, _ = _cell_to_xy(row_idx, end, lines, grid_cols)
                length = (end - start + 1) * CELL_X
                runs.append(
                    {
                        "x": (x_start + x_end) / 2.0,
                        "y": y,
                        "z": WALL_HEIGHT / 2.0 if ch == "0" else OBSTACLE_HEIGHT / 2.0,
                        "size_x": length,
                        "size_y": WALL_THICKNESS if ch == "0" else OBSTACLE_THICKNESS,
                        "size_z": WALL_HEIGHT if ch == "0" else OBSTACLE_HEIGHT,
                        "kind": ch,
                    }
                )
            col += 1
    return runs


def _points_for(lines: Sequence[str], char: str) -> List[Tuple[int, int]]:
    return [
        (row_idx, col_idx)
        for row_idx, line in enumerate(lines)
        for col_idx, ch in enumerate(line)
        if ch == char
    ]


def _find_marker(lines: Sequence[str], marker: str) -> Optional[Tuple[int, int]]:
    marker = marker.lower()
    for row_idx, line in enumerate(lines):
        for col_idx, ch in enumerate(line):
            if ch.lower() == marker:
                return row_idx, col_idx
    return None


def _neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    row, col = cell
    return [(row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)]


def _dynamic_components(lines: Sequence[str], char: str) -> List[List[Tuple[int, int]]]:
    remaining = set(_points_for(lines, char))
    components = []
    while remaining:
        stack = [min(remaining)]
        component = []
        remaining.remove(stack[0])
        while stack:
            cell = stack.pop()
            component.append(cell)
            for neighbor in _neighbors(cell):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _ordered_component_cells(component: Sequence[Tuple[int, int]], reverse: bool) -> List[Tuple[int, int]]:
    cells = set(component)
    adjacency = {cell: sorted(neighbor for neighbor in _neighbors(cell) if neighbor in cells) for cell in cells}
    endpoints = sorted(cell for cell, neighbors in adjacency.items() if len(neighbors) <= 1)
    start = endpoints[0] if endpoints else min(cells)

    ordered = [start]
    visited = {start}
    previous = None
    current = start
    while True:
        candidates = [cell for cell in adjacency[current] if cell != previous and cell not in visited]
        if not candidates:
            break
        next_cell = candidates[0]
        ordered.append(next_cell)
        visited.add(next_cell)
        previous, current = current, next_cell

    for cell in sorted(cells - visited):
        ordered.append(cell)
    if reverse:
        ordered.reverse()
    return ordered


def _compress_collinear(cells: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(cells) <= 2:
        return list(cells)
    compressed = [cells[0]]
    for idx in range(1, len(cells) - 1):
        prev_row, prev_col = compressed[-1]
        row, col = cells[idx]
        next_row, next_col = cells[idx + 1]
        if (row - prev_row, col - prev_col) == (next_row - row, next_col - col):
            continue
        compressed.append(cells[idx])
    compressed.append(cells[-1])
    return compressed


def _component_waypoints(
    component: Sequence[Tuple[int, int]], lines: Sequence[str], grid_cols: int, reverse: bool
) -> List[Tuple[float, float]]:
    ordered = _compress_collinear(_ordered_component_cells(component, reverse))
    if len(ordered) == 1:
        row, col = ordered[0]
        x, y = _cell_to_xy(row, col, lines, grid_cols)
        return [(x, y - 0.4), (x, y + 0.4)]
    return [_cell_to_xy(row, col, lines, grid_cols) for row, col in ordered]


def _material_xml(material_name: str, ambient: str = "") -> str:
    ambient_line = f"\n            <ambient>{ambient}</ambient>" if ambient else ""
    return f"""<material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>{material_name}</name>
            </script>{ambient_line}
          </material>"""


def _box_model(
    name: str,
    x: float,
    y: float,
    z: float,
    sx: float,
    sy: float,
    sz: float,
    material_name: str,
    ambient: str = "",
    yaw: float = 0.0,
) -> str:
    return f"""    <model name='{name}'>
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          {_material_xml(material_name, ambient)}
        </visual>
      </link>
    </model>"""


def _outer_wall_models(lines: Sequence[str], grid_cols: int) -> List[str]:
    min_row, max_row, min_col, max_col = _active_bounds(lines)
    x_min, y_top = _cell_to_xy(min_row, min_col, lines, grid_cols)
    x_max, y_bottom = _cell_to_xy(max_row, max_col, lines, grid_cols)
    left_x = x_min - OUTER_WALL_MARGIN
    right_x = x_max + OUTER_WALL_MARGIN
    top_y = y_top + OUTER_WALL_MARGIN
    bottom_y = y_bottom - OUTER_WALL_MARGIN
    center_x = (left_x + right_x) / 2.0
    center_y = (top_y + bottom_y) / 2.0
    width = right_x - left_x
    height = top_y - bottom_y
    z = WALL_HEIGHT / 2.0
    return [
        _box_model("map_outer_wall_top", center_x, top_y, z, width, WALL_THICKNESS, WALL_HEIGHT, WOOD_MATERIAL, "1 1 1 1"),
        _box_model("map_outer_wall_bottom", center_x, bottom_y, z, width, WALL_THICKNESS, WALL_HEIGHT, WOOD_MATERIAL, "1 1 1 1"),
        _box_model("map_outer_wall_left", left_x, center_y, z, height, WALL_THICKNESS, WALL_HEIGHT, WOOD_MATERIAL, "1 1 1 1", 1.5708),
        _box_model("map_outer_wall_right", right_x, center_y, z, height, WALL_THICKNESS, WALL_HEIGHT, WOOD_MATERIAL, "1 1 1 1", 1.5708),
    ]


def _person_model(idx: int, waypoints: Sequence[Tuple[float, float]], velocity: float) -> str:
    waypoint_xml = "\n".join(f"          <waypoint>{x:.3f} {y:.3f} 0.0</waypoint>" for x, y in waypoints)
    return f"""    <model name="person_map_{idx}_0.5_0.5_1.8">
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <collision name="bottom">
          <pose>0 0 0.01 0 0 1.57</pose>
          <geometry><box><size>0.35 0.75 0.02</size></box></geometry>
        </collision>
        <visual name="visual">
          <pose>0 0 -0.02 0 0 1.57</pose>
          <geometry><mesh><uri>model://person/meshes/walking.dae</uri></mesh></geometry>
        </visual>
      </link>
      <plugin name="obstacle_motion" filename="libobstaclePathPlugin.so">
        <loop>0</loop>
        <velocity>{velocity:.3f}</velocity>
        <angular_velocity>0.8</angular_velocity>
        <path>
{waypoint_xml}
        </path>
      </plugin>
    </model>"""


def _world_xml(lines: Sequence[str], world_name: str, ground_size_x: float, ground_size_y: float) -> str:
    grid_cols = _grid_cols(lines)
    static_models = _outer_wall_models(lines, grid_cols)
    for idx, run in enumerate(_static_runs(lines, grid_cols)):
        material_name = WOOD_MATERIAL if run["kind"] == "0" else STATIC_OBSTACLE_MATERIAL
        ambient = "1 1 1 1" if run["kind"] == "0" else "0.921569 0.807843 0.615686 1"
        static_models.append(
            _box_model(
                f"map_static_{idx}",
                run["x"],
                run["y"],
                run["z"],
                run["size_x"],
                run["size_y"],
                run["size_z"],
                material_name,
                ambient,
            )
        )

    people_models = []
    idx = 1
    for char, reverse in ((DYNAMIC_FORWARD_CHAR, False), (DYNAMIC_REVERSE_CHAR, True)):
        for component in _dynamic_components(lines, char):
            waypoints = _component_waypoints(component, lines, grid_cols, reverse)
            people_models.append(_person_model(idx, waypoints, 1.1))
            idx += 1

    models = "\n\n".join(static_models + people_models)
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
          <geometry><plane><normal>0 0 1</normal><size>{ground_size_x:.1f} {ground_size_y:.1f}</size></plane></geometry>
          <surface>
            <contact>
              <collide_bitmask>65535</collide_bitmask>
              <ode/>
            </contact>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
              <torsional>
                <ode/>
              </torsional>
            </friction>
            <bounce/>
          </surface>
          <max_contacts>10</max_contacts>
        </collision>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry><plane><normal>0 0 1</normal><size>{ground_size_x:.1f} {ground_size_y:.1f}</size></plane></geometry>
          {_material_xml(GROUND_MATERIAL)}
        </visual>
      </link>
    </model>

{models}

    <gravity>0 0 -9.8</gravity>
    <physics type='ode'>
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>1</shadows>
    </scene>
  </world>
</sdf>
"""


def _launch_xml(world_rel_path: str, spawn_x: float, spawn_y: float, spawn_z: float, spawn_yaw: float) -> str:
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
        args="-urdf -param robot_description -model quadcopter -x {spawn_x:.3f} -y {spawn_y:.3f} -z {spawn_z:.3f} -Y {spawn_yaw:.3f}"
        respawn="false" />

  <include file="$(find uav_simulator)/launch/setupTF.launch">
    <arg name="pose_topic" value="/CERLAB/quadcopter/pose" />
  </include>

  <node name="keyboard_control" pkg="uav_simulator" type="keyboard_control" />
</launch>
"""


def _copy_to_catkin(world_path: Path, launch_path: Path, catkin_uav_sim: Path) -> None:
    if not catkin_uav_sim.exists():
        return
    catkin_world_dir = catkin_uav_sim / "worlds" / "generated_maps"
    catkin_launch_dir = catkin_uav_sim / "launch"
    catkin_world_dir.mkdir(parents=True, exist_ok=True)
    catkin_launch_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(world_path, catkin_world_dir / world_path.name)
    shutil.copy2(launch_path, catkin_launch_dir / launch_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gazebo world/launch from an ASCII txt map or xlsx map.")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH, help="ASCII map or .xlsx map path")
    parser.add_argument("--world-name", default="txt_map_benchmark", help="Generated world name without .world")
    parser.add_argument("--launch-name", default="txt_map_benchmark", help="Generated launch name without .launch")
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--spawn-z", type=float, default=0.1)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--ground-size-x", type=float, default=120.0)
    parser.add_argument("--ground-size-y", type=float, default=60.0)
    parser.add_argument("--catkin-uav-sim", type=Path, default=DEFAULT_CATKIN_UAV_SIM)
    parser.add_argument("--no-sync-catkin", action="store_true", help="Do not copy outputs into catkin_ws package")
    args = parser.parse_args()

    lines = _read_map(args.map)
    grid_cols = _grid_cols(lines)
    start_marker = _find_marker(lines, "x")
    goal_marker = _find_marker(lines, "y")
    marker_spawn_x = -8.0
    marker_spawn_y = 0.0
    if start_marker is not None:
        marker_spawn_x, marker_spawn_y = _cell_to_xy(start_marker[0], start_marker[1], lines, grid_cols)
    spawn_x = marker_spawn_x if args.spawn_x is None else args.spawn_x
    spawn_y = marker_spawn_y if args.spawn_y is None else args.spawn_y

    WORLD_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LAUNCH_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    world_path = WORLD_OUT_ROOT / f"{args.world_name}.world"
    launch_path = LAUNCH_OUT_ROOT / f"{args.launch_name}.launch"
    ascii_path = WORLD_OUT_ROOT / f"{args.world_name}.txt"

    world_path.write_text(
        _world_xml(lines, args.world_name, args.ground_size_x, args.ground_size_y),
        encoding="utf-8",
    )
    ascii_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rel_world = world_path.relative_to(UAV_SIM_ROOT).as_posix()
    launch_path.write_text(
        _launch_xml(rel_world, spawn_x, spawn_y, args.spawn_z, args.spawn_yaw),
        encoding="utf-8",
    )

    print(f"[ok] world:  {world_path}")
    print(f"[ok] launch: {launch_path}")
    print(f"[ok] ascii:  {ascii_path}")
    print(f"[info] start: ({spawn_x:.3f}, {spawn_y:.3f}, {args.spawn_z:.3f})")
    if goal_marker is not None:
        goal_x, goal_y = _cell_to_xy(goal_marker[0], goal_marker[1], lines, grid_cols)
        print(f"[info] goal:  ({goal_x:.3f}, {goal_y:.3f}, {args.spawn_z:.3f})")
    if not args.no_sync_catkin:
        _copy_to_catkin(world_path, launch_path, args.catkin_uav_sim)
        print(f"[ok] synced to: {args.catkin_uav_sim}")


if __name__ == "__main__":
    main()
