from pathlib import Path
import re


MAP_PATH = Path("/home/lzx/project/NavRL/map")
WORLD_PATHS = [
    Path("/home/lzx/catkin_ws/src/ros1/uav_simulator/worlds/corridor/corridor_dynamic_10.world"),
    Path("/home/lzx/project/NavRL/ros1/uav_simulator/worlds/corridor/corridor_dynamic_10.world"),
]
STATIC_CHARS = {"0", "1"}
ROOM_EDGE_MATERIAL = "Gazebo/Wood"
DEFAULT_OBSTACLE_MATERIAL = "Gazebo/CeilingTiled"
ROOM_WALL_THICKNESS = 0.15
ROOM_WALL_HEIGHT = 2.5
OBSTACLE_THICKNESS = 1.0
OBSTACLE_HEIGHT = 2.0


def build_one_runs(lines, grid_cols, cell_x, x0):
    runs = []
    for row_idx, line in enumerate(lines):
        row = line.ljust(grid_cols)
        col = 0
        while col < grid_cols:
            if row[col] == "1":
                start = col
                while col + 1 < grid_cols and row[col + 1] == "1":
                    col += 1
                end = col
                y_center = (len(lines) - 1) / 2 - row_idx
                x_start = x0 + start * cell_x
                x_end = x0 + end * cell_x
                x_center = (x_start + x_end) / 2
                length = (end - start + 1) * cell_x
                runs.append((x_center, y_center, length, "1", "H"))
            col += 1
    return runs


def build_zero_runs(lines, grid_cols, cell_x, x0):
    zero_points = [
        (row_idx, col_idx)
        for row_idx, line in enumerate(lines)
        for col_idx, ch in enumerate(line.ljust(grid_cols))
        if ch == "0"
    ]
    if not zero_points:
        return []

    min_row = min(r for r, _ in zero_points)
    max_row = max(r for r, _ in zero_points)
    min_col = min(c for _, c in zero_points)
    max_col = max(c for _, c in zero_points)

    top_y = (len(lines) - 1) / 2 - min_row
    bottom_y = (len(lines) - 1) / 2 - max_row
    left_x = x0 + min_col * cell_x
    right_x = x0 + max_col * cell_x

    # Use centerline-to-centerline spans so the inner walls meet cleanly at the
    # four corners instead of extending past the splice points.
    horizontal_length = (max_col - min_col) * cell_x
    vertical_length = (max_row - min_row) * cell_x
    center_x = (left_x + right_x) / 2
    center_y = (top_y + bottom_y) / 2

    return [
        (center_x, top_y, horizontal_length, "0", "H"),
        (center_x, bottom_y, horizontal_length, "0", "H"),
        (left_x, center_y, vertical_length, "0", "V"),
        (right_x, center_y, vertical_length, "0", "V"),
    ]


def load_lines():
    return MAP_PATH.read_text().splitlines()


def get_grid_cols(lines):
    return max(
        c for line in lines for c, ch in enumerate(line) if ch in STATIC_CHARS.union({"2"})
    ) + 1


def build_runs():
    lines = load_lines()
    grid_cols = get_grid_cols(lines)
    cell_x = 1.01
    x0 = -((grid_cols - 1) / 2) * cell_x
    return build_one_runs(lines, grid_cols, cell_x, x0) + build_zero_runs(lines, grid_cols, cell_x, x0)


def build_main_models(runs):
    models = []
    for idx, (x_center, y_center, length, run_char, orientation) in enumerate(runs):
        model_x = x_center + 0.85
        model_y = y_center + 1.44
        length_str = f"{length:.3f}"
        material_name = ROOM_EDGE_MATERIAL if run_char == "0" else DEFAULT_OBSTACLE_MATERIAL
        thickness = ROOM_WALL_THICKNESS if run_char == "0" else OBSTACLE_THICKNESS
        height = ROOM_WALL_HEIGHT if run_char == "0" else OBSTACLE_HEIGHT
        z_center = height / 2
        yaw = "-1.5708" if orientation == "V" else "0"
        models.append(
            f"""    <model name='box_map_{idx}'>
      <pose>{model_x:.3f} {model_y:.3f} 0 0 -0 0</pose>
      <link name='Wall_0'>
        <collision name='Wall_0_Collision'>
          <geometry>
            <box>
              <size>{length_str} {thickness:.2f} {height:.2f}</size>
            </box>
          </geometry>
          <pose>0 0 {z_center:.2f} 0 -0 0</pose>
          <max_contacts>10</max_contacts>
          <surface>
            <contact>
              <ode/>
            </contact>
            <bounce/>
            <friction>
              <torsional>
                <ode/>
              </torsional>
              <ode/>
            </friction>
          </surface>
        </collision>
        <visual name='Wall_0_Visual'>
          <pose>0 0 {z_center:.2f} 0 -0 0</pose>
          <geometry>
            <box>
              <size>{length_str} {thickness:.2f} {height:.2f}</size>
            </box>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>{material_name}</name>
            </script>
            <ambient>0.921569 0.807843 0.615686 1</ambient>
          </material>
          <meta>
            <layer>0</layer>
          </meta>
        </visual>
        <pose>-0.85 -1.44 0 0 -0 {yaw}</pose>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
      <static>1</static>
    </model>"""
        )
    return "\n".join(models) + "\n\n"


def build_state_models(runs):
    models = []
    for idx, (x_center, y_center, length, run_char, orientation) in enumerate(runs):
        model_x = x_center + 0.85
        model_y = y_center + 1.44
        yaw = "-1.5708" if orientation == "V" else "0"
        models.append(
            f"""      <model name='box_map_{idx}'>
        <pose>{model_x:.3f} {model_y:.3f} 0 0 -0 0</pose>
        <scale>1 1 1</scale>
        <link name='Wall_0'>
          <pose>{x_center:.3f} {y_center:.3f} 0 0 -0 {yaw}</pose>
          <velocity>0 0 0 0 -0 0</velocity>
          <acceleration>0 0 0 0 -0 0</acceleration>
          <wrench>0 0 0 0 -0 0</wrench>
        </link>
      </model>"""
        )
    return "\n".join(models) + "\n"


def remove_old_static_models(section):
    pattern = re.compile(
        r"\s*<model name='(?:box_[^']+|box_map_(?:[0-9]|1[0-9]|2[0-1])|box_group_[^']+|box_fill_[^']+|box_dense_[^']+)'>.*?</model>\s*",
        re.S,
    )
    return pattern.sub("\n", section)


def rebuild_world(path: Path, main_block: str, state_block: str):
    text = path.read_text()
    main, tail = text.split("<state world_name='default'>", 1)
    state, after_state = tail.split("</state>", 1)

    main = remove_old_static_models(main).rstrip() + "\n" + main_block
    state = remove_old_static_models(state)

    state_match = re.search(r"^\s*<model name='corridor'>", state, re.M)
    if state_match is None:
        raise RuntimeError(f"Could not find corridor state block in {path}")
    insert_at = state_match.start()
    state = state[:insert_at] + state_block + state[insert_at:]

    rebuilt = main + "    <state world_name='default'>" + state + "</state>" + after_state
    rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt)
    path.write_text(rebuilt)


def main():
    runs = build_runs()
    main_block = build_main_models(runs)
    state_block = build_state_models(runs)
    for world_path in WORLD_PATHS:
        rebuild_world(world_path, main_block, state_block)
        print(f"Updated {world_path}")


if __name__ == "__main__":
    main()
