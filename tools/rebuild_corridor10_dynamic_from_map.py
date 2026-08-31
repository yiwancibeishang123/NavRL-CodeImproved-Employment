from pathlib import Path
import re


MAP_PATH = Path("/home/lzx/project/NavRL/map")
WORLD_PATHS = [
    Path("/home/lzx/catkin_ws/src/ros1/uav_simulator/worlds/corridor/corridor_dynamic_10.world"),
    Path("/home/lzx/project/NavRL/ros1/uav_simulator/worlds/corridor/corridor_dynamic_10.world"),
]

CELL_SIZE_X = 1.01
ROOM_HALF_EXTENT = 11.0
PERSON_HALF_X = 0.375
PERSON_HALF_Y = 0.175
STATIC_CHARS = {"0", "1"}
DYNAMIC_FORWARD_CHAR = "2"
DYNAMIC_REVERSE_CHAR = "3"


PERSON_TEMPLATE = """    <model name="person{idx}_0.5_0.5_1.8">
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
        <loop>0</loop>
        <velocity>0.5</velocity>
        <angular_velocity>0.8</angular_velocity>
        <path>
          <waypoint>{x1:.3f} {y1:.3f} 0.0</waypoint>
          <waypoint>{x2:.3f} {y2:.3f} 0.0</waypoint>
        </path>
      </plugin>
    </model>"""


def load_points(chars):
    lines = MAP_PATH.read_text().splitlines()
    return {
        (r, c)
        for r, line in enumerate(lines)
        for c, ch in enumerate(line)
        if ch in chars
    }


def load_lines():
    return MAP_PATH.read_text().splitlines()


def get_grid_cols(lines):
    return max(
        c
        for line in lines
        for c, ch in enumerate(line)
        if ch in STATIC_CHARS.union({DYNAMIC_FORWARD_CHAR, DYNAMIC_REVERSE_CHAR})
    ) + 1


def grow_run(points, start, axis):
    r, c = start
    cells = [start]
    if axis == "H":
        cc = c - 1
        while (r, cc) in points:
            cells.insert(0, (r, cc))
            cc -= 1
        cc = c + 1
        while (r, cc) in points:
            cells.append((r, cc))
            cc += 1
    else:
        rr = r - 1
        while (rr, c) in points:
            cells.insert(0, (rr, c))
            rr -= 1
        rr = r + 1
        while (rr, c) in points:
            cells.append((rr, c))
            rr += 1
    return cells


def load_segments(chars):
    remaining = set(load_points(chars))
    segments = []
    while remaining:
        start = min(remaining)
        horizontal = grow_run(remaining, start, "H")
        vertical = grow_run(remaining, start, "V")

        if len(horizontal) >= len(vertical):
            chosen = horizontal
        else:
            chosen = vertical

        for cell in chosen:
            remaining.remove(cell)
        segments.append(chosen)
    return segments


def component_to_waypoints(comp):
    lines = load_lines()
    grid_cols = get_grid_cols(lines)
    x0 = -((grid_cols - 1) / 2) * CELL_SIZE_X
    rows = sorted(r for r, _ in comp)
    cols = sorted(c for _, c in comp)
    if len(set(rows)) == 1:
        row = rows[0]
        col1, col2 = cols[0], cols[-1]
        y = (len(lines) - 1) / 2 - row
        x1 = x0 + col1 * CELL_SIZE_X
        x2 = x0 + col2 * CELL_SIZE_X
        return x1, y, x2, y
    if len(set(cols)) == 1:
        col = cols[0]
        row1, row2 = rows[0], rows[-1]
        x = x0 + col * CELL_SIZE_X
        y1 = (len(lines) - 1) / 2 - row1
        y2 = (len(lines) - 1) / 2 - row2
        return x, y1, x, y2
    if len(comp) == 1:
        row, col = comp[0]
        x = x0 + col * CELL_SIZE_X
        y = (len(lines) - 1) / 2 - row
        return x, y - 0.25, x, y + 0.25
    raise ValueError(f"Unsupported dynamic segment: {comp}")


def build_static_boxes():
    lines = load_lines()
    grid_cols = get_grid_cols(lines)
    x0 = -((grid_cols - 1) / 2) * CELL_SIZE_X
    boxes = []
    for row_idx, line in enumerate(lines):
        row = line.ljust(grid_cols)
        col = 0
        while col < grid_cols:
            if row[col] in STATIC_CHARS:
                start = col
                while col + 1 < grid_cols and row[col + 1] in STATIC_CHARS:
                    col += 1
                end = col
                y_center = (len(lines) - 1) / 2 - row_idx
                x_start = x0 + start * CELL_SIZE_X
                x_end = x0 + end * CELL_SIZE_X
                x_center = (x_start + x_end) / 2
                width = (end - start + 1) * CELL_SIZE_X
                boxes.append((x_center, y_center, width, 1.0))
            col += 1
    return boxes


def is_segment_safe(segment, boxes):
    x1, y1, x2, y2 = segment
    min_x, max_x = sorted((x1, x2))
    min_y, max_y = sorted((y1, y2))
    min_x -= PERSON_HALF_X
    max_x += PERSON_HALF_X
    min_y -= PERSON_HALF_Y
    max_y += PERSON_HALF_Y

    if (
        min_x < -ROOM_HALF_EXTENT
        or max_x > ROOM_HALF_EXTENT
        or min_y < -ROOM_HALF_EXTENT
        or max_y > ROOM_HALF_EXTENT
    ):
        return False

    for box_x, box_y, box_w, box_h in boxes:
        if not (
            max_x < box_x - box_w / 2
            or min_x > box_x + box_w / 2
            or max_y < box_y - box_h / 2
            or min_y > box_y + box_h / 2
        ):
            return False
    return True


def find_safe_segment(segment, axis, boxes):
    offsets = [
        0.0,
        0.25,
        -0.25,
        0.5,
        -0.5,
        0.75,
        -0.75,
        1.0,
        -1.0,
        1.25,
        -1.25,
        1.5,
        -1.5,
        1.75,
        -1.75,
        2.0,
        -2.0,
    ]
    for offset in offsets:
        if axis == "H":
            candidate = (segment[0], segment[1] + offset, segment[2], segment[3] + offset)
        else:
            candidate = (segment[0] + offset, segment[1], segment[2] + offset, segment[3])
        if is_segment_safe(candidate, boxes):
            return candidate
    raise ValueError(f"Could not place safe dynamic track for {segment}")


def build_people_block():
    boxes = build_static_boxes()
    blocks = []
    idx = 1
    for chars, reverse in (((DYNAMIC_FORWARD_CHAR,), False), ((DYNAMIC_REVERSE_CHAR,), True)):
        for comp in load_segments(chars):
            base = component_to_waypoints(comp)
            axis = "H" if abs(base[1] - base[3]) < 1e-9 else "V"
            x1, y1, x2, y2 = find_safe_segment(base, axis, boxes)
            if reverse:
                x1, y1, x2, y2 = x2, y2, x1, y1
            blocks.append(PERSON_TEMPLATE.format(idx=idx, x1=x1, y1=y1, x2=x2, y2=y2))
            idx += 1
    return "\n\n".join(blocks) + "\n\n"


def rebuild_world(path: Path, people_block: str):
    text = path.read_text()
    world_marker = "  <world name='default'>"
    light_marker = "   <light name='sun' type='directional'>"
    if world_marker not in text:
        raise RuntimeError(f"Could not find world marker in {path}")
    if light_marker not in text:
        raise RuntimeError(f"Could not find light marker in {path}")
    head, rest = text.split(world_marker, 1)
    dynamic_region, suffix = rest.split(light_marker, 1)
    rebuilt = head + world_marker + "\n" + people_block + light_marker + suffix
    rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt)
    path.write_text(rebuilt)


def main():
    people_block = build_people_block()
    for world_path in WORLD_PATHS:
        rebuild_world(world_path, people_block)
        print(f"Updated {world_path}")


if __name__ == "__main__":
    main()
