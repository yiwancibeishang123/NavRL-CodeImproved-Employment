# Scenario Tables

Use Excel/WPS to edit these CSV files, then generate a Gazebo `.world` and matching `.launch`.

## TXT/XLSX Map Workflow

You can also use `/home/lzx/project/NavRL-decoder/map` as an ASCII map, or an `.xlsx` map edited in Excel/LibreOffice.

Map symbols:

- `0`: thin room/corridor wall
- `1`: static block obstacle
- `2`: dynamic person path, forward direction
- `3`: dynamic person path, reverse direction
- `x`: UAV start marker
- `y`: goal marker
- space: free space

Generate from the txt/xlsx map and sync the generated files into `/home/lzx/catkin_ws/src/ros1/uav_simulator`:

```bash
cd /home/lzx/project/NavRL-decoder
python ros1/uav_simulator/tools/generate_world_from_txt_map.py \
  --map /home/lzx/project/NavRL-decoder/map \
  --world-name txt_map_benchmark \
  --launch-name txt_map_benchmark
```

For an Excel map:

```bash
cd /home/lzx/project/NavRL-decoder
python ros1/uav_simulator/tools/generate_world_from_txt_map.py \
  --map '/media/lzx/F94C-2F04/地图.xlsx' \
  --world-name xlsx_map_benchmark \
  --launch-name xlsx_map_benchmark
```

Run the generated scene:

```bash
source /opt/ros/noetic/setup.bash
source /home/lzx/catkin_ws/devel/setup.bash
roslaunch uav_simulator txt_map_benchmark.launch
```

## Workflow

1. Copy `indoor_dynamic_benchmark_template/` to a new folder, for example:

   - `ros1/uav_simulator/scenario_tables/indoor_dynamic_benchmark_v1/`

2. Edit the CSV files with Excel/WPS.

3. Generate files:

```bash
cd /home/lzx/project/NavRL-decoder
python ros1/uav_simulator/tools/generate_world_from_tables.py indoor_dynamic_benchmark_v1
```

4. Generated outputs:

   - `ros1/uav_simulator/worlds/generated_tables/<world_name>.world`
   - `ros1/uav_simulator/launch/generated_tables/<launch_name>.launch`

## CSV Files

### `scenario_config.csv`

Global settings such as output names and UAV spawn pose.

### `static_boxes.csv`

Static box obstacles used to shape the corridor.

### `dynamic_obstacles.csv`

Dynamic obstacle definitions.

Supported `model_type`:

- `person`
- `box`
- `cylinder`

### `dynamic_waypoints.csv`

Path waypoints for each dynamic obstacle.

Each obstacle can have multiple rows, ordered by the `order` column.
