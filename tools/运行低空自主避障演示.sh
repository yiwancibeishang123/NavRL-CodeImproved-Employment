#!/usr/bin/env bash
set -euo pipefail

# The container entrypoint owns Gazebo. This script only restarts that world,
# starts the ROS navigation stack, waits for odometry, and publishes one goal.
container_name="${1:-navrl_gazebo_avoid_1787925921}"
run_label="低空自主避障六航点客户演示"
metrics_log="/tmp/low_altitude_obstacle_demo_full_route.jsonl"

if ! docker inspect "$container_name" >/dev/null 2>&1; then
  echo "找不到容器: $container_name" >&2
  exit 1
fi

echo "重启仿真容器并加载 simple_box 世界..."
docker restart "$container_name" >/dev/null

echo "等待 ROS/Gazebo 就绪..."
for _ in $(seq 1 60); do
  if docker exec "$container_name" bash -lc 'source /opt/ros/noetic/setup.bash; rostopic list >/dev/null 2>&1'; then
    break
  fi
  sleep 1
done

echo "启动 PPO + LiDAR + 安全盾导航节点..."
docker exec -d "$container_name" bash -lc \
  "source /opt/ros/noetic/setup.bash; source /workspace/catkin_ws/devel/setup.bash; \
   roslaunch navigation_runner low_altitude_obstacle_demo.launch rviz:=true \
   run_label:='$run_label' metrics_log:='$metrics_log' \
   > /tmp/low_altitude_obstacle_demo_customer.launch.log 2>&1"

echo "等待无人机完成自动起飞..."
if ! docker exec "$container_name" bash -lc \
  'source /opt/ros/noetic/setup.bash; timeout 45 rostopic echo -n1 /CERLAB/quadcopter/odom >/dev/null'; then
  echo "未收到里程计，请查看容器内 /tmp/low_altitude_obstacle_demo_customer.launch.log" >&2
  exit 1
fi

echo "发布六个主航点和返程安全子点，开始完整自主绕障..."
docker exec -d "$container_name" bash -lc \
  'source /opt/ros/noetic/setup.bash; source /workspace/catkin_ws/devel/setup.bash; \
   rosrun navigation_runner publish_full_obstacle_route.py \
   > /tmp/low_altitude_obstacle_demo_full_route.log 2>&1'

echo "演示已开始。Gazebo 窗口显示真实闭环轨迹，轨迹话题为 /rl_navigation/flight_trajectory。"
echo "原始指标: $container_name:$metrics_log"
