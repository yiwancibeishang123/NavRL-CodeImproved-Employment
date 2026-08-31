# NavRL 实物部署交接文档

这份文档用于代码交付后的现场实施。现场人员必须先完成地面检查，再逐步放开控制权限。本文不替代无人机厂商的飞控安全规范，也不承诺未验证环境下的自动飞行安全。

## 现场需要补齐的信息

1. 无人机型号、机体半径和质量。
2. PX4 固件版本、飞控串口/UDP 地址和 MAVROS 配置。
3. 机载电脑 Ubuntu 版本、GPU 型号、CUDA 版本。
4. 相机型号、深度和彩色话题、分辨率、深度单位。
5. 相机到机体的实测外参矩阵。
6. 定位话题和坐标系：`map`、`odom`、`base_link`。
7. LiDAR/raycast 服务来源和实际水平/垂直采样数量。
8. 动态检测服务是否发布位置、速度、尺寸和时间戳。
9. 地图文件位置（若使用预建图）。

## 上机前文件检查

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl_ws/devel/setup.bash
conda activate NavRL
python3 ~/navrl_ws/src/navigation_runner/scripts/validate_deployment.py \
  --checkpoint ~/navrl_ws/src/navigation_runner/scripts/ckpts/step3-80new.pt
```

确认 YOLO 权重和相机参数：

```bash
rosparam get /onboard_detector/depth_intrinsics
rosparam get /onboard_detector/body_to_camera
ls ~/navrl_ws/src/onboard_detector/scripts/yolo_detector/weights/
```

连接飞控后先运行只读预检，不会解锁或切换模式：

```bash
rosrun navigation_runner px4_preflight_check.py
```

## 推荐启动顺序

1. 启动飞控、MAVROS 和定位节点。
2. 确认 `/mavros/state`、`/mavros/local_position/odom` 有稳定数据。
3. 启动相机驱动和深度图。
4. 启动 `onboard_detector` 和 YOLO。
5. 启动地图/raycast 服务。
6. 启动安全盾。
7. 以 `auto_takeoff:=false` 启动 NavRL。
8. 用 RViz 或目标话题发送目标，先观察日志和可视化，不立即解锁。

启动命令模板：

```bash
roslaunch navigation_runner navrl_deployment.launch \
  real:=true device:=cuda:0 \
  model_checkpoint:=/home/uav/navrl_ws/src/navigation_runner/scripts/ckpts/step3-80new.pt \
  use_safety_shield:=true auto_takeoff:=false \
  map_file:=/home/uav/maps/scene.pcd \
  run_label:=physical metrics_log:=/home/uav/navrl_logs/physical.jsonl
```

## 地面检查

- 检查相机彩色和深度画面是否有数据。
- 检查深度单位和距离范围是否正确。
- 检查机体坐标到相机坐标外参方向。
- 检查动态目标位置、速度和尺寸是否在机体/地图坐标中一致。
- 检查 LiDAR/raycast 是否固定返回 144 个点。
- 检查模型日志中的 checkpoint、设备和推理频率。
- 检查安全盾启动且 `/rl_navigation/toggle_safety_stop` 可用。
- 检查关闭导航节点后是否停止发布速度指令。

## 逐级放开控制

1. 桨叶拆除，只检查 ROS 消息和动作方向。
2. 桨叶安装但不解锁，检查 OFFBOARD 前置 setpoint。
3. 低高度、低速度、空旷环境进行手动保护测试。
4. 再测试静态障碍物和动态目标。
5. 任何传感器超时、模型异常、定位丢失或安全服务退出，都应停止测试并回滚到手动控制。

## 回滚

旧模型仅用于回归对比，不应在新模型部署失败时静默替换。需要显式指定：

```bash
roslaunch navigation_runner navrl_deployment.launch \
  real:=true model_checkpoint:=/path/to/known_good_checkpoint.pt
```

保留每次部署的 checkpoint、YAML、启动命令和 ROS 日志，便于复现问题。

## 审稿意见第七条的真机日志

若现场允许做真机试验，必须使用与仿真相同的 `metrics_log` 口径，并另行汇总成功率、碰撞率、完成时间与路径长度。再运行 `navigation_runner/scripts/sim_real_gap_analyze.py` 与 clean/perturbed 仿真日志比较。尚未完成现场测试时，不应创建或填写 `physical.jsonl`，也不能将接口验证描述为实机实验结果。
