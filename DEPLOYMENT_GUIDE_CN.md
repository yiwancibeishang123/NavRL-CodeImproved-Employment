# NavRL 改进模型部署指南

本文档对应 `ros1/navigation_runner` 的部署代码。当前交付目标是：在 Ubuntu 20.04 + ROS Noetic 工作空间中完成编译，使用训练端的 `step3-80new` PPO checkpoint，启动感知、地图、安全和导航节点。本文档不替代具体飞控和传感器厂商手册。

## 1. 目录和模型

- `ros1/navigation_runner/scripts/ppo.py`：与 Isaac 训练端同源的推理网络。
- `ros1/navigation_runner/scripts/observation.py`：状态、LiDAR 和动态障碍物输入构造。
- `ros1/navigation_runner/scripts/ckpts/step3-80new.pt`：部署 checkpoint，可用 `~model_checkpoint` 覆盖。
- `ros1/navigation_runner/launch/navrl_deployment.launch`：统一入口。
- `isaac-training/setup_deployment_ubuntu.sh`：部署 Conda 环境安装脚本。

## 2. Ubuntu 环境

系统建议 Ubuntu 20.04、ROS Noetic、Python 3.8、CUDA 11.x（若使用 GPU）。Python 3.8 与 Noetic 的 ROS 二进制扩展 ABI 一致；不要在 ROS1 真机环境中把 Conda Python 3.10 与系统 `cv_bridge` 混用。完整的可复制安装流程见仓库根目录 `UBUNTU_DEPLOYMENT_GUIDE_CN.md`。

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-catkin-tools \
  ros-noetic-mavros ros-noetic-mavros-extras \
  ros-noetic-cv-bridge ros-noetic-image-transport \
  ros-noetic-vision-msgs ros-noetic-pcl-conversions \
  ros-noetic-tf ros-noetic-tf2-ros ros-noetic-message-filters
```

在 `isaac-training` 目录执行。该脚本为 ROS Noetic 的 Python 3.8 创建独立部署环境，并固定 NumPy 1.24.4、PyTorch 2.1.2、TensorDict/TorchRL 0.4；不要复用 Python 3.10 训练环境：

```bash
bash setup_deployment_ubuntu.sh
conda activate NavRL
```

如果使用 PX4/MAVROS，还要安装 GeographicLib 的 geoid 数据集；MAVROS 在启动时会读取它，缺失时即使 PX4 SITL 已经启动也无法连接：

```bash
sudo /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh
```

ROS 2 Humble 的对应路径为 `/opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh`。脚本会安装在系统共享目录，首次部署后执行一次即可。

## 3. 编译 ROS 工作空间

```bash
mkdir -p ~/navrl_ws/src
cp -r ros1/* ~/navrl_ws/src/
cd ~/navrl_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

验证安装文件：

```bash
python3 ~/navrl_ws/src/navigation_runner/scripts/validate_deployment.py
```

该命令检查 checkpoint、配置、Torch、TensorDict 和 TorchRL；它不需要连接飞控。

## 4. 仿真启动

先启动 Gazebo 和仿真传感器，再启动统一入口：

```bash
roslaunch uav_simulator start.launch
roslaunch navigation_runner navrl_deployment.launch \
  real:=false use_px4:=false use_real_detector:=false \
  device:=cpu use_safety_shield:=true
```

启用 YOLO 和真实检测节点：

```bash
roslaunch navigation_runner navrl_deployment.launch \
  real:=false use_real_detector:=true device:=cuda:0
```

导航节点收到 `/move_base_simple/goal` 后才开始控制。默认 `auto_takeoff` 为关闭；仿真需要自动起飞时显式设置 `auto_takeoff:=true`。

## 5. 真机代码入口

代码交付使用同一个入口，但真机参数必须由硬件方填写：

```bash
roslaunch navigation_runner navrl_deployment.launch \
  real:=true device:=cuda:0 \
  model_checkpoint:=/path/to/step3-80new.pt \
  use_safety_shield:=true auto_takeoff:=false
```

第一次接入必须保持 `auto_takeoff:=false`，先确认里程计、相机、深度、检测服务和安全服务均正常。

## 6. 参数说明

导航节点私有参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `~model_checkpoint` | `step3-80new.pt` | 模型权重路径 |
| `~device` | `cuda:0` | `cuda:0` 或 `cpu` |
| `~control_period` | `0.05` | 控制周期，单位秒 |
| `~sensor_timeout` | `0.5` | 里程计和 LiDAR 最大允许数据年龄 |
| `~obstacle_timeout` | `0.5` | 动态目标数据最大允许年龄 |
| `~robot_size` | `0.30` | 机体安全半径 |
| `~use_policy_server` | `false` | 是否改用独立推理服务 |
| `~height_control` | `false` | 是否输出 z 方向速度 |
| `~auto_takeoff` | `false` | 是否由节点自动起飞 |
| `~metrics_log` | 空 | JSONL 控制指标输出路径 |
| `~observation_trace_log` | 空 | JSONL 观测回放输出路径 |
| `~sim_real_perturbation_enabled` | `false` | 是否启用 sim-to-sim 观测扰动 |
| `~metrics_resource_stride` | `20` | 每 N 个控制周期记录一次 RSS/GPU 内存快照 |

训练和部署必须保持以下输入约束：状态 8 维、LiDAR `36x4`、动态障碍物 `5x10`、动作 3 维。动态障碍物 10 维顺序由 `observation.py` 统一生成，不要在其他节点重复实现。

## 7. 相机和深度标定

`cfg/detection/real/real_detector_px4.yaml` 已从仿真深度单位改为常见 RealSense 深度单位，并提供起始内参和外参。实际部署前必须用相机标定结果替换：

- `depth_intrinsics: [fx, fy, cx, cy]`
- `color_intrinsics: [fx, fy, cx, cy]`
- `depth_scale_factor`
- `body_to_camera`
- `body_to_camera_color`

外参矩阵按行优先的 4x4 齐次矩阵填写，坐标方向必须与 `map`、`odom` 和机体坐标约定一致。代码交付不包含现场标定数据，因此这里的默认值只是可编辑模板。

## 8. 安全策略

- 模型加载失败直接退出，不使用旧模型静默运行。
- LiDAR 点数不等于 `36x4`、传感器数据过期、动态检测服务异常时发布停止/保持命令。
- 模型输出 NaN/Inf 或超过速度上限时被截断或停止。
- `rosservice call /rl_navigation/toggle_safety_stop "data: true"` 可触发软件急停。
- 安全盾参数位于 `cfg/safety/safe_action_param.yaml`。

安全盾不可用时不建议在陌生环境中放开速度控制。代码只提供异常兜底，不保证未经现场验证的飞行安全。

## 9. sim-to-sim 回放

可以将 Isaac 或 Gazebo 导出的 JSON 观测序列输入部署网络，检查部署网络与训练网络的动作一致性：

```bash
python3 scripts/sim_to_sim_replay.py observations.json \
  --checkpoint scripts/ckpts/step3-80new.pt --device cpu
```

每条记录必须包含 `state`、`direction`、`lidar`、`dynamic_obstacle`，数组长度分别为 `8`、`3`、`36*4`、`5*10`。

## 10. 常见故障

**Checkpoint mismatch**：确认使用 `step3-80new.pt` 和同目录 `ppo.yaml`，不要混用旧版 `navrl_checkpoint.pt`。

**LiDAR observation size error**：检查水平分辨率和垂直 beam 数，部署节点要求固定 36x4 点阵。

**YOLO weights not found**：通过 launch 的 `yolo_weight:=...` 指定权重，文件名应使用实际存在的下划线版本。

**没有动作输出**：检查 odom、目标、raycast 和动态障碍物服务是否在 timeout 内更新。

**CPU 很慢**：将 `device:=cuda:0`，或降低控制频率并确认 PyTorch CUDA 可用；不要删除过期数据保护。

## 11. 仿真与实际应用差距分析

`sim_real_gap.launch` 可在相同任务上生成干净仿真和受扰动仿真的同口径 JSONL 日志；`sim_real_gap_analyze.py` 生成延迟、观测新鲜度、安全盾介入和任务统计的对照表。使用方式和论文结论边界见根目录 `SIM_REAL_GAP_ANALYSIS_CN.md`。在没有真机日志时，只能报告差距分析和受扰动仿真结果，不能宣称实机性能。
