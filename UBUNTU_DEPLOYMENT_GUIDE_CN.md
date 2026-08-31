# Ubuntu 部署与后续联调指南

## 适用范围

本交付包针对 Ubuntu 20.04、ROS Noetic、Python 3.8。选择 Python 3.8 是为了和 Noetic 的 `rospy`、`cv_bridge`、MAVROS 等二进制扩展保持 ABI 一致；模型权重可在该版本直接加载。Ubuntu 22.04/ROS2 不属于本 ROS1 包的直接支持范围。

运行安装脚本前，Ubuntu 必须已配置 ROS Noetic 的 apt 软件源，并安装 Miniconda 或 Anaconda。`--with-apt` 会安装 ROS base 和本项目依赖，但不会替用户添加 ROS 软件源或安装 Conda。

部署环境将安装 Python 3.8 兼容的 NumPy 1.24.4、PyTorch 2.1.2 和 torchvision 0.16.2；这是包内 TensorDict/TorchRL 0.4 源码声明的最低 PyTorch 版本。不要将 Python 3.10 训练环境中的 NumPy 1.26.4 或旧版 PyTorch 2.0.1 固定配置混入本 ROS1 部署环境。

## 1. 解压与安装

将压缩包复制到 Ubuntu 后执行：

```bash
tar -xzf NavRL-deployment-ubuntu-20260824.tar.gz
cd NavRL-decoder-deployment
bash INSTALL_UBUNTU.sh --with-apt
```

脚本不会覆盖已有的 `~/navrl_ws/src/navigation_runner`。已有工作空间请显式指定新目录，例如：

```bash
bash INSTALL_UBUNTU.sh --workspace=$HOME/navrl_ws_delivery
```

首次使用 MAVROS 前，还须按 MAVROS 官方步骤安装 GeographicLib 数据集。完成后，每个新终端执行：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl_ws/devel/setup.bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate NavRL
```

## 2. 代码级验收

以下检查不连接飞控、不解锁：

```bash
python3 ~/navrl_ws/src/navigation_runner/scripts/validate_deployment.py
python3 ~/navrl_ws/src/onboard_detector/scripts/check_calibration.py \
  ~/navrl_ws/src/navigation_runner/cfg/detection/real/real_detector_px4.yaml
```

预期结果是 checkpoint 架构一致、依赖完整、标定 YAML 格式有效。第二项只验证格式；默认相机内外参是模板，不是现场标定结果。

## 3. 常规仿真启动

终端 A：

```bash
source ~/navrl_ws/src/uav_simulator/gazeboSetup.bash
roslaunch uav_simulator start.launch
```

无桌面服务器或容器可使用 `roslaunch uav_simulator start.launch gui:=false keyboard_control:=false`，并以 `rviz:=false` 启动 `sim_real_gap.launch`。

终端 B：

```bash
roslaunch navigation_runner navrl_deployment.launch \
  real:=false use_px4:=false device:=cpu \
  auto_takeoff:=false use_safety_shield:=true
```

在 RViz 发布 `/move_base_simple/goal` 后，导航节点才会请求 raycast、动态目标并输出动作。`auto_takeoff` 默认关闭，避免在未检查 MAVROS 状态时产生飞控动作。

## 4. 审稿意见第七条的可复现实验

先分别启动同一场景、同一初始状态和同一目标的干净仿真及受扰动仿真。每组建议至少 3 个随机种子、每个种子足够多的任务回合；任务成功、碰撞、时间和路径长度由外部评测脚本汇总到 `TASK_SUMMARY_TEMPLATE.json` 同格式文件。

干净组：

```bash
roslaunch navigation_runner sim_real_gap.launch \
  run_label:=clean_sim metrics_log:=$HOME/navrl_logs/clean.jsonl \
  observation_trace_log:=$HOME/navrl_logs/clean_observations.jsonl
```

受扰动组的示例参数仅用于压力测试，不能称作真机拟合。实测拟合参数应由传感器、检测器、状态估计器日志统计获得：

```bash
roslaunch navigation_runner sim_real_gap.launch \
  run_label:=perturbed_sim metrics_log:=$HOME/navrl_logs/perturbed.jsonl \
  sim_real_perturbation_enabled:=true sim_real_perturbation_seed:=7 \
  lidar_noise_std:=0.05 lidar_dropout_ratio:=0.05 \
  obstacle_position_noise_std:=0.10 obstacle_velocity_noise_std:=0.10 \
  obstacle_dropout_ratio:=0.10 state_noise_std:=0.02 observation_delay_steps:=2
```

将同口径的真机运行日志留作第三组后，生成论文表格：

```bash
python3 ~/navrl_ws/src/navigation_runner/scripts/sim_real_gap_analyze.py \
  --clean $HOME/navrl_logs/clean.jsonl \
  --perturbed $HOME/navrl_logs/perturbed.jsonl \
  --physical $HOME/navrl_logs/physical.jsonl \
  --out-dir $HOME/navrl_logs/gap_report
```

脚本生成 `sim_real_gap_summary.md`、CSV 和 JSON。若还没有真机日志，省略 `--physical`，报告会明确标注真实组缺失，不会伪造结论。

观测 trace 可直接回放，验证部署推理网络对记录输入的确定性输出：

```bash
python3 ~/navrl_ws/src/navigation_runner/scripts/sim_to_sim_replay.py \
  $HOME/navrl_logs/clean_observations.jsonl \
  --checkpoint ~/navrl_ws/src/navigation_runner/scripts/ckpts/step3-80new.pt \
  --device cpu
```

## 5. 参数和故障排查

- 相机内外参、深度比例：`navigation_runner/cfg/detection/real/real_detector_px4.yaml`。
- 安全距离和速度相关参数：`navigation_runner/cfg/safety/safe_action_param.yaml`。
- 仿真/实际差距扰动配置说明：`navigation_runner/cfg/sim_real_gap_profiles.yaml`。
- 传感器超时、LiDAR 点数不为 144、动态检测服务失效、模型输出 NaN/Inf 时，导航节点发布保持/停止指令，并在 `metrics_log` 中留下原因。
- `rosservice call /rl_navigation/toggle_safety_stop "data: true"` 触发软件急停；恢复前必须先检查传感器状态。

更多现场流程见 `PHYSICAL_DEPLOYMENT_GUIDE_CN.md`；部署侧模型和 ROS 启动细节见 `DEPLOYMENT_GUIDE_CN.md`。
