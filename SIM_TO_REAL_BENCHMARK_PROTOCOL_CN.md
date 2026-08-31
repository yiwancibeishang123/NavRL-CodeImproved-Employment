# 仿真到实物差距与计算开销实验协议

本文件是对审稿意见第 7 条的可执行补充，供论文实验、Gazebo 联调和现场飞行使用。它区分三类证据：模型前向基准、闭环仿真鲁棒性、真机端到端结果。三者不能互相替代。

## 1. 结论边界

当前交付包含 ROS1、PX4/MAVROS、深度相机、YOLO/跟踪和安全层的接口，以及干净仿真和受扰动仿真的可复现日志链路。默认相机参数和扰动参数是模板或压力测试参数，不是现场标定值。

因此，在没有已完成的现场日志前，论文可陈述以下事实：

- 策略在保持 checkpoint、地图、初始状态、目标和安全阈值不变时，可接受受控的观测噪声、丢包和时延注入测试。
- 部署端对观测超时、LiDAR 尺寸异常、策略 NaN/Inf 和安全服务失效有保持/停止保护，并记录其触发比例。
- 控制节点对连续超周期或 ROS scheduler lag 超过阈值主动降级：连续两个异常周期进入 0.5 s 保持窗口，PX4 分支保持当前位姿，ROS1 非 PX4 分支发布零速度；`control_timing_degraded` 原因和保持比例必须纳入论文表格。
- 真机成功率、碰撞率、能耗和最终 sim-to-real 性能差异尚待同口径现场试验测量，不能由仿真外推。

## 2. 物理部署必须补齐的模块

| 模块 | 仿真与实物差距 | 真机必须完成的工作 | 证据/输出 |
|---|---|---|---|
| 状态估计与传感器融合 | 仿真里位置、姿态、速度和时间戳理想；实机有 IMU 偏置、视觉/激光里程计漂移和 EKF 重置 | 配置 PX4 EKF2 或外部 VIO/LiDAR 里程计输入，验证估计器状态、协方差、失效行为和时间同步 | odom/pose 时间戳、估计误差、延迟和重置日志 |
| 深度相机 | 内参、深度比例、曝光、反光和遮挡与 Gazebo 不同 | 标定 `fx, fy, cx, cy`、深度比例和有效距离；保存原始图像与深度样本 | 标定重投影误差、深度误差随距离的曲线 |
| 相机-机体外参 | 仿真 SDF 给出刚性变换；实机安装存在平移、俯仰和振动偏差 | 测得 `T_body_camera`，用静态靶标验证变换方向和轴定义 | 平移/角度残差与复验记录 |
| 动态目标感知 | 仿真目标无纹理、漏检和 ID 切换；实机 YOLO、深度和跟踪存在完整链路时延 | 记录检测 precision/recall、跟踪 ID switch、位置与速度误差和失效持续时间 | 检测/跟踪 bag、速度估计误差、目标年龄 |
| 控制执行 | Gazebo 动力学不含全部风场、推力曲线、地效和通信抖动 | 检查 MAVROS setpoint、OFFBOARD 前置流、限速、飞控模式和人工接管 | 指令-速度跟踪误差、控制时延、failsafe 触发 |
| 计算资源 | 开发机 GPU/CPU 与机载设备不同，ROS 线程竞争也不同 | 在目标机以实际分辨率、控制频率和并发节点测量 | 推理、检测、端到端 p50/p95、内存、超周期率 |

## 3. 坐标与标定验证

所有任务先确认一个唯一的局部坐标定义。推荐在 ROS 侧使用 `map`/`odom` 的 ENU 表达和 `base_link` 的 FLU 机体表达；PX4 原生使用 NED/FRD，MAVROS 负责消息转换。不能因为 `PositionTarget.FRAME_LOCAL_NED` 的名称而手工重复交换轴或取反。

在未解锁、桨叶拆除的台架状态执行以下检查：

1. 沿 ROS `+x`、`+y`、`+z` 分别发布很小的测试 setpoint，只比较 MAVROS/PX4 报告的方向，不让飞机起飞。
2. 在视野中放置已知位置的静态靶标，使用 `T_body_camera` 将深度点转换到机体/地图坐标，与独立测量结果比较。
3. 平移无人机和移动目标各一次，核对检测器输出的位置、速度和导航器使用的相对速度符号。
4. 将以上检查的坐标约定、外参矩阵、相机固件版本和时间同步方法写入实验附录；变更任一项后重新验证。

## 4. 三组可复现实验

三个组必须使用同一个 checkpoint、场景、随机种子、起点、目标、控制周期、安全阈值、最大速度和任务终止定义。每个场景至少覆盖静态障碍、横穿动态目标和迎面动态目标；每个条件使用多个种子和独立任务回合。论文中报告回合数而不只报告均值。

| 组别 | 用途 | 可变因素 | 不可替代的证据 |
|---|---|---|---|
| `clean_sim` | 理想闭环基线 | 无观测扰动 | Gazebo JSONL、任务汇总、checkpoint 哈希 |
| `perturbed_sim` | sim-to-sim 鲁棒性 | 由现场传感器/估计误差拟合的噪声、丢帧、时延；未拟合时只能称压力测试 | 同一任务的 JSONL 与扰动配置 |
| `physical` | 真实端到端验证 | 实机传感器、状态估计、飞控和算力 | 同口径 JSONL、rosbag、任务汇总和安全记录 |

压力测试参数在 `ros1/navigation_runner/cfg/sim_real_gap_profiles.yaml` 中标记为 `stress_example_not_calibrated`。只有根据真实传感器和估计器日志拟合后，才可将该组描述为“校准受扰动仿真”。

## 5. 计算开销口径

### 5.1 纯策略前向

在每种目标硬件上执行一次，先预热再测量。该程序使用已交付 checkpoint 和部署输入形状：状态 8、LiDAR 1x36x4、方向 1x3、动态障碍 1x5x10。

```bash
python3 ros1/navigation_runner/scripts/benchmark_policy.py \
  --checkpoint ros1/navigation_runner/scripts/ckpts/step3-80new.pt \
  --device cuda:0 --warmup 100 --samples 1000 --control-period-ms 50 \
  --out navrl_logs/policy_profile_cuda.json
```

输出包含 checkpoint SHA-256、PyTorch/CUDA/GPU 版本、参数量、预热次数、p50/p95/max 前向墙钟时间、CUDA kernel 时间、串行吞吐率、p95 控制预算余量、超周期比例和进程/GPU 内存。`inference_wall_ms` 是控制线程必须等待的时间；`inference_device_ms` 只是 GPU kernel 时间，不能用后者替代端到端延迟。

### 5.2 闭环仿真与真机

导航节点每个控制周期记录 `metrics_log`。其中包含策略前向、端到端 `control_ms`、控制周期抖动、ROS 定时器滞后、里程计/raycast/动态障碍物年龄、保持比例、安全层介入、RSS、GPU 内存和超周期标志。资源快照默认每 20 个周期记录一次，可通过 `~metrics_resource_stride` 调整。

YOLO 节点发布 `/yolo_detector/yolo_time`（预处理、推理、NMS、框后处理总时间）、`/yolo_detector/yolo_forward_time`（CUDA 同步后的网络前向墙钟时间）、`/yolo_detector/yolo_device_time`（GPU kernel 秒数；CPU 时为 `-1`）和 `/yolo_detector/yolo_image_age`（图像消息至完成检测时的年龄）。仿真/现场期间记录这些话题，计算检测端到端 p50/p95、有效帧率和图像至动态障碍物服务的年龄；不要只报告网络 forward 时间。

## 6. 推荐运行顺序

先在 Ubuntu 20.04 + ROS Noetic 工作空间完成依赖、checkpoint 和标定 YAML 静态检查。Gazebo 启动后，按每个随机种子分别运行下列两组；每次运行使用新的日志文件，避免追加混合。

```bash
# clean_sim
roslaunch navigation_runner sim_real_gap.launch \
  run_label:=clean_sim metrics_log:=$HOME/navrl_logs/clean_seed07.jsonl \
  observation_trace_log:=$HOME/navrl_logs/clean_seed07_observations.jsonl

# stress example only; replace with fitted values before paper claims
roslaunch navigation_runner sim_real_gap.launch \
  run_label:=perturbed_sim metrics_log:=$HOME/navrl_logs/perturbed_seed07.jsonl \
  sim_real_perturbation_enabled:=true sim_real_perturbation_seed:=7 \
  lidar_noise_std:=0.05 lidar_dropout_ratio:=0.05 \
  obstacle_position_noise_std:=0.10 obstacle_velocity_noise_std:=0.10 \
  obstacle_dropout_ratio:=0.10 state_noise_std:=0.02 observation_delay_steps:=2
```

无显示服务器的容器/服务器运行 Gazebo 时，可先使用：

```bash
roslaunch uav_simulator start.launch gui:=false keyboard_control:=false
```

`sim_real_gap.launch` 默认 `rviz:=false`，因此不会尝试启动 RViz。桌面调试时显式传入 `gui:=true` 或 `rviz:=true` 即可。

将每组所有完成回合汇总为 `TASK_SUMMARY_TEMPLATE.json` 的同字段 JSON。模板中的 `null` 明确表示未测量，不能填 0。随后汇总：

```bash
python3 ros1/navigation_runner/scripts/sim_real_gap_analyze.py \
  --clean $HOME/navrl_logs/clean_seed07.jsonl \
  --perturbed $HOME/navrl_logs/perturbed_seed07.jsonl \
  --clean-task-summary clean_tasks.json \
  --perturbed-task-summary perturbed_tasks.json \
  --out-dir $HOME/navrl_logs/gap_report_seed07
```

现场通过地面检查和低风险逐级放开后，才增加 `--physical` 和对应任务汇总。汇总程序输出 CSV、Markdown 和 JSON，并在 JSON 的 `comparison_to_clean` 中给出绝对差与相对百分比。

## 7. 论文表格与审稿回复

论文至少报告：回合数、成功率、碰撞率、完成时间、路径长度、平均速度、航点误差、策略前向 p50/p95、控制 p50/p95、检测 p50/p95、传感器年龄 p95、超周期率、主动降级保持比例、普通保持比例和安全层介入比例。所有表格标注硬件、GPU、软件版本、控制周期、相机分辨率及扰动是否由现场拟合。单目标视觉场景只能报告可见真值帧漏检率；误检率必须增加无目标/干扰目标场景后再报告。

建议回复表述：

> 感谢审稿人的建议。我们新增了仿真与实际部署差距分析，明确说明了状态估计与传感器融合、深度相机内外参与深度比例、检测/跟踪误差、坐标系对齐、控制执行时延和机载计算资源的影响。我们设计了 clean simulation、参数化受扰动 simulation 和 physical 三组同口径实验；部署端统一记录策略前向、端到端控制、传感器新鲜度、安全层介入和资源开销。受扰动仿真仅作为 sim-to-sim 鲁棒性证据，只有在取得同口径现场日志后才报告真机指标与 sim-to-real 数值差异。
