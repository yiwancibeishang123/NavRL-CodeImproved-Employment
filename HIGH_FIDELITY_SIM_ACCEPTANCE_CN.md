# 高保真仿真验收说明

本说明覆盖没有实物无人机时的第 3、4、5 项替代验收。它是可执行的高保真仿真验证，不是实飞结果，也不能替代现场标定或飞行安全结论。

## 覆盖范围

| 工作 | 仿真栈 | 真值或通过条件 | 原始输出 |
|---:|---|---|---|
| 3 | ROS Noetic、Gazebo Classic、RGB-D 相机、移动行人模型、部署 YOLO 权重和动态目标跟踪器 | 必须有相机和深度消息、YOLO 正检测、跟踪服务结果；以 `/gazebo/model_states` 计算水平（XY）位置和速度误差 | `hifi_perception_acceptance.json` |
| 4 | 同上，Gazebo `CameraInfo`、`32FC1` 浮点深度和 TF | 内参必须与场景定义一致；深度编码必须为 `32FC1`；必须取得 `base_link -> camera_link` 外参 | `hifi_calibration_acceptance.json` |
| 5 | PX4 SITL、Gazebo Garden `gz_x500`、MAVROS、MAVLink UDP | FCU 连接、预发送 setpoint、进入 OFFBOARD、解锁、1 m 悬停、`AUTO.LAND` | `px4_mavros_sitl_acceptance.json` |

高保真场景中目标不是向检测器注入的虚拟检测框。YOLO 直接读取 Gazebo 渲染的彩色图像，目标深度从同步浮点深度图投影；速度由连续 Gazebo 真值位置差分得到。第 5 项使用隔离的临时 PX4 SITL 根目录，运行结束自动清理，不会写入开发者已有的 PX4 参数。

## 复现

视觉、深度、标定和跟踪验收需要 Noetic 容器：

```bash
bash docker/run_hifi_perception_acceptance.sh
```

最新 clean 闭环证据位于 `benchmark_results/hifi_perception_20260828_r8/`：381 个可见目标帧全部检测，检测前向 p95 21.32 ms、回调 p95 24.65 ms；跟踪位置/速度误差 p95 0.167 m / 0.403 m/s，并在同一进程中完成组合退化 -> clean 自动恢复。六种条件挑战和空场景负对照仍引用 `benchmark_results/hifi_perception_20260827_r7/`：共 1301 个可见目标帧，247 个漏检（18.99%）；combined 条件 173/173 漏检且 watchdog 151/151 周期报警，空场景阳性 0/247（0.00%）。这些是可复现的仿真域挑战和负对照，不等价于真实环境 precision/recall；r7 目录保留为挑战集原始记录。

PX4/MAVROS 验收使用宿主已安装的 PX4 v1.15、Gazebo Garden 和 ROS 2 Humble MAVROS：

```bash
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
bash tools/run_px4_mavros_sitl_acceptance.sh
```

PX4 v1.15 的 onboard MAVLink 端口为本地 `14580`、远端 `14540`；MAVROS 使用 `udp://:14540@localhost:14580`。验收脚本会保存 PX4 和 MAVROS 日志、服务往返时间、setpoint 间隔和悬停高度误差。

完整 NavRL sim-to-sim 闭环与 PPO 计算开销另行执行：

```bash
bash docker/run_sim_acceptance.sh
python3 tools/render_acceptance_report.py
```

## 与实物的差距

这个替代验证仍缺少相机的真实噪声与畸变、滚动快门、深度失效模式、机体震动、时间同步误差、无线链路延迟、气动和桨流、电源压降及真实 EKF 漂移。因此实物阶段必须重新测量内外参与深度比例，并执行台架、系留和监督飞行门禁。具体测量表和停止条件见 `SIM_TO_REAL_BENCHMARK_PROTOCOL_CN.md` 与 `PHYSICAL_DEPLOYMENT_GUIDE_CN.md`。
