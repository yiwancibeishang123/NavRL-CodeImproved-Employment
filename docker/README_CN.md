# Noetic 容器验证

宿主为 Ubuntu 22.04/ROS Humble 时，不应将 ROS Noetic 二进制直接安装到宿主。`Dockerfile.noetic` 使用 Ubuntu 20.04 Noetic 用户空间，安装编译本 ROS1 工程所需的 Gazebo、MAVROS、PCL、视觉消息和部署 Python 依赖。

执行离线构建、checkpoint 校验、标定 YAML 校验和 CPU 策略基准：

```bash
bash docker/run_noetic_checks.sh
```

该脚本不启动 PX4、不解锁飞控，也不把容器中的工作空间写回源码目录。CPU 基准会写入宿主 `benchmark_results/policy_profile_cpu_noetic.json`，终端同时输出完整 JSON。

图形 Gazebo 和 GPU 测试需要主机安装 NVIDIA Container Toolkit，并显式使用 `--gpus all`、X11/Wayland 显示转发或 headless Gazebo。通过 `docker run --gpus all ... nvidia-smi` 确认容器可见 GPU 后，再把 `benchmark_policy.py --device cuda:0` 的输出保存为论文计算开销证据。

## 无头 Gazebo 闭环验收

以下命令使用 Xvfb 软件渲染启动 Gazebo 深度相机，连续执行干净组和受扰动组（约两分钟），并在宿主生成完整控制 JSONL、观测 trace 及差距汇总：

```bash
bash docker/run_sim_acceptance.sh
```

默认结果目录为 `benchmark_results/sim_latest/`。受扰动参数是未标定的压力测试，不能当作真机精度或实飞安全结论；详见 `SIM_TO_REAL_BENCHMARK_PROTOCOL_CN.md`。

## 高保真视觉、标定与跟踪验收

以下命令启动独立的 Gazebo RGB-D 场景和移动行人，让部署 YOLO 对实际渲染图像推理，并将检测、跟踪、位置/速度误差和推理时间写入宿主结果目录：

```bash
bash docker/run_hifi_perception_acceptance.sh
```

默认输出为 `benchmark_results/hifi_perception/`；为保留某次结果，可设置 `NAVRL_PERCEPTION_RESULTS=benchmark_results/hifi_perception_YYYYMMDD_rN`。相机内参、深度编码和机体到相机 TF 的独立真值校验也会一并生成。PX4/MAVROS 的 Gazebo Garden 验收入口和边界见 `HIGH_FIDELITY_SIM_ACCEPTANCE_CN.md`。
