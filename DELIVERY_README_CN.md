# NavRL 仿真交付包

本包包含 ROS1 Noetic/Gazebo 闭环、PPO 推理、观测与 LiDAR 接口、动态目标/YOLO 跟踪、相机标定、PX4/MAVROS SITL、安全看门狗、sim-to-sim 证据、长时资源监控及部署文档。

## 快速验收

```bash
bash docker/run_noetic_checks.sh
NAVRL_SIM_REPEATS=5 NAVRL_SIM_MISSION_SECONDS=6 bash docker/run_sim_acceptance.sh
```

仿真脚本会输出每次任务的 JSONL、任务汇总 JSON、速度/航点误差/碰撞代理和资源指标。`comparison/train_policy_variants.sh` 与 `comparison/evaluate_policy_variants.sh` 是四变体 Isaac-Sim 消融入口；训练端必须在安装 Isaac-Sim/omni_drones 的主机执行。

## 证据索引

- `ACCEPTANCE_REPORT_CN.md`: 九项交付验收总表。
- `SIM_REAL_GAP_ANALYSIS_CN.md`: 仿真与实际应用差距分析。
- `benchmark_results/hifi_perception_20260828_r8/`: 感知退化、恢复和延迟。
- `benchmark_results/hifi_px4_mavros/`: PX4/MAVROS SITL。
- `benchmark_results/long_run_20260828_r1/`: 20 分钟资源/实时性曲线。
- `benchmark_results/ablation_results.json`: 消融数据完整性状态和复现命令。
