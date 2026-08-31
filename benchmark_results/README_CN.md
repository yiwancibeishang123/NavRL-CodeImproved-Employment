# 验收证据索引

这里保留的是可以随源码一起复核的最小证据集。原始数据统一使用 JSON、JSONL、CSV 和 SVG，旧的重复运行、缓存和中间录屏没有纳入仓库。

## 快速查看

| 内容 | 路径 |
| --- | --- |
| 九项工作总看板 | `acceptance_dashboard.svg` |
| 原版/改进版 NavRL 对照 | `navrl_baseline_comparison.json`、`navrl_baseline_comparison.svg` |
| 最终六障碍物自主避障航线 | `自主避障演示验收/` |
| 计算开销与推理效率 | `自主避障演示验收/计算开销与推理效率分析.md` |
| RGB-D、YOLO、标定和退化恢复 | `hifi_perception_20260827_r7/`、`hifi_perception_20260828_r8/` |
| PX4/MAVROS SITL | `hifi_px4_mavros/` |
| LiDAR 独立退让 | `perception_safety_20260828_r1/` |
| 20 分钟长时稳定性 | `long_run_20260828_r1/` |
| clean/stress sim-to-sim | `sim_latest_20260826_r2/`、`sim_latest_20260828_r5/` |
| 五次任务统计 | `final_clean_repeated.json`、`final_pressure_repeated.json` |

## 统计口径

`policy_profile_cpu_noetic.json` 是纯模型前向基准，不包含 ROS、Gazebo、感知和安全层。完整闭环的 `control` JSONL 会额外记录控制耗时、控制间隔、调度滞后、传感器年龄、安全盾介入和进程 RSS。任务级 `mission_complete` 记录完成时间、路径长度、速度、航点误差和最小 LiDAR 距离。

`自主避障演示验收/六障碍物完整自主避障_视频同步原始记录.jsonl` 是最终客户演示的权威原始记录；对应汇总和表格只统计 `full_route_video` 任务窗口。压力组标记为 `stress_example_not_calibrated`，只能用于 sim-to-sim 敏感性测试，不能当作真实传感器标定结果。

## 复现入口

从仓库根目录执行：

```bash
bash docker/run_noetic_checks.sh
NAVRL_SIM_REPEATS=5 bash docker/run_sim_acceptance.sh
```

更完整的实验口径、实物部署门禁和论文表格要求见根目录的 `SIM_REAL_GAP_ANALYSIS_CN.md` 与 `SIM_TO_REAL_BENCHMARK_PROTOCOL_CN.md`。
