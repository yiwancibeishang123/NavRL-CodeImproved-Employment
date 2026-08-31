# 原版 NavRL 与改进版量化对照

本对照解决“改进量未量化”的验收缺口。两份 W&B 摘要使用相同训练配置：1024 个并行环境、80 个动态障碍、最大回合长度 2200、seed 0。这里的数值是训练端同配置评估摘要，不把 ROS clean/stress 控制日志误写成导航任务成功率。

| 指标 | 原版 NavRL | 改进版 NavRL | 改进版 - 原版 | 判定 |
|---|---:|---:|---:|---|
| 到达成功率 | 71.39% | 84.86% | +13.48 个百分点 | 改善 |
| 碰撞率 | 25.98% | 11.04% | -14.94 个百分点（相对降低 57.58%） | 改善 |
| 平均回合长度 | 1858.17 | 2064.83 | +206.66 | 辅助指标 |
| 平均回报 | 9713.33 | 9971.29 | +257.95 | 辅助指标 |
| 平均速度 | N/A | N/A | N/A | 历史摘要未记录 |
| 航点误差 | N/A | N/A | N/A | 历史摘要未记录 |

证据原件是 `experiment/train/selected_runs/NavRL-80/files/wandb-summary.json` 和 `experiment/train/selected_runs/step3-80new/files/wandb-summary.json`，机器可读汇总见 `benchmark_results/navrl_baseline_comparison.json`，柱状图见 `benchmark_results/navrl_baseline_comparison.svg`。

## 缺失指标的处理

平均速度和航点误差不能由成功率、碰撞率或回合长度反推，因此本报告不填估算值。`isaac-training/training/scripts/eval_checkpoint_worker.py` 和环境统计接口已经改为逐回合输出 `average_speed_mps`、`waypoint_error_m`，并另外输出成功回合的到达时间、路径长度和速度。在满足 Isaac-Sim 运行时要求的环境中可执行：

```bash
bash comparison/evaluate_policy_variants.sh \
  baseline_ppo=/abs/path/navrl_checkpoint.pt \
  ours_constrained=/abs/path/step3-80new.pt \
  -- --num-runs 20 --seed-start 100 --output comparison/eval_results.json
```

当前交付主机没有 Isaac-Sim 运行时，所以不会把缺失项伪装成已测量数据；拿到该运行环境后，使用同一 seed 集合即可补齐四指标对照。

本机硬件为 RTX 3050 4 GB 显存、约 15 GB 内存。NVIDIA Isaac Sim 4.0 官方最低建议为 32 GB RAM、RTX 3070/8 GB VRAM；因此直接安装容器也无法稳定运行刚体/渲染评估，本次采用已通过的 Gazebo RGB-D/PX4 高保真链路，并把 Isaac 评估器字段和复现入口保留在包内。该环境边界和官方要求见 [NVIDIA Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/4.0.0/installation/requirements.html)。

## 解释边界

这份对照证明改进训练端在同配置摘要中的成功率和碰撞率变化，但不等价于真机性能提升。真机仍需按照 `PHYSICAL_DEPLOYMENT_GUIDE_CN.md` 进行标定、系留和受监督飞行验收。
