# 消融与仿真证据说明

## 结论口径

当前包中可审计的训练产物只有两项：原版 `NavRL-80` checkpoint 与完整改进版 `step3-80new` checkpoint。因此可以严格报告二者的同配置训练端摘要对照：成功率 71.39% -> 84.86%，碰撞率 25.98% -> 11.04%。这不是四项逐项消融。

机器可读状态见 `benchmark_results/navrl_ablation_status.json`。其中明确列出已归档的两项结果、缺失的中间 checkpoint 以及可复现实验入口；缺失 checkpoint 补齐前，报告不会填入逐项消融百分比。

四项模块的独立 checkpoint 未随训练归档保存，不能把完整模型的权重强行关掉某个模块来伪造消融结果。那会造成结构不匹配或未训练参数参与推理，数值没有科学含义。

## 已实现的可复现流程

| 变体 | 几何编码 | 历史/风险预测 | cost critic | Lagrange | 入口 |
|---|---:|---:|---:|---:|---|
| `baseline_ppo` | 关 | 关 | 关 | 关 | `comparison/train_policy_variants.sh baseline_ppo` |
| `geom_encoder_ppo` | 开 | 关 | 关 | 关 | `comparison/train_policy_variants.sh geom_encoder_ppo` |
| `risk_aux_ppo` | 开 | 开 | 关 | 关 | `comparison/train_policy_variants.sh risk_aux_ppo` |
| `ours_constrained` | 开 | 开 | 开 | 开 | `comparison/train_policy_variants.sh ours_constrained` |

训练完成后，按相同地图、种子集合和评估器执行：

```bash
bash comparison/evaluate_policy_variants.sh \
  baseline_ppo=/abs/path/baseline.pt \
  geom_encoder_ppo=/abs/path/geom.pt \
  risk_aux_ppo=/abs/path/risk.pt \
  ours_constrained=/abs/path/ours.pt \
  -- --num-runs 20 --seed-start 100 --output comparison/eval_results.json
```

输出要求至少包含成功率、碰撞率、平均速度、到达时间、路径长度和航点误差，并按 clean 与挑战场景拆开。只有所有四个由各自结构训练而来的 checkpoint 都存在时，才可将该表写入论文主文。

## Gazebo 指标补齐

Gazebo 部署侧不依赖 Isaac Sim 才能测效率。`navigation.py` 对每个目标任务记录：起止时间、里程计路径长度、路径时间平均速度、里程计瞬时速度均值、终点到航点的 3D 距离与任务结果。`mission_metrics_analyze.py` 从原始 JSONL 导出汇总；失败任务也保留速度、路径和终点误差，到达时间则按成功任务单独统计，绝不用失败超时替代到达时间。

第 r5 次 Gazebo 采集的样本位于 `benchmark_results/sim_latest_20260828_r5/`，可视化见其中的 `task_metrics.svg`。干净组 1 次任务在窗口结束时未完成，路径 1.622 m、路径平均速度 0.049 m/s、终点误差 2.678 m；受扰动组 1 次任务成功，完成时间 2.919 s、路径 1.374 m、路径平均速度 0.471 m/s、终点误差 0.985 m。样本量为 1，且初始状态存在动力学差异，不能据此比较 clean/stress 性能优劣；它证明指标已由 Gazebo 真值里程计实际采集，而不是由 Isaac 依赖缺失而留空。

## 视觉故障安全边界

视觉组合退化时，系统不再复用旧的 YOLO/跟踪目标状态。若独立的深度占据图 LiDAR 在 1.4 m 内看见障碍物，控制器调用安全盾后沿反方向以最多 0.45 m/s 退让；若 LiDAR 也无近距返回，才进入位置保持。`benchmark_results/perception_safety_20260828_r1/lidar_fallback_acceptance.json` 在视觉 watchdog 已告警后放置 0.9 m 障碍物，记录到 63 条 LiDAR-only 退让命令，首条延迟 1.3384 s，局部速度约 `[-0.45, 0, 0]` m/s。

这证明“视觉失效 + LiDAR 可见障碍”可独立退让；不证明 LiDAR 视场外、透明/弱反射物体或速度极快目标的安全性。这样的限制在实物阶段仍需传感器覆盖和制动距离验证。
