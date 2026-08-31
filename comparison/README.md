# NavRL Comparison Experiments

这个目录用于把论文里的控制效果对比跑得更可复现。当前工程已经把你的改进拆成了三个开关：

| 变体 | 含义 | Hydra overrides |
|---|---|---|
| `baseline_ppo` | 原始 NavRL/PPO 风格 baseline，不启用新增模块 | `step1=false, step2=false, step3=false` |
| `geom_encoder_ppo` | 只启用动态障碍几何编码器 | `step1=true, step2=false, step3=false` |
| `risk_aux_ppo` | 几何编码器 + 短时风险场预测辅助损失 | `step1=true, step2=true, step3=false` |
| `ours_constrained` | 完整方法：几何编码器 + 风险预测 + constrained PPO/cost critic | `step1=true, step2=true, step3=true` |

## 训练

跑单个变体：

```bash
bash comparison/train_policy_variants.sh baseline_ppo seed=0 env_dyn.num_obstacles=64
```

跑完整四组：

```bash
bash comparison/train_policy_variants.sh all seed=0 env_dyn.num_obstacles=64
```

脚本会继续使用 `isaac-training/training/cfg/train.yaml` 和 `ppo.yaml` 里的公共训练参数，只替换算法结构开关，并把 W&B 名称写成 `comparison/<variant>`。

## 评估

训练结束后，把每个 checkpoint 路径传给评估脚本：

```bash
bash comparison/evaluate_policy_variants.sh \
  baseline_ppo=/path/to/baseline/checkpoint_final.pt \
  geom_encoder_ppo=/path/to/geom/checkpoint_final.pt \
  risk_aux_ppo=/path/to/risk_aux/checkpoint_final.pt \
  ours_constrained=/path/to/ours/checkpoint_final.pt \
  -- --num-runs 20 --seed-start 100 --output comparison/eval_results.json
```

评估脚本会调用 `isaac-training/training/scripts/eval_models.py`，并为每个 checkpoint 自动附加对应的 Hydra 结构开关，避免加载 checkpoint 时出现网络结构不匹配。

交付包中的同配置原版/改进版对照见 `../NAVRL_BASELINE_COMPARISON_CN.md` 和 `../benchmark_results/navrl_baseline_comparison.json`。两份 W&B 摘要已量化成功率、碰撞率和相对变化；平均速度、航点误差由 Isaac-Sim worker 输出，但在 Isaac-Sim 运行时缺失的主机上保持 `N/A`，不能从回合长度反推。

`kinematic_policy_eval.py` 仅是可选的同观测运动学回放，明确标记为非 Isaac-Sim、非刚体动力学，不能用于论文主表或宣称飞行成功率；主表只接受 `eval_models.py` 在 Isaac-Sim/OmniDrones 中输出的字段。

## 论文表格建议

主表建议报告：

- `reach_goal`: 成功率，越高越好。
- `collision`: 碰撞率，越低越好。
- `episode_len`: 到达效率或失败拖尾，结合成功率一起解释。
- `return`: 综合训练目标，只作为辅助指标。

如果要再补传统规划类对比，优先在 ROS/Gazebo 部署侧接入已有 planner，例如 EGO-Planner、ViGO/B-spline 或 DPMPC-Planner；Isaac 训练侧先保留上面四组消融，公平性会更强。
