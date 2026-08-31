# NavRL 改进的可解释差异

## 对照边界

原版和改进版使用同一任务环境、同一原始观测接口（8 维状态、36x4 LiDAR、5 个动态目标槽位）、同一动作上限 2 m/s、同一奖励函数和同一训练规模（1024 并行环境、80 个动态障碍、最大回合 2200、seed 0）。因此成功率变化不能归因于换地图、换奖励或只调安全阈值。

| 组件 | 原版 NavRL | 改进版 NavRL | 改动目的 |
|---|---|---|---|
| 动态目标输入 | 单帧/基础编码 | `current_geometry`：相对位置、速度、尺寸标准化编码 | 让网络直接区分接近、横穿和远离目标 |
| 短期预测 | 无 | 4 步历史 MLP（每步 7 维）+ 8 步风险场预测，8x8、4 m 局部网格 | 用未来占用风险补偿检测延迟和速度估计噪声 |
| 策略融合 | 仅 actor/critic | 共享风险 embedding（32 维）供风险 critic 使用 | 将感知风险转成可解释的价值条件 |
| 约束目标 | 标准 PPO reward critic | 独立 128 隐层 cost critic + Lagrange 乘子（预算 0.003） | 直接压低碰撞/近障碍代价，而不是用回报间接替代安全 |
| 训练辅助 | 只有策略损失 | 风险预测辅助损失系数 0.1，预测学习率 8e-5 | 让表示学习受真实未来风险监督，降低仅靠调参过拟合的可能 |

实现证据：`isaac-training/training/scripts/ppo.py` 中的 `DynamicObstacleCurrentGeometryEncoder`、`RiskFieldPredictor` 和 `CostCritic`；部署端通过同一 `ppo.py` 加载 `step3-80new.pt`。原版和改进版的 W&B 配置原件分别为 `experiment/train/selected_runs/NavRL-80/files/config.yaml` 与 `experiment/train/selected_runs/step3-80new/files/config.yaml`。

## 量化结果

同配置 W&B 摘要给出：成功率 71.39% -> 84.86%（+13.48 个百分点），碰撞率 25.98% -> 11.04%（-14.94 个百分点，相对降低 57.58%）。改进版平均回合长度由 1858.17 增至 2064.83 步，说明安全性提升伴随更长的决策过程，不能宣称“无代价加速”。

平均速度、路径长度、到达时间和航点误差已加入 `NavigationEnv` 的逐回合统计契约，并由 `eval_checkpoint_worker.py` 输出；只有在具备 Isaac-Sim/OmniDrones 运行时的机器执行同一 seed 集合后，才将这些字段填入原版/改进版对照表，不从成功率或回合长度反推。

## 消融和复现

`comparison/evaluate_policy_variants.sh` 支持四个可解释变体：`baseline_ppo`、`geom_encoder_ppo`、`risk_aux_ppo`、`ours_constrained`。它们固定 checkpoint、地图、seed 和评估器，只切换 Step 1/2/3 开关，可将提升拆分为表示、预测和约束三部分。当前交付主机因 Isaac-Sim 官方最低约 32 GB RAM、RTX 3070/8 GB VRAM 级别而无法运行该 GPU 仿真；Gazebo 高保真闭环已独立验证感知和控制接口，不能冒充 Isaac 任务成功率。
