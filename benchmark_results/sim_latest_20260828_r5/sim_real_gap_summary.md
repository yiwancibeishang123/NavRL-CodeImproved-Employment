# 仿真与实际应用差距汇总

本文件由 `sim_real_gap_analyze.py` 自动生成。`N/A` 表示日志未提供该指标，不能据此推断性能。

| 指标 | clean_sim | perturbed_sim |
|---|---:|---:|
| action_delta_norm_max | 2.06754 | 2 |
| action_delta_norm_p50 | 1.45252 | 1.82101 |
| action_delta_norm_p95 | 1.97509 | 2 |
| average_speed_mps | 0.0490076 | 0.470687 |
| command_rate_hz | 19.0789 | 18.3463 |
| command_records | 59 | 149 |
| completion_time_s | N/A | 2.919 |
| configured_control_rate_hz | 20 | 20 |
| control_ms_max | 193.215 | 179.417 |
| control_ms_p50 | 0 | 0 |
| control_ms_p95 | 25.5537 | 34.8868 |
| control_overrun_rate | 0.00253165 | 0.00126103 |
| control_period_jitter_ms_max | 143.834 | 129.717 |
| control_period_jitter_ms_p50 | 0.9344 | 0.81275 |
| control_period_jitter_ms_p95 | 6.5615 | 7.59785 |
| control_rate_hz | 19.9293 | 19.9371 |
| control_records | 790 | 793 |
| cuda_allocated_mib_max | N/A | N/A |
| cuda_allocated_mib_p50 | N/A | N/A |
| cuda_allocated_mib_p95 | N/A | N/A |
| cuda_reserved_mib_max | N/A | N/A |
| cuda_reserved_mib_p50 | N/A | N/A |
| cuda_reserved_mib_p95 | N/A | N/A |
| hold_rate | 0.912658 | 0.785624 |
| inference_device_ms_max | N/A | N/A |
| inference_device_ms_p50 | N/A | N/A |
| inference_device_ms_p95 | N/A | N/A |
| inference_wall_ms_max | 71.4318 | 28.8463 |
| inference_wall_ms_p50 | 5.0372 | 4.7461 |
| inference_wall_ms_p95 | 15.4784 | 16.4871 |
| obstacle_age_s_max | 0.173 | 0.178 |
| obstacle_age_s_p50 | 0.024 | 0.011 |
| obstacle_age_s_p95 | 0.047 | 0.046 |
| odom_age_s_max | 0.129 | 0.104 |
| odom_age_s_p50 | 0.016 | 0.013 |
| odom_age_s_p95 | 0.038 | 0.036 |
| path_length_m | 1.62171 | 1.37394 |
| policy_inference_records | 59 | 149 |
| process_rss_mib_max | 456.676 | 456.266 |
| process_rss_mib_p50 | 456.438 | 456.109 |
| process_rss_mib_p95 | 456.648 | 456.266 |
| raycast_age_s_max | 0.167 | 0.176 |
| raycast_age_s_p50 | 0.014 | 0.009 |
| raycast_age_s_p95 | 0.034 | 0.028 |
| safety_intervention_rate | 1 | 1 |
| scheduler_lag_ms_max | 227 | 213 |
| scheduler_lag_ms_p50 | 140 | 125 |
| scheduler_lag_ms_p95 | 142 | 126 |
| serial_policy_throughput_hz | 116.417 | 131.674 |
| stale_observation_hold_rate | 0.167089 | 0.172762 |
| success_rate | 0 | 1 |
| timing_degrade_hold_rate | 0.74557 | 0.612863 |
| timing_degrade_rate | 0.74557 | 0.612863 |
| waypoint_error_m | 2.67845 | 0.985186 |

## 运行环境

- clean_sim: device=cpu; torch=2.1.2+cu121; platform=Linux-6.8.0-138-generic-x86_64-with-glibc2.29; python=3.8.10
- perturbed_sim: device=cpu; torch=2.1.2+cu121; platform=Linux-6.8.0-138-generic-x86_64-with-glibc2.29; python=3.8.10

## 解释边界

- `inference_wall_ms` 是控制节点观察到的前向延迟；CUDA 下 `inference_device_ms` 仅是 GPU kernel 时间，两者不可互换。
- `control_ms` 包含观测构造、安全服务和指令发布路径；`control_overrun_rate` 依据记录的控制周期判断。
- `timing_degrade_rate`/`timing_degrade_hold_rate` 表示控制节点检测到连续超周期或 scheduler lag 超阈值后主动保持的记录比例；保持动作不复用迟到的 PPO 动作。
- clean_sim 与 perturbed_sim 的差异量化可控噪声、丢包和时延敏感性。未传入 physical 日志时，报告不包含也不推断真机性能。
- 成功率、碰撞率、完成时间和路径长度只能来自独立任务汇总 JSON，不能由控制日志替代。
