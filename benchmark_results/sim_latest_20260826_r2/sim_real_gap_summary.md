# 仿真与实际应用差距汇总

本文件由 `sim_real_gap_analyze.py` 自动生成。`N/A` 表示日志未提供该指标，不能据此推断性能。

| 指标 | clean_sim | perturbed_sim |
|---|---:|---:|
| action_delta_norm_max | 1.99052 | 3.7894 |
| action_delta_norm_p50 | 1.58649 | 0.96189 |
| action_delta_norm_p95 | 1.92499 | 3.62258 |
| command_rate_hz | 16.2698 | 15.3251 |
| command_records | 206 | 112 |
| configured_control_rate_hz | 20 | 20 |
| control_ms_max | 120.955 | 128.569 |
| control_ms_p50 | 16.1731 | 0 |
| control_ms_p95 | 29.6759 | 40.3071 |
| control_overrun_rate | 0.00307692 | 0.0153257 |
| control_period_jitter_ms_max | 224.659 | 119.479 |
| control_period_jitter_ms_p50 | 84.3325 | 1.30125 |
| control_period_jitter_ms_p95 | 103.788 | 20.8092 |
| control_rate_hz | 20 | 19.9271 |
| control_records | 325 | 783 |
| cuda_allocated_mib_max | N/A | N/A |
| cuda_allocated_mib_p50 | N/A | N/A |
| cuda_allocated_mib_p95 | N/A | N/A |
| cuda_reserved_mib_max | N/A | N/A |
| cuda_reserved_mib_p50 | N/A | N/A |
| cuda_reserved_mib_p95 | N/A | N/A |
| hold_rate | 0.190769 | 0.810983 |
| inference_device_ms_max | N/A | N/A |
| inference_device_ms_p50 | N/A | N/A |
| inference_device_ms_p95 | N/A | N/A |
| inference_wall_ms_max | 14.2311 | 104.727 |
| inference_wall_ms_p50 | 3.2817 | 9.8802 |
| inference_wall_ms_p95 | 7.6141 | 32.6083 |
| obstacle_age_s_max | 0.049 | 0.19 |
| obstacle_age_s_p50 | 0 | 0.022 |
| obstacle_age_s_p95 | 0.049 | 0.046 |
| odom_age_s_max | 0.05 | 0.134 |
| odom_age_s_p50 | 0.012 | 0.014 |
| odom_age_s_p95 | 0.028 | 0.0369 |
| policy_inference_records | 206 | 112 |
| process_rss_mib_max | 457.965 | 458.383 |
| process_rss_mib_p50 | 457.836 | 458.262 |
| process_rss_mib_p95 | 457.948 | 458.375 |
| raycast_age_s_max | 0.049 | 0.174 |
| raycast_age_s_p50 | 0 | 0.011 |
| raycast_age_s_p95 | 0.047 | 0.03615 |
| safety_intervention_rate | 1 | 1 |
| scheduler_lag_ms_max | 6 | 228 |
| scheduler_lag_ms_p50 | 0 | 143 |
| scheduler_lag_ms_p95 | 1 | 144 |
| serial_policy_throughput_hz | 253.871 | 76.1203 |
| stale_observation_hold_rate | 0.190769 | 0.210728 |
| timing_degrade_hold_rate | 0 | 0.600255 |
| timing_degrade_rate | 0 | 0.600255 |

## 运行环境

- clean_sim: device=cpu; torch=2.1.2+cu121; platform=Linux-6.8.0-138-generic-x86_64-with-glibc2.29; python=3.8.10
- perturbed_sim: device=cpu; torch=2.1.2+cu121; platform=Linux-6.8.0-138-generic-x86_64-with-glibc2.29; python=3.8.10

## 解释边界

- `inference_wall_ms` 是控制节点观察到的前向延迟；CUDA 下 `inference_device_ms` 仅是 GPU kernel 时间，两者不可互换。
- `control_ms` 包含观测构造、安全服务和指令发布路径；`control_overrun_rate` 依据记录的控制周期判断。
- `timing_degrade_rate`/`timing_degrade_hold_rate` 表示控制节点检测到连续超周期或 scheduler lag 超阈值后主动保持的记录比例；保持动作不复用迟到的 PPO 动作。
- clean_sim 与 perturbed_sim 的差异量化可控噪声、丢包和时延敏感性。未传入 physical 日志时，报告不包含也不推断真机性能。
- 成功率、碰撞率、完成时间和路径长度只能来自独立任务汇总 JSON，不能由控制日志替代。
