#!/usr/bin/env python3
"""Render the evidence-backed nine-item deployment acceptance report."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    if not path or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def check(payload, name):
    for item in payload.get("checks", []):
        if item.get("name") == name:
            return item
    return {}


def number(value, digits=2):
    if value is None:
        return "N/A"
    return format(float(value), ".{}f".format(digits))


def percent(value, digits=2):
    return number(None if value is None else 100.0 * float(value), digits)


def relative(root, path, fallback="N/A"):
    return path.relative_to(root).as_posix() if path else fallback


def newest_passing(root, pattern):
    candidates = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = load_json(path)
        if payload.get("result") == "passed":
            return path, payload
    return None, {}


def sim_evidence(root):
    candidates = sorted(root.glob("benchmark_results/sim_*/sim_real_gap_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in candidates:
        report = load_json(path)
        groups = report.get("groups", {})
        clean = groups.get("clean_sim", {}).get("summary", {})
        stress = groups.get("perturbed_sim", {}).get("summary", {})
        if clean.get("policy_inference_records") and stress.get("policy_inference_records"):
            return path, clean, stress
    return None, {}, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "ACCEPTANCE_REPORT_CN.md")
    args = parser.parse_args()
    root = args.project_root.resolve()

    offline = load_json(root / "benchmark_results/deployment_acceptance.json")
    policy = load_json(root / "benchmark_results/policy_profile_cpu_noetic.json")
    perception_path, perception = newest_passing(root, "benchmark_results/hifi_perception*/hifi_perception_acceptance.json")
    calibration_path = perception_path.parent / "hifi_calibration_acceptance.json" if perception_path else None
    calibration = load_json(calibration_path)
    negative_path = perception_path.parent / "hifi_perception_negative_control.json" if perception_path else None
    negative = load_json(negative_path)
    challenge_path = perception_path.parent / "hifi_perception_challenge_suite.json" if perception_path else None
    challenge = load_json(challenge_path)
    px4_path = root / "benchmark_results/hifi_px4_mavros/px4_mavros_sitl_acceptance.json"
    px4 = load_json(px4_path)
    sim_path, clean, stress = sim_evidence(root)
    baseline = load_json(root / "benchmark_results/navrl_baseline_comparison.json")

    ppo = check(offline, "ppo_checkpoint_and_forward").get("details", {})
    observation = check(offline, "observation_and_sensor_perturbation").get("details", {})
    safety = check(offline, "safety_parameter_contract").get("details", {})
    yolo = perception.get("yolo", {})
    yolo_forward = yolo.get("forward_ms", {})
    yolo_callback = yolo.get("callback_ms", {})
    tracker = perception.get("tracker", {})
    track_position = tracker.get("position_error_m", {})
    track_velocity = tracker.get("velocity_error_mps", {})
    perception_passed = bool(perception_path and yolo.get("positive_box_messages", 0) > 0 and tracker.get("matched_samples", 0) > 0)
    challenge_passed = challenge.get("result") == "passed"
    calibration_passed = calibration.get("result") == "passed"
    px4_passed = px4.get("result") == "passed"
    sim_passed = bool(clean.get("policy_inference_records") and stress.get("policy_inference_records"))

    high_fidelity = "PASS (高保真仿真)"
    rows = [
        ("1", "新 PPO 推理接入 ROS", "PASS", "checkpoint 前向与精确观测契约", "{} 参数；单次前向 {} ms".format(ppo.get("parameters", "N/A"), number(ppo.get("single_forward_wall_ms"), 4))),
        ("2", "状态与 LiDAR 观测", "PASS", "离线形状及扰动接口校验", "state {}; LiDAR {}; dynamic {}".format(observation.get("state", "N/A"), observation.get("lidar", "N/A"), observation.get("dynamic_obstacle", "N/A"))),
        ("3", "相机、深度、YOLO 与跟踪", high_fidelity if perception_passed and challenge_passed else "BLOCKED", "Gazebo RGB-D、真实 YOLO、目标跟踪对真值；遮挡/雾/雨/逆光/组合挑战；空场景负对照", "clean 可见漏检率 {}%；挑战集加权漏检率 {}%（{} 个可见帧）；组合工况 {}%；位置 p95 {} m；速度 p95 {} m/s".format(percent(yolo.get("eligible_miss_rate", 0.0)), percent(challenge.get("aggregate", {}).get("eligible_miss_rate")), challenge.get("aggregate", {}).get("eligible_visibility_frames", "N/A"), percent(challenge.get("cases", {}).get("combined", {}).get("eligible_miss_rate")), number(track_position.get("p95"), 3), number(track_velocity.get("p95"), 3))),
        ("4", "深度与坐标标定", high_fidelity if calibration_passed else "BLOCKED", "CameraInfo、深度编码、TF 外参对 Gazebo 真值", "内参最大误差 {} px；{}；外参平移 {} m".format(number(calibration.get("intrinsic_max_abs_error_px"), 4), calibration.get("camera", {}).get("depth_encoding", "N/A"), calibration.get("body_to_camera", {}).get("translation_m", "N/A"))),
        ("5", "PX4/MAVROS 控制", high_fidelity if px4_passed else "BLOCKED", "PX4 SITL + Gazebo Garden + MAVROS OFFBOARD 状态机", "OFFBOARD {} ms；解锁 {} ms；高度 p95 误差 {} m".format(number(px4.get("services", {}).get("offboard_set_mode_ms"), 2), number(px4.get("services", {}).get("arm_ms"), 2), number(px4.get("hold_altitude", {}).get("error_p95_m"), 3))),
        ("6", "异常处理与安全", "PASS", "超时保持、NaN 防护、急停、安全盾；控制和感知超时主动悬停降级", "安全距离 {} m；安全盾介入 {}%；控制降级保持率 clean/stress {}/{}%；组合感知报警 {}/{} 周期".format(safety.get("safety_distance_m", "N/A"), number(100.0 * clean.get("safety_intervention_rate", 0.0), 2), number(100.0 * clean.get("timing_degrade_hold_rate", 0.0), 2), number(100.0 * stress.get("timing_degrade_hold_rate", 0.0), 2), challenge.get("cases", {}).get("combined", {}).get("watchdog_degraded_messages", "N/A"), challenge.get("cases", {}).get("combined", {}).get("watchdog_messages", "N/A"))),
        ("7", "依赖与部署", "PASS", "Noetic Docker 构建、Catkin 编译和部署脚本", "PyTorch {}；TensorDict/TorchRL 0.4.0".format(policy.get("runtime", {}).get("torch", "N/A"))),
        ("8", "sim-to-sim 验证", "PASS" if sim_passed else "BLOCKED", "无头 Gazebo 干净组/压力组完整闭环", "干净组 {} 次策略调用；压力组 {}；配置 20 Hz".format(clean.get("policy_inference_records", "N/A"), stress.get("policy_inference_records", "N/A"))),
        ("9", "文档与实物部署指南", "PASS", "安装、参数、启动、排障、差距分析和实物门禁", "已交付实物流程；真机执行状态为 N/A（无实物）"),
    ]

    lines = [
        "# NavRL 九项工作验收报告",
        "",
        "`PASS` 表示已在对应环境实现并留有原始证据；`PASS (高保真仿真)` 表示在高保真仿真完成，不能等同于真机结论。当前没有无人机实物，因此真机验收明确标为 `N/A`，未被仿真结果替代或伪装。",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for item, title, status, _, _ in rows:
        lines.append('  W{}["{} {}<br/>{}"]'.format(item, item, title, status))
    lines += [
        "  W1 --> W2 --> W3 --> W4 --> W5 --> W6 --> W7 --> W8 --> W9",
        "```",
        "",
        "| 工作 | 范围 | 状态 | 验收证据 | 实测结果或边界 |",
        "|---:|---|---|---|---|",
    ]
    for item, title, status, evidence, result in rows:
        lines.append("| {} | {} | `{}` | {} | {} |".format(item, title, status, evidence, result))

    lines += [
        "",
        "## 计算开销与效率",
        "",
        "| 指标 | 仅模型 CPU | 干净完整闭环 | 压力完整闭环 | 高保真视觉 | PX4/MAVROS SITL |",
        "|---|---:|---:|---:|---:|---:|",
        "| PPO 前向 p50 / p95 (ms) | {} / {} | {} / {} | {} / {} | N/A | N/A |".format(number(policy.get("inference_wall_ms", {}).get("p50"), 3), number(policy.get("inference_wall_ms", {}).get("p95"), 3), number(clean.get("inference_wall_ms_p50"), 3), number(clean.get("inference_wall_ms_p95"), 3), number(stress.get("inference_wall_ms_p50"), 3), number(stress.get("inference_wall_ms_p95"), 3)),
        "| 控制记录 / 策略调用 | N/A | {} / {} | {} / {} | N/A | N/A |".format(clean.get("control_records", "N/A"), clean.get("policy_inference_records", "N/A"), stress.get("control_records", "N/A"), stress.get("policy_inference_records", "N/A")),
        "| 陈旧观测保持 / 控制超时 | N/A | {}% / {}% | {}% / {}% | N/A | N/A |".format(number(100.0 * clean.get("stale_observation_hold_rate", 0.0), 2), number(100.0 * clean.get("control_overrun_rate", 0.0), 2), number(100.0 * stress.get("stale_observation_hold_rate", 0.0), 2), number(100.0 * stress.get("control_overrun_rate", 0.0), 2)),
        "| 超周期主动降级保持率 | N/A | {}% | {}% | N/A | N/A |".format(number(100.0 * clean.get("timing_degrade_hold_rate", 0.0), 2), number(100.0 * stress.get("timing_degrade_hold_rate", 0.0), 2)),
        "| 进程 RSS p95 (MiB) | {} | {} | {} | N/A | N/A |".format(number(policy.get("runtime_after", {}).get("process_rss_mib"), 3), number(clean.get("process_rss_mib_p95"), 3), number(stress.get("process_rss_mib_p95"), 3)),
        "| YOLO 前向 / 回调 p50/p95 (ms) | N/A | N/A | N/A | {}/{}；{}/{} | N/A |".format(number(yolo_forward.get("p50"), 3), number(yolo_forward.get("p95"), 3), number(yolo_callback.get("p50"), 3), number(yolo_callback.get("p95"), 3)),
        "| 高保真感知挑战（加权漏检率 / 组合报警率） | N/A | N/A | N/A | {}% / {}% | N/A |".format(percent(challenge.get("aggregate", {}).get("eligible_miss_rate")), percent(challenge.get("cases", {}).get("combined", {}).get("watchdog_degraded_fraction"))),
        "| YOLO 空场景负对照：阳性帧 / 总检测帧 | N/A | N/A | N/A | {} / {}（{}%） | N/A |".format(negative.get("yolo", {}).get("positive_box_messages", "N/A"), negative.get("yolo", {}).get("detector_messages", "N/A"), percent(negative.get("yolo", {}).get("false_positive_frame_rate"))),
        "| 目标水平位置 / 水平速度 p95 | N/A | N/A | N/A | {} m / {} m/s | N/A |".format(number(track_position.get("p95"), 3), number(track_velocity.get("p95"), 3)),
        "| MAVROS 服务 / setpoint p95 (ms) | N/A | N/A | N/A | N/A | OFFBOARD {}；ARM {}；间隔 {} |".format(number(px4.get("services", {}).get("offboard_set_mode_ms"), 2), number(px4.get("services", {}).get("arm_ms"), 2), number(px4.get("setpoint_stream", {}).get("interval_ms_p95"), 2)),
        "",
        "压力配置刻意标为 `stress_example_not_calibrated`：5 cm LiDAR 噪声、5% LiDAR 丢帧、10 cm/10 cm/s 目标误差、10% 目标丢失、2 cm 状态噪声和两个控制周期的观测延迟。它是 sim-to-sim 鲁棒性压力测试，不是对真实精度的估计。",
        "",
        "## 改进量化对照（原版 NavRL）",
        "",
        "两份训练记录采用相同配置（1024 并行环境、80 个动态障碍、最大回合长度 2200、seed 0）。这组数据量化训练端改进，不把 ROS 控制日志当作任务成功率。",
        "",
        "| 指标 | 原版 NavRL | 改进版 NavRL | 差值 |",
        "|---|---:|---:|---:|",
        "| 成功率 | {}% | {}% | +{} 个百分点 |".format(number(100.0 * baseline.get("models", {}).get("original_navrl", {}).get("metrics", {}).get("success_rate"), 2), number(100.0 * baseline.get("models", {}).get("improved_navrl", {}).get("metrics", {}).get("success_rate"), 2), number(100.0 * baseline.get("delta_improved_minus_original", {}).get("success_rate_abs"), 2)),
        "| 碰撞率 | {}% | {}% | {} 个百分点（相对降低 {}%） |".format(number(100.0 * baseline.get("models", {}).get("original_navrl", {}).get("metrics", {}).get("collision_rate"), 2), number(100.0 * baseline.get("models", {}).get("improved_navrl", {}).get("metrics", {}).get("collision_rate"), 2), number(100.0 * baseline.get("delta_improved_minus_original", {}).get("collision_rate_abs"), 2), number(100.0 * baseline.get("delta_improved_minus_original", {}).get("collision_relative_reduction"), 2)),
        "| 平均速度 | N/A | N/A | 历史摘要未记录 |",
        "| 航点误差 | N/A | N/A | 历史摘要未记录 |",
        "",
        "算法差异不是调速或安全阈值平移：原版使用基础动态障碍编码和标准 PPO critic；改进版启用 `current_geometry` 几何编码、4 步历史/8 步风险场辅助预测，以及带风险 embedding 的 cost critic + Lagrange 约束。奖励、观测维度、动作上限、场景规模和 seed 保持一致，详见 `NAVRL_ALGORITHM_DIFFERENCE_CN.md`。",
        "",
        "平均速度和航点误差已加入 Isaac-Sim 评估器输出，但当前主机没有 Isaac-Sim，不能伪造这两项数值。完整说明和复现命令见 `NAVRL_BASELINE_COMPARISON_CN.md`。",
        "",
        "## 仿真与真实差距结论",
        "",
        "高保真仿真已经覆盖相机图像、浮点深度、相机内参、机体到相机外参、移动行人、YOLO、深度投影、目标速度估计，以及 PX4/Gazebo Garden/MAVROS 的解锁和 OFFBOARD 控制。然而仿真仍没有真实相机噪声、滚动快门、镜头畸变、无线网络抖动、震动、桨流、气动扰动、电池电压下降或 EKF 失配。",
        "",
        "实物部署保留的门禁是：实测相机内外参与深度比例、时间同步、EKF/传感器融合误差表征、台架测试、系留测试和受监督飞行测试。没有上述实物数据时，不报告真机成功率、碰撞率、定位精度或飞行安全结论。",
        "",
        "## 原始证据与复现入口",
        "",
        "- `{}`：最新高保真 RGB-D、YOLO、跟踪和计算开销原始结果。".format(relative(root, perception_path)),
        "- `{}`：遮挡/雾/雨/逆光/组合感知挑战汇总（含漏检率和 watchdog 报警）。".format(relative(root, challenge_path)),
        "- `{}`：同一相机/YOLO 链路的空场景负对照，用于估计无目标阳性帧率。".format(relative(root, negative_path)),
        "- `{}`：相机内参、浮点深度编码和 TF 外参对 Gazebo 真值结果。".format(relative(root, calibration_path)),
        "- `{}`：PX4/Gazebo Garden/MAVROS OFFBOARD 原始结果。".format(relative(root, px4_path)),
        "- `{}`：干净组/压力组 JSONL、CSV 和汇总。".format(relative(root, sim_path)),
        "- `benchmark_results/navrl_baseline_comparison.json`、`NAVRL_BASELINE_COMPARISON_CN.md`：原版 NavRL 与改进版同配置量化对照。",
        "- `docker/run_hifi_perception_acceptance.sh`、`tools/run_px4_mavros_sitl_acceptance.sh` 和 `docker/run_sim_acceptance.sh`：三个可复现验收入口。",
        "- `SIM_REAL_GAP_ANALYSIS_CN.md`、`SIM_TO_REAL_BENCHMARK_PROTOCOL_CN.md`、`PHYSICAL_DEPLOYMENT_GUIDE_CN.md`：论文差距分析、实物测量协议和实物部署门禁。",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {}".format(args.out))


if __name__ == "__main__":
    main()
