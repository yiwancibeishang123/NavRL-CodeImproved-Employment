#!/usr/bin/env python3
"""Offline acceptance checks for the NavRL deployment package.

The script covers model/observation/perception/calibration/safety/dependency
contracts without commanding a vehicle.  PX4 transport is checked separately
by ``px4_mavros_mock_test.py`` because it requires a ROS master.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROS1_DIR = next(parent for parent in SCRIPT_DIR.parents if (parent / "onboard_detector").is_dir())
ROOT_DIR = ROS1_DIR.parent
DETECTOR_DIR = ROS1_DIR / "onboard_detector" / "scripts" / "yolo_detector"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_DIR))


def load_yaml(path):
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("{} must contain a YAML mapping".format(path))
    return data


def run_check(report, name, callback):
    started = time.perf_counter()
    try:
        details = callback() or {}
        report["checks"].append({
            "name": name,
            "result": "passed",
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "details": details,
        })
    except Exception as exc:
        report["checks"].append({
            "name": name,
            "result": "failed",
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": "{}: {}".format(type(exc).__name__, exc),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR / "ckpts" / "step3-80new.pt")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "cfg" / "ppo.yaml")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT_DIR,
        help="Delivery root containing documentation and docker/ (defaults to the source-tree parent).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROS1_DIR / "navigation_runner" / "cfg" / "detection" / "real" / "real_detector_px4.yaml",
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=ROS1_DIR / "navigation_runner" / "cfg" / "safety" / "safe_action_param.yaml",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "kind": "navrl_deployment_acceptance",
        "started_at_unix_s": time.time(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checks": [],
        "boundary": "Offline checks do not establish camera field calibration, PX4 flight behavior, or physical safety.",
    }

    def check_runtime():
        import torch
        import tensordict
        import torchrl

        if not torch.__version__.startswith("2.1."):
            raise RuntimeError("expected Torch 2.1.x, got {}".format(torch.__version__))
        return {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "tensordict": tensordict.__version__,
            "torchrl": torchrl.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }

    def check_policy_forward():
        import torch
        from benchmark_policy import build_observation, load_policy
        from deployment_metrics import timed_call
        from omegaconf import OmegaConf

        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        cfg_data = load_yaml(args.config)
        algo = OmegaConf.create(cfg_data["algo"])
        hbeams = int(360 / float(cfg_data.get("sensor", {}).get("lidar_hres", 10.0)))
        vbeams = int(cfg_data.get("sensor", {}).get("lidar_vbeams", 4))
        policy, dyn_obs_num = load_policy(algo, args.checkpoint, torch.device("cpu"), hbeams, vbeams)
        observation = build_observation(torch.device("cpu"), hbeams, vbeams, dyn_obs_num)

        def forward():
            with torch.no_grad():
                return policy.act(observation)

        output, wall_ms, _ = timed_call(forward, torch, torch.device("cpu"))
        action = output["agents", "action"]
        if action.numel() != 3 or not torch.isfinite(action).all():
            raise RuntimeError("policy action must be finite with three elements")
        return {
            "observation": {"state": 8, "lidar": [1, hbeams, vbeams], "dynamic_obstacle": [1, dyn_obs_num, 10]},
            "action": [round(float(value), 6) for value in action.reshape(-1)],
            "single_forward_wall_ms": round(float(wall_ms), 4),
            "parameters": sum(parameter.numel() for parameter in policy.parameters()),
        }

    def check_observation_and_perturbation():
        import torch
        from observation import (
            build_dynamic_obstacle_observation,
            build_lidar_observation,
            build_state_observation,
        )
        from sensor_perturbation import SensorPerturber

        position = torch.tensor([0.0, 0.0, 1.0])
        raypoints = [[3.0, 0.0, 1.0] for _ in range(36 * 4)]
        lidar = build_lidar_observation(raypoints, position, 5.0, 36, 4)
        state, direction = build_state_observation(
            position, torch.zeros(3), torch.tensor([4.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0])
        )
        obstacle = build_dynamic_obstacle_observation(
            position,
            direction.reshape(-1),
            torch.tensor([[1.0, 0.5, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.zeros(5, 3),
            torch.tensor([[0.5, 0.5, 1.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            0.3,
        )
        first = SensorPerturber(enabled=True, seed=17, lidar_noise_std=0.01, lidar_dropout_ratio=0.1, delay_steps=1)
        second = SensorPerturber(enabled=True, seed=17, lidar_noise_std=0.01, lidar_dropout_ratio=0.1, delay_steps=1)
        first_out = first.apply(state, lidar, obstacle)
        second_out = second.apply(state, lidar, obstacle)
        if not all(torch.isfinite(value).all() for value in (state, lidar, direction, obstacle)):
            raise RuntimeError("observation contains non-finite values")
        if not all(torch.equal(left, right) for left, right in zip(first_out[:3], second_out[:3])):
            raise RuntimeError("seeded sensor perturbation is not reproducible")
        if tuple(lidar.shape) != (1, 1, 36, 4) or tuple(obstacle.shape) != (1, 1, 5, 10):
            raise RuntimeError("deployment observation shape mismatch")
        return {"state": list(state.shape), "lidar": list(lidar.shape), "dynamic_obstacle": list(obstacle.shape)}

    def check_calibration():
        calibration_dir = ROS1_DIR / "onboard_detector" / "scripts"
        if str(calibration_dir) not in sys.path:
            sys.path.insert(0, str(calibration_dir))
        from check_calibration import load_and_validate

        return load_and_validate(args.calibration)

    def check_detector():
        output = args.out.parent / "yolo_detector_smoke.json"
        command = [sys.executable, str(DETECTOR_DIR / "detector_smoke_test.py"), "--out", str(output)]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return json.loads(output.read_text(encoding="utf-8"))

    def check_safety_config():
        safety = load_yaml(args.safety_config)
        for key in ("time_horizon", "time_step", "safety_distance"):
            if not math.isfinite(float(safety.get(key, 0.0))) or float(safety[key]) <= 0.0:
                raise ValueError("{} must be finite and positive".format(key))
        if float(safety["min_height"]) >= float(safety["max_height"]):
            raise ValueError("min_height must be below max_height")
        policy = load_yaml(args.config)
        action_limit = float(policy["algo"]["actor"]["action_limit"])
        if action_limit <= 0.0:
            raise ValueError("PPO action_limit must be positive")
        return {"safety_distance_m": float(safety["safety_distance"]), "action_limit_mps": action_limit}

    def check_documentation_and_launches():
        required = [
            args.project_root / "SIM_TO_REAL_BENCHMARK_PROTOCOL_CN.md",
            args.project_root / "PHYSICAL_DEPLOYMENT_GUIDE_CN.md",
            ROS1_DIR / "navigation_runner" / "launch" / "navrl_deployment.launch",
            ROS1_DIR / "navigation_runner" / "launch" / "sim_real_gap.launch",
            args.project_root / "docker" / "run_noetic_checks.sh",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        return {"required_files": len(required)}

    run_check(report, "runtime_dependency_contract", check_runtime)
    run_check(report, "ppo_checkpoint_and_forward", check_policy_forward)
    run_check(report, "observation_and_sensor_perturbation", check_observation_and_perturbation)
    run_check(report, "camera_depth_calibration_contract", check_calibration)
    run_check(report, "yolo_detector_forward", check_detector)
    run_check(report, "safety_parameter_contract", check_safety_config)
    run_check(report, "deployment_documentation_and_launches", check_documentation_and_launches)

    report["result"] = "passed" if all(check["result"] == "passed" for check in report["checks"]) else "failed"
    report["finished_at_unix_s"] = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["result"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
