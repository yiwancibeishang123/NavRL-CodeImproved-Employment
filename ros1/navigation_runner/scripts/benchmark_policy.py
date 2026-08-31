#!/usr/bin/env python3
"""Profile the exact deployed PPO checkpoint with its deployment observation shape.

This is intentionally a model-only measurement.  It complements, but does not
replace, the JSONL metrics recorded by ``navigation.py`` during Gazebo or field
tests: the latter include ROS services, safety filtering and scheduling.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from deployment_metrics import runtime_metadata, runtime_snapshot, timed_call


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def describe(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(cfg, checkpoint, device, hbeams, vbeams):
    import torch
    from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from ppo import PPO

    dyn_obs_num = int(cfg.feature_extractor.dyn_obs_num)
    observation_spec = CompositeSpec({"agents": CompositeSpec({"observation": CompositeSpec({
        "state": UnboundedContinuousTensorSpec((8,), device=device),
        "lidar": UnboundedContinuousTensorSpec((1, hbeams, vbeams), device=device),
        "direction": UnboundedContinuousTensorSpec((1, 3), device=device),
        "dynamic_obstacle": UnboundedContinuousTensorSpec((1, dyn_obs_num, 10), device=device),
    })}).expand(1)}, shape=[1], device=device)
    action_spec = CompositeSpec({"agents": CompositeSpec({
        "action": UnboundedContinuousTensorSpec((3,), device=device),
    })}).expand(1, 3).to(device)
    policy = PPO(cfg, observation_spec, action_spec, device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture mismatch: missing={}, unexpected={}".format(
                list(missing)[:10], list(unexpected)[:10]
            )
        )
    policy.eval()
    return policy, dyn_obs_num


def build_observation(device, hbeams, vbeams, dyn_obs_num):
    import torch
    from tensordict import TensorDict

    # Keep a nonzero, finite observation so the policy takes its normal path.
    dynamic_obstacle = torch.zeros((1, 1, dyn_obs_num, 10), dtype=torch.float32, device=device)
    dynamic_obstacle[..., 0, 0:3] = torch.tensor([1.5, 0.2, 0.0], device=device)
    dynamic_obstacle[..., 0, 5:8] = torch.tensor([-0.2, 0.0, 0.0], device=device)
    dynamic_obstacle[..., 0, 8:10] = torch.tensor([0.5, 1.6], device=device)
    return TensorDict({"agents": TensorDict({"observation": TensorDict({
        "state": torch.zeros((1, 8), dtype=torch.float32, device=device),
        "lidar": torch.full((1, 1, hbeams, vbeams), 3.0, dtype=torch.float32, device=device),
        "direction": torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32, device=device),
        "dynamic_obstacle": dynamic_obstacle,
    })})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "ckpts" / "step3-80new.pt")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "cfg" / "ppo.yaml")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device such as cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--control-period-ms", type=float, default=50.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.samples <= 0 or args.warmup < 0 or args.control_period_ms <= 0.0:
        raise SystemExit("--samples must be positive; --warmup must be non-negative; control period must be positive")
    if not args.checkpoint.is_file() or not args.config.is_file():
        raise SystemExit("Checkpoint or config is missing")

    import torch
    import yaml
    from omegaconf import OmegaConf
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; pass --device cpu")
    cfg_data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = OmegaConf.create(cfg_data["algo"])
    hbeams = int(360 / float(cfg_data.get("sensor", {}).get("lidar_hres", 10.0)))
    vbeams = int(cfg_data.get("sensor", {}).get("lidar_vbeams", 4))
    policy, dyn_obs_num = load_policy(cfg, args.checkpoint, device, hbeams, vbeams)
    observation = build_observation(device, hbeams, vbeams, dyn_obs_num)

    def forward():
        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            return policy.act(observation) if hasattr(policy, "act") else policy(observation)

    for _ in range(args.warmup):
        timed_call(forward, torch, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_ms = []
    device_ms = []
    started_at = time.time()
    for _ in range(args.samples):
        result, wall, kernel = timed_call(forward, torch, device)
        action = result["agents", "action"]
        if not torch.isfinite(action).all():
            raise RuntimeError("Policy emitted a non-finite action during benchmark")
        wall_ms.append(wall)
        if kernel is not None:
            device_ms.append(kernel)

    wall_summary = describe(wall_ms)
    device_summary = describe(device_ms)
    deadline_misses = sum(value > args.control_period_ms for value in wall_ms)
    report = {
        "schema_version": 1,
        "kind": "policy_forward_benchmark",
        "started_at_unix_s": started_at,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "config": str(args.config.resolve()),
        "runtime": runtime_metadata(torch, device),
        "observation": {
            "state": 8,
            "lidar": [1, hbeams, vbeams],
            "direction": [1, 3],
            "dynamic_obstacle": [1, dyn_obs_num, 10],
            "action": 3,
        },
        "model_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "warmup_iterations": args.warmup,
        "sample_iterations": args.samples,
        "control_period_ms": args.control_period_ms,
        "inference_wall_ms": wall_summary,
        "inference_device_ms": device_summary,
        "serial_throughput_hz": None if not wall_summary["mean"] else 1000.0 / wall_summary["mean"],
        "deadline_miss_rate": deadline_misses / float(args.samples),
        "p95_budget_margin_ms": None if wall_summary["p95"] is None else args.control_period_ms - wall_summary["p95"],
        "runtime_after": runtime_snapshot(torch, device),
    }
    if device.type == "cuda":
        report["runtime_after"]["cuda_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 3
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote", args.out)
    print("p50/p95 wall latency (ms): {:.4f}/{:.4f}".format(wall_summary["p50"], wall_summary["p95"]))
    print("serial throughput (Hz): {:.2f}".format(report["serial_throughput_hz"]))


if __name__ == "__main__":
    main()
