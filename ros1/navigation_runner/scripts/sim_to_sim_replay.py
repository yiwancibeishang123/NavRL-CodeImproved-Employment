#!/usr/bin/env python3
"""Replay a JSON observation trace through the deployment model.

The trace format is intentionally hardware-independent and can be generated
from Isaac/Gazebo or a rosbag converter later:
{state, lidar, direction, dynamic_obstacle}.
"""
import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type

from ppo import PPO


def load_policy(cfg, checkpoint, device, hbeams, vbeams):
    dyn_obs_num = int(cfg.feature_extractor.dyn_obs_num)
    obs_spec = CompositeSpec({"agents": CompositeSpec({"observation": CompositeSpec({
        "state": UnboundedContinuousTensorSpec((8,), device=device),
        "lidar": UnboundedContinuousTensorSpec((1, hbeams, vbeams), device=device),
        "direction": UnboundedContinuousTensorSpec((1, 3), device=device),
        "dynamic_obstacle": UnboundedContinuousTensorSpec((1, dyn_obs_num, 10), device=device),
    })}).expand(1)}, shape=[1], device=device)
    action_spec = CompositeSpec({"agents": CompositeSpec({"action": UnboundedContinuousTensorSpec((3,), device=device)})}).expand(1, 3).to(device)
    policy = PPO(cfg, obs_spec, action_spec, device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    policy.load_state_dict(state)
    policy.eval()
    return policy, dyn_obs_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hbeams", type=int, default=36)
    parser.add_argument("--vbeams", type=int, default=4)
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(__file__).with_name("cfg").joinpath("ppo.yaml").read_text())
    from omegaconf import OmegaConf
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; pass --device cpu")
    policy, dyn_obs_num = load_policy(OmegaConf.create(cfg["algo"]), args.checkpoint, device, args.hbeams, args.vbeams)
    trace_text = args.trace.read_text(encoding="utf-8")
    if args.trace.suffix.lower() == ".jsonl":
        records = [
            record for record in (json.loads(line) for line in trace_text.splitlines() if line.strip())
            if record.get("event") == "observation"
        ]
    else:
        records = json.loads(trace_text)
    if not records:
        raise SystemExit("No observation records found in {}".format(args.trace))
    outputs = []
    with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
        for item in records:
            obs = TensorDict({"agents": TensorDict({"observation": TensorDict({
                "state": torch.tensor(item["state"], dtype=torch.float32, device=device).reshape(1, 8),
                "lidar": torch.tensor(item["lidar"], dtype=torch.float32, device=device).reshape(1, 1, args.hbeams, args.vbeams),
                "direction": torch.tensor(item["direction"], dtype=torch.float32, device=device).reshape(1, 3),
                "dynamic_obstacle": torch.tensor(item["dynamic_obstacle"], dtype=torch.float32, device=device).reshape(1, 1, dyn_obs_num, 10),
            })})})
            result = policy.act(obs)["agents", "action"].reshape(-1).tolist()
            outputs.append(result)
    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
