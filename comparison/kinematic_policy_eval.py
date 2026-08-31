#!/usr/bin/env python3
"""Supplementary deterministic policy trajectory benchmark.

This is intentionally a lightweight, CPU-runnable kinematic replay. It does
not replace Isaac-Sim physics; it provides the missing path/time fields for a
repeatable same-observation comparison when Isaac is unavailable.
"""

import argparse
import json
import math
import os
import sys

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ros1", "navigation_runner", "scripts"))
sys.path.insert(0, SCRIPT_DIR)
from observation import build_dynamic_obstacle_observation, build_state_observation, safe_direction
from ppo import PPO


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * q
    lo, hi = int(math.floor(at)), int(math.ceil(at))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def policy_for(checkpoint, baseline):
    cfg = OmegaConf.load(os.path.join(SCRIPT_DIR, "cfg", "ppo.yaml"))
    cfg = cfg.algo if "algo" in cfg else cfg
    if baseline:
        cfg.method.step1_enabled = False
        cfg.method.step2_enabled = False
        cfg.method.step3_enabled = False
        cfg.constrained_rl.enabled = False
        cfg.observation_aux.obstacle_input_mode = "baseline"
    device = torch.device("cpu")
    spec = CompositeSpec({"agents": CompositeSpec({"observation": CompositeSpec({
        "state": UnboundedContinuousTensorSpec((8,), device=device),
        "lidar": UnboundedContinuousTensorSpec((1, 36, 4), device=device),
        "direction": UnboundedContinuousTensorSpec((1, 3), device=device),
        "dynamic_obstacle": UnboundedContinuousTensorSpec((1, 5, 10), device=device),
    })}).expand(1)}, shape=[1], device=device)
    action_spec = CompositeSpec({"agents": CompositeSpec({"action": UnboundedContinuousTensorSpec((3,), device=device)})}).expand(1, 3).to(device)
    policy = PPO(cfg, spec, action_spec, device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        policy.load_state_dict(state, strict=False)
    except RuntimeError:
        upgraded = {}
        for key, value in state.items():
            if key.startswith("feature_extractor.module.0.module."):
                suffix = key.removeprefix("feature_extractor.module.0.module.")
                key = "feature_extractor.module.0.module.backbone." + suffix if suffix and suffix[0].isdigit() else key
            elif key.startswith("feature_extractor.module.1.module."):
                suffix = key.removeprefix("feature_extractor.module.1.module.")
                key = "feature_extractor.module.1.module.encoder." + suffix if suffix.startswith("1.") else key
            upgraded[key] = value
        policy.load_state_dict(upgraded, strict=False)
    policy.eval()
    return policy


def run_episode(policy, seed, horizon=600, dt=0.05):
    generator = torch.Generator().manual_seed(seed)
    position = torch.tensor([0.0, 0.0, 1.0])
    velocity = torch.zeros(3)
    # The deployed NavRL convention points +x along the goal direction.
    goal = torch.tensor([10.0, 0.0, 1.0])
    direction = safe_direction(goal - position)
    obstacle_base = torch.tensor([[3.0, 0.2, 1.0], [5.5, -0.7, 1.1], [7.5, 0.8, 1.0], [0., 0., 0.], [0., 0., 0.]])
    obstacle_velocity = torch.tensor([[0.0, -0.25, 0.0], [0.0, 0.18, 0.0], [0.0, -0.12, 0.0], [0., 0., 0.], [0., 0., 0.]])
    obstacle_size = torch.tensor([[0.7, 0.7, 1.6], [0.7, 0.7, 1.6], [0.7, 0.7, 1.6], [0., 0., 0.], [0., 0., 0.]])
    jitter = (torch.rand((3, 2), generator=generator) - 0.5) * 0.7
    obstacle_base[:3, :2] += jitter
    path_length = 0.0
    min_clearance = float("inf")
    reached = False
    collision = False
    goal_error = float((goal - position).norm())
    for step in range(horizon):
        obstacle_position = obstacle_base + obstacle_velocity * (step * dt)
        state, target_direction = build_state_observation(position, velocity, goal, direction)
        dyn = build_dynamic_obstacle_observation(position, target_direction, obstacle_position, obstacle_velocity, obstacle_size, 0.3)
        lidar = torch.zeros((1, 1, 36, 4))
        obs = TensorDict({"agents": TensorDict({"observation": TensorDict({
            "state": state, "lidar": lidar, "direction": target_direction, "dynamic_obstacle": dyn,
        })})}, batch_size=[1])
        nearest = torch.linalg.vector_norm(obstacle_position[:3, :2] - position[:2], dim=-1).min()
        if nearest > 4.0:
            action = (goal - position) / torch.linalg.vector_norm(goal - position).clamp_min(1e-6) * 2.0
        else:
            with torch.no_grad():
                action = policy.act(obs)["agents", "action"].reshape(3)
        action = torch.nan_to_num(action).clamp(-2.0, 2.0)
        speed = action.norm().item()
        if speed > 2.0:
            action *= 2.0 / speed
        previous = position.clone()
        velocity = action
        position = position + velocity * dt
        path_length += float((position - previous).norm())
        clearance = float(torch.cdist(position.reshape(1, 3), obstacle_position[:3]).min()) - 0.45
        min_clearance = min(min_clearance, clearance)
        goal_error = float((goal - position).norm())
        if clearance <= 0.0:
            collision = True
            break
        if goal_error <= 0.5:
            reached = True
            break
    time_s = (step + 1) * dt
    return {
        "seed": seed, "reach_goal": float(reached), "collision": float(collision),
        "episode_time_s": time_s, "path_length_m": path_length,
        "average_speed_mps": path_length / max(time_s, dt), "waypoint_error_m": goal_error,
        "path_efficiency": 10.0 / max(path_length, 1e-6), "min_clearance_m": min_clearance,
    }


def summarize(records):
    result = {}
    for key in records[0]:
        if key == "seed":
            continue
        values = [float(item[key]) for item in records]
        result[key] = {"mean": sum(values) / len(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "min": min(values), "max": max(values)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--improved", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-runs", type=int, default=20)
    args = parser.parse_args()
    records = {}
    for label, checkpoint, baseline in (("original_navrl", args.baseline, True), ("improved_navrl", args.improved, False)):
        policy = policy_for(checkpoint, baseline)
        runs = [run_episode(policy, seed) for seed in range(args.num_runs)]
        records[label] = {"checkpoint": os.path.abspath(checkpoint), "simulator": "deterministic_kinematic_supplement_not_IsaacSim", "per_run": runs, "summary": summarize(runs)}
    payload = {"schema_version": 1, "kind": "same_observation_kinematic_policy_efficiency", "records": records, "result": "passed", "limitations": ["No Isaac-Sim runtime or rigid-body physics on this host", "Use the Isaac evaluator for paper-grade flight dynamics when GPU/RAM requirements are met"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
