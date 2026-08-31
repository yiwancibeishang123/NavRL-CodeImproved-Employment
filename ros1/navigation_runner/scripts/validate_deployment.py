#!/usr/bin/env python3
"""Static deployment validation; does not require a connected aircraft."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(Path(__file__).parent / "ckpts" / "step3-80new.pt"))
    parser.add_argument("--config", default=str(Path(__file__).parent / "cfg" / "ppo.yaml"))
    args = parser.parse_args()

    required = [args.checkpoint, args.config, Path(__file__).with_name("navigation.py"), Path(__file__).with_name("observation.py")]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing deployment files:\n" + "\n".join(missing))

    try:
        import torch
        import yaml
        import tensordict  # noqa: F401
        import torchrl  # noqa: F401
        from omegaconf import OmegaConf
        from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
    except Exception as exc:
        raise SystemExit("Python dependencies are incomplete: {}".format(exc))

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    algo = cfg.get("algo", {})
    obs = algo.get("observation_aux", {})
    if obs.get("obstacle_input_mode") != "current_geometry":
        raise SystemExit("Deployment config must use current_geometry")
    if int(algo.get("feature_extractor", {}).get("dyn_obs_num", 0)) <= 0:
        raise SystemExit("dyn_obs_num must be positive")
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise SystemExit("Checkpoint is not a non-empty state dict")
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from ppo import PPO
    algo_cfg = OmegaConf.create(cfg["algo"])
    hbeams = int(360 / float(cfg.get("sensor", {}).get("lidar_hres", 10.0)))
    vbeams = int(cfg.get("sensor", {}).get("lidar_vbeams", 4))
    device = torch.device("cpu")
    observation_spec = CompositeSpec({"agents": CompositeSpec({"observation": CompositeSpec({
        "state": UnboundedContinuousTensorSpec((8,), device=device),
        "lidar": UnboundedContinuousTensorSpec((1, hbeams, vbeams), device=device),
        "direction": UnboundedContinuousTensorSpec((1, 3), device=device),
        "dynamic_obstacle": UnboundedContinuousTensorSpec((1, int(algo_cfg.feature_extractor.dyn_obs_num), 10), device=device),
    })}).expand(1)}, shape=[1], device=device)
    action_spec = CompositeSpec({"agents": CompositeSpec({"action": UnboundedContinuousTensorSpec((3,), device=device)})}).expand(1, 3).to(device)
    policy = PPO(algo_cfg, observation_spec, action_spec, device)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit("Checkpoint architecture mismatch: missing={}, unexpected={}".format(list(missing)[:10], list(unexpected)[:10]))
    print("deployment validation passed")
    print("checkpoint:", os.path.abspath(args.checkpoint))
    print("parameters:", len(state))


if __name__ == "__main__":
    main()
