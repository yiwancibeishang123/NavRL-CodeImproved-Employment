import argparse
import inspect
import json
import os
import sys
import traceback
from statistics import mean, pstdev

import torch
from hydra import compose, initialize_config_dir
from omni.isaac.kit import SimulationApp
from torchrl.envs.transforms import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType


def _prepend_repo_scripts(repo_root: str):
    scripts_dir = os.path.join(repo_root, "isaac-training", "training", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return scripts_dir


def _load_cfg(repo_root: str, overrides: list[str]):
    cfg_dir = os.path.join(repo_root, "isaac-training", "training", "cfg")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    return cfg


def _summarize(records: list[dict], metric: str):
    values = [float(r[metric]) for r in records if r.get(metric) is not None]
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _upgrade_legacy_feature_keys(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map older checkpoint names to the current wrapped feature modules.

    Older NavRL checkpoints saved the LiDAR and dynamic-obstacle encoders as
    plain Sequential modules. The current code wraps those modules in named
    containers (`backbone` / `encoder`) without changing the tensor shapes.
    This conversion is evaluation-only compatibility glue.
    """
    upgraded = {}
    for key, value in checkpoint.items():
        new_key = key
        if key.startswith("feature_extractor.module.0.module."):
            suffix = key.removeprefix("feature_extractor.module.0.module.")
            if suffix and suffix[0].isdigit():
                new_key = f"feature_extractor.module.0.module.backbone.{suffix}"
        elif key.startswith("feature_extractor.module.1.module."):
            suffix = key.removeprefix("feature_extractor.module.1.module.")
            if suffix.startswith("1."):
                new_key = f"feature_extractor.module.1.module.encoder.{suffix}"
        upgraded[new_key] = value
    return upgraded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    _prepend_repo_scripts(args.repo_root)

    overrides = [
        "headless=true",
        "wandb.mode=disabled",
    ]
    overrides.extend(args.override)
    cfg = _load_cfg(args.repo_root, overrides)

    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    env = None
    transformed_env = None
    try:
        from env import NavigationEnv
        from ppo import PPO
        from utils import evaluate
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController

        env = NavigationEnv(cfg)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        vel_transform = VelController(controller, yaw_control=False)
        transformed_env = TransformedEnv(env, Compose(vel_transform)).train()
        transformed_env.set_seed(cfg.seed)

        checkpoint = torch.load(args.checkpoint, map_location=cfg.device)
        use_decoder_predictor = any(
            k.startswith("risk_predictor.project.") or k.startswith("risk_predictor.decoder.")
            for k in checkpoint.keys()
        )
        obs_aux = None
        if "observation_aux" in cfg.algo:
            obs_aux = cfg.algo.observation_aux
        elif "observation_aux" in cfg:
            obs_aux = cfg.observation_aux
        if obs_aux is not None and "risk_predictor_decoder_enabled" in obs_aux:
            obs_aux.risk_predictor_decoder_enabled = use_decoder_predictor
            predictor_name = "decoder" if use_decoder_predictor else "mlp"
            print(f"[EvalWorker] detected {predictor_name}-style RiskFieldPredictor checkpoint", flush=True)

        policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
        try:
            policy.load_state_dict(checkpoint)
        except RuntimeError as exc:
            upgraded_checkpoint = _upgrade_legacy_feature_keys(checkpoint)
            if upgraded_checkpoint == checkpoint:
                raise
            print("[EvalWorker] retrying checkpoint load with legacy feature-key mapping", flush=True)
            try:
                policy.load_state_dict(upgraded_checkpoint)
            except RuntimeError:
                raise exc

        results = []
        evaluate_signature = inspect.signature(evaluate)
        supports_history_ablation = "history_ablation_mode" in evaluate_signature.parameters
        for seed in range(args.seed_start, args.seed_start + args.num_runs):
            print(f"[EvalWorker] {args.label}: seed {seed}", flush=True)
            eval_kwargs = dict(
                env=transformed_env,
                policy=policy,
                seed=seed,
                cfg=cfg,
                exploration_type=ExplorationType.MEAN,
            )
            if supports_history_ablation:
                eval_kwargs["history_ablation_mode"] = "none"
            eval_info = evaluate(**eval_kwargs)
            def metric(name):
                value = eval_info.get("eval/stats." + name)
                return None if value is None else float(value)

            result = {
                "seed": seed,
                "return": metric("return"),
                "reach_goal": metric("reach_goal"),
                "collision": metric("collision"),
                "episode_len": metric("episode_len"),
                "truncated": metric("truncated"),
                "average_speed_mps": metric("average_speed_mps"),
                "waypoint_error_m": metric("waypoint_error_m"),
                "episode_time_s": metric("episode_time_s"),
                "path_length_m": metric("path_length_m"),
                "path_efficiency": metric("path_efficiency"),
                "initial_goal_distance_m": metric("initial_goal_distance_m"),
                "success_episode_time_s": metric("success_episode_time_s"),
                "success_path_length_m": metric("success_path_length_m"),
                "success_average_speed_mps": metric("success_average_speed_mps"),
                "success_waypoint_error_m": metric("success_waypoint_error_m"),
            }
            results.append(result)

        payload = {
            "label": args.label,
            "repo_root": args.repo_root,
            "checkpoint": args.checkpoint,
            "num_runs": args.num_runs,
            "seed_start": args.seed_start,
            "overrides": args.override,
            "per_run": results,
            "summary": {
                "return": _summarize(results, "return"),
                "reach_goal": _summarize(results, "reach_goal"),
                "collision": _summarize(results, "collision"),
                "episode_len": _summarize(results, "episode_len"),
                "truncated": _summarize(results, "truncated"),
                "average_speed_mps": _summarize(results, "average_speed_mps"),
                "waypoint_error_m": _summarize(results, "waypoint_error_m"),
                "episode_time_s": _summarize(results, "episode_time_s"),
                "path_length_m": _summarize(results, "path_length_m"),
                "path_efficiency": _summarize(results, "path_efficiency"),
                "initial_goal_distance_m": _summarize(results, "initial_goal_distance_m"),
                "success_episode_time_s": _summarize(results, "success_episode_time_s"),
                "success_path_length_m": _summarize(results, "success_path_length_m"),
                "success_average_speed_mps": _summarize(results, "success_average_speed_mps"),
                "success_waypoint_error_m": _summarize(results, "success_waypoint_error_m"),
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if transformed_env is not None:
            transformed_env.close()
        elif env is not None:
            env.close()
        sim_app.close()


if __name__ == "__main__":
    main()
