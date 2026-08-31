import datetime
import os
import traceback

import torch
import wandb
from omegaconf import DictConfig
from omni.isaac.kit import SimulationApp
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl import EpisodeStats, SyncDataCollector
from omni_drones.utils.torchrl.transforms import VelController
from ppo import PPO
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType

from utils import evaluate


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


def save_checkpoint(run, policy, checkpoint_name, tag):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_files_dir = run.dir
    run_root_dir = os.path.dirname(run_files_dir)
    save_dir = os.path.join(
        run_root_dir,
        f"{os.path.basename(run_root_dir)}-{tag}-saved-{timestamp}",
    )
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, checkpoint_name)
    torch.save(policy.state_dict(), ckpt_path)
    return ckpt_path


def run_ppo_family_training(cfg: DictConfig, label: str):
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    run = None
    collector = None
    transformed_env = None
    env = None
    last_ckpt_path = None
    training_succeeded = False

    try:
        wandb_kwargs = {
            "project": cfg.wandb.project,
            "name": f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            "entity": cfg.wandb.entity,
            "config": cfg,
            "mode": cfg.wandb.mode,
        }
        if cfg.wandb.run_id is not None:
            wandb_kwargs.update({"id": cfg.wandb.run_id, "resume": "must"})
        run = wandb.init(**wandb_kwargs)

        from env import NavigationEnv

        env = NavigationEnv(cfg)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed_env = TransformedEnv(
            env,
            Compose(VelController(controller, yaw_control=False), InitTracker()),
        ).train()
        transformed_env.set_seed(cfg.seed)

        policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
        if hasattr(policy, "reset_temporal_state"):
            policy.reset_temporal_state()

        if cfg.evaluation.checkpoint_path:
            checkpoint = torch.load(cfg.evaluation.checkpoint_path, map_location=cfg.device)
            policy.load_state_dict(checkpoint)
            last_ckpt_path = cfg.evaluation.checkpoint_path
            print(f"[{label}]: checkpoint loaded: ", cfg.evaluation.checkpoint_path)

        if cfg.evaluation.eval_only:
            print(f"[{label}]: start evaluation-only run.")
            info = {}
            for history_ablation_mode in cfg.evaluation.history_ablation_modes:
                info.update(
                    evaluate(
                        env=transformed_env,
                        policy=policy,
                        seed=cfg.seed,
                        cfg=cfg,
                        exploration_type=ExplorationType.MEAN,
                        history_ablation_mode=history_ablation_mode,
                    )
                )
            run.log(info)
            training_succeeded = True
            print(f"[{label}]: evaluation-only run done.")
            return

        episode_stats_keys = [
            k
            for k in transformed_env.observation_spec.keys(True, True)
            if isinstance(k, tuple) and k[0] == "stats"
        ]
        episode_stats = EpisodeStats(episode_stats_keys)

        collector = SyncDataCollector(
            transformed_env,
            policy=policy,
            frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,
            total_frames=int(cfg.max_frame_num),
            device=cfg.device,
            return_same_td=True,
            exploration_type=ExplorationType.RANDOM,
        )

        for i, data in enumerate(collector):
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            policy.set_training_progress(int(collector._frames))
            info.update(policy.train(data))

            episode_stats.add(data)
            if len(episode_stats) >= transformed_env.num_envs:
                stats = {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                    for k, v in episode_stats.pop().items(True, True)
                }
                info.update(stats)

            if i % cfg.eval_interval == 0:
                print(f"[{label}]: start evaluating policy at training step: ", i)
                if hasattr(policy, "reset_temporal_state"):
                    policy.reset_temporal_state()
                for history_ablation_mode in cfg.evaluation.history_ablation_modes:
                    info.update(
                        evaluate(
                            env=transformed_env,
                            policy=policy,
                            seed=cfg.seed,
                            cfg=cfg,
                            exploration_type=ExplorationType.MEAN,
                            history_ablation_mode=history_ablation_mode,
                        )
                    )
                print(f"\n[{label}]: evaluation done.")

            run.log(info)

            if i % cfg.save_interval == 0:
                last_ckpt_path = save_checkpoint(run, policy, f"checkpoint_{i}.pt", f"step-{i}")
                print(f"[{label}]: model saved at training step: ", i)
                print(f"[{label}]: checkpoint path: ", last_ckpt_path)

        last_ckpt_path = save_checkpoint(run, policy, "checkpoint_final.pt", "final")
        training_succeeded = True
        print(f"[{label}]: training succeeded.")
        print(f"[{label}]: final checkpoint path: ", last_ckpt_path)
    except Exception:
        print(f"[{label}]: training failed.")
        print(f"[{label}]: last checkpoint path: ", last_ckpt_path)
        traceback.print_exc()
        raise
    finally:
        if collector is not None:
            collector.shutdown()
        if transformed_env is not None:
            transformed_env.close()
        elif env is not None:
            env.close()
        if run is not None:
            wandb.finish()
        print(f"[{label}]: training success flag: ", training_succeeded)
        if last_ckpt_path is not None:
            print(f"[{label}]: checkpoint available at: ", last_ckpt_path)
        sim_app.close()
