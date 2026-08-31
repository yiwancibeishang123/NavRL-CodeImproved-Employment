import hydra

from ppo_family_train import FILE_PATH, run_ppo_family_training


@hydra.main(config_path=FILE_PATH, config_name="train_lag_ppo", version_base=None)
def main(cfg):
    run_ppo_family_training(cfg, "PPO-Lagrangian")


if __name__ == "__main__":
    main()
