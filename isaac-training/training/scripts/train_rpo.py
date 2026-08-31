import hydra

from ppo_family_train import FILE_PATH, run_ppo_family_training


@hydra.main(config_path=FILE_PATH, config_name="train_rpo", version_base=None)
def main(cfg):
    run_ppo_family_training(cfg, "RPO")


if __name__ == "__main__":
    main()
