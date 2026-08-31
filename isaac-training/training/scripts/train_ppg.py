import hydra

from ppo_family_train import FILE_PATH, run_ppo_family_training


@hydra.main(config_path=FILE_PATH, config_name="train_p3o", version_base=None)
def main(cfg):
    run_ppo_family_training(cfg, "P3O")


if __name__ == "__main__":
    main()
