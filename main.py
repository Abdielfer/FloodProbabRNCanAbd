# To test Hydra
import hydra
import os
from omegaconf import DictConfig

@hydra.main(config_path="./config", config_name="config")
def func(cfg: DictConfig):
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")

    # To access elements of the config
    print(f"The batch size is {cfg.batch_size}")
    print(f"The learning rate is {cfg['lr']}")

if __name__ == "__main__":
    func()