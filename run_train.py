# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch

import hydra
from omegaconf import OmegaConf

from scripts.speech_enhancement.unpack import unpack
from scripts.speech_enhancement.train import train


@hydra.main(version_base=None, config_path="config/train", config_name="main")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.unpack.run:
        unpack(cfg.unpack)

    if cfg.train.run:
        train(
            cfg,
            cfg.dataloading,
            cfg.iotypes,
            cfg.model,
            cfg.train,
            cfg.train_dir,
            cfg.debug,
            cfg.wandb.entity,
            cfg.wandb.project,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    main()  # noqa pylint: disable=no-value-for-parameter
