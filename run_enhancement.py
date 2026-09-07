# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.speech_enhancement.enhance import enhance_development


@hydra.main(version_base=None, config_path="config/enhancement", config_name="main")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    enhance_development(cfg.exp_dir, cfg.exp_name, cfg.dataset, cfg.checkpoint)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
