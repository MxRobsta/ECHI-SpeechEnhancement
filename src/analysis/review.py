"""
Review the results from all the experiments
"""

import json
import numpy as np
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None, config_path="../../config/analysis", config_name="review"
)
def main(cfg: DictConfig):

    root = Path(cfg.allexps_dir)

    metrics = ["stoi", "pesq", "fwsegsnr", "Csig", "Cbak", "Covl", "Nsamp"]

    devices = cfg.devices
    if isinstance(devices, str):
        devices = [devices]

    exps = []
    for dev in devices:
        exps += list(root.glob(f"{dev}*"))
    exps = sorted(exps, key=lambda x: ".".join(x.stem.split(".")[1:]))

    system_scores = {}
    for entry in exps:

        if cfg.evaltype == "sum":
            matchstr = f"metrics/{cfg.segtype}-json/{cfg.dataset}*sum*"
        elif cfg.evaltype == "individual":
            matchstr = f"metrics/{cfg.segtype}-json/{cfg.dataset}*P*"
        else:
            raise ValueError(f"No evaltype={cfg.evaltype}")

        met_files = list(entry.glob(matchstr))
        if len(met_files) == 0:
            continue
        scores = []
        for file in met_files:
            with open(file, "r") as f:
                scores += json.load(f)

        system_scores[entry.name] = []
        for met in metrics[:-1]:
            system_scores[entry.name].append(
                np.mean([x[met] for x in scores if not np.isnan(x[met])])
            )
        system_scores[entry.name].append(
            len([x[met] for x in scores if not np.isnan(x[met])])
        )

    metstring = "{:>10}"
    fstring = "{:70}" + metstring * len(metrics)
    print(fstring.format("system", *metrics))
    fstring = fstring.replace(metstring, metstring[:-1] + ".2f" + "}")

    for sys, sc in system_scores.items():
        print(fstring.format(sys, *sc))


if __name__ == "__main__":
    main()
