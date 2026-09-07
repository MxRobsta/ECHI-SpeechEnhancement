"""
Compute statistical signficance
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import hydra
import logging
from omegaconf import DictConfig

from shared.core_utils import read_json

METRICS = ["stoi", "pesq", "fwsegsnr", "Csig", "Cbak", "Covl"]


def load_all_scores(exp_dir: Path, segtype: str, dataset: str):
    """Load the scores for this exp into one list"""

    all_scores = []
    for file in exp_dir.glob(f"metrics/{segtype}-json/{dataset}*sum*.json"):
        all_scores += read_json(file)

    for x in all_scores:
        x["wav"] = Path(x["wav"]).stem + "." + x["pid"]

    return all_scores


@hydra.main(version_base=None, config_path="../../config/analysis", config_name="stats")
def main(cfg: DictConfig):
    """
    Loads in the scores for all the experiments and computes p-values using
    ANOVA and pairwise t-tests
    """

    devices = cfg.devices
    if isinstance(devices, str):
        devices = [devices]

    if cfg.exps == "main":
        logging.info("Stats for the main experiments")
        exp_titles = cfg.mainexps
    else:
        logging.info("Stats for the vadmask experiments")
        exp_titles = cfg.vadexps

    all_exps = []
    for dev in devices:
        all_exps += [Path(cfg.allexps_dir) / f"{dev}.{e}" for e in exp_titles]

    exp_scores = {}
    exp_names = []
    all_keys = set()
    for exp_dir in all_exps:

        scores = load_all_scores(exp_dir, cfg.segtype, cfg.dataset)
        exp_name = exp_dir.name
        exp_scores[exp_name] = pd.DataFrame(scores).fillna(1)

        if len(scores) == 0:
            continue

        exp_names.append(exp_name)

        these_keys = [x["wav"] for x in scores]
        if len(all_keys) == 0:
            all_keys = set(these_keys)
        else:
            all_keys = all_keys.intersection(these_keys)
    ida, idb = -2, -1
    print(exp_names[ida])
    print(exp_names[idb])
    for met in METRICS:
        thing = {}
        for exp_name, df in exp_scores.items():
            df = df[["wav", met]]
            df = df[df["wav"].isin(all_keys)]  # type: pd.DataFrame
            df = df.sort_values("wav")
            thing[exp_name] = np.array(df[met])
        thing = list(thing.values())

        score = stats.ttest_ind(thing[ida], thing[idb])
        print(met, score)


if __name__ == "__main__":
    main()
