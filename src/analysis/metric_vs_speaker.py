import torch
import json
from pathlib import Path
import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt


def metric_vs_speaker(
    dataset: str,
    embedding_type: str,
    rainbow_template: str,
    metric_dir: str,
    target_metric: str,
    spkemb_dist: str,
):
    metric_files = Path(metric_dir).glob("*sum*.json")

    metrics_names = ["stoi", "pesq", "fwsegsnr", "Csig", "Cbak", "Covl"]

    cossim = torch.nn.CosineSimilarity()
    mse = torch.nn.MSELoss()

    rainbow_template = rainbow_template.replace("wav", "pt")

    all_scores = {"session": [], "pid": [], "cossim": [], "mse": []}
    for m in metrics_names:
        all_scores[m] = []

    for mfile in metric_files:
        with open(mfile, "r") as file:
            all_metrics = json.load(file)

        all_pids = list(set([x["pid"] for x in all_metrics]))

        for pid in all_pids:
            session = mfile.stem.split(".")[0]
            dataset = session.split("_")[0]

            metrics = [x for x in all_metrics if x["pid"] == pid]
            rainbow = torch.load(
                rainbow_template.format(
                    enrolsource="rainbow",
                    enroltype=embedding_type,
                    dataset=dataset,
                    pid=pid,
                )
            ).unsqueeze(0)
            ref = torch.load(
                rainbow_template.format(
                    enrolsource="dataocean",
                    enroltype=embedding_type,
                    dataset=dataset,
                    pid=pid,
                )
            ).unsqueeze(0)

            rainbow = torch.nn.functional.normalize(rainbow, p=2, dim=1)
            ref = torch.nn.functional.normalize(ref, p=2, dim=1)

            this_cossim = cossim(rainbow, ref).item()
            this_mse = mse(rainbow, ref).item()

            all_scores["session"].append(session)
            all_scores["pid"].append(pid)
            all_scores["cossim"].append(this_cossim)
            all_scores["mse"].append(this_mse)
            for met in metrics_names:
                all_scores[met].append(
                    np.mean([x[met] for x in metrics if not np.isnan(x[met])])
                )

    df = pd.DataFrame(all_scores)
    df.to_csv("scratch/cossims.csv")

    plt.scatter(all_scores["cossim"], all_scores[target_metric])
    plt.xlabel(f"{spkemb_dist} for rainbow vs ref")
    plt.ylabel(target_metric)

    plt.show()


@hydra.main(version_base=None, config_path="../../config/analysis", config_name="main")
def main(cfg: DictConfig):
    met_cfg = cfg.metric_vs_speaker
    metric_vs_speaker(
        met_cfg.dataset,
        met_cfg.embedding_type,
        met_cfg.rainbow_path,
        met_cfg.metric_dir,
        met_cfg.target_metric,
        met_cfg.spkemd_dist,
    )


if __name__ == "__main__":
    main()
