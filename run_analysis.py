import hydra
from omegaconf import DictConfig

from analysis.ownvoice import ownvoice_scores
from analysis.rainbow_similarity import rainbow_similarity
from analysis.metric_time import metric_time


@hydra.main(version_base=None, config_path="config/analysis", config_name="main")
def main(cfg: DictConfig):

    if cfg.ownvoice.run:
        ownvoice_scores(cfg.ownvoice.metrics)

    if cfg.rainbow_similarity.run:
        rainbow_cfg = cfg.rainbow_similarity
        rainbow_similarity(
            rainbow_cfg.dataset,
            rainbow_cfg.embedding_type,
            rainbow_cfg.rainbow_path,
            rainbow_cfg.ref_file,
            rainbow_cfg.ref_embedding_file,
            rainbow_cfg.sessions_file,
            rainbow_cfg.segments_file,
        )

    if cfg.metric_time.run:
        metric_time(
            cfg.metric_time.sessions_file,
            cfg.metric_time.segment_file,
            cfg.metric_time.device,
            cfg.metric_time.ref_file,
            cfg.metric_time.exp_dir,
            cfg.metric_time.model_file,
            cfg.metric_time.metric_file,
        )


if __name__ == "__main__":
    main()
