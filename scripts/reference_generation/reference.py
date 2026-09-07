import hydra
import json
import logging
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

from refgen.room_sim import simulate_reference
from shared.core_utils import get_session_dicts


@hydra.main(version_base=None, config_path="pkg://config/refgen", config_name="main")
def main(cfg: DictConfig):
    """Denoise all the close talk mics for the ECHI dataset"""

    datasets = cfg.datasets
    if isinstance(datasets, str):
        datasets = [datasets]

    # Load sessions file
    session_dicts = get_session_dicts(
        cfg.sessions_file, cfg.devices, cfg.datasets, True
    )

    # Get paths
    noisy_ftemp = cfg.chime_device
    ref_source = cfg.ref_source
    den_name = ref_source.split(".")[0]
    denoised_ftemp = cfg.ct_denoised
    ref_ftemp = cfg.ref_session
    tracker_temp = cfg.chime_tracker

    # Load session alignment info
    with open(cfg.session_alignment, "r") as file:
        session_alignment = {x["session"]: x for x in json.load(file)}

    # Split for multirun
    if cfg.n_ranges > 1:
        first = int(len(session_dicts) * cfg.range_id / cfg.n_ranges)
        if cfg.range_id == cfg.n_ranges - 1:
            last = len(session_dicts)
        else:
            last = int(len(session_dicts) * (cfg.range_id + 1) / cfg.n_ranges)

        session_dicts = session_dicts[first:last]
    logging.debug(f"Simulating references for {len(session_dicts)} sessions")
    logging.debug(f" - Sessions: {json.dumps(session_dicts, indent=4)}")

    # Run for all sessions
    for sinfo in tqdm(session_dicts):

        noisy_fpath = Path(
            noisy_ftemp.format(
                session=sinfo["session"],
                dataset=sinfo["dataset"],
                device=sinfo["device"],
            )
        )
        denoised_fpath = Path(
            denoised_ftemp.format(
                session=sinfo["session"],
                dataset=sinfo["dataset"],
                dentype=den_name,
                device="ct",
                pid=sinfo["target_pos"],
            )
        )
        ref_fpath = Path(
            ref_ftemp.format(
                source=ref_source,
                dataset=sinfo["dataset"],
                session=sinfo["session"],
                device=sinfo["device"],
                pid=sinfo["target_pos"],
            )
        )
        refpid_fpath = Path(
            ref_ftemp.format(
                source=ref_source,
                dataset=sinfo["dataset"],
                session=sinfo["session"],
                device=sinfo["device"],
                pid=sinfo["target_pid"],
            )
        )
        wearer_tracker_fpath = Path(
            tracker_temp.format(
                dataset=sinfo["dataset"],
                session=sinfo["session"],
                pos=sinfo["wearer_pos"],
            )
        )
        target_tracker_fpath = Path(
            tracker_temp.format(
                dataset=sinfo["dataset"],
                session=sinfo["session"],
                pos=sinfo["target_pos"],
            )
        )

        if not denoised_fpath.exists():
            continue
        elif ref_fpath.exists() and not cfg.overwrite:
            logging.info(
                f"Reference exists and will not be overwritten for session {sinfo}"
            )
            continue

        if not ref_fpath.parent.exists():
            ref_fpath.parent.mkdir(parents=True)

        if "clock_drift" not in session_alignment[sinfo["session"]]:
            if sinfo["device"] == "aria":
                continue
            session_alignment[sinfo["session"]]["clock_drift"] = []

        simulate_reference(
            noisy_fpath,
            denoised_fpath,
            target_tracker_fpath,
            wearer_tracker_fpath,
            cfg.head_locations,
            ref_fpath,
            refpid_fpath,
            cfg.sample_rate,
            cfg.ref_source.split(".")[1],
            cfg.collar,
            session_alignment[sinfo["session"]]["clock_drift"],
            sinfo,
        )


if __name__ == "__main__":
    main()
