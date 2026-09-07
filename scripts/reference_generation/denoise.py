import csv
import hydra
from omegaconf import DictConfig
from pathlib import Path
import torchaudio
from tqdm import tqdm

from refgen.denoisers import load_denoiser


@hydra.main(version_base=None, config_path="pkg://config/refgen", config_name="main")
def main(cfg: DictConfig):
    """Denoise all the close talk mics for the ECHI dataset"""

    datasets = cfg.datasets
    if isinstance(datasets, str):
        datasets = [datasets]

    # Load sessions file
    session_infos = []
    sessions_file = cfg.sessions_file
    for ds in datasets:
        with open(sessions_file.format(dataset=ds), "r") as file:
            session_infos += list(csv.DictReader(file))

    # Load the denoiser
    den_name = cfg.ref_source.split(".")[0]
    denoiser = load_denoiser(den_name)

    # Get paths
    input_ftemp = cfg.chime_ct
    output_ftemp = cfg.ct_denoised

    # Run for all sessions
    for sinfo in tqdm(session_infos):
        session = sinfo["session"]
        datasets = session.split("_")[0]

        for i in range(1, 5):
            input_file = input_ftemp.format(session=session, dataset=datasets, i=i)
            audio, fs = denoiser.denoise(input_file)

            output_file = Path(
                output_ftemp.format(
                    session=session,
                    dataset=datasets,
                    dentype=den_name,
                    device="ct",
                    pid=f"pos{i}",
                )
            )

            # Save audio
            if not output_file.parent.exists():
                output_file.parent.mkdir(parents=True)

            torchaudio.save(output_file, audio.detach().cpu(), fs)


if __name__ == "__main__":
    main()
