import hydra
import json
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
import torchaudio
from tqdm import tqdm
from typing import Optional

from train_help.time_dataset import collate_time

from shared.core_utils import get_model, get_devset, get_dev_sessions, get_device
from shared.signal_utils import STFTWrapper, prep_audio, pad_tolength


def find_ckptpath(ckpt_dir: Path, ckpt_name: Optional[str], train_log: list[dict]):
    if ckpt_name is None or ckpt_dir == "last":
        options = sorted(ckpt_dir.glob("epoch*.pt"))
        ckpt = options[-1]
        logging.info(f"Loading last checkpoint from {str(ckpt)}")
    elif ckpt_name == "loss":
        best = min(train_log, key=lambda x: x["val_loss"])
        ckpt = ckpt_dir / f"epoch{best['epoch']:03d}.pt"
    elif ckpt_name == "stoi":
        best = max(train_log, key=lambda x: x["val_stoi"])
        ckpt = ckpt_dir / f"epoch{best['epoch']:03d}.pt"
    else:
        ckpt = ckpt_dir / ckpt_name

    if not ckpt.exists():
        raise ValueError(
            f"Checkpoint could not be found:\nSpecified {ckpt_name} in config but no checkpoint found at {str(ckpt)}"
        )

    return ckpt


def enhance_development(allexperiments_dir, exp_name, dataset, checkpoint):

    torch_device = get_device()
    audio_device = exp_name.split(".")[0]
    if audio_device not in ["aria", "ha"]:
        raise ValueError(
            f"Audio device could not be inferred from exp_name: {exp_name}"
        )

    # Setup diretory and config stuff
    allexperiments_dir = Path(allexperiments_dir)
    thisexperiment_dir = allexperiments_dir / exp_name

    exp_cfg = OmegaConf.load(thisexperiment_dir / "hydra/.hydra/config.yaml")

    model_cfg = exp_cfg.model
    data_cfg = exp_cfg.dataloading
    with open(thisexperiment_dir / "train_log.json", "r") as file:
        train_log = json.load(file)

    # Model
    ckpt = find_ckptpath(thisexperiment_dir / "checkpoints", checkpoint, train_log)
    model = get_model(model_cfg, ckpt).to(torch_device)

    # Data
    devset = get_devset(dataset, data_cfg, False)
    devsessions = get_dev_sessions(
        data_cfg.sessions_file, data_cfg.device, dataset, False
    )

    # Output
    output_dir = thisexperiment_dir / "enhancement" / ckpt.stem
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Convenient definitions
    input_channels = model_cfg.input.channels
    input_sr = model_cfg.input.sample_rate
    input_rms = model_cfg.input.rms
    use_aux = exp_cfg.iotypes.in_srcs > 0

    # STFT Stuff
    if model_cfg.input.type == "stft":
        do_stft = True
        stft = STFTWrapper(**model_cfg.input.stft, device=torch_device)
    elif model_cfg.input.type != "wave":
        logging.error(f"Unrecognised model input type {model_cfg.input.type}")
    else:
        do_stft = False

    if torch_device == "cpu":
        data_cfg.loader.dev.num_workers = 0

    with torch.no_grad():
        for info in devsessions:
            devset.make_manifest(info["session"])

            output_store = devset.get_val_output(info["session"], False)
            ref_store = devset.get_val_output(info["session"], False)
            noisy_store = torch.zeros(
                [
                    model_cfg.input.channels,
                    max(x.shape[-1] for x in output_store.values()),
                ]
            ).to(torch_device)

            for key in output_store:
                output_store[key] = output_store[key].to(torch_device)
                ref_store[key] = ref_store[key].to(torch_device)

            loader = DataLoader(devset, **data_cfg.loader.dev, collate_fn=collate_time)
            loader = tqdm(loader, info["session"])

            for batch in loader:
                noisy = batch["noisy"].to(
                    torch_device, non_blocking=True
                )  # [batch, channels, time]
                aux = batch["aux"].to(
                    torch_device, non_blocking=True
                )  # [batch, (spks), time]
                target = batch["target"].to(
                    torch_device, non_blocking=True
                )  # [batch, (spks), time]
                noisy = prep_audio(
                    noisy,
                    batch["fs"],
                    input_channels,
                    input_sr,
                    input_rms,
                    True,
                )
                target = prep_audio(
                    target,
                    batch["fs"],
                    target.shape[1],
                    input_sr,
                    input_rms,
                    True,
                )

                if do_stft:
                    noisy = stft(noisy)  # [batch, channels, time, freqs, 2]
                    if use_aux and model_cfg.input.rainbow_type == "audio":
                        aux = stft(aux)  # [batch, (spks), time, freqs, 2]
                        batch["aux_lens"] = (
                            batch["aux_lens"] - stft.n_fft
                        ) // stft.hop_length

                processed, spk_pred = model(noisy, aux, batch["aux_lens"])
                # processed = processed.squeeze(1)

                if do_stft:
                    processed = stft.inverse(processed)

                for i in range(noisy.shape[0]):
                    key = batch["key"][i]
                    start, end = batch["start"][i], batch["end"][i]
                    length = end - start

                    this_noisy = batch["noisy"][i].to(torch_device)
                    if this_noisy.shape[-1] < length:
                        this_noisy = pad_tolength(this_noisy, length)
                    noisy_store[:, start:end] += this_noisy

                    this_proc = processed[i]
                    if this_proc.shape[-1] < length:
                        this_proc = pad_tolength(this_proc, length)
                    output_store[key][:, start:end] += this_proc

                    this_ref = target[i]
                    if this_ref.shape[-1] < length:
                        this_ref = pad_tolength(this_ref, length)
                    ref_store[key][:, start:end] += this_ref

            if len(output_store) == 1:
                audio = list(output_store.values())[0]
                if audio.shape[0] > 1:
                    pids = info["targets"]
                    output_store = {
                        pid: chan.unsqueeze(0) for pid, chan in zip(pids, audio)
                    }

            sum_audio = None
            for name, audio in output_store.items():
                output_fpath = (
                    output_dir / f"{info['session']}.{audio_device}.{name}.wav"
                )
                torchaudio.save(
                    output_fpath, audio.detach().cpu(), model_cfg.input.sample_rate
                )
                if sum_audio is None:
                    sum_audio = audio
                else:
                    sum_audio += audio
            output_fpath = output_dir / f"{info['session']}.{audio_device}.sum.wav"
            torchaudio.save(
                output_fpath, sum_audio.detach().cpu(), model_cfg.input.sample_rate
            )


@hydra.main(
    version_base=None, config_path="pkg://config/enhancement", config_name="main"
)
def main(cfg: DictConfig):
    enhance_development(cfg.exp_dir, cfg.exp_name, cfg.dataset, cfg.checkpoint)


if __name__ == "__main__":

    main()
