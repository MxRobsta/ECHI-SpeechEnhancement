import csv
import itertools
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, Iterable
import torch
from torch.utils.data.dataloader import DataLoader

from models.CausalMCxTFGridNet import MCxTFGridNet
from models.PreSpkMCxTFGridnet import PreSpkMCxTFGridNet
from models.DenTFGridNet import DenTFGridNet
from models.DenMSX import DenMSX
from models.passthrough import Passthrough
from models.SwishTFGridNet import SwishTFGridNet

from shared.file_utils import read_json

from train_help.speech_dataset import ECHISpeech, collate_speech
from train_help.time_dataset import ECHITime, collate_time


def get_model(
    cfg: DictConfig, ckpt_path: Optional[Path | str] = None
) -> torch.nn.Module:
    # Old exps might not have twostage in their configs
    if "twostage" in cfg:
        twostage = cfg.twostage
    else:
        twostage = False

    # Load twostage model if required
    if twostage:
        return get_twostage_model(cfg, ckpt_path)
    else:
        return get_single_model(cfg, ckpt_path)


def get_twostage_model(
    cfg: DictConfig, ckpt_path: Optional[Path | str] = None
) -> torch.nn.Module:

    dnn1_dir = Path(cfg.dnn1.exp_dir)
    dnn1_cfg = OmegaConf.load(dnn1_dir / "hydra/.hydra/config.yaml")
    dnn1_epoch = find_best_epoch(dnn1_dir / "train_log.json", cfg.dnn1.checkpoint)

    dnn1_ckpt = dnn1_dir / f"checkpoints/epoch{dnn1_epoch:03d}.pt"

    dnn1 = get_single_model(dnn1_cfg.model, dnn1_ckpt)

    if cfg.twostage.dnn2_intype == "noisy.den":
        cfg.dnn2.params.n_imics = cfg.twostage.dnn1_inmics + 1
    elif cfg.twostage.dnn2_intype == "den":
        cfg.dnn2.params.n_imics = 1
    else:
        raise ValueError(f"Can't use dnn2_intype={cfg.twostage.dnn2_intype}")

    dnn2 = get_single_model(cfg.dnn2)

    model = DenMSX(dnn1, dnn2, **cfg.twostage)

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=get_device())
        model.load_state_dict(ckpt)
        logging.info(
            f"Loaded twostage model {cfg.name} from checkpoint {str(ckpt_path)}"
        )
    else:
        logging.info(f"Loaded twostage model {cfg.name} without a checkpoint")

    return model


def get_single_model(
    cfg: DictConfig, ckpt_path: Optional[Path | str] = None
) -> torch.nn.Module:

    if cfg.name == "new-tfgridnet":
        model = MCxTFGridNet(**cfg.params)
    elif cfg.name == "prespk-tfgridnet":
        model = PreSpkMCxTFGridNet(**cfg.params)
    elif cfg.name == "den-tfgridnet":
        model = DenTFGridNet(**cfg.params)
    elif cfg.name == "swish-tfgridnet":
        model = SwishTFGridNet(**cfg.params)
    elif cfg.name == "passthrough":
        model = Passthrough(**cfg.params)
    else:
        raise ValueError(f"Model {cfg.name} not recognised. Add code here!")

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=get_device())
        model.load_state_dict(ckpt)

        logging.info(f"Loaded single model {cfg.name} from checkpoint {str(ckpt_path)}")
    else:
        logging.info(f"Loaded single model {cfg.name} without a checkpoint")

    return model


def find_best_epoch(trainlog_fpath: str | Path, etype: str | None):
    """
    Work out which epoch to take from a training log

    :param trainlog_fpath: Path to the training log
    :param etpye: how to decide which epoch to pick
    """

    trainlog_fpath = Path(trainlog_fpath)
    if not trainlog_fpath.exists():
        raise ValueError(f"No training log found at {str(trainlog_fpath)}")

    train_log = read_json(trainlog_fpath)

    if etype == "last":
        epoch = train_log[-1]["epoch"]
    elif etype == "stoi":
        epoch = max(train_log, key=lambda x: x["val_stoi"])["epoch"]
    elif etype == "loss":
        epoch = min(train_log, key=lambda x: x["val_loss"])["epoch"]
    else:
        raise ValueError(f"Epoch type {etype} not recognised")

    return epoch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def stack_varlengths(things: Iterable[torch.Tensor]):
    max_length = max(t.shape[-1] for t in things)

    new_things = []
    lengths = []
    for thing in things:
        if thing.shape[-1] == max_length:
            new_things.append(thing)
            lengths.append(max_length)
        else:
            this_len = thing.shape[-1]
            new_things.append(
                torch.nn.functional.pad(thing, (0, max_length - this_len))
            )
            lengths.append(this_len)
    return torch.stack(new_things, dim=0), torch.tensor(lengths)


def check_memusage():
    from pympler import muppy, summary

    all_objects = muppy.get_objects()
    suma = summary.summarize(all_objects)
    summary.print_(suma)


def get_dev_sessions(sessions_file, device, dataset, debug):
    """Get session tuples for the specified datasets and devices."""

    with open(sessions_file.format(dataset=dataset), "r") as f:
        sessions = list(csv.DictReader(f))

    output = []

    for session in sessions:
        new = {"session": session["session"]}
        wearer_pos = int(session[f"{device}_pos"])
        wearer = session[f"pos{wearer_pos}"]
        targets = [session[f"pos{i}"] for i in range(1, 5) if i != wearer_pos]
        new["wearer"] = wearer
        new["targets"] = targets
        output.append(new)
        if debug:
            break

    return output


def get_trainset(split: str, data_cfg: DictConfig, debug: bool):

    if get_device() == "cpu":
        data_cfg.loader.train.pin_memory = False
        data_cfg.loader.train.num_workers = 0

    if data_cfg.segment_type[:4] == "time":
        data = ECHITime(
            split,
            "all",
            data_cfg.device,
            data_cfg.segment_type,
            data_cfg.ref_type,
            data_cfg.aux_type,
            data_cfg.device_segment,
            data_cfg.ref_segment,
            data_cfg.ref_source,
            data_cfg.seg_fs,
            data_cfg.audio_fs,
            data_cfg.enrol_file,
            data_cfg.enrol_type,
            data_cfg.enrol_source,
            data_cfg.manifest_file,
            data_cfg.segments_file,
            data_cfg.shuffle_targets,
            debug,
        )
        collate = collate_time

    else:
        data = ECHISpeech(
            split,
            "all",
            data_cfg.device,
            data_cfg.segment_type,
            data_cfg.nonspeech_split,
            data_cfg.ref_type,
            data_cfg.aux_type,
            data_cfg.device_segment,
            data_cfg.ref_segment,
            data_cfg.ref_source,
            data_cfg.seg_fs,
            data_cfg.audio_fs,
            data_cfg.enrol_file,
            data_cfg.enrol_type,
            data_cfg.enrol_source,
            data_cfg.manifest_file,
            data_cfg.segments_file,
            debug,
        )
        collate = collate_speech

    loader = DataLoader(data, **data_cfg.loader[split], collate_fn=collate)

    data_len = len(data)
    samples = [data.__getitem__(i * data_len // 5)["id"] for i in range(1, 4)]
    return loader, samples


def get_devset(split: str, data_cfg: DictConfig, debug: bool):

    if get_device() == "cpu":
        data_cfg.loader.dev.pin_memory = False
        data_cfg.loader.dev.num_workers = 0

    if data_cfg.segment_type == "speech":
        stypes = "time.5"
    else:
        stypes = data_cfg.segment_type

    data = ECHITime(
        split,
        "all",
        data_cfg.device,
        stypes,
        data_cfg.ref_type,
        data_cfg.aux_type,
        data_cfg.device_segment,
        data_cfg.ref_segment,
        "chime9.delay",
        data_cfg.seg_fs,
        data_cfg.audio_fs,
        data_cfg.enrol_file,
        data_cfg.enrol_type,
        data_cfg.enrol_source,
        data_cfg.manifest_file,
        data_cfg.segments_file,
        data_cfg.shuffle_targets,
        debug,
    )

    return data


def get_segments(segment_file):
    with open(segment_file, "r") as file:
        segments = list(csv.DictReader(file, fieldnames=["index", "start", "end"]))

    segments = [{a: int(b) for a, b in x.items()} for x in segments]
    return segments


def get_dataocean(segment_file: Path | str, sample_rate: int) -> list[dict]:
    """Load the dataocean transcripts"""
    raw_segments = read_json(Path(segment_file))

    segments = []

    def get_time(timestring: str):
        hour, minute, sec = timestring.split(":")
        hour = int(hour)
        minute = int(minute)
        sec = float(sec)
        return hour * 60 * 60 + minute * 60 + sec

    for i, seg in enumerate(raw_segments):
        start_time = get_time(seg["start_time"]["original"])
        end_time = get_time(seg["end_time"]["original"])

        start = int(start_time * sample_rate)
        end = int(end_time * sample_rate)
        segments.append({"index": i, "start": start, "end": end})

    return segments


POSITIONS = ["pos1", "pos2", "pos3", "pos4"]


def get_session_dicts(session_file, devices, datasets, include_wearer=False):
    """Get session tuples for the specified datasets and devices."""
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(devices, str):
        devices = [devices]

    sessions = []

    for ds in datasets:
        with open(session_file.format(dataset=ds), "r") as f:
            sessions += list(csv.DictReader(f))

    session_device_pid_tuples = []

    for device, session in itertools.product(devices, sessions):
        device_pos = "pos" + session[f"{device}_pos"]

        if device_pos not in POSITIONS:
            logging.warning(f"Device {device} not found for session {session}")
            continue

        for pos in POSITIONS:
            if pos != device_pos or include_wearer:
                session_device_pid_tuples.append(
                    {
                        "dataset": session["session"].split("_")[0],
                        "session": session["session"],
                        "device": device,
                        "wearer_pos": device_pos,
                        "wearer_pid": session[device_pos],
                        "target_pos": pos,
                        "target_pid": session[pos],
                    }
                )

    return session_device_pid_tuples


def get_dataset_pids(session_ftemp: str, datasets: list[str] | str):
    """Load all pids and their dataset"""

    if isinstance(datasets, str):
        datasets = [datasets]

    output = []
    for ds in datasets:
        with open(session_ftemp.format(dataset=ds), "r") as file:
            session_info = list(csv.DictReader(file))

        for sinfo in session_info:
            output += [(ds, sinfo["session"], sinfo[f"pos{i}"]) for i in range(1, 5)]

    return output
