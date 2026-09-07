"""Setup the ECHI data for use in experiments"""

import logging
import csv
import json
import numpy as np
import soundfile as sf
import soxr
import torch
from tqdm import tqdm
from pathlib import Path

import hydra
from omegaconf import DictConfig

from shared.signal_utils import (
    resample,
    get_resmblyzer_fpath,
    get_rawnet_fpath,
    get_rawnet_array,
    get_resemblyzer_array,
)
from shared.file_utils import get_unpack_outfiles
from shared.core_utils import get_dataocean, get_segments, get_device


def segment_speech(
    device: str,
    dataset: str,
    sessions_file: str,
    segments_ftemplate: str,
    noisy_template: str,
    chime_ref_ftemp: str,
    new_ref_ftemp: str,
    noisy_out_template: str,
    ref_out_template: str,
    segtype: str,
    ref_source: str,
    manifest_file: str,
    target_sr: int,
    overwrite: bool,
):

    def get_speech_segments():
        segments_file = segments_ftemplate.format(
            dataset=this_dataset, session=session, device=device, pid=this_pid
        )
        with open(segments_file, "r") as file:
            segments_csv = list(
                csv.DictReader(file, fieldnames=["index", "start", "end"])
            )

        segments = []

        for seg in segments_csv:
            start = int(seg["start"])
            end = int(seg["end"])

            if start < first_sample:
                continue
            elif end > session_length:
                break

            segments.append(
                {"index": int(seg["index"]), "start": start, "end": end, "speech": True}
            )
        n_speech = len(segments)
        for i in range(len(segments_csv) - 1):
            start = max(first_sample, int(segments_csv[i]["end"]))
            end = min(session_length, int(segments_csv[i + 1]["start"]))

            if end < first_sample:
                continue
            if start >= session_length:
                break

            segments.append(
                {"index": n_speech + i, "start": start, "end": end, "speech": False}
            )
        return segments

    mani = manifest_file.format(
        content=segtype, device=device, dataset=dataset, reftype=ref_source
    )
    mani = Path(mani)
    if Path(mani).exists() and not overwrite:
        logging.info(f"Manifest found at {mani}. Skipping {dataset}...")
        return

    with open(sessions_file.format(dataset=dataset), "r") as file:
        session_info = list(csv.DictReader(file))

    first_sample = 30 * target_sr

    if ref_source == "chime9.delay":
        ref_template = chime_ref_ftemp
    else:
        ref_template = new_ref_ftemp

    manifest = []  # type list[dict]
    for info in tqdm(session_info):

        if info[f"{device}_pos"] == "":
            continue

        session = info["session"]
        this_dataset = session.split("_")[0]
        wearer = info[f"{device}_pos"]
        wearer = info[f"pos{wearer}"]

        noisy_fpath = noisy_template.format(
            device=device, dataset=this_dataset, session=session
        )

        noisy_audio, nfs = sf.read(noisy_fpath)
        if nfs != target_sr:
            noisy_audio = soxr.resample(noisy_audio, nfs, target_sr)

        for i in range(1, 5):
            this_pid = info[f"pos{i}"]
            ref_fpath = ref_template.format(
                reftype=ref_out_template,
                device=device,
                dataset=this_dataset,
                session=session,
                pid=this_pid,
            )
            ref_audio, rfs = sf.read(ref_fpath)
            if rfs != target_sr:
                ref_audio = soxr.resample(ref_audio, rfs, target_sr)

            session_length = min(noisy_audio.shape[0], ref_audio.shape[0])
            save_segments = get_speech_segments()

            for seg in save_segments:
                ref_snip = ref_audio[seg["start"] : seg["end"]]
                noisy_snip = noisy_audio[seg["start"] : seg["end"]]
                if ref_snip.shape[0] == 0 or noisy_snip.shape[0] == 0:
                    continue

                noisy_outfile = noisy_out_template.format(
                    segtype=segtype,
                    dataset=this_dataset,
                    session=session,
                    device=device,
                    pid=this_pid,
                    segid=seg["index"],
                )

                ref_outfile = ref_out_template.format(
                    segtype=segtype,
                    source=ref_source,
                    dataset=this_dataset,
                    session=session,
                    device=device,
                    pid=this_pid,
                    segid=seg["index"],
                )

                manifest.append(
                    {
                        "session": session,
                        "index": seg["index"],
                        "start_sample": seg["start"],
                        "end_sample": seg["end"],
                        "wearer": wearer,
                        "targets": this_pid,
                        "speech": seg["speech"],
                    }
                )

                if (
                    Path(noisy_outfile).exists()
                    and Path(ref_outfile).exists()
                    and not overwrite
                ):
                    continue

                if not Path(noisy_outfile).parent.exists():
                    Path(noisy_outfile).parent.mkdir(parents=True)

                sf.write(noisy_outfile, noisy_snip, target_sr)

                if not Path(ref_outfile).parent.exists():
                    Path(ref_outfile).parent.mkdir(parents=True)

                sf.write(ref_outfile, ref_snip, target_sr)

    mani = manifest_file.format(
        content=segtype, device=device, dataset=dataset, reftype=ref_source
    )

    if not Path(mani).parent.exists():
        Path(mani).parent.mkdir(parents=True)

    with open(mani, "w") as file:
        json.dump(manifest, file, indent=4)


def segment_time(
    device: str,
    datasets: str | list[str],
    sessions_file: str,
    noisy_template: str,
    chime_ref_ftemp: str,
    new_ref_ftemp: str,
    noisy_out_template: str,
    ref_out_template: str,
    content: str,
    ref_source: str,
    manifest_file: str,
    target_sr: int,
):
    if isinstance(datasets, str):
        datasets = [datasets]

    if ref_source == "chime9.delay":
        ref_template = chime_ref_ftemp
    else:
        ref_template = new_ref_ftemp

    existing_datasets = []
    for ds in datasets:
        mani = manifest_file.format(
            content=content, device=device, dataset=ds, reftype=ref_source
        )
        if Path(mani).exists():
            logging.info(f"Manifest found at {mani}. Skipping {ds}...")
            existing_datasets.append(ds)

    datasets = [d for d in datasets if d not in existing_datasets]

    if len(datasets) == 0:
        return

    segment_seconds = int(content.split(".")[1])

    session_info = []
    for ds in datasets:
        with open(sessions_file.format(dataset=ds), "r") as file:
            session_info += list(csv.DictReader(file))

    segment_samples = target_sr * segment_seconds
    first_sample = 30 * target_sr

    manifest = {a: [] for a in datasets}  # type dict[str, list[Any]]
    for info in tqdm(session_info):

        if info[f"{device}_pos"] == "":
            continue

        session = info["session"]
        this_dataset = session.split("_")[0]

        noisy_fpath = noisy_template.format(
            device=device, dataset=this_dataset, session=session
        )
        noisy_audio, nfs = sf.read(noisy_fpath)
        if nfs != target_sr:
            noisy_audio = soxr.resample(noisy_audio, nfs, target_sr)
        session_length = noisy_audio.shape[0]

        ref_audio = {}
        for i in range(1, 5):
            this_pid = info[f"pos{i}"]
            ref_fpath = ref_template.format(
                source=ref_source,
                device=device,
                dataset=this_dataset,
                session=session,
                pid=this_pid,
            )
            if not Path(ref_fpath).exists():
                raise LookupError(f"Cannot find audio at {ref_fpath}")
            ref_audio[this_pid], rfs = sf.read(ref_fpath)
            if rfs != target_sr:
                ref_audio[this_pid] = soxr.resample(ref_audio[this_pid], rfs, target_sr)

            if ref_audio[this_pid].shape[0] < session_length:
                session_length = ref_audio[this_pid].shape[0]

        assert len(ref_audio) == 4

        segment_id = -1
        for start in range(first_sample, session_length, segment_samples):

            if start >= session_length:
                break

            end = min(start + segment_samples, session_length)
            segment_id += 1

            wearer = info[f"pos{info[f'{device}_pos']}"]
            targets = [pid for pid in ref_audio.keys() if pid != wearer]

            manifest[this_dataset].append(
                {
                    "session": session,
                    "index": segment_id,
                    "start_sample": start,
                    "end_sample": end,
                    "wearer": wearer,
                    "targets": targets,
                }
            )

            outfiles = get_unpack_outfiles(
                noisy_out_template,
                ref_out_template,
                ref_source,
                content,
                device,
                manifest[this_dataset][-1],
            )

            noisy_outfile = outfiles["noisy"]

            if not Path(noisy_outfile).parent.exists():
                Path(noisy_outfile).parent.mkdir(parents=True)

            sf.write(noisy_outfile, noisy_audio[start:end, :], target_sr)

            for pid, audio in ref_audio.items():
                ref_outfile = outfiles[pid]

                if not Path(ref_outfile).parent.exists():
                    Path(ref_outfile).parent.mkdir(parents=True)

                sf.write(ref_outfile, audio[start:end], target_sr)

    for ds in datasets:
        mani = manifest_file.format(
            content=content, device=device, dataset=ds, reftype=ref_source
        )
        if not Path(mani).parent.exists():
            Path(mani).parent.mkdir(parents=True)
        with open(mani, "w") as file:
            json.dump(manifest[ds], file, indent=4)


def prep_rainbow(
    rainbow_template,
    out_template,
    embed_type,
    model_sample_rate,
    datasets,
    sessions_file,
):

    if isinstance(datasets, str):
        datasets = [datasets]

    torch_device = get_device()

    split_pids = []
    for ds in datasets:
        with open(sessions_file.format(dataset=ds), "r") as file:
            session_info = list(csv.DictReader(file))

        split_pids += [
            (ds, info[f"pos{i}"]) for info in session_info for i in range(1, 5)
        ]

    for split, pid in split_pids:

        in_fpath = rainbow_template.format(dataset=split, pid=pid)
        out_fpath = Path(
            out_template.format(
                enroltype=embed_type, enrolsource="rainbow", dataset=split, pid=pid
            )
        )

        if not Path(in_fpath).exists():
            logging.warning(f"No rainbow file found at {in_fpath}. Skipping...")
            continue

        if embed_type != "audio":
            out_fpath = out_fpath.with_suffix(".pt")

        if not out_fpath.parent.exists():
            out_fpath.parent.mkdir(parents=True)
        elif out_fpath.exists():
            continue

        if embed_type == "audio":
            audio = resample(in_fpath, model_sample_rate)
            sf.write(out_fpath, audio, model_sample_rate)
        elif embed_type == "resemblyzer":
            embed = get_resmblyzer_fpath(in_fpath, torch_device)
            out_fpath = out_fpath
            torch.save(embed, out_fpath)
        elif embed_type == "rawnet":
            embed = get_rawnet_fpath(in_fpath, torch_device)
            out_fpath = out_fpath
            torch.save(embed, out_fpath)
        else:
            raise NotImplementedError(f"Rainbow prep type not recognise: {embed_type}")


def prep_ref_spkemb(
    ref_template: str,
    out_template: str,
    embed_type,
    enrol_source,
    datasets,
    sessions_file,
    segments_template,
):
    logging.info("prep_ref_spkemb")
    N_SEGMENTS = 5
    MIN_DURATION = 3 * 16000
    RNG = np.random.default_rng(1362713)

    torch_device = get_device()

    if isinstance(datasets, str):
        datasets = [datasets]

    split_sess_pids = []
    for ds in datasets:
        logging.info("Opening csv file at " + sessions_file.format(dataset=ds))
        with open(sessions_file.format(dataset=ds), "r") as file:
            session_info = list(csv.DictReader(file))
        logging.info("read csv file successfully")
        split_sess_pids += [
            (ds, info["session"], info[f"pos{i}"])
            for info in session_info
            for i in range(1, 5)
        ]

    for split, sess, pid in split_sess_pids:

        input_fpath = ref_template.format(
            dataset=split, device="ha", session=sess, pid=pid
        )
        out_fpath = Path(
            out_template.format(
                enrolsource=enrol_source, enroltype=embed_type, dataset=split, pid=pid
            )
        )

        if embed_type != "audio":
            out_fpath = out_fpath.with_suffix(".pt")
        if out_fpath.exists():
            continue
        elif not out_fpath.parent.exists():
            out_fpath.parent.mkdir(parents=True)

        segments_file = segments_template.format(
            dataset=split, device="ha", session=sess, pid=pid
        )

        if enrol_source == "refspk":
            segments = get_segments(segments_file)
        else:
            segments = get_dataocean(segments_file, 16000)

        audio, fs = sf.read(input_fpath)
        if fs != 16000:
            audio = soxr.resample(audio, fs, 16000)
        segments = [
            s
            for s in segments
            if int(s["end"]) < audio.shape[0]
            and (int(s["end"]) - int(s["start"])) >= MIN_DURATION
        ]

        if len(segments) < N_SEGMENTS:
            raise RuntimeError(f"Not enough segments for {pid} in {input_fpath}")

        chosen_segments = RNG.choice(segments, N_SEGMENTS)  # type: ignore

        wavs = []
        for seg in chosen_segments:
            start = int(seg["start"])
            end = int(seg["end"])
            wavs.append(audio[start:end])

        if embed_type == "resemblyzer":
            embed = get_resemblyzer_array(wavs, torch_device)
        elif embed_type == "rawnet":
            embed = get_rawnet_array(wavs, torch_device)
        else:
            raise NotImplementedError(f"Embedding type {embed_type} not recognised")

        torch.save(embed, out_fpath)


def unpack(cfg):
    logging.info("Preparing the ECHI dataset")

    if cfg.enrol_type is None:
        pass
    elif cfg.enrol_source == "rainbow":
        prep_rainbow(
            cfg.chime_rainbow,
            cfg.enrol_file,
            cfg.enrol_type,
            cfg.model_sample_rate,
            cfg.dataset,
            cfg.sessions_file,
        )
    elif cfg.enrol_source == "refspk" or cfg.enrol_source == "dataocean":

        if cfg.enrol_source == "refspk":
            segfile = cfg.segment_info_file
        else:
            segfile = cfg.dataocean_file

        prep_ref_spkemb(
            cfg.chime_ref,
            cfg.enrol_file,
            cfg.enrol_type,
            cfg.enrol_source,
            cfg.dataset,
            cfg.sessions_file,
            segfile,
        )
    else:
        raise NotImplementedError(
            f"Source for speaker embeddings not valid: {cfg.enrol_source}.\n"
            + "Valid options: [rainbow, refspk]"
        )

    if cfg.ref_source == "chime.delay":
        cfg.ref_session = cfg.chime_ref

    if cfg.segment_type[:4] == "time":
        # Segment by time
        segment_time(
            cfg.device,
            cfg.dataset,
            cfg.sessions_file,
            cfg.chime_device,
            cfg.chime_ref,
            cfg.ref_session,
            cfg.device_segment,
            cfg.ref_segment,
            cfg.segment_type,
            cfg.ref_source,
            cfg.manifest_file,
            cfg.model_sample_rate,
        )
    elif cfg.segment_type == "speech":
        # Segment dev set by time
        segment_time(
            cfg.device,
            "dev",
            cfg.sessions_file,
            cfg.chime_device,
            cfg.chime_ref,
            cfg.ref_session,
            cfg.device_segment,
            cfg.ref_segment,
            "time.5",
            cfg.ref_source,
            cfg.manifest_file,
            cfg.model_sample_rate,
        )
        # Segment train by speech
        segment_speech(
            cfg.device,
            "train",
            cfg.sessions_file,
            cfg.segments_file,
            cfg.chime_device,
            cfg.chime_ref,
            cfg.ref_session,
            cfg.device_segment,
            cfg.ref_segment,
            "speech",
            cfg.ref_source,
            cfg.manifest_file,
            cfg.model_sample_rate,
            cfg.overwrite,
        )
    else:
        raise NotImplementedError(f"Processing for {cfg.segment_type} not implemented")


@hydra.main(version_base=None, config_path="pkg://config/train", config_name="main")
def main(cfg: DictConfig) -> None:
    unpack(cfg.unpack)


if __name__ == "__main__":
    main()
