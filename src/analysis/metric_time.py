import json
import hydra
import logging
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from pesq import pesq
from pysepm import composite, fwSNRseg
from pystoi.stoi import stoi
import soundfile as sf
import soxr
from tqdm import tqdm


from shared.core_utils import (
    get_dataocean,
    get_dev_sessions,
    get_segments,
    find_best_epoch,
)

MIN_STOI_SAMPLES = 0


def get_session_score(
    dataset,
    session,
    pids,
    audio_device,
    ref_audio_ftemplate,
    ref_type,
    model_audio_ftemplate,
    segment_ftemplate,
    use_silero,
    output_ftemplate,
    epoch,
):

    ref_audio = {}
    for pid in pids:
        if pid == "sum":
            audio = sum(x for x in ref_audio.values())
        else:
            ref_file = ref_audio_ftemplate.format(
                dataset=dataset,
                session=session,
                device=audio_device,
                pid=pid,
                source=ref_type,
            )
            audio, _ = sf.read(ref_file)
        ref_audio[pid] = audio

    model_audio = {}
    model_files = [
        Path(
            model_audio_ftemplate.format(
                dataset=dataset,
                session=session,
                device=audio_device,
                epoch=epoch,
                pid=pid,
            )
        )
        for pid in pids
    ]
    model_files = []
    for pid in pids:
        expected_file = Path(
            model_audio_ftemplate.format(
                dataset=dataset,
                session=session,
                device=audio_device,
                epoch=epoch,
                pid=pid,
            )
        )
        main_file = Path(
            model_audio_ftemplate.format(
                dataset=dataset,
                session=session,
                device=audio_device,
                epoch=epoch,
                pid="main",
            )
        )
        if expected_file.exists():
            model_files.append(expected_file)
        elif main_file.exists():
            model_files.append(main_file)
        elif len(model_files) > 0:
            model_files.append(None)
        else:
            raise ValueError(f"No file at {expected_file} or {main_file}")

    for mf, pid in zip(model_files, pids):
        if pid == "sum":
            model_audio[pid] = sum(x for x in model_audio.values())
        else:
            audio, fs = sf.read(mf)
            if fs != 16000:
                audio = soxr.resample(audio, fs, 16000)
            model_audio[pid] = audio

    sum_segments = []
    for pid in model_audio.keys():
        proc = model_audio[pid]
        target = ref_audio[pid]

        if target.ndim == 2:
            target = np.sum(target, axis=1)

        if pid == "sum":
            segments = sum_segments
        else:
            segments = segment_ftemplate.format(
                dataset=dataset, session=session, device=audio_device, pid=pid
            )
            if use_silero:
                segments = get_segments(segments)
            else:
                segments = get_dataocean(segments, 16000)
            for i in range(len(segments)):
                segments[i]["pid"] = pid
            sum_segments += segments

        output_file = output_ftemplate.format(
            session=session,
            device=audio_device,
            pid=pid,
            seg="{seg}",
            filetype="{filetype}",
        )
        scores = process_session(target, proc, segments, output_file)

        output_file = output_file.format(seg="metrics", filetype="json")

        with open(output_file, "w") as f:
            json.dump(scores, f, indent=4)


def process_session(target, proc, segments, output_file):
    score_output = []

    output_sfile = Path(output_file.format(seg="metrics", filetype="json"))
    existing_scores = []
    merge = False

    if output_sfile.exists():
        merge = True
        with open(output_sfile, "r") as file:
            existing_scores = json.load(file)
    elif not output_sfile.parent.exists():
        output_sfile.parent.mkdir(parents=True)

    norm_factor = 0.05 / np.sqrt(np.mean(np.square(proc)))

    index = 0
    for i, segment in tqdm(enumerate(segments), total=len(segments)):
        audio_file = output_file.format(seg=segment["index"], filetype="wav")

        if not Path(audio_file).parent.exists():
            Path(audio_file).parent.mkdir(parents=True)

        start, end = segment["start"], segment["end"]

        if end > target.shape[0] or end > proc.shape[0]:
            continue

        target_snip = target[start:end]
        proc_snip = proc[start:end]

        existing_dict = {}
        if merge:
            existing_dict = existing_scores[index]
            assert existing_dict["wav"] == audio_file
        index += 1

        if "fwsegsnr" in existing_dict:
            segment["fwsegsnr"] = existing_dict["fwsegsnr"]
        else:
            segment["fwsegsnr"] = fwSNRseg(target_snip, proc_snip, 16000)
        if "stoi" in existing_dict:
            segment["stoi"] = existing_dict["stoi"]
        else:
            this_targ = target_snip / np.max(np.abs(target_snip))
            this_proc = proc_snip / np.max(np.abs(proc_snip))
            segment["stoi"] = stoi(this_targ, this_proc, 16000, extended=False)

        if "Csig" in existing_dict:
            segment["Csig"] = existing_dict["Csig"]
            segment["Cbak"] = existing_dict["Cbak"]
            segment["Covl"] = existing_dict["Covl"]
        else:
            try:
                sig, bak, ovl = composite(target_snip, proc_snip, 16000)
            except Exception:
                logging.warning(f"Error processing {Path(audio_file).name}")
                sig, bak, ovl = 1, 1, 1

            segment["Csig"] = sig
            segment["Cbak"] = bak
            segment["Covl"] = ovl

        if "pesq" in existing_dict:
            segment["pesq"] = existing_dict["pesq"]
        else:
            try:
                segment["pesq"] = pesq(16000, target_snip, proc_snip)
            except Exception:
                logging.error(f"PESQ failed on {audio_file}")
                segment["pesq"] = 1

        if "sil_rms" in existing_dict:
            segment["sil_rms"] = existing_dict["sil_rms"]
        else:
            last_seg = i - 1
            if i < 0:
                continue
            last_end = segments[last_seg]["end"]
            sil_snip = proc[last_end:start] * norm_factor
            sil_rms = 10 * np.log10(np.sqrt(np.mean(np.square(sil_snip))) + 1e-8)
            segment["sil_rms"] = sil_rms

        segment["duration"] = (end - start) / 16000
        score_output.append(segment)

        segment["wav"] = audio_file
        # sf.write(audio_file, proc_snip, 16000)

    return score_output


def metric_time(
    dataset,
    sessions_file,
    silero_ftemplate,
    dataocean_ftemplate,
    transcript_type,
    audio_device,
    chime_ref_ftemp,
    new_ref_ftemp,
    ref_type,
    exp_name,
    trainlog_fpath,
    best_epoch_type,
    model_audio_ftemplate,
    output_ftemplate,
):
    """Compute metrics for each session"""

    # If device isn't specified, try to infer it from the exp_name
    if audio_device is None:
        audio_device = exp_name.split(".")[0]
    assert audio_device in [
        "aria",
        "ha",
    ], f"Device not specified correctly and cannot be inferred: {audio_device}"

    dev_sessions = get_dev_sessions(sessions_file, audio_device, dataset, False)
    if "{epoch" in model_audio_ftemplate:
        best_epoch = find_best_epoch(trainlog_fpath, best_epoch_type)
    else:
        best_epoch = None

    if ref_type == "chime9.delay":
        ref_audio_ftemp = chime_ref_ftemp
    else:
        ref_audio_ftemp = new_ref_ftemp

    if transcript_type == "silero":
        segment_ftemp = silero_ftemplate
        use_silero = True
    elif transcript_type == "dataocean":
        segment_ftemp = dataocean_ftemplate
        use_silero = False

    for info in dev_sessions:
        session = info["session"]

        pids = info["targets"] + ["sum"]

        get_session_score(
            dataset,
            session,
            pids,
            audio_device,
            ref_audio_ftemp,
            ref_type,
            model_audio_ftemplate,
            segment_ftemp,
            use_silero,
            output_ftemplate,
            best_epoch,
        )


@hydra.main(version_base=None, config_path="../../config/analysis", config_name="main")
def main(cfg: DictConfig):
    """Entry point for computing metrics"""
    metric_time(
        cfg.metric_time.dataset,
        cfg.metric_time.sessions_file,
        cfg.metric_time.silero_segment_file,
        cfg.metric_time.dataocean_file,
        cfg.metric_time.transcripts,
        cfg.metric_time.device,
        cfg.metric_time.chime_ref,
        cfg.metric_time.new_ref,
        cfg.metric_time.ref_type,
        cfg.metric_time.exp_name,
        cfg.metric_time.train_log,
        cfg.metric_time.best_epoch,
        cfg.metric_time.model_file,
        cfg.metric_time.metric_file,
    )


if __name__ == "__main__":
    main()
