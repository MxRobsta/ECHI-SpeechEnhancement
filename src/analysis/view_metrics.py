import streamlit as st
import json
import hydra
from omegaconf import DictConfig
import csv
from pathlib import Path

EXP_NAMES = []


@st.cache_data
def load_sessions(sessions_file):
    with open(sessions_file, "r") as file:
        sessions = list(csv.DictReader(file))
    return sessions


def build_table(
    sessions, metrics_file, exp_names, target_session="dev_11", sort_col="index"
):

    metrics = {}
    keys = []
    for exp in exp_names:
        metrics[exp] = []
        for info in sessions:
            session = info["session"]
            pids = [info[f"pos{i}"] for i in range(1, 5) if i != int(info["aria_pos"])]

            if session != target_session:
                continue

            for pid in pids:
                fpath = metrics_file.format(
                    exp_name=exp, session=session, device="aria", pid=pid
                )
                with open(fpath, "r") as file:
                    metrics[exp] += json.load(file)

        metrics[exp] = {Path(a["wav"]).stem: a for a in metrics[exp]}
        keys = list(metrics[exp].keys())

    print_mets = ["stoi", "fwsegsnr"]
    table = [["key", "duration"] + print_mets * len(exp_names)]
    for k in keys:
        row = [k, metrics[exp_names[0]][k]["duration"]]
        for exp in exp_names:
            for met in print_mets:
                row.append(metrics[exp][k][met])
        table.append(row)
    title = ["info"] + exp_names

    return tuple(title), tuple([tuple(x) for x in table])


def write_table(title, table):

    cols = st.columns(len(title))
    for c, t in zip(cols, title):
        c.write(t)

    for row in table:
        cols = st.columns(len(table[0]) + 1)
        for c, r in zip(cols, row):
            if isinstance(r, str):
                c.write(r)
            else:
                c.write(f"{r:.2f}")

        if cols[-1].button("Load", key=row[0]):
            load_audio(row[0])


@st.cache_data
def load_audio(wav_name):
    noisy_fpath = f"data/analysis/speech_segments/aria/dev/{wav_name}.wav"
    ref_fpath = f"data/analysis/speech_segments/ref/dev/{wav_name}.wav"

    cols = st.columns(2 + len(EXP_NAMES))

    cols[0].write("noisy")
    cols[0].audio(noisy_fpath, format="audio/wav")
    cols[1].write("ref")
    cols[1].audio(ref_fpath, format="audio/wav")

    for i, exp in enumerate(EXP_NAMES):
        cols[2 + i].write(exp)
        exp_fpath = f"data/scratch/experiments/{exp}/enhancement/metrics/{wav_name}.wav"
        cols[2 + i].audio(exp_fpath, format="audio/wav")


def view_metrics(exp_names, metrics_file, sessions_file):

    global EXP_NAMES
    EXP_NAMES = exp_names

    sessions = load_sessions(sessions_file)

    title, table = build_table(sessions, metrics_file, exp_names)

    write_table(title, table)


@hydra.main(version_base=None, config_path="../../config/analysis", config_name="main")
def main(cfg: DictConfig):
    vm = cfg.view_metrics
    view_metrics(vm.exp_names, vm.metrics_file, vm.sessions_file)


if __name__ == "__main__":
    main()
