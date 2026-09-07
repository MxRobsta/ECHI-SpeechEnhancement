import json
import hydra
from omegaconf import DictConfig
from pathlib import Path
import streamlit as st

from shared.core_utils import get_dev_sessions


@st.cache_data
def get_exps(metric_ftemplate):
    exps = Path(
        metric_ftemplate.format(
            exp_name="", session="", device="", pid="", seg="", filetype=""
        )
    ).parent.parent.parent
    exps = list(exps.glob("[a-zA-Z]*"))
    return exps


@st.cache_data
def get_devs(sessions_file, device):
    return get_dev_sessions(sessions_file, device, "dev", False)


@st.cache_data
def load_audio(src, fpath):
    st.write(src)
    st.audio(fpath, format="audio/wav")


def metric_viewer(metric_ftemplate, sessions_file):

    exps = get_exps(metric_ftemplate)
    print_mets = ["STOI", "ESTOI", "FWSegSNR"]
    sort_options = ["index"] + print_mets
    sessions = get_devs(sessions_file, "aria")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        this_exp = st.selectbox("Experiment", exps, format_func=lambda x: x.stem)
    with col2:
        this_session = st.selectbox(
            "Session", sessions, format_func=lambda x: x["session"]
        )
    with col3:
        this_device = st.selectbox("Device", ["aria", "ha"])
    with col4:
        this_sort = st.selectbox("Sorting", sort_options)

    metrics = []
    for pid in this_session["targets"]:
        metric_file = metric_ftemplate.format(
            exp_name=this_exp.name,
            session=this_session["session"],
            device=this_device,
            pid=pid,
            seg="metrics",
            filetype="json",
        )

        with open(metric_file, "r") as file:
            metrics += json.load(file)

    metrics = sorted(
        metrics, key=lambda x: x[this_sort.lower()], reverse=this_sort != "index"
    )[:10]

    for metric in metrics:
        st.write(Path(metric["wav"]).stem)
        cols = st.columns(2)
        wav_name = str(Path(metric["wav"]).name)
        device = "aria" if "aria" in wav_name else "ha"

        load_audio("noisy", f"data/analysis/speech_segments/{device}/dev/{wav_name}")
        load_audio("processed", metric["wav"])
        load_audio("ref", f"data/analysis/speech_segments/ref/dev/{wav_name}")

        cols = st.columns(len(print_mets))
        for i, met in enumerate(print_mets):
            cols[i].write(f"{met}: {metric[met.lower()]:.2f}")

        st.markdown("---")


@hydra.main(version_base=None, config_path="../../config/analysis", config_name="main")
def main(cfg: DictConfig):
    metric_viewer(cfg.metric_viewer.metric_file, cfg.metric_viewer.sessions_file)


if __name__ == "__main__":
    main()
