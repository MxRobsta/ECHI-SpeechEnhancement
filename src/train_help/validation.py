from torch.utils.data import Dataset
import torch
import torchaudio
import csv
import soxr

from shared.core_utils import get_device
from shared.signal_utils import stack_varlength


def get_dev_sessions(session_file, device, dataset, debug):
    """Get session tuples for the specified datasets and devices."""

    with open(session_file.format(dataset=dataset), "r") as f:
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


class DevSet(Dataset):
    def __init__(
        self,
        audio_device: str,
        model_fs: int,
        noisy_template: str,
        ref_template: str,
        rainbow_template: str,
        segment_template: str,
        aux_type: str,
        ref_type: str,
        segment_length: int,
        min_stoi_len: float,
        debug: bool,
    ) -> None:
        super().__init__()

        self.torch_device = get_device()
        self.audio_device = audio_device

        self.model_fs = model_fs

        self.noisy_template = noisy_template
        self.ref_template = ref_template
        self.rainbow_tamplate = rainbow_template
        self.segment_template = segment_template

        self.aux_type = aux_type
        self.ref_type = ref_type

        self.segment_samples = self.model_fs * segment_length
        self.start_sample = self.model_fs * 30
        self.min_stoi_samples = self.model_fs * min_stoi_len

        self.manifest = []

        self.debug = debug

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        info = self.manifest[index]
        start, end = info["start"], info["end"]

        output = {
            "start": start,
            "end": end,
            "noisy": self.noisy[:, start:end],
            "fs": self.model_fs,
        }

        if self.ref_type == "1spk":
            output["target"] = self.refs[info["target"]][:, start:end]
            if self.aux_type == "target":
                output["aux"] = self.auxs[info["target"]]
                output["aux_lens"] = output["aux"].shape[-1]
            else:
                this_aux = [self.auxs[info["target"]], self.auxs[info["wearer"]]]
                output["aux"], output["aux_lens"] = stack_varlength(this_aux)
            output["key"] = info["target"]
            output["vad"] = self.vad[info["target"]][start:end]
        elif self.ref_type in ["3spk", "3spksum"]:
            output["target"], _ = stack_varlength(
                [self.refs[p][:, start:end] for p in info["target"]]
            )
            if self.aux_type == "target":
                this_aux = [self.auxs[p][0] for p in info["target"]]
            elif self.aux_type == "target-wearer":
                this_aux = [self.auxs[p][0] for p in info["target"]]
                this_aux.append(self.auxs[info["wearer"][0]])
            elif self.aux_type == "wearer":
                this_aux = [self.auxs[info["wearer"]][0]]
            elif self.aux_type is None:
                this_aux = []
            output["aux"], output["aux_lens"] = stack_varlength(this_aux)

            output["vad"], _ = stack_varlength(
                [self.vad[p][start:end] for p in info["target"]]
            )

            if self.ref_type == "3spk":
                output["key"] = info["target"]
            else:
                output["key"] = "mix"
                output["vad"] = torch.sum(output["vad"], dim=0)
                output["target"] = torch.sum(output["target"], dim=0)

        if not isinstance(output["aux_lens"], torch.Tensor):
            output["aux_lens"] = torch.tensor([output["aux_lens"]])

        return output

    def load_audio(self, fpath):
        audio, fs = torchaudio.load(fpath)

        if fs == self.model_fs:
            return audio

        audio = audio.detach().to("cpu").numpy()
        audio = soxr.resample(audio.T, fs, self.model_fs).T
        audio = torch.from_numpy(audio)

        return audio

    def set_session(self, session, targets, wearer):

        self.noisy = self.load_audio(
            self.noisy_template.format(
                dataset="dev", session=session, device=self.audio_device
            )
        )

        session_len = self.noisy.shape[-1]

        self.refs = {}
        for pid in targets:
            self.refs[pid] = self.load_audio(
                self.ref_template.format(
                    dataset="dev", session=session, device=self.audio_device, pid=pid
                )
            )
            session_len = min(self.refs[pid].shape[-1], session_len)

        self.auxs = {}
        for pid in targets + [wearer]:
            self.auxs[pid] = self.load_audio(
                self.rainbow_tamplate.format(
                    dataset="dev", session=session, device=self.audio_device, pid=pid
                )
            )

        self.vad = {}
        self.all_speech_segments = []
        scalar = self.model_fs / 16000
        for pid in targets + [wearer]:
            with open(
                self.segment_template.format(
                    dataset="dev", device=self.audio_device, session=session, pid=pid
                ),
                "r",
            ) as file:
                speech_segments = list(
                    csv.DictReader(file, fieldnames=["index", "start", "end"])
                )
            self.vad[pid] = torch.zeros(session_len)
            for seg in speech_segments:
                start = int(int(seg["start"]) * scalar)
                end = int(int(seg["end"]) * scalar)
                self.vad[pid][start:end] = 1
                self.all_speech_segments.append([start, end])

        self.manifest = []
        for start in range(self.start_sample, session_len, self.segment_samples):
            end = min(start + self.segment_samples, session_len)

            if self.ref_type == "1spk":
                for pid in targets:
                    self.manifest.append(
                        {"start": start, "end": end, "target": pid, "wearer": wearer}
                    )
            else:
                self.manifest.append(
                    {"start": start, "end": end, "target": targets, "wearer": wearer}
                )

        if self.debug:
            self.manifest = self.manifest[:10]
            session_len = max(x["end"] for x in self.manifest)

        if self.ref_type in ["1spk", "3spk"]:
            evaluation = {pid: torch.zeros([session_len]) for pid in targets}
        else:
            evaluation = {"mix": torch.zeros([session_len])}

        return evaluation
