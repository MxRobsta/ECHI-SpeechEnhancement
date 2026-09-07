import torchaudio
from torch.utils.data import Dataset
import json
import torch
import numpy as np
import logging
from glob import glob

from shared.signal_utils import stack_varlength

from typing import Any, Optional, Tuple

RNG = np.random.default_rng(42)

SPK_EMBEDDINGS = ["resemblyzer", "rawnet"]
VALID_RAINBOWTYPES = ["audio"] + SPK_EMBEDDINGS

MAX_SECONDS = 5
MIN_SECONDS = 0.5


def collate_speech(batch: list[dict[str, Any]]):
    new_out: dict[str, Any] = {
        "id": [x["id"] for x in batch],
        "fs": batch[0]["fs"],
        "key": [x["key"] for x in batch],
        "start": [x["start"] for x in batch],
        "end": [x["end"] for x in batch],
    }

    for audio_type in ["noisy", "target", "aux"]:
        audio = [x[audio_type] for x in batch]
        audio, lens = stack_varlength(audio)
        new_out[audio_type] = audio

        if isinstance(batch[0][f"{audio_type}_len"], int):
            new_out[f"{audio_type}_lens"] = torch.tensor(
                [x[f"{audio_type}_len"] for x in batch]
            ).unsqueeze(0)
        else:
            new_out[f"{audio_type}_lens"] = torch.stack(
                [x[f"{audio_type}_len"] for x in batch], dim=0
            )

    new_out["vad"] = torch.tensor([x["vad"] for x in batch])
    return new_out


class ECHISpeech(Dataset):
    def __init__(
        self,
        subset: str,
        session: str,
        audio_device: str,
        unpack_type: str,
        nonspeech_split: float,
        ref_type: str,
        aux_type: str,
        noisy_signal: str,
        ref_signal: str,
        ref_source: str,
        seg_fs: int,
        audio_fs: int,
        rainbow_signal: str,
        rainbow_type: str,
        spkemb_source: str,
        manifest_file: str,
        segments_file: str,
        debug: bool,
    ):
        super().__init__()
        self.subset = subset
        self.audio_device = audio_device

        self.unpack_type = unpack_type
        self.nonspeech_split = nonspeech_split

        self.ref_type = ref_type
        self.ref_source = ref_source
        self.aux_type = aux_type

        with open(
            manifest_file.format(
                content=unpack_type,
                split=subset,
                device=audio_device,
                dataset=subset,
                reftype=ref_source,
            ),
            "r",
        ) as f:
            self.unpack_manifest = json.load(f)

        if session in [m["session"] for m in self.unpack_manifest]:
            self.unpack_manifest = [
                m for m in self.unpack_manifest if m["session"] == session
            ]

        self.trim_unpack_manifest()

        self.segments_file = segments_file

        self.enrol_type = rainbow_type
        if self.enrol_type in SPK_EMBEDDINGS:
            rainbow_signal = rainbow_signal.replace(".wav", ".pt")
        elif self.enrol_type not in VALID_RAINBOWTYPES:
            raise NotImplementedError(f"Rainbow type {self.enrol_type} not implemented")
        self.enrol_source = spkemb_source

        self.signal_paths = {
            "noisy": noisy_signal,
            "target": ref_signal,
            "aux": rainbow_signal,
        }

        self.seg_fs = seg_fs
        self.audio_fs = audio_fs

        self.debug = debug

        self.rainbow_files = []
        self.get_rainbow_files()

        self.manifest: list[dict]
        self.make_manifest()

    def trim_unpack_manifest(self):

        self.unpack_manifest = [
            x for x in self.unpack_manifest if x["wearer"] != x["targets"]
        ]

        speech = [x for x in self.unpack_manifest if bool(x["speech"])]
        nonspeech = [x for x in self.unpack_manifest if not bool(x["speech"])]

        self.unpack_manifest = speech

        if self.nonspeech_split > 0:
            ns_samples = min(len(nonspeech), int(len(speech) * self.nonspeech_split))
            nonspeech = RNG.choice(nonspeech, ns_samples)
            self.unpack_manifest += list(nonspeech)

    def make_manifest(self, session: Optional[str] = None):
        logging.info(f"Setting up data manifest for {self.subset}")
        self.manifest = []

        for meta in self.unpack_manifest:

            if session is not None and meta["session"] != session:
                continue

            start = int(meta["start_sample"])
            end = int(meta["end_sample"])
            meta["start_sample"] = start
            meta["end_sample"] = end

            if (end - start) < (MIN_SECONDS * self.audio_fs):
                continue

            if self.ref_type == "1spk":
                pid = meta["targets"]
                new = {a: b for a, b in meta.items()}
                new["ref_pids"] = [pid]

                new["aux_pids"] = [pid]
                if self.aux_type == "target-wearer":
                    new["aux_pids"].append(meta["wearer"])

                new["vad"] = int(bool(meta["speech"]))
            else:
                raise NotImplementedError(
                    "For speech only training, must provide"
                    + f"ref_type=1spk, not {self.ref_type}"
                )

            if self.check_rainbow(new):
                self.manifest.append(new)

        if self.debug:
            if len(self.manifest) > 10:
                self.manifest = self.manifest[:10]

    def get_rainbow_files(self):

        mpath = self.signal_paths["aux"].format(
            pid="*",
            dataset=self.subset,
            enroltype=self.enrol_type,
            enrolsource=self.enrol_source,
        )
        self.rainbow_files = glob(mpath)

    def check_rainbow(self, meta):

        good = True
        for pid in meta["aux_pids"]:
            this_fpath = self.signal_paths["aux"].format(
                dataset=self.subset,
                pid=pid,
                enroltype=self.enrol_type,
                enrolsource=self.enrol_source,
            )
            if this_fpath not in self.rainbow_files:
                good = False
                break

        return good

    def get_batch(
        self, meta
    ) -> Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], int]:

        noisy_file = self.signal_paths["noisy"].format(
            segtype="speech",
            split=self.subset,
            session=meta["session"],
            device=self.audio_device,
            segid=meta["index"],
            content=self.unpack_type,
            dataset=self.subset,
            pid=meta["targets"],
        )
        noisy, main_fs = torchaudio.load(noisy_file)

        target, tfs = [], []
        for targ_pid in meta["ref_pids"]:
            targ_file = self.signal_paths["target"].format(
                segtype="speech",
                source=self.ref_source,
                dataset=self.subset,
                session=meta["session"],
                device=self.audio_device,
                segid=meta["index"],
                content=self.unpack_type,
                pid=targ_pid,
            )
            this, this_fs = torchaudio.load(str(targ_file))

            target.append(this)
            tfs.append(this_fs)

        target = torch.stack(target, dim=0)
        target = target.squeeze(1)

        aux, afs = [], []
        for aux_pid in meta["aux_pids"]:
            aux_file = self.signal_paths["aux"].format(
                session=meta["session"],
                device=self.audio_device,
                segment=meta["index"],
                content=self.unpack_type,
                pid=aux_pid,
                dataset=self.subset,
                enroltype=self.enrol_type,
                enrolsource=self.enrol_source,
            )

            if self.enrol_type == "audio":
                this, this_fs = torchaudio.load(str(aux_file))
                afs.append(this_fs)
            elif self.enrol_type in SPK_EMBEDDINGS:
                this = torch.load(aux_file, weights_only=True)

            aux.append(this)

        for f in tfs + afs:
            assert f == main_fs

        return noisy, target, aux, main_fs

    def __getitem__(self, index):

        meta = self.manifest[index]

        out = {"id": f"{meta['session']}.{meta['index']}"}  # type: dict[str, Any]

        if self.ref_type == "1spk":
            out["key"] = meta["ref_pids"][0]
        else:
            out["key"] = "main"

        noisy, target, aux, main_fs = self.get_batch(meta)

        min_length = min(noisy.shape[-1], target.shape[-1], MAX_SECONDS * self.audio_fs)
        noisy = noisy[:, :min_length]
        target = target[:, :min_length]

        noisy = noisy[..., :min_length]
        target = target[..., :min_length]

        if len(aux) == 0:
            aux = torch.tensor([])
            aux_lengths = torch.tensor([])
        elif self.enrol_type == "audio":
            aux, aux_lengths = stack_varlength(aux)
            aux = aux.squeeze(1)
        else:
            aux = torch.stack(aux, dim=0)
            aux_lengths = torch.tensor([])

        out["noisy"] = noisy
        out["noisy_len"] = noisy.shape[-1]
        out["target"] = target
        out["target_len"] = target.shape[-1]
        out["aux"] = aux
        out["aux_len"] = aux_lengths
        out["vad"] = meta["vad"]
        out["fs"] = main_fs
        out["start"] = meta["start_sample"]
        out["end"] = meta["end_sample"]

        return out

    def __len__(self):
        return len(self.manifest)
