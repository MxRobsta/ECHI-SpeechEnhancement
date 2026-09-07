import torchaudio
from torch.utils.data import Dataset
import csv
import json
import torch
import numpy as np
import logging
import gc
from glob import glob

from shared.signal_utils import stack_varlength

from typing import Any, Optional, Tuple

RNG = np.random.default_rng(42)

SPK_EMBEDDINGS = ["resemblyzer", "rawnet"]
VALID_RAINBOWTYPES = ["audio"] + SPK_EMBEDDINGS


def collate_time(batch: list[dict[str, Any]]):
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

    new_out["vad"] = torch.stack([x["vad"] for x in batch], dim=0)
    return new_out


class ECHITime(Dataset):
    def __init__(
        self,
        subset: str,
        session: str,
        audio_device: str,
        unpack_type: str,
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
        shuffle_targets: bool,
        debug: bool,
    ):
        super().__init__()
        self.dataset = subset
        self.audio_device = audio_device

        self.unpack_type = unpack_type

        self.ref_type = ref_type
        self.ref_source = ref_source
        self.aux_type = aux_type

        with open(
            manifest_file.format(
                content=unpack_type,
                dataset=subset,
                device=audio_device,
                reftype=ref_source,
            ),
            "r",
        ) as f:
            self.unpack_manifest = json.load(f)

        if session in [m["session"] for m in self.unpack_manifest]:
            self.unpack_manifest = [
                m for m in self.unpack_manifest if m["session"] == session
            ]

        self.segments_file = segments_file

        self.enrol_type = rainbow_type
        if self.enrol_type in SPK_EMBEDDINGS:
            rainbow_signal = rainbow_signal.replace(".wav", ".pt")
        elif self.enrol_type not in VALID_RAINBOWTYPES and self.enrol_type is not None:
            raise NotImplementedError(f"Rainbow type {self.enrol_type} not implemented")
        self.enrol_source = spkemb_source

        self.signal_paths = {
            "noisy": noisy_signal,
            "target": ref_signal,
            "aux": rainbow_signal,
        }

        segment_length = int(self.unpack_type.split(".")[-1])

        self.seg_fs = seg_fs
        self.audio_fs = audio_fs
        self.segment_samples = segment_length * self.audio_fs

        self.debug = debug
        self.shuffle_targets = shuffle_targets

        self.rainbow_files = []
        self.get_rainbow_files()

        self.manifest: list[dict]
        self.make_manifest()

    def make_manifest(self, session: Optional[str] = None):
        logging.info(f"Setting up data manifest for {self.dataset}")

        self.manifest = []
        vad_dict = self.make_vad_dict()

        for meta in self.unpack_manifest:

            if session is not None and meta["session"] != session:
                continue

            start = int(meta["start_sample"])
            end = int(meta["end_sample"])
            meta["start_sample"] = start
            meta["end_sample"] = end

            if self.ref_type == "1spk":
                new = []
                for pid in meta["targets"]:
                    new.append({a: b for a, b in meta.items()})
                    new[-1]["ref_pids"] = [pid]

                    new[-1]["aux_pids"] = [pid]
                    if self.aux_type == "target-wearer":
                        new[-1]["aux_pids"].append(meta["wearer"])

                    new[-1]["vad"] = (
                        vad_dict[meta["session"]][pid][start:end]
                        .unsqueeze(0)
                        .contiguous()
                        .clone()
                    )

            elif self.ref_type in ["3spk", "3spksum"]:
                new = {a: b for a, b in meta.items()}
                new["ref_pids"] = meta["targets"]

                if self.aux_type == "target":
                    new["aux_pids"] = meta["targets"]
                elif self.aux_type == "target-wearer":
                    new["aux_pids"] = meta["targets"] + [meta["wearer"]]
                elif self.aux_type == "wearer":
                    new["aux_pids"] = [meta["wearer"]]
                elif self.aux_type is None:
                    new["aux_pids"] = []
                else:
                    raise NotImplementedError(
                        f"Ref type {self.ref_type} not compatible with {self.aux_type}"
                    )

                new["vad"] = [
                    vad_dict[meta["session"]][pid][start:end].contiguous().clone()
                    for pid in new["ref_pids"]
                ]
                new["vad"] = torch.stack(new["vad"], dim=0)
            elif self.ref_type == "4spk":
                new = {a: b for a, b in meta.items()}
                new["ref_pids"] = meta["targets"] + [meta["wearer"]]
                new["aux_pids"] = meta["targets"] + [meta["wearer"]]
                new["vad"] = [
                    vad_dict[meta["session"]][pid][start:end].contiguous().clone()
                    for pid in new["ref_pids"]
                ]
                new["vad"] = torch.stack(new["vad"], dim=0)
            else:
                raise NotImplementedError(f"Ref type {self.ref_type} not recognised")

            if isinstance(new, dict):
                new = [new]

            for thing in new:
                if self.check_rainbow(thing):
                    self.manifest.append(thing)

        del vad_dict
        gc.collect()

        if self.debug:
            if len(self.manifest) > 10:
                self.manifest = self.manifest[:10]

    def make_vad_dict(self):

        self.session_pids = {}
        self.session_targets = {}
        self.session_lens = {}
        for meta in self.unpack_manifest:
            if meta["session"] not in self.session_pids:
                self.session_pids[meta["session"]] = meta["targets"] + [meta["wearer"]]
                self.session_targets[meta["session"]] = meta["targets"]
                self.session_lens[meta["session"]] = max(
                    m["end_sample"]
                    for m in self.unpack_manifest
                    if m["session"] == meta["session"]
                )

        self.session_speech_segments = {}
        vad_dict = {}
        for session, pids in self.session_pids.items():
            split = session.split("_")[0]
            vad_dict[session] = {}
            self.session_speech_segments[session] = []

            for pid in pids:

                speech_segfile = self.segments_file.format(
                    dataset=split, device=self.audio_device, session=session, pid=pid
                )
                with open(speech_segfile, "r") as file:
                    speech_segments = list(
                        csv.DictReader(file, fieldnames=["index", "start", "end"])
                    )

                scalar = self.audio_fs / self.seg_fs

                vad = torch.zeros(self.session_lens[session], dtype=torch.bool)

                for seg in speech_segments:
                    start = int(int(seg["start"]) * scalar)
                    end = int(int(seg["end"]) * scalar)

                    vad[start:end] = True

                    self.session_speech_segments[session].append([pid, start, end])

                vad_dict[session][pid] = vad
        return vad_dict

    def get_rainbow_files(self):

        mpath = self.signal_paths["aux"].format(
            pid="*",
            dataset=self.dataset,
            enroltype=self.enrol_type,
            enrolsource=self.enrol_source,
        )
        self.rainbow_files = glob(mpath)

    def check_rainbow(self, meta):

        good = True
        for pid in meta["aux_pids"]:
            this_fpath = self.signal_paths["aux"].format(
                dataset=self.dataset,
                pid=pid,
                enroltype=self.enrol_type,
                enrolsource=self.enrol_source,
            )
            if this_fpath not in self.rainbow_files:
                good = False
                break

        return good

    def get_val_output(self, session, debug):

        if debug:
            session_length = max(x["end_sample"] for x in self.manifest)
        else:
            session_length = self.session_lens[session]

        if self.ref_type == "1spk":
            pids = self.session_targets[session]
            output = {p: torch.zeros([1, session_length]) for p in pids}
        elif self.ref_type == "3spk":
            output = {"main": torch.zeros([3, session_length])}
        elif self.ref_type == "4spk":
            output = {"main": torch.zeros([4, session_length])}
        elif self.ref_type == "3spksum":
            output = {"main": torch.zeros([1, session_length])}
        else:
            raise NotImplementedError(f"Unrecognised ref type {self.ref_type}")

        return output

    def get_val_speech_segments(self, session):

        return self.session_speech_segments[session], self.manifest[0]["ref_pids"]

    def get_batch(
        self, meta
    ) -> Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], int]:

        noisy_file = self.signal_paths["noisy"].format(
            split=self.dataset,
            session=meta["session"],
            device=self.audio_device,
            segid=meta["index"],
            segtype=self.unpack_type,
            dataset=self.dataset,
            pid="_",
        )
        noisy, main_fs = torchaudio.load(noisy_file)

        target, tfs = [], []
        for targ_pid in meta["ref_pids"]:
            targ_file = self.signal_paths["target"].format(
                split=self.dataset,
                session=meta["session"],
                device=self.audio_device,
                segid=meta["index"],
                segtype=self.unpack_type,
                pid=targ_pid,
                dataset=self.dataset,
                source=self.ref_source,
            )

            this, this_fs = torchaudio.load(str(targ_file))
            if this.shape[0] == 2:
                this = torch.mean(this, dim=0, keepdim=True)

            target.append(this)
            tfs.append(this_fs)

        target = torch.stack(target, dim=0)
        target = target.squeeze(1)

        aux, afs = [], []
        for aux_pid in meta["aux_pids"]:
            aux_file = self.signal_paths["aux"].format(
                split=self.dataset,
                session=meta["session"],
                device=self.audio_device,
                segid=meta["index"],
                segtype=self.unpack_type,
                pid=aux_pid,
                dataset=self.dataset,
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

        if len(aux) == 0:
            aux = torch.tensor([])
            aux_lengths = torch.tensor([])
        elif self.enrol_type == "audio":
            aux, aux_lengths = stack_varlength(aux)
            aux = aux.squeeze(1)
        else:
            aux = torch.stack(aux, dim=0)
            aux_lengths = torch.tensor([])

        vad = meta["vad"].to(torch.float)
        if self.ref_type == "3spksum":
            target = torch.sum(target, dim=0, keepdim=True)
            vad = torch.sum(vad, dim=0, keepdim=True)

        if noisy.shape[-1] > self.segment_samples:
            noisy = noisy[..., : self.segment_samples]
            target = target[..., : self.segment_samples]

        if self.shuffle_targets:
            ntarg = target.shape[0]
            naux = aux.shape[0]
            nvad = vad.shape[0]
            assert ntarg == nvad
            if ntarg >= 3 or naux >= 3:
                # only shuffle when extracting multiple
                order = torch.randperm(3)
                sortarg = target[order]
                sortvad = vad[order]
                if ntarg == 4:
                    target = torch.concat([sortarg, target[3:]])
                    vad = torch.concat([sortvad, vad[3:]])
                else:
                    target = sortarg
                    vad = sortvad

                sortaux = aux[order]
                if naux == 4:
                    aux = torch.concat([sortaux, aux[3:]])
                else:
                    aux = sortaux

        out["noisy"] = noisy
        out["noisy_len"] = noisy.shape[-1]
        out["target"] = target
        out["target_len"] = target.shape[-1]
        out["aux"] = aux
        out["aux_len"] = aux_lengths
        out["vad"] = vad
        out["fs"] = main_fs
        out["start"] = meta["start_sample"]
        out["end"] = meta["end_sample"]

        return out

    def __len__(self):
        return len(self.manifest)
