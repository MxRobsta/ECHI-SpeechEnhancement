from denoiser import pretrained
from denoiser.dsp import convert_audio

from pathlib import Path
import torch
import torchaudio
from typing import Protocol


class Denoiser(Protocol):
    """
    Protocol class for running a denoiser
    """

    def denoise(self, fpath: str | Path) -> torch.Tensor: ...


def load_denoiser(name: str) -> Denoiser:
    """Load a denoiser module from the options listed below"""
    if name.lower() == "meta-denoiser":
        return MetaDenoiser()
    else:
        raise ValueError(f"Denoiser {name} not implemented")


class MetaDenoiser(Denoiser):
    def __init__(self) -> None:
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.den = pretrained.dns64().to(self.device)

    def denoise(self, fpath: str | Path) -> tuple[torch.Tensor, int]:
        audio, fs = torchaudio.load(fpath)
        audio = audio.to(self.device)  # type: ignore

        audio = convert_audio(
            audio, fs, self.den.sample_rate, self.den.chin
        ).contiguous()
        output = []
        splits = 3
        split_samples = audio.shape[-1] // splits

        if (audio.shape[-1] % split_samples) < 10 * self.den.sample_rate:
            split_samples += audio.shape[-1] % split_samples

        with torch.no_grad():
            for start in range(0, audio.shape[1], split_samples):
                end = min(start + split_samples, audio.shape[1])
                output.append(self.den(audio[:, start:end])[0])

        return torch.concatenate(output, dim=1), self.den.sample_rate
