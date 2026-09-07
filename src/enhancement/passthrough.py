from typing import Dict
import torch
import soxr

from enhancement.registry import register_enhancement, Enhancement


@register_enhancement("passthrough")
class Passthrough(Enhancement):
    def __init__(self, output_sample_rate: int) -> None:
        self.output_sample_rate = output_sample_rate

    def process_session(
        self,
        device_audio: torch.Tensor,
        device_fs: int,
        spkid_audio: torch.Tensor,
        spkid_fs: int,
        kwargs: Dict | None = None,
    ) -> torch.Tensor:
        output = soxr.resample(
            device_audio[0].detach().cpu().numpy(), device_fs, self.output_sample_rate
        )
        return torch.from_numpy(output)
