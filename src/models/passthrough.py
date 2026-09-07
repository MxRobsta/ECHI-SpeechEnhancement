import torch

from typing import Tuple


class Passthrough(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        A "model" which does nothing, just returns the audio.
        Good for debugging training loop on CPU
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dummy = torch.nn.Parameter(torch.ones(2))

    def get_test_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            torch.rand([1, self.in_channels, 125, 65, 2]),
            torch.rand([1, self.out_channels, 125 * 30, 65, 2]),
            torch.tensor([[125 * 30 for _ in range(self.in_channels)]]),
        )

    def forward(
        self, spec: torch.Tensor, spk: torch.Tensor, spk_lens: torch.Tensor
    ) -> tuple[torch.Tensor, None]:

        assert spec.shape[1] == self.in_channels

        while spec.shape[1] < self.out_channels:
            spec = torch.stack([spec, spec], dim=1)

        return spec[:, : self.out_channels] * self.dummy, None
