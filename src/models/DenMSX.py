import torch
from typing import Tuple


class DenMSX(torch.nn.Module):
    """
    Two stage system which first denoises the audio, then runs source separation
    """

    def __init__(
        self,
        dnn1: torch.nn.Module,
        dnn2: torch.nn.Module,
        dnn1_insrcs: str | None,
        dnn1_inmics: int,
        dnn1_freeze: bool,
        dnn2_insrcs: int,
        dnn2_outsrcs: int,
        dnn2_intype: str,
    ):
        super().__init__()

        self.dnn1 = dnn1
        self.dnn2 = dnn2

        self.dnn1_insrcs = dnn1_insrcs
        self.dnn1_inmics = dnn1_inmics
        self.dnn1_freeze = dnn1_freeze

        self.dnn2_insrcs = dnn2_insrcs
        self.dnn2_outsrcs = dnn2_outsrcs
        self.dnn2_intype = dnn2_intype

        if self.dnn1_freeze:
            self.dnn1.eval()

    def get_test_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            torch.rand([1, self.dnn1_inmics, 125, 65, 2]),
            torch.rand([1, self.dnn2_insrcs, 192]),
            torch.tensor([[125 * 30 for _ in range(self.dnn2_insrcs)]]),
        )

    def forward(
        self, spec: torch.Tensor, spk: torch.Tensor, spk_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Construct input data for DNN1
        dnn1_audio = spec
        if self.dnn1_insrcs is None:
            # Don't use any spk info for dnn1
            dnn1_spk = None
            dnn1_spklens = None
        elif self.dnn1_insrcs == 1:
            # Wearer spk info is always the last from dataloader
            dnn1_spk = torch.narrow(spk, 1, -1, 1)
            dnn1_spklens = torch.narrow(spk_lens, 1, -1, 1)
        else:
            raise ValueError(f"Cannot handle dnn1_insrcs={self.dnn1_insrcs}")

        # Process data with DNN1 and detach if we don't want to update params
        if self.dnn1_freeze:
            with torch.no_grad():
                dnn1_out, _ = self.dnn1(dnn1_audio, dnn1_spk, dnn1_spklens)
        else:
            dnn1_out, _ = self.dnn1(dnn1_audio, dnn1_spk, dnn1_spklens)

        if self.dnn1_freeze:
            dnn1_out = dnn1_out.detach()

        # Prep input for DNN2
        #   Audio Input:
        if self.dnn2_intype == "noisy.den":
            dnn2_audio = torch.cat([spec, torch.view_as_real(dnn1_out)], dim=1)
        elif self.dnn2_intype == "den":
            dnn2_audio = dnn1_out
        else:
            raise ValueError(f"Can't use dnn2_intype={self.dnn2_intype}")

        #   Spk input:
        if self.dnn2_insrcs == 4:
            # Use all spk info
            dnn2_spk = spk
            dnn2_spklens = spk_lens
        elif self.dnn2_insrcs == 3:
            dnn2_spk = spk[:, :3]
            dnn2_spklens = spk[:, :3]
        else:
            raise ValueError(f"Can't run DNN2 with dnn2_insrcs={self.dnn2_insrcs}")

        dnn2_out, dnn2_spkout = self.dnn2(dnn2_audio, dnn2_spk, dnn2_spklens)

        return dnn2_out, dnn2_spkout
