from ci_sdr.pt import ci_sdr_loss
from omegaconf import DictConfig
import torch
from torch.optim.optimizer import Optimizer
import torchaudio
from typing import Tuple

import RobAuraLoss

from shared.core_utils import get_device

EPS = 1e-5


def get_loss(params: DictConfig) -> Tuple[torch.nn.Module, list[str]]:
    """
    Get the loss function for the specified architecture.

    Parameters
    ----------
    name : str
        The name of the loss function.
    params : dict
        The parameters for the loss function.

    Returns
    -------
    torch.nn.Module
        The loss function for the specified architecture.
    """
    name = params.name
    if name == "sisnr":
        loss, extras = LossWrapper(RobAuraLoss.time.SISDRLoss(eps=EPS)), []
    elif name == "auraloss.spec":
        loss, extras = LossWrapper(RobAuraLoss.freq.STFTLoss(**params.kwargs)), []  # type: ignore
    elif name == "random.vadmask.auraloss.spec":
        loss = RobAuraLoss.freq.STFTLoss(**params.kwargs)
        loss, extras = VADMaskLoss(loss, params.vad_thresh), ["targ_vad"]
    elif name == "denmsx.auraloss.spec":
        denloss, dextras = get_loss(params.den_loss)
        msxloss, mextras = get_loss(params.msx_loss)
        loss = DenoiseMSX(denloss, msxloss, params.weights)
        loss, extras = loss, dextras + mextras
    elif name == "vadmask.cisdr":
        loss, extras = VADmaskCISDR(), ["targ_vad"]  # type: ignore
    else:
        raise ValueError(f"Unknown loss name: {name}")

    vpw = params.get("vad_power_weights", None)
    if vpw is None:
        return loss, extras

    loss = VADPowerLoss(loss, vpw)
    if "targ_vad" not in extras:
        extras.append("targ_vad")
    return loss, extras


def get_distance(name: str, reduction: str) -> torch.nn.Module:
    name = name.lower()
    if name == "l1":
        return torch.nn.L1Loss(reduction=reduction)
    elif name == "mse" or name == "l2":
        return torch.nn.MSELoss(reduction=reduction)
    raise NotImplementedError(f"Distance {name} not implemented here")


def get_lrmethod(
    name: str, optim, data_len, params
) -> tuple[torch.optim.lr_scheduler.LRScheduler, bool]:
    """
    Return a configured learning-rate scheduler and a flag indicating its stepping frequency.
    This helper constructs and returns a PyTorch learning-rate scheduler object based on a
    short string identifier and the provided optimizer. It also returns a boolean flag that
    indicates how the caller should advance the scheduler during training:
    - If the flag is True, the scheduler should be stepped on every optimizer update (per-step).
    - If the flag is False, the scheduler should be stepped less frequently (e.g., once per
        epoch) or with a validation metric (as required by ReduceLROnPlateau).
    Parameters
    ----------
    name : str
            Identifier of the LR scheduler to construct. Supported values include:
            - "plateau_reduce" : torch.optim.lr_scheduler.ReduceLROnPlateau
            - "warmup_cos"     : custom WarmupCosine scheduler
    optim
            An instantiated torch.optim.Optimizer whose learning rate the scheduler will manage.
    params
            A mapping (e.g., dict) of keyword arguments to pass to the chosen scheduler's constructor.
    Returns
    -------
    tuple[torch.optim.lr_scheduler.LRScheduler, bool]
            A tuple (scheduler, per_step) where `scheduler` is the constructed scheduler object
            and `per_step` is True if the caller should call scheduler.step() on every optimizer
            update, or False if scheduler.step() should be called less frequently or with a metric.
    Raises
    ------
    ValueError
            If `name` does not match a supported scheduler identifier.
    Examples
    --------
    # Example usage:
    # scheduler, per_step = get_lrmethod("plateau_reduce", optimizer, {"mode": "min", "patience": 2})
    # if per_step:
    #     # call scheduler.step() every optimizer step
    # else:
    #     # call scheduler.step(metric) or call once per epoch
    """

    if name == "plateau_reduce":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **params), False
    elif name == "warmup_cosine":
        return WarmupCosine(optim, steps_per_epoch=data_len, **params), True
    else:
        raise ValueError(f"LR Scheduler {name} not implemented. Add code here")


def vad_mask(
    proc_speech: torch.Tensor, targ_speech: torch.Tensor, targ_vad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Masks the speech signals with the vad array, zeroing any parts which are
    not speech
    """

    vad_success = targ_vad.sum(dim=-1) > 0

    if torch.all(~vad_success):
        return torch.tensor(torch.nan), torch.tensor(torch.nan)

    min_len = min(proc_speech.shape[-1], targ_speech.shape[-1], targ_vad.shape[-1])

    targ_speech = targ_speech[..., :min_len] * targ_vad[..., :min_len]
    proc_speech = proc_speech[..., :min_len] * targ_vad[..., :min_len]

    return proc_speech, targ_speech


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        n_epochs: int,
        steps_per_epoch: int,
        start_factor: float,
        linear_epochs: int | float,
        last_epoch: int = -1,
    ):

        self.steps = 0

        self.lin_steps = int(steps_per_epoch * linear_epochs)
        self.linear = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1,
            total_iters=self.lin_steps,
        )

        self.cos_steps = int((n_epochs - linear_epochs) * steps_per_epoch)
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.cos_steps
        )
        super().__init__(optimizer, last_epoch)

    def step(self):
        if self.steps < self.lin_steps:
            self.linear.step()
        else:
            self.cosine.step()
        self.steps += 1


class LinSpec_SpkCE(torch.nn.Module):
    def __init__(self, weights=[1, 1]):
        super().__init__()
        self.spec_weight = weights[0]
        self.spk_weight = weights[1]
        self.spec = RobAuraLoss.freq.STFTLoss(
            w_sc=0.0, w_lin_mag=1.0, w_log_mag=0.0, w_phs=0.0
        )
        self.bce = torch.nn.BCELoss()

    def forward(self, proc_speech, targ_speech, proc_spk, targ_spk):
        spec = self.spec(proc_speech, targ_speech) * self.spec_weight
        spk = self.bce(proc_spk, targ_spk) * self.spk_weight

        return spec + spk


class LossWrapper(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, proc_speech, targ_speech):
        return self.loss(proc_speech, targ_speech)


class VADMaskLoss(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module, threshold: float):
        """
        Apply VAD Masking to the loss function
        params:
            loss: the loss function to compute
            threshold: the probability of using the VAD Mask
        """
        super().__init__()
        self.loss = loss
        self.thresh = threshold

    def forward(
        self,
        proc_speech: torch.Tensor,
        targ_speech: torch.Tensor,
        targ_vad: torch.Tensor,
    ):

        if self.thresh == 0.0:
            mask = True
        elif self.thresh == 1.0:
            mask = False
        else:
            mask = torch.rand(1).item() < self.thresh

        if mask:
            proc_speech, targ_speech = vad_mask(proc_speech, targ_speech, targ_vad)

        if torch.any(proc_speech.isnan()):
            return torch.tensor(torch.nan)

        return self.loss(proc_speech, targ_speech)


class VADPowerLoss(torch.nn.Module):
    def __init__(
        self, speech_loss: torch.nn.Module, weights: list[float], eps: float = 1e-6
    ):
        super().__init__()
        self.speech_loss = speech_loss
        self.speech_weight, self.silence_weight = weights
        self.eps = eps

    def forward(
        self,
        proc_speech: torch.Tensor,
        targ_speech: torch.Tensor,
        targ_vad: torch.Tensor,
    ):

        speech_loss = self.speech_loss(proc_speech, targ_speech, targ_vad=targ_vad)

        silence_mask = 1 - targ_vad
        silence, _ = vad_mask(proc_speech, targ_speech, silence_mask)
        silence_sums = silence_mask.sum(dim=-1) + 1

        norms = silence.square().sum(dim=-1)
        powers = (norms.sum(dim=-1) / silence_sums) + self.eps
        silence_loss = (10 * powers.log10()).mean()

        return speech_loss * self.speech_weight + silence_loss * self.silence_weight


class DenoiseMSX(torch.nn.Module):
    def __init__(
        self, den_loss: torch.nn.Module, msx_loss: torch.nn.Module, weights=[0.5, 0.5]
    ):
        super().__init__()
        self.den_loss = den_loss
        self.msx_loss = msx_loss
        self.den_weight = weights[0]
        self.msx_weight = weights[1]

    def forward(
        self,
        proc_speech: torch.Tensor,
        targ_speech: torch.Tensor,
        **kwargs,
    ):
        assert (
            proc_speech.ndim == 3
        ), f"Expected 'proc_speech' shape [batch, chans, samples] but found {proc_speech.shape}"
        assert (
            targ_speech.ndim == 3
        ), f"Expected 'targ_speech' shape [batch, chans, samples] but found {targ_speech.shape}"

        proc_den = proc_speech.sum(dim=1, keepdim=True)
        targ_den = targ_speech.sum(dim=1, keepdim=True)

        den_loss = self.den_loss(proc_den, targ_den) * self.den_weight
        msx_loss = self.msx_loss(proc_speech, targ_speech, **kwargs) * self.msx_weight

        return den_loss + msx_loss


class Wav2Vec2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(
            get_device()
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.dist = torch.nn.CosineSimilarity(dim=2)

    def forward(self, proc_speech, targ_speech, **kwargs):

        min_frames = min(proc_speech.shape[-1], targ_speech.shape[-1])
        proc_speech = proc_speech[..., :min_frames]
        targ_speech = targ_speech[..., :min_frames]

        if proc_speech.ndim == 3:
            assert targ_speech.ndim == 3, "Target speech doesn't have enough dimensions"
            proc_speech = proc_speech.reshape(-1, min_frames)
            targ_speech = targ_speech.reshape(-1, min_frames)

        proc_feats, _ = self.model.extract_features(proc_speech)  # type: ignore
        targ_feats, _ = self.model.extract_features(targ_speech)  # type: ignore

        loss = self.dist(proc_feats[-1], targ_feats[-1])
        return -loss.mean()


class VADmaskCISDR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        proc_speech: torch.Tensor,
        targ_speech: torch.Tensor,
        targ_vad: torch.Tensor,
    ):
        min_len = min(proc_speech.shape[-1], targ_speech.shape[-1], targ_vad.shape[-1])

        targ_speech = targ_speech[..., :min_len] * targ_vad[..., :min_len]
        proc_speech = proc_speech[..., :min_len] * targ_vad[..., :min_len]

        vad_success = targ_vad.sum(dim=-1) > 0

        if torch.all(~vad_success):
            return torch.tensor(torch.nan)

        targ_speech = targ_speech[vad_success]
        proc_speech = proc_speech[vad_success]

        targ_speech = targ_speech.reshape(-1, targ_speech.shape[-1])
        proc_speech = proc_speech.reshape(-1, proc_speech.shape[-1])

        loss = ci_sdr_loss(proc_speech, targ_speech, compute_permutation=False)

        return loss.mean()
