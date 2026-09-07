import torch
import torch_complex.functional as FC
from torch_complex.tensor import ComplexTensor
from omegaconf import DictConfig

from shared.signal_utils import STFTWrapper


class iNeuBe(torch.nn.Module):
    def __init__(
        self,
        dnn1: torch.nn.Module,
        dnn2: torch.nn.Module,
        stft_params: DictConfig,
        mfmcwf_chunks: int,
        device: str,
        return_stage: str = "dnn2",
    ):
        super().__init__()
        self.dnn1 = dnn1
        self.dnn2 = dnn2
        self.stft = STFTWrapper(**stft_params, device=device)
        self.mfmcwf_chunks = mfmcwf_chunks
        self.return_stage = return_stage

    def forward(
        self, spec: torch.Tensor, spk: torch.Tensor, spk_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        esti1, spk_pred = self.dnn1(spec, spk, spk_lens)

        esti1 = esti1.detach().squeeze(1).squeeze(1)

        if self.return_stage == "dnn1":
            return esti1, spk_pred

        esti1 = self.stft(self.stft.inverse(esti1))

        esti_wf = self.mfmcwf(spec, esti1, self.mfmcwf_chunks, 1e-8)

        if self.return_stage == "mcwf":
            return esti_wf, spk_pred

        in_dnn2 = torch.cat([spec, esti1.unsqueeze(1), esti_wf.unsqueeze(1)], dim=1)
        esti2, spk_pred = self.dnn2(in_dnn2, spk, spk_lens)

        return esti2, spk_pred

    def unfold(self, tf_rep, chunk_size):
        """unfolding STFT representation to add context in the mics channel.

        Args:
            mixture (torch.Tensor): 3D tensor (monaural complex STFT)
                of shape [B, T, F] batch, frames, microphones, frequencies.
            n_chunks (int): number of past and future to consider.

        Returns:
            est_unfolded (torch.Tensor): complex 3D tensor STFT with context channel.
                shape now is [B, T, C, F] batch, frames, context, frequencies.
                Basically same shape as a multi-channel STFT with C microphones.

        """
        bsz, freq, _ = tf_rep.shape
        if chunk_size == 0:
            return tf_rep

        est_unfolded = torch.nn.functional.unfold(
            torch.nn.functional.pad(
                tf_rep, (chunk_size, chunk_size), mode="constant"
            ).unsqueeze(-1),
            kernel_size=(2 * chunk_size + 1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )
        n_chunks = est_unfolded.shape[-1]
        est_unfolded = est_unfolded.reshape(bsz, freq, 2 * chunk_size + 1, n_chunks)
        est_unfolded = est_unfolded.transpose(1, 2)
        return est_unfolded

    def mfmcwf(self, mixture, estimate, n_chunks, tik_eps):
        """multi-frame multi-channel wiener filter.

        Args:
            mixture (torch.Tensor): multi-channel STFT complex mixture tensor,
                of shape [B, M, F, T, 2] batch, mics, freqs, frames, complex
            estimate (torch.Tensor): monaural STFT complex estimate
                of target source [B, F, T, 2] batch, frequencies, frames, complex.
            n_chunks (int): number of past and future mfMCWF frames.
                If 0 then standard MCWF.
            tik_eps (float): diagonal loading for matrix inversion in MCWF computation.

        Returns:
            beamformed (torch.Tensor): monaural STFT complex estimate
                of target source after MFMCWF [B, T, F] batch, frames, frequencies.
        """

        mixture = torch.complex(mixture[..., 0], mixture[..., 1])
        estimate = torch.complex(estimate[..., 0], estimate[..., 1])

        bsz, mics, _, frames = mixture.shape

        mix_unfolded = self.unfold(
            mixture.reshape(bsz * mics, -1, frames), n_chunks
        ).reshape(bsz, mics * (2 * n_chunks + 1), -1, frames)

        mix_unfolded = to_double(mix_unfolded)
        estimate1 = to_double(estimate)

        zeta = torch.einsum("bmft, bft->bmf", mix_unfolded, estimate1.conj())
        scm_mix = torch.einsum("bmft, bnft->bmnf", mix_unfolded, mix_unfolded.conj())
        inv_scm_mix = torch.inverse(
            tik_reg(scm_mix.permute(0, 3, 1, 2), tik_eps)
        ).permute(0, 2, 3, 1)
        bf_vector = torch.einsum("bmnf, bnf->bmf", inv_scm_mix, zeta)

        beamformed = torch.einsum("...mf,...mft->...ft", bf_vector.conj(), mix_unfolded)
        beamformed = beamformed.to(mixture)

        return torch.stack([beamformed.real, beamformed.imag], dim=-1)


def to_double(c):
    if not isinstance(c, ComplexTensor) and torch.is_complex(c):
        return c.to(dtype=torch.complex128)
    else:
        return c.double()


def tik_reg(mat, reg: float = 1e-8, eps: float = 1e-8):
    """Perform Tikhonov regularization (only modifying real part).

    Args:
        mat (torch.complex64/ComplexTensor): input matrix (..., C, C)
        reg (float): regularization factor
        eps (float)
    Returns:
        ret (torch.complex64/ComplexTensor): regularized matrix (..., C, C)
    """
    # Add eps
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    shape = [1 for _ in range(mat.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(*mat.shape[:-2], 1, 1)
    with torch.no_grad():
        epsilon = FC.trace(mat).real[..., None, None] * reg
        # in case that correlation_matrix is all-zero
        epsilon = epsilon + eps
    mat = mat + epsilon * eye
    return mat


if __name__ == "__main__":
    from src.shared.CausalMCxTFGridNet import MCxTFGridNet

    dnn1 = MCxTFGridNet(n_srcs=1, n_imics=4)
    dnn2 = MCxTFGridNet(n_srcs=1, n_imics=6)

    stft_params = DictConfig(
        {
            "n_fft": 128,
            "win_length": 128,
            "hop_length": 64,
            "window": "hann",
        }
    )

    model = iNeuBe(dnn1, dnn2, stft_params, 5, "cpu")

    audio = torch.rand([1, 4, 65, 250, 2])
    spk = torch.rand([1, 65, 100, 2])
    spk_lens = torch.tensor([100])

    model(audio, spk, spk_lens)
