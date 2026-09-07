import numpy as np
import torch
import torchaudio
import logging
from pathlib import Path
import soundfile as sf
import soxr

RESEMBLYZER = None
RAWNET3 = None


def resample(fpath: str | Path, target_fs: int):
    """Loads audio and resamples to desired frequency"""
    audio, fs = sf.read(str(fpath))
    if fs == target_fs:
        return audio
    return soxr.resample(audio, fs, target_fs)


def noisy_reference_channel(audio: torch.Tensor, ref_channel: int | list):
    """
    Returns reference channel of the noisy device (Aria/HA)
    Assumes shape [batch, channel, ...]
    Time-domain or spectrograms
    """
    if audio.ndim < 3:
        raise ValueError(f"Not enough dimensions in audio: {audio.shape}")

    if isinstance(ref_channel, int):
        # Return just the target channel but keep dim
        return torch.narrow(audio, 1, ref_channel, 1)  # Use narrow to keep dim
    elif isinstance(ref_channel, (list, tuple)):
        # Return average of the ref channels
        audio = audio[:, ref_channel, :]
        audio = audio.sum(dim=1, keepdim=True) / len(ref_channel)
        return audio

    raise ValueError(f"Ref channel argument '{ref_channel}' not recognised")


def get_resmblyzer_fpath(fpath: str | Path, device):
    """Gets resemblyzer embedding from a file path"""
    import resemblyzer

    global RESEMBLYZER

    if RESEMBLYZER is None:
        RESEMBLYZER = resemblyzer.VoiceEncoder(device=device)

    wav = resemblyzer.preprocess_wav(Path(fpath))

    embed = RESEMBLYZER.embed_utterance(wav)

    return torch.from_numpy(embed)


def get_resemblyzer_array(wavs: np.ndarray | list[np.ndarray], device):
    """Gets resemblyzer embedding from a (list of) wav arrays"""
    import resemblyzer

    global RESEMBLYZER

    if isinstance(wavs, np.ndarray):
        wavs = [wavs]

    if RESEMBLYZER is None:
        RESEMBLYZER = resemblyzer.VoiceEncoder(device=device)

    embed = RESEMBLYZER.embed_speaker(wavs)
    return torch.from_numpy(embed)


def get_rawnet_array(wavs: np.ndarray | list[np.ndarray], device):
    from espnet2.bin.spk_inference import Speech2Embedding

    global RAWNET3

    if RAWNET3 is None:
        RAWNET3 = Speech2Embedding.from_pretrained(
            model_tag="espnet/voxcelebs12_rawnet3", device=device
        )
    if isinstance(wavs, np.ndarray):
        wavs = [wavs]
    prev_level = logging.disable(logging.WARNING)
    embeddings = []
    for wav in wavs:
        this_emb = RAWNET3(wav)
        embeddings.append(this_emb / torch.norm(this_emb, p=0, dim=1))
    embed = torch.concat(embeddings, dim=0).mean(dim=0).to("cpu")
    logging.basicConfig(level=prev_level)
    return embed


def get_rawnet_fpath(fpath: str | Path, device):
    from espnet2.bin.spk_inference import Speech2Embedding

    global RAWNET3

    if RAWNET3 is None:
        RAWNET3 = Speech2Embedding.from_pretrained(
            model_tag="espnet/voxcelebs12_rawnet3", device=device
        )

    wav, fs = sf.read(fpath)
    if fs != 16000:
        wav = soxr.resample(wav, fs, 16000)

    embed = get_rawnet_array(wav, device)
    return embed.squeeze(0)


def get_rms(signal: torch.Tensor) -> torch.Tensor:
    """
    Calculate the RMS of a signal.
    Args:
        signal (torch.Tensor): The input signal.
    Returns:
        torch.Tensor: The RMS of the signal.
    """
    return torch.sqrt(torch.mean(signal**2))


def rms_normalize(signal: torch.Tensor, target_rms: float) -> torch.Tensor:
    """
    Normalize the RMS of a signal to 1.
    Args:
        signal (torch.Tensor): The input signal.
    Returns:
        torch.Tensor: The normalized signal.
    """
    rms = get_rms(signal)
    if rms > 0:
        return signal * target_rms / rms
    else:
        return signal


def get_file_length(fpath: str | Path, unit: str = "samples"):
    with sf.SoundFile(fpath, "r") as file:
        samples = file.frames
        fs = file.samplerate

    if unit == "samples":
        return samples
    elif unit == "seconds":
        return samples / fs
    else:
        return samples


def prep_audio(
    audio: torch.Tensor,
    sample_rate: int,
    target_channels: int,
    target_sr: int,
    target_rms: float,
    batched: bool = True,
):
    input_ndim = audio.ndim

    # Ensure second dimension is the channel dimension
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
        assert not batched, "Batched audio must be 2D or 3D"

    if audio.ndim == 2:
        audio = audio.unsqueeze(int(batched))

    if audio.ndim > 3:
        raise ValueError(
            f"Audio cannot have more than 3 dimensions. Found shape {audio.shape}"
        )

    # Audio shape: [batch, audio_channels, samples]

    # Strip excess channels
    if target_channels > audio.shape[1]:
        logging.warning(
            f"Unexpected number of audio channels. Found {audio.shape[1]} channels in audio, but want to return {target_channels}"
        )
    else:
        audio = audio[:, :target_channels, :]

    # Resample
    if sample_rate != target_sr:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sr)

    # Normalize
    if target_rms > 0.0:
        audio = rms_normalize(audio, target_rms)

    # Restore to original shape
    if batched:
        # input was 2D or 3D. Squeeze channel dim if necessary
        if input_ndim == 2:
            audio = audio.squeeze(1)
    else:
        # input was 1D or 2D. Squeeze batch dim
        audio = audio.squeeze(0)
        if input_ndim == 1:
            # Squeeze channel dim
            audio = audio.squeeze(0)

    return audio


def match_length(audio0, audio1):
    """
    Pads the shorter of two audio tensors along the last dimension so that both have the same length.
    Parameters:
        audio0 (torch.Tensor): The first audio tensor.
        audio1 (torch.Tensor): The second audio tensor.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the two audio tensors, both padded to the same length along the last dimension.
    """

    if audio0.shape[-1] > audio1.shape[-1]:
        pad_len = audio0.shape[-1] - audio1.shape[-1]
        audio1 = torch.nn.functional.pad(audio1, (0, pad_len))
    elif audio1.shape[-1] > audio0.shape[-1]:
        pad_len = audio1.shape[-1] - audio0.shape[-1]
        audio0 = torch.nn.functional.pad(audio0, (0, pad_len))
    return audio0, audio1


def pad_samples(audio: torch.Tensor, samples: int):
    """
    Pads the input audio tensor with a specified number of zeros at the end.
    Args:
        audio (torch.Tensor): The input audio tensor to be padded.
        samples (int): The number of zero samples to pad at the end of the audio tensor.
    Returns:
        torch.Tensor: The padded audio tensor. If samples is 0, returns the original tensor.
    """

    if samples == 0:
        return audio
    audio = torch.nn.functional.pad(audio, (0, samples), mode="constant", value=0.0)
    return audio


def pad_tolength(audio: torch.Tensor, target_length: int):
    """
    Pads the input audio tensor to the specified target length.
    If the target length is less than or equal to the current length of the audio tensor,
    the original audio is returned. If the target length is greater, the audio is padded
    with zeros (or as defined by `pad_samples`) to reach the target length.
    Args:
        audio (torch.Tensor): The input audio tensor to be padded.
        target_length (int): The desired length of the output tensor.
    Returns:
        torch.Tensor: The padded audio tensor if padding is needed, otherwise the original tensor.
    Raises:
        Logs an error if the target length is shorter than the audio length.
    """

    if target_length < audio.shape[-1]:
        logging.error("Target length shorter than audio len")
        return audio
    elif target_length == audio.shape[-1]:
        return audio
    else:
        return pad_samples(audio, target_length - audio.shape[-1])


def stack_varlength(audio: list[torch.Tensor]):
    if len(audio) == 0:
        return torch.tensor([[]]), torch.tensor([])

    lens = [x.shape[-1] for x in audio]

    if len(set(lens)) == 1:
        return torch.stack(audio), torch.tensor(lens)
    max_len = max(lens)
    new_audio = []
    for x in audio:
        new_audio.append(pad_tolength(x, max_len))
    return torch.stack(new_audio), torch.tensor(lens)


class STFTWrapper(torch.nn.Module):
    def __init__(
        self, n_fft=1024, hop_length=256, win_length=None, window=None, device="cpu"
    ):
        super(STFTWrapper, self).__init__()

        self.device = device

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = torch.hann_window(self.win_length).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        do_reshape = False
        if x.ndim == 3:
            do_reshape = True
            batch, chan, samp = x.shape
            x = x.reshape(batch * chan, samp)

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        X = torch.view_as_real(X)

        if do_reshape:
            _, F, T, comp = X.shape
            X = X.reshape(batch, chan, F, T, comp)

        return X.transpose(-2, -3)  # Swap shape to be [batch?, time, freq, comp]

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        if not X.is_complex():
            X = X.contiguous()
            X = torch.complex(X[..., 0], X[..., 1])

        do_reshape = X.ndim == 4
        if do_reshape:
            # Multichannel input
            batch, chan, frames, freq = X.shape
            X = X.reshape(batch * chan, frames, freq)

        X = X.transpose(-1, -2)

        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )

        if do_reshape:
            x = x.reshape(batch, chan, -1)

        return x


if __name__ == "__main__":
    stft = STFTWrapper()
    audio = torch.rand(1, 2, 16000)
    s = stft(audio)
    print(s.shape)
