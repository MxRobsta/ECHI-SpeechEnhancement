import csv
import cv2 as cv
import logging
import numpy as np
from pathlib import Path
import pandas as pd
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
import soundfile as sf
import soxr
import torch
from typing import Callable, Optional, Union

import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    Cardioid,
    DirectionVector,
    MeasuredDirectivityFile,
    Rotation3D,
)


VICON_FS = 250
SEGMENT_FS = 16000
ROOM_DIMENSTIONS = np.array([9.2, 7.0, 2.8])
ROOM_CENTRE = ROOM_DIMENSTIONS / 2
ROOM_CENTRE[2] = 0
ENERGY_ABSORPTION = 0.3
MAX_ORDER = 40

SELF_AZIMS = {
    "pos1": np.array([0, 0, 3 * np.pi / 4]),
    "pos2": np.array([0, 0, np.pi / 4]),
    "pos3": np.array([0, 0, -np.pi / 4]),
    "pos4": np.array([0, 0, -3 * np.pi / 4]),
}


def get_room_params(sim_type: str, sample_rate: int):
    """Get the pyroomacoustics room parameters"""

    output = {"p": ROOM_DIMENSTIONS, "fs": sample_rate, "max_order": MAX_ORDER}

    if sim_type == "pyroom-normed":
        # Default setting
        output["materials"] = pra.Material(ENERGY_ABSORPTION)
    elif sim_type == "pyroom-plaster":
        # Plaster board in room
        output["materials"] = pra.Material("plasterboard")
    else:
        raise NotImplementedError(f"Sim type {sim_type} not recognised")

    return output


def load_csv(
    fpath: str | Path, dtype, fieldnames: Optional[list[str]] = None
) -> list[dict]:
    """Load csv file and cast the values to the specified dtype"""

    with open(fpath, "r") as file:
        data = csv.DictReader(file, fieldnames)

        output = []
        for row in data:
            output.append({a: dtype(b) for a, b in row.items()})

    return output


def get_vicon_loc(vicon: pd.DataFrame, start: int, end: int):
    """Finds the average vicon position for a segment"""
    if start >= len(vicon):
        return [], []
    if end >= len(vicon):
        end = len(vicon)

    snip = vicon.iloc[start : end + 1]
    positions = np.array(snip[["TX", "TY", "TZ"]].mean().values) / 1000 + ROOM_CENTRE

    rotations = np.array(snip[["RX", "RY", "RZ"]].mean().values)

    if np.any(np.isnan(positions)) or np.any(np.isnan(rotations)):
        return [], []

    return positions, rotations


def get_avg_loc(avg_headlocs_file: str | Path, target_pos: str, wearer_pos: str):
    # Load average loacations from the refgen repo
    headlocs = pd.read_csv(avg_headlocs_file, skipinitialspace=True, index_col="pos")

    target_loc = np.array(headlocs.loc[target_pos]) / 1000 + ROOM_CENTRE
    wearer_loc = np.array(headlocs.loc[wearer_pos]) / 1000 + ROOM_CENTRE

    if target_pos != wearer_pos:
        # Target is not wearer
        direction = wearer_loc - target_loc
        direction /= np.linalg.norm(direction)
        azim = np.arctan2(direction[1], direction[0])

        return (
            target_loc,
            np.array([0, 0, azim]),
            wearer_loc,
            np.array([0, 0, azim + np.pi]),
        )
    else:
        # Target is wearer
        return (target_loc, SELF_AZIMS[wearer_pos], target_loc, SELF_AZIMS[wearer_pos])


def make_drift_regressor_poly1d(
    clock_drift_estimates: list[dict],
) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """Create a linear regression model to correct clock drift using np.poly1d."""
    THRESHOLD = 0.05
    delays = [elem["offset"] for elem in clock_drift_estimates]
    times = [elem["analysis_time"] for elem in clock_drift_estimates]
    weights = [elem["max_corr"] for elem in clock_drift_estimates]
    weights = [1 if w > THRESHOLD else 0 for w in weights]

    # Filter out data points with zero weight
    valid_indices = [i for i, w in enumerate(weights) if w > 0]
    if not valid_indices:
        # Handle case with no valid data points
        logging.warning(
            "No data points with sufficient correlation to perform regression."
        )
        return lambda t: 0.0  # Or some other sensible default

    filtered_times = np.array([times[i] for i in valid_indices])
    filtered_delays = np.array([delays[i] for i in valid_indices])
    filtered_weights = np.array([weights[i] for i in valid_indices])

    if len(filtered_times) < 2:  # Need at least 2 points for a line
        logging.warning("Warning: Not enough data points to perform regression.")
        mean_delay = np.mean(filtered_delays) if filtered_delays.size > 0 else 0.0
        return lambda t: float(mean_delay)

    coefficients = np.polyfit(filtered_times, filtered_delays, 1, w=filtered_weights)
    regressor_func = np.poly1d(coefficients)
    return regressor_func


def vad_norm(audio: np.ndarray, fs: int, target_rms=0.05):
    """Normalise the signal for an avg rms during speech"""

    audio *= target_rms / np.sqrt(np.mean(np.square(audio)))
    audio_tensor = torch.from_numpy(audio[:, 0]).float()

    silero = load_silero_vad()
    segments = get_speech_timestamps(audio_tensor, silero, sampling_rate=fs)
    total_rms = 0
    total_samples = 0
    for seg in segments:
        start = seg["start"]
        end = seg["end"]

        samples = end - start
        total_samples += samples
        total_rms += np.sqrt(np.mean(np.square(audio[start:end])))

    total_rms /= total_samples

    audio *= target_rms / total_rms

    return audio


def simulate_reference(
    noisy_fpath: str | Path,
    den_fpath: str | Path,
    target_vicon_fpath: str | Path,
    wearer_vicon_fpath: str | Path,
    avg_headlocs: str | Path,
    refpos_fpath: str | Path,
    refpid_fpath: str | Path,
    sample_rate: int,
    sim_type: str,
    collar: float,
    drift_info: list[dict],
    session_info: dict,
):
    """Use pyroomacoustics to simulate data to produce more realistic references"""

    collar_samples = int(collar * sample_rate)

    # Load the audio data and resample if necessary
    noisy, nfs = sf.read(noisy_fpath)
    denoised, dfs = sf.read(den_fpath)

    # Extract the correct channels
    if session_info["device"] == "aria":
        noisy = noisy[:, 3]
    elif session_info["device"] == "ha":
        noisy = np.sum(noisy[:, :2], axis=1)

    # Resample to target freq
    if nfs != sample_rate:
        noisy = soxr.resample(noisy, nfs, sample_rate)
    if dfs != sample_rate:
        denoised = soxr.resample(denoised, dfs, sample_rate)

    # Trim fies to be same length
    if denoised.shape[0] > noisy.shape[0]:
        denoised = denoised[: noisy.shape[0]]
    elif denoised.shape[0] < noisy.shape[0]:
        noisy = noisy[: denoised.shape[0]]

    # Load segment information
    silero = load_silero_vad()
    segments = get_speech_timestamps(
        read_audio(str(den_fpath)), silero, sampling_rate=sample_rate
    )

    # Check for and load the vicon data
    if not Path(target_vicon_fpath).exists() or not Path(wearer_vicon_fpath).exists():
        # If vicon can't be found, use averages from refgen repo
        logging.warning(
            f"Missing VICON data:\n - {target_vicon_fpath}: {Path(target_vicon_fpath).exists()}"
            + f"\n - {wearer_vicon_fpath}: {Path(wearer_vicon_fpath).exists()}"
        )
        targ_coords, targ_rot, wear_coords, wear_rot = get_avg_loc(
            avg_headlocs, session_info["target_pos"], session_info["wearer_pos"]
        )
        use_avgvicon = True
    else:
        target_vicon = pd.read_csv(target_vicon_fpath, skipinitialspace=True)
        wearer_vicon = pd.read_csv(wearer_vicon_fpath, skipinitialspace=True)
        use_avgvicon = False

    # Get room parameters
    room_params = get_room_params(sim_type, sample_rate)

    # Load a standard HRTF
    hrtf = MeasuredDirectivityFile(
        path="mit_kemar_normal_pinna.sofa",  # SOFA file is in the database
        fs=sample_rate,
        interp_order=12,  # interpolation order
        interp_n_points=1000,  # number of points in the interpolation grid
    )

    # Make drift regressor for clock drift
    if session_info["device"] == "aria":
        regressor = make_drift_regressor_poly1d(drift_info)

    # Process each segment
    reference = np.zeros((denoised.shape[0], 2), dtype=denoised.dtype)

    logging.debug(f"Simulating for session {session_info['session']}, {session_info}")

    for seg in segments:

        start = seg["start"]
        end = seg["end"]
        if end >= reference.shape[0] or end >= noisy.shape[0]:
            logging.warning("Segment is after the reference")
            continue

        den_snip = denoised[start : end + 1]

        # get vicon data for the segment
        if not use_avgvicon:
            vstart = int(start * VICON_FS / SEGMENT_FS)
            vend = int(end * VICON_FS / SEGMENT_FS)
            targ_coords, targ_rot = get_vicon_loc(target_vicon, vstart, vend)
            wear_coords, wear_rot = get_vicon_loc(wearer_vicon, vstart, vend)
            this_collar = collar_samples
        else:
            this_collar = collar_samples * 3

        # if loading vicon fails, use average locations
        if any(len(x) == 0 for x in [targ_coords, wear_coords, targ_rot, wear_rot]):
            targ_coords, targ_rot, wear_coords, wear_rot = get_avg_loc(
                avg_headlocs, session_info["target_pos"], session_info["wearer_pos"]
            )
            this_collar = collar_samples * 3

        # Create the room
        room = pra.ShoeBox(**room_params)

        # Position the wearer
        wori = Rotation3D(wear_rot, "xyz", False)
        room.add_microphone(
            wear_coords, sample_rate, hrtf.get_mic_directivity("left", wori)
        )
        room.add_microphone(
            wear_coords, sample_rate, hrtf.get_mic_directivity("right", wori)
        )

        # Add target source
        target_direction = Cardioid(DirectionVector(azimuth=targ_rot[2]), gain=1.0)
        room.add_source(targ_coords, den_snip, 0, target_direction)

        room.simulate()

        output = room.mic_array.signals.T  # type: ignore

        # Delay start for the aria glasses
        if session_info["device"] == "aria":
            start_time = start / sample_rate
            drift = int(regressor(start_time) * sample_rate)
            start += drift

        # Correlate with device signal to fine-align
        oend = start + output.shape[0]

        # fix if segment end is after end of noisy audio
        if oend + this_collar > noisy.shape[0]:
            noisy_corr = noisy[start - this_collar :]
        else:
            noisy_corr = noisy[start - this_collar : oend + this_collar]

        noisy_corr /= np.linalg.norm(noisy_corr)
        out_corr = np.sum(output, axis=1) / np.linalg.norm(np.sum(output, axis=1))

        noisy_corr = noisy_corr[np.newaxis, :].astype(np.float32)
        out_corr = out_corr[np.newaxis, :].astype(np.float32)

        res = cv.matchTemplate(noisy_corr, out_corr, cv.TM_CCOEFF_NORMED)
        _, _, min_loc, max_loc = cv.minMaxLoc(res)

        if session_info["device"] == "aria":
            # Aria has the same polarity as the CT mic
            offset_samples = collar_samples - max_loc[0]
        else:
            # HA have the opposite polarity to the CT mic
            offset_samples = collar_samples - min_loc[0]

        start += offset_samples
        oend = start + output.shape[0]
        if oend > reference.shape[0]:
            olen = reference.shape[0] - start
            output = output[:olen]
            oend = start + olen

        reference[start:oend] += output

    reference = vad_norm(reference, sample_rate)

    if not Path(refpos_fpath).parent.exists():
        Path(refpos_fpath).parent.mkdir(parents=True)

    sf.write(refpos_fpath, reference, sample_rate)

    if not Path(refpid_fpath).exists():
        Path(refpid_fpath).symlink_to(Path(refpos_fpath).name)
