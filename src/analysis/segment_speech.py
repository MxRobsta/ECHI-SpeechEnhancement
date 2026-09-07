import soundfile as sf
from pathlib import Path
import csv
import soxr
import numpy as np
from tqdm import tqdm

input_audio_dir = Path("data/chime9_echi")
output_audio_dir = Path("data/analysis/speech_segments")

for csv_file in tqdm((input_audio_dir / "metadata/ref/dev").glob("*P*"), total=80):

    with open(csv_file, "r") as file:
        segments = list(csv.DictReader(file, fieldnames=["index", "start", "end"]))

    session, device, pid = csv_file.stem.split(".")

    ref_fpath = input_audio_dir / f"ref/dev/{session}.{device}.{pid}.wav"
    noisy_fpath = input_audio_dir / f"{device}/dev/{session}.{device}.wav"

    for atype, fpath in zip(["noisy", "ref"], [noisy_fpath, ref_fpath]):
        if atype == "noisy":
            in_fpath = input_audio_dir / f"{device}/dev/{session}.{device}.wav"
        else:
            in_fpath = input_audio_dir / f"ref/dev/{session}.{device}.{pid}.wav"

        audio, fs = sf.read(in_fpath)
        if fs != 16000:
            audio = soxr.resample(audio, fs, 16000)

        if audio.ndim == 2:
            audio = np.sum(audio, axis=1)

        for seg in segments:
            start = int(seg["start"])
            end = int(seg["end"])

            if atype == "noisy":
                outfpath = (
                    output_audio_dir
                    / f"{device}/dev/{session}.{device}.{pid}.{seg['index']}.wav"
                )
            else:
                outfpath = (
                    output_audio_dir
                    / f"ref/dev/{session}.{device}.{pid}.{seg['index']}.wav"
                )

            outfpath.parent.mkdir(exist_ok=True, parents=True)
            snippet = audio[start:end]
            sf.write(outfpath, snippet, 16000)
