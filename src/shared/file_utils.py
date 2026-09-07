import json
import soundfile as sf
from pathlib import Path


def read_json(path: Path | str):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def write_json(path: Path, data: list[dict] | dict):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def load_transcript(path: Path | str):
    transcript = read_json(path)

    for x in transcript:
        for thing in ["start", "end"]:
            this_string = x[f"{thing}_time"]["original"].split(":")
            time = (
                60 * 60 * int(this_string[0])
                + 60 * int(this_string[1])
                + float(this_string[2])
            )
            x[f"{thing}_time"] = time

    return transcript


def get_audio_len(path: Path, unit: str):
    """
    Calculates the duration (in seconds) of an audio file.
    Args:
        path (Path): The path to the audio file.
        unit (str): The unit of the duration (seconds, samples)
    Returns:
        float: The length of the audio file in seconds.
    Raises:
        RuntimeError: If the file cannot be opened or read as an audio file.
    """
    if not path.exists():
        raise RuntimeError(f"File path does not exist:\n{str(path)}")
    with sf.SoundFile(str(path)) as file:
        length = file.frames
        if unit == "seconds":
            length /= file.samplerate
    return length


def get_unpack_outfiles(
    noisy_template: str,
    ref_template: str,
    refsource: str,
    segtype: str,
    device: str,
    manitem: dict,
):

    outfiles = {}

    dataset = manitem["session"].split("_")[0]

    outfiles["noisy"] = noisy_template.format(
        segtype=segtype,
        device=device,
        dataset=dataset,
        session=manitem["session"],
        segid=manitem["index"],
        pid="_",
    )

    for pid in manitem["targets"] + [manitem["wearer"]]:
        outfiles[pid] = ref_template.format(
            source=refsource,
            segtype=segtype,
            device=device,
            dataset=dataset,
            session=manitem["session"],
            segid=manitem["index"],
            pid=pid,
        )

    return outfiles
