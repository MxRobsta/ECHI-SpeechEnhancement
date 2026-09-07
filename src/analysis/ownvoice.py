import numpy as np
import csv
import logging
import matplotlib.pyplot as plt

with open("data/chime9_echi/metadata/sessions.dev.csv", "r") as file:
    sessions_file = list(csv.DictReader(file))

segments_fstring = "data/chime9_echi/metadata/ref/dev/{session}.{device}.{pid}.csv"
results_fstring = "data/results/{system}/evaluation/reports/report.dev.{device}.{ref_type}.{session}.{pid}.csv"

systems = ["base150", "pass150"]
devices = ["aria", "ha"]
positions = [f"pos{i}" for i in range(1, 5)]
segment_key = "{session}.{device}.{pid}.{index:03d}.{start}_{end}.wav"

min_speech = 0.5 * 16000


def get_segfile(session, device, pid):
    with open(
        segments_fstring.format(session=session, device=device, pid=pid), "r"
    ) as file:
        segments = list(csv.DictReader(file, fieldnames=["index", "start", "end"]))
    return segments


def get_score_file(system, device, ref_type, session="_", pid="_"):
    with open(
        results_fstring.format(
            system=system, device=device, ref_type=ref_type, session=session, pid=pid
        ),
        "r",
    ) as file:
        scores = list(csv.DictReader(file))
    return scores


def ownvoice_overlap():

    overlaps = {}
    overlap_arrays = {}

    for sess_data in sessions_file:
        for device in devices:
            dev_pos = "pos" + str(sess_data[f"{device}_pos"])

            if dev_pos == "pos":
                logging.warning(f"No {device} found for {sess_data['session']}")
                continue

            wearer = sess_data[dev_pos]
            wearer_segments = get_segfile(sess_data["session"], device, wearer)

            wearer_speech = np.zeros(int(wearer_segments[-1]["end"]), dtype=int)
            for seg in wearer_segments:
                start = int(seg["start"])
                end = int(seg["end"])
                wearer_speech[start:end] = 1

            overlap_arrays[sess_data["session"] + "." + device] = wearer_speech

            for pos in positions:
                if pos == dev_pos:
                    continue

                pid = sess_data[pos]
                segments = get_segfile(sess_data["session"], device, pid)

                for seg in segments:
                    start = int(seg["start"])
                    end = int(seg["end"])
                    length = end - start
                    if length < min_speech:
                        continue

                    key = segment_key.format(
                        session=sess_data["session"],
                        device=device,
                        pid=pid,
                        index=int(seg["index"]),
                        start=start,
                        end=end,
                    )
                    olap = wearer_speech[start:end].sum()

                    overlaps[key] = [int(olap), length]

    return overlaps, overlap_arrays


def process_overlaps():
    overlaps, _ = ownvoice_overlap()

    percents = [100 * a / b for a, b in overlaps.values()]

    plt.figure(figsize=[8, 8])
    plt.hist(percents, rwidth=0.95)
    plt.title("Wearer speech overlapping with target")
    plt.xlabel("Overlap (%)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("data/analysis/overlap_hist.png")
    plt.close()


def ownvoice_scores(target_metrics="stoi"):

    if isinstance(target_metrics, str):
        target_metrics = [target_metrics]

    _, overlap_arrays = ownvoice_overlap()

    ylims = {"stoi": [-0.3, 1], "pysepm_fwsegsnr": [-10, 15]}

    for system in systems:
        for device in devices:
            scores = get_score_file(system, device, "summed")

            overlap_percs = []
            targets = {met: [] for met in target_metrics}

            for score_row in scores:

                for met in target_metrics:
                    targets[met].append(float(score_row[met]))

                session, device, _, _, start_end, _ = score_row["key"].split(".")
                start, end = start_end.split("_")
                start = int(start)
                end = int(end)

                olap = overlap_arrays[f"{session}.{device}"][start:end].sum()
                overlap_percs.append(100 * olap / (end - start))

            for met in target_metrics:
                plt.figure(figsize=[8, 8])
                plt.scatter(overlap_percs, targets[met], marker="x", alpha=0.5)
                plt.xlabel("Overlap (%)")
                plt.ylabel(met)
                plt.title(f"{system} {device} {met}")

                if met in ylims:
                    plt.ylim(ylims[met])

                plt.savefig(f"data/analysis/{system}.{device}.{met}.scatter.png")
                plt.close()

                _, bins, patches = plt.hist(overlap_percs, rwidth=0.95, density=False)

                mets_bins = []
                for i in range(len(bins) - 1):
                    start, end = bins[i], bins[i + 1]
                    if start == 0:
                        start = -0.1
                    if end == 100:
                        end = 101
                    # Get the values of targets[met] where overlap_percs is between start and end
                    bin_targets = [
                        t
                        for o, t in zip(overlap_percs, targets[met])
                        if start < o <= end
                    ]
                    mets_bins.append(np.mean(bin_targets))

                mets_bins = [f"{m:.2f}" for m in mets_bins]
                plt.bar_label(patches, mets_bins)
                plt.title(f"{system} {device} {met}")
                plt.xlabel("Overlap (%)")
                plt.ylabel("Density")
                plt.savefig(f"data/analysis/{system}.{device}.{met}.hist.png")
                plt.close()


if __name__ == "__main__":
    ownvoice_scores(["stoi", "pysepm_fwsegsnr", "pesq"])
