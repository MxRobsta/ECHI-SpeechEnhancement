import csv
import hydra
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from sklearn.manifold import TSNE
import soundfile as sf
import torch

from shared.core_utils import get_device
from train_help.signal_prep import get_resemblyzer_array, get_rawnet_array

RNG = np.random.default_rng(666)
COSSIM = torch.nn.CosineSimilarity(dim=0)
MSE = torch.nn.MSELoss()


def rescale_emb(emb: torch.Tensor):
    """Scale embedding to range [0,1]"""

    mini = emb.min()
    maxi = emb.max()
    emb += mini
    emb /= maxi - mini
    return emb


def get_control_rainbow(rainbow_fpath, embed_type):
    audio, fs = sf.read(rainbow_fpath)
    import soxr

    audio = soxr.resample(audio, fs, 16000)
    audio_one, audio_two = np.split(audio, 2)

    if embed_type == "resemblyzer":
        embed1 = get_resemblyzer_array(audio_one)
        embed2 = get_resemblyzer_array(audio_two)
    elif embed_type == "rawnet":
        embed1 = get_rawnet_array(audio_one)
        embed2 = get_rawnet_array(audio_two)
    else:
        raise NotImplementedError(f"Embedding type {embed_type} not recognised")

    return embed1, embed2


def get_ref_embedding(
    embed_type, ref_audio_path, segment_info, segments=4, min_length=2
):

    audio, fs = sf.read(ref_audio_path)
    min_samples = min_length * fs
    min_total = 20 * fs
    max_total = 30 * fs

    possible_segments = [
        seg
        for seg in segment_info
        if int(seg["end"]) - int(seg["start"]) > min_samples
        and int(seg["end"]) < audio.shape[0]
    ]

    if len(possible_segments) <= segments:
        chosen_segments = possible_segments
    else:
        this_len = 0
        chosen_segments = []
        while this_len < min_total:
            newseg = possible_segments.pop(RNG.integers(0, len(possible_segments)))
            seglen = int(newseg["end"]) - int(newseg["start"])
            if this_len + seglen < max_total:
                this_len += seglen
                chosen_segments.append(newseg)

    ref_snippets = []
    for seg in chosen_segments:
        start = int(seg["start"])
        end = int(seg["end"])

        ref_snippets.append(audio[start:end])

    if embed_type == "resemblyzer":
        embed = get_resemblyzer_array(ref_snippets)
    elif embed_type == "rawnet":
        embed = get_rawnet_array(ref_snippets)
    else:
        raise NotImplementedError(f"Embedding type {embed_type} not recognised")

    return embed


def plot_confusion_matrix(
    session_pids, rows, cols, embeds, dist, embedding_type, fpath=None
):
    global COSSIM

    m = len(session_pids) // 2
    n = 2

    fig = plt.figure(figsize=(8, 4))
    gs = GridSpec(
        n, m + 1, figure=fig, width_ratios=[1] * m + [0.1], wspace=0.8, hspace=0.05
    )

    ims = []
    for plot_id, (sess, pids) in enumerate(session_pids.items()):
        thing = np.zeros([len(pids), len(pids)])
        for i, pid in enumerate(pids):
            rainbow_embed = embeds[rows][pid]
            for j, pid in enumerate(pids):
                ref_embed = embeds[cols][pid]

                if dist == "cossim":
                    this_score = COSSIM(rainbow_embed, ref_embed)
                elif dist == "mse":
                    this_score = -MSE(rainbow_embed, ref_embed)
                thing[i, j] = this_score.item()
        row = plot_id // (len(session_pids) // 2)
        col = plot_id % (len(session_pids) // 2)

        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(thing, cmap="viridis")  # , vmin=0.5, vmax=1)
        ax.set_xticks(range(3, -1, -1), labels=pids, rotation=90)
        ax.set_yticks(range(3, -1, -1), labels=pids)
        ax.set_title(sess)
        ims.append(im)

    cbar_ax = fig.add_subplot(gs[:, -1])  # full height of last column
    fig.colorbar(im, cax=cbar_ax)

    if dist == "mse":
        dist = "-mse"
    fig.suptitle(f"{embedding_type}: {dist} of {cols} vs {rows}")

    if fpath is not None:
        plt.savefig(fpath)

    plt.show()
    plt.close()


def plot_tsne(embeds):

    if isinstance(embeds, dict):
        embeds = list(embeds.values())
    embeds = np.stack([x.numpy() for x in embeds])

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    embeddings_2d = tsne.fit_transform(embeds)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="blue", s=50)

    # Label each point (optional)
    for i in range(embeddings_2d.shape[0]):
        plt.text(
            embeddings_2d[i, 0] + 0.5, embeddings_2d[i, 1] + 0.5, str(i), fontsize=9
        )

    plt.title("t-SNE of 40 Speaker Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


def rainbow_control(
    datasets: str | list[str],
    embedding_type: str,
    rainbow_ftemplate: str,
    sessions_ftemplate: str,
):
    torch_device = get_device()

    cossim = torch.nn.CosineSimilarity(dim=0).to(torch_device)
    mse = torch.nn.MSELoss().to(torch_device)

    if isinstance(datasets, str):
        datasets = [datasets]

    ds_pids = []
    session_pids = {}
    for dataset in datasets:
        with open(sessions_ftemplate.format(dataset=dataset), "r") as file:
            info = list(csv.DictReader(file))
        ds_pids += [
            (dataset, thing["session"], thing[f"pos{i}"])
            for thing in info
            for i in range(1, 5)
        ]

        for thing in info:
            session_pids[thing["session"]] = [thing[f"pos{i}"] for i in range(1, 5)]

    ds_pids = sorted(ds_pids, key=lambda x: x[2])

    scores = {}
    embeds = {"first": {}, "second": {}}

    for ds, sess, pid in ds_pids:
        rainbow_fpath = rainbow_ftemplate.format(
            rainbow_type="audio", dataset=ds, pid=pid
        )

        emb1, emb2 = get_control_rainbow(rainbow_fpath, embedding_type)
        embeds["first"][pid] = emb1
        embeds["second"][pid] = emb2

        scores[pid] = [cossim(emb1, emb2), mse(emb1, emb2)]

    mean_cossim = np.mean([x[0] for x in scores.values()])
    mean_nmse = np.mean([x[1] for x in scores.values()])

    print("SCORES")
    print(f"Cossim: {mean_cossim:.2f}")
    print(f"MSE:    {mean_nmse:.2f}")

    plot_confusion_matrix(
        session_pids,
        "first",
        "second",
        embeds,
        cossim,
        embedding_type,
        f"data/analysis/{embedding_type}_half.rainbow.png",
    )


def rainbow_similarity(
    datasets: str | list[str],
    embedding_type: str,
    rainbow_ftemplate: str,
    ref_audio_ftemplate: str,
    ref_embed_ftemplate: str,
    sessions_ftemplate: str,
    segments_ftemplate: str,
):
    torch_device = get_device()

    rainbow_ftemplate = rainbow_ftemplate.replace(".wav", ".pt")
    cossim = torch.nn.CosineSimilarity(dim=0).to(torch_device)
    mse = torch.nn.MSELoss().to(torch_device)

    if isinstance(datasets, str):
        datasets = [datasets]

    ds_pids = []
    session_pids = {}
    for dataset in datasets:
        with open(sessions_ftemplate.format(dataset=dataset), "r") as file:
            info = list(csv.DictReader(file))
        ds_pids += [
            (dataset, thing["session"], thing[f"pos{i}"])
            for thing in info
            for i in range(1, 5)
        ]

        for thing in info:
            session_pids[thing["session"]] = [thing[f"pos{i}"] for i in range(1, 5)]

    ds_pids = sorted(ds_pids, key=lambda x: x[2])

    scores = {}
    embeds = {"ref": {}, "rainbow": {}}

    for ds, sess, pid in ds_pids:
        rainbow_fpath = rainbow_ftemplate.format(
            enroltype=embedding_type, dataset=ds, pid=pid, enrolsource="rainbow"
        )
        rainbow_embed = torch.load(
            rainbow_fpath, weights_only=False, map_location=torch_device
        )

        ref_embed_file = ref_embed_ftemplate.format(
            enroltype=embedding_type, dataset=ds, pid=pid, enrolsource="dataocean"
        )
        if Path(ref_embed_file).exists():
            ref_embed = torch.load(
                ref_embed_file, weights_only=False, map_location=torch_device
            )
        else:
            ref_audio_fpath = ref_audio_ftemplate.format(
                dataset=ds, session=sess, device="ha", pid=pid
            )
            segments_file = segments_ftemplate.format(
                dataset=ds, session=sess, device="ha", pid=pid
            )
            with open(segments_file, "r") as file:
                segments = list(
                    csv.DictReader(file, fieldnames=["index", "start", "end"])
                )

            ref_embed = get_ref_embedding(embedding_type, ref_audio_fpath, segments)

            if not Path(ref_embed_file).parent.exists():
                Path(ref_embed_file).parent.mkdir(parents=True)

            torch.save(ref_embed, ref_embed_file)

        rainbow_embed = torch.nn.functional.normalize(rainbow_embed, p=2, dim=0)
        ref_embed = torch.nn.functional.normalize(ref_embed, p=2, dim=0)

        # rainbow_embed = rescale_emb(rainbow_embed)
        # ref_embed = rescale_emb(ref_embed)

        embeds["rainbow"][pid] = rainbow_embed
        embeds["ref"][pid] = ref_embed

        this_cossim = cossim(rainbow_embed, ref_embed)
        this_mse = mse(rainbow_embed, ref_embed)

        scores[pid] = [this_cossim, this_mse]

        # print(f"{pid}: {rainbow_embed.min()} {rainbow_embed.max()}")

    mean_cossim = np.mean([x[0] for x in scores.values()])
    mean_nmse = np.mean([x[1] for x in scores.values()])

    print("SCORES")
    print(f"Cossim: {mean_cossim:.2f}")
    print(f"MSE:    {mean_nmse}")

    # confusion matrix
    distance = "cossim"
    plot_confusion_matrix(
        session_pids,
        "rainbow",
        "ref",
        embeds,
        distance,
        embedding_type,
        f"data/analysis/spk_sim/{distance}_{embedding_type}_rainbow.vs.ref.png",
    )

    plot_confusion_matrix(
        session_pids,
        "ref",
        "ref",
        embeds,
        distance,
        embedding_type,
        f"data/analysis/spk_sim/{distance}_{embedding_type}_ref.vs.ref.png",
    )

    distance = "mse"
    plot_confusion_matrix(
        session_pids,
        "rainbow",
        "ref",
        embeds,
        distance,
        embedding_type,
        f"data/analysis/spk_sim/{distance}_{embedding_type}_rainbow.vs.ref.png",
    )

    plot_confusion_matrix(
        session_pids,
        "ref",
        "ref",
        embeds,
        distance,
        embedding_type,
        f"data/analysis/spk_sim/{distance}_{embedding_type}_ref.vs.ref.png",
    )

    # tsne plots
    # plot_tsne(embeds["rainbow"])


@hydra.main(version_base=None, config_path="../../config/analysis", config_name="main")
def main(cfg: DictConfig):
    rainbow_cfg = cfg.rainbow_similarity

    # rainbow_control(
    #     rainbow_cfg.dataset,
    #     rainbow_cfg.embedding_type,
    #     rainbow_cfg.rainbow_path,
    #     rainbow_cfg.sessions_file,
    # )

    rainbow_similarity(
        rainbow_cfg.dataset,
        rainbow_cfg.embedding_type,
        rainbow_cfg.rainbow_path,
        rainbow_cfg.ref_file,
        rainbow_cfg.ref_embedding_file,
        rainbow_cfg.sessions_file,
        rainbow_cfg.segments_file,
    )


if __name__ == "__main__":
    main()
