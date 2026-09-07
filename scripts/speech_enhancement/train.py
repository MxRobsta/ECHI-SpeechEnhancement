import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
from pystoi import stoi
import torch
from torch.utils.data.dataloader import DataLoader
import torchinfo

from shared.core_utils import (
    get_model,
    get_device,
    get_trainset,
    get_devset,
    get_dev_sessions,
)
from shared.signal_utils import STFTWrapper, match_length, prep_audio, pad_tolength

from train_help.gromit import Gromit
from train_help.losses import get_loss, get_lrmethod
from train_help.time_dataset import collate_time

torch.manual_seed(666)


def save_sample(
    sample_rate: int,
    processed: torch.Tensor,
    batch_scenes: list,
    save_scenes: list,
    split: str,
    epoch: int,
    noisy: torch.Tensor,
    target: torch.Tensor,
    gromit: Gromit,
):
    saves = list(set(batch_scenes) & set(save_scenes))
    if not saves:
        return None

    processed = processed.detach().cpu()
    if epoch == 0:
        noisy = noisy.detach().cpu()
        target = target.detach().cpu()
    for i, scene in enumerate(batch_scenes):
        if scene in save_scenes:
            gromit.save_sample(
                processed[i],
                sample_rate,
                split,
                epoch,
                scene,
                "proc",
            )
            if epoch == 0:
                gromit.save_sample(
                    noisy[i],
                    sample_rate,
                    split,
                    epoch,
                    scene,
                    "noisy",
                )
                gromit.save_sample(
                    target[i],
                    sample_rate,
                    split,
                    epoch,
                    scene,
                    "target",
                )


def check_lengths(
    scene: list[str],
    processed: torch.Tensor,
    target: torch.Tensor,
    split: str,
    do_stft: bool,
):
    use_val = True
    if processed.shape[-1] != target.shape[-1]:
        len_diff = abs(processed.shape[-1] - target.shape[-1])
        if not do_stft and len_diff > 1000:
            # Difference not due to stft
            logging.error(
                f"Time samples mismatch ({split}). Batch: {scene}. Proc: {processed.shape[-1]}. Targ: {target.shape[-1]}"
            )
            use_val = False
        processed, target = match_length(processed, target)
    return processed, target, use_val


def reject_batch(vad: torch.Tensor, min_prop: float, threshold: float):

    prop = vad.sum() / vad.numel()

    if prop > min_prop:
        return False
    return torch.rand(1) > threshold


def mix_target(
    noisy: torch.Tensor,
    target: torch.Tensor,
    current_epoch: int,
    max_epoch: int,
    rms_norm: float,
):

    _, nspk, _ = target.shape
    noisy = noisy[:, :1, :].repeat(1, nspk, 1)

    noisy_rms = noisy.square().mean(dim=(1, 2), keepdim=True).sqrt()
    target_rms = target.square().mean(dim=(1, 2), keepdim=True).sqrt()

    noisy_normed = (
        noisy * rms_norm * 2 / (noisy_rms + 1e-8)
    )  # Normalise noisy signal louder to account for noise
    target_normed = target * rms_norm / (target_rms + 1e-8)

    if current_epoch == 0:
        return noisy_normed
    elif current_epoch == max_epoch - 1:
        return target_normed

    mix_factor = current_epoch / max_epoch
    output = noisy_normed * (1 - mix_factor) + target_normed * mix_factor
    return output


def model_summary(model: torch.nn.Module):

    sample = model.get_test_sample()  # type: ignore

    torchinfo.summary(model, input_data=sample)


def print_batch(batch):
    for i, j in batch.items():
        if isinstance(j, torch.Tensor):
            print(i, j.shape)
        else:
            print(i, j)


def train(
    cfg,
    data_cfg,
    io_cfg,
    model_cfg,
    train_cfg,
    exp_dir,
    debug,
    wandb_entity=None,
    wandb_project=None,
):

    device = get_device()

    # Training helper
    gromit = Gromit(
        cfg,
        data_cfg.device,
        data_cfg.segment_type,
        train_cfg.epochs,
        train_cfg.loss.name,
        exp_dir,
        debug,
        wandb_entity,
        wandb_project,
    )

    # Model and training bits and bobs

    if model_cfg.input.type == "stft":
        do_stft = True
        stft = STFTWrapper(**model_cfg.input.stft, device=device)
    elif model_cfg.input.type != "wave":
        logging.error(f"Unrecognised model input type {model_cfg.input.type}")
    else:
        do_stft = False

    model = get_model(model_cfg, None)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    loss_fn, extras = get_loss(train_cfg.loss)
    use_aux = io_cfg.in_srcs > 0

    trainloader, trainsaves = get_trainset("train", data_cfg, debug)

    # Validation stuff
    dev_sessions = get_dev_sessions(
        data_cfg.sessions_file, data_cfg.device, "dev", debug
    )
    devset = get_devset("dev", data_cfg, debug)

    ckpt_interval = train_cfg.checkpoint_interval

    # Epoch progression stuff
    do_lrschedule = train_cfg.schedule_lr is not None
    lr_per_step = False
    if do_lrschedule:
        lr_scheduler, lr_per_step = get_lrmethod(
            train_cfg.schedule_lr.name,
            optimizer,
            len(trainloader),
            train_cfg.schedule_lr.params,
        )

    # Convenient definitions
    input_channels = model_cfg.input.channels
    input_sr = model_cfg.input.sample_rate
    input_rms = model_cfg.input.rms

    if debug:
        model_summary(model)

    model.to(device)
    loss_fn.to(device)

    gromit.start_training()

    trainset_len = len(trainloader)
    mini_report_check = max(trainset_len // 100, 1)

    # Update for debug
    if debug:
        epochs = 2
    else:
        epochs = train_cfg.epochs

    # Train this fine chap
    total_steps = 0
    for epoch in range(epochs):
        model.train()

        if debug:
            loader = tqdm(trainloader, desc="Training loop")
        else:
            loader = trainloader
        epoch_steps = 0
        for batch in loader:

            if debug and epoch_steps == 0 and epoch == 0:
                print_batch(batch)

            epoch_steps += 1
            total_steps += 1

            noisy_time = batch["noisy"].to(
                device, non_blocking=True
            )  # [batch, channels, time]
            aux = batch["aux"].to(device, non_blocking=True)  # [batch, (spks), time]
            target = batch["target"].to(
                device, non_blocking=True
            )  # [batch, (spks), time]
            noisy_time = prep_audio(
                noisy_time, batch["fs"], input_channels, input_sr, input_rms, True
            )
            target = prep_audio(
                target, batch["fs"], target.shape[1], input_sr, input_rms, True
            )

            if do_stft:
                noisy = stft(noisy_time)  # [batch, channels, time, freqs, 2]
                if use_aux and model_cfg.input.rainbow_type == "audio":
                    aux = stft(aux)  # [batch, (spks), time, freqs, 2]
                    batch["aux_lens"] = (
                        batch["aux_lens"] - stft.n_fft
                    ) // stft.hop_length
            else:
                noisy = noisy_time

            processed, spk_pred = model(noisy, aux, batch["aux_lens"])

            if do_stft:
                processed = stft.inverse(processed)

            processed, target, use_val = check_lengths(
                batch["id"], processed, target, "train", do_stft
            )
            if not use_val:
                continue

            loss_input = {"proc_speech": processed, "targ_speech": target}
            if "targ_vad" in extras:
                loss_input["targ_vad"] = batch["vad"].to(device)

            loss = loss_fn(**loss_input)  # type: torch.Tensor

            if loss.isnan():
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
            optimizer.step()

            gromit.train_loss.update(loss.detach())

            if epoch_steps % mini_report_check == 0 and epoch_steps > 0:
                gromit.mini_train_report(epoch + (epoch_steps / trainset_len))

            optimizer.zero_grad()

            if do_lrschedule and lr_per_step:
                lr_scheduler.step()

            save_sample(
                model_cfg.input.sample_rate,
                processed,
                batch["id"],
                trainsaves,
                "train",
                epoch,
                batch["noisy"],
                batch["target"],
                gromit,
            )

        do_checkpoint = (epoch % ckpt_interval == 0) or ((epoch + 1) == epochs)

        if do_checkpoint:
            logging.info("Running validation")
            thing = False
            model.eval()
            with torch.no_grad():
                for session in dev_sessions:
                    devset.make_manifest(session["session"])
                    output_store = devset.get_val_output(session["session"], debug)
                    ref_store = devset.get_val_output(session["session"], debug)
                    noisy_store = torch.zeros(
                        [
                            model_cfg.input.channels,
                            max(x.shape[-1] for x in output_store.values()),
                        ]
                    ).to(device)

                    for key in output_store:
                        output_store[key] = output_store[key].to(device)
                        ref_store[key] = ref_store[key].to(device)

                    loader = DataLoader(
                        devset, **data_cfg.loader.dev, collate_fn=collate_time
                    )

                    if debug:
                        loader = tqdm(loader, session["session"])

                    for batch in loader:
                        if debug and not thing and epoch == 0:
                            thing = True
                            print_batch(batch)
                        noisy = batch["noisy"].to(
                            device, non_blocking=True
                        )  # [batch, channels, time]
                        aux = batch["aux"].to(
                            device, non_blocking=True
                        )  # [batch, (spks), time]
                        target = batch["target"].to(
                            device, non_blocking=True
                        )  # [batch, (spks), time]
                        noisy = prep_audio(
                            noisy,
                            batch["fs"],
                            input_channels,
                            input_sr,
                            input_rms,
                            True,
                        )
                        target = prep_audio(
                            target,
                            batch["fs"],
                            target.shape[1],
                            input_sr,
                            input_rms,
                            True,
                        )

                        if do_stft:
                            noisy = stft(noisy)  # [batch, channels, time, freqs, 2]
                            if use_aux and model_cfg.input.rainbow_type == "audio":
                                aux = stft(aux)  # [batch, (spks), time, freqs, 2]
                                batch["aux_lens"] = (
                                    batch["aux_lens"] - stft.n_fft
                                ) // stft.hop_length

                        processed, spk_pred = model(noisy, aux, batch["aux_lens"])
                        # processed = processed.squeeze(1)

                        if do_stft:
                            processed = stft.inverse(processed)

                        loss_input = {"proc_speech": processed, "targ_speech": target}
                        if "targ_vad" in extras:
                            loss_input["targ_vad"] = batch["vad"].to(device)

                        loss = loss_fn(**loss_input)

                        if not loss.isnan():
                            gromit.val_loss.update(loss)

                        for i in range(noisy.shape[0]):
                            key = batch["key"][i]
                            start, end = batch["start"][i], batch["end"][i]
                            length = end - start

                            this_noisy = batch["noisy"][i].to(device)
                            if this_noisy.shape[-1] < length:
                                this_noisy = pad_tolength(this_noisy, length)
                            noisy_store[:, start:end] += this_noisy

                            this_proc = processed[i]
                            if this_proc.shape[-1] < length:
                                this_proc = pad_tolength(this_proc, length)
                            output_store[key][:, start:end] += this_proc

                            this_ref = target[i]
                            if this_ref.shape[-1] < length:
                                this_ref = pad_tolength(this_ref, length)
                            ref_store[key][:, start:end] += this_ref

                    # Sum channels for stoi computation
                    output_store = list(output_store.values())
                    ref_store = list(ref_store.values())

                    if len(output_store) == 1:
                        output_store = output_store[0]
                        ref_store = ref_store[0]
                    else:
                        output_store = torch.concat(output_store, dim=0)
                        ref_store = torch.concat(ref_store, dim=0)

                    if output_store.shape[0] == 4:
                        output_store = output_store[:3]
                        ref_store = ref_store[:3]

                    output_store = torch.sum(output_store, dim=0, keepdim=True)
                    ref_store = torch.sum(ref_store, dim=0, keepdim=True)

                    ref_store = ref_store.to("cpu").detach()
                    output_store = output_store.to("cpu").detach()

                    speech_segments, pids = devset.get_val_speech_segments(
                        session["session"]
                    )

                    for pid, start, end in speech_segments:

                        if end > output_store.shape[-1]:
                            break
                        this_src = output_store[:, start:end] / (
                            output_store[:, start:end].abs().max() + 1e-8
                        )
                        this_stoi = torch.tensor(
                            stoi(
                                ref_store[0, start:end].numpy(),
                                this_src[0].numpy(),
                                16000,
                            )
                        )
                        gromit.val_stoi.update(this_stoi.squeeze(0))

                    gromit.save_valsample(
                        epoch,
                        session["session"],
                        noisy_store,
                        output_store,
                        ref_store,
                        devset.audio_fs,
                    )

            if do_lrschedule and not lr_per_step:
                lr_scheduler.step(gromit.val_loss.get_average())  # type: ignore

        gromit.epoch_report(
            epoch, do_checkpoint, model, optimizer.param_groups[0]["lr"]
        )


@hydra.main(version_base=None, config_path="pkg://config/train", config_name="main")
def main(cfg: DictConfig) -> None:
    train(
        cfg,
        cfg.dataloading,
        cfg.iotypes,
        cfg.model,
        cfg.train,
        cfg.train_dir,
        cfg.debug,
        cfg.wandb.entity,
        cfg.wandb.project,
    )


if __name__ == "__main__":
    main()
