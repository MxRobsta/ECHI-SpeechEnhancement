import logging
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import torch
import torchaudio
import wandb

from shared.file_utils import read_json, write_json


class LossTracker:
    def __init__(self):
        self.loss = torch.tensor(0.0)
        self.steps = 0
        self.history = []

    def update(self, loss: torch.Tensor):
        self.loss += loss.to(self.loss.device)
        self.steps += 1

    def get_average(self) -> float:
        return self.loss.item() / self.steps if self.steps > 0 else 0

    def reset(self, epoch):
        self.history.append((epoch, self.get_average()))
        self.loss = torch.tensor(0.0)
        self.steps = 0


class Gromit:
    def __init__(
        self,
        config,
        audio_device,
        segment_types,
        epochs,
        loss_name,
        output_path,
        debug,
        wandb_entity,
        wandb_project,
    ):
        self.config = config
        self.exp_name = config.shared.exp_name
        self.debug = debug
        self.audio_device = audio_device
        self.epochs = epochs
        self.loss_name = loss_name

        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        self.use_wandb = (
            (not self.debug)
            and (self.wandb_entity is not None)
            and (self.wandb_project is not None)
        )

        self.train_loss = LossTracker()
        self.val_loss = LossTracker()
        self.val_stoi = LossTracker()

        self.output_path = Path(output_path)
        self.json_name = self.output_path / "train_log.json"

        self.train_sampledir = self.output_path / "train_samples"
        self.dev_sampledir = self.output_path / "val_samples"
        self.train_sampledir.mkdir(parents=True, exist_ok=True)
        self.dev_sampledir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.output_path / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.is_summed = isinstance(segment_types, str) and "summed" in segment_types
        if self.is_summed:
            self.log_pref = "summed"
        else:
            self.log_pref = "indi"

    def start_training(self):
        write_json(self.json_name, [])

        if self.use_wandb:
            wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.exp_name,
                dir=self.output_path,
                config=OmegaConf.to_container(self.config, resolve=True),  # type: ignore
            )

        logging.info("Training")

    def epoch_report(
        self,
        epoch,
        do_ckpt,
        model: torch.nn.Module,
        lr: float,
        delim=" ",
    ):

        train_log = read_json(self.json_name)
        new_log = {
            "epoch": epoch,
            "train_loss": np.nan,
            "val_loss": np.nan,
            "val_stoi": np.nan,
            "lr": lr,
        }

        wlog = {"completion": epoch + 1}

        train_loss = self.train_loss.get_average()
        out_string = f"{delim}Epoch {epoch+1} report"
        out_string += f"{delim}Train loss: {train_loss:.2f}"
        self.train_loss.reset(epoch)

        wlog[f"{self.log_pref}_train_{self.loss_name}"] = train_loss
        new_log["train_loss"] = train_loss

        if do_ckpt:
            val_loss = self.val_loss.get_average()
            val_stoi = self.val_stoi.get_average()
            out_string += f"{delim}Val loss: {val_loss:.2f}"
            out_string += f"{delim}Val stoi: {val_stoi:.2f}"
            self.val_loss.reset(epoch)
            self.val_stoi.reset(epoch)

            wlog[f"{self.log_pref}_val_{self.loss_name}"] = val_loss
            wlog[f"{self.log_pref}_new_stoi"] = val_stoi
            wlog["lr"] = lr

            new_log["val_loss"] = val_loss
            new_log["val_stoi"] = val_stoi

            ckpt_path = self.get_ckpt_path(epoch)
            torch.save(model.state_dict(), ckpt_path)

        out_string += f"{delim}LR: {lr}"

        logging.info(out_string)
        train_log.append(new_log)
        write_json(self.json_name, train_log)

        if self.use_wandb:
            wandb.log(wlog)

    def mini_train_report(self, completion):
        if self.use_wandb:
            wandb.log(
                {
                    f"step_train_{self.loss_name}": self.train_loss.get_average(),
                    "completion": completion,
                }
            )

    def save_sample(
        self,
        sample: torch.Tensor,
        fs: int,
        split: str,
        epoch: int,
        scene: str,
        audio_type: str,
    ):
        if split == "train":
            audio_dir = self.train_sampledir
        elif split == "dev":
            audio_dir = self.dev_sampledir
        else:
            raise ValueError(f"Unknown split: {split}")

        audio_fname = f"{scene}_{audio_type}.wav"
        if audio_type == "proc":
            audio_fname = f"epoch{str(epoch).zfill(3)}_" + audio_fname
        audio_path = str(audio_dir / audio_fname)

        if sample.ndim == 1:
            sample = sample.unsqueeze(0)
        elif sample.ndim == 3:
            sample = sample.squeeze(0)

        torchaudio.save(audio_path, sample, fs)

    def save_valsample(self, epoch, session, noisy, processed, ref, fs):
        start = processed.shape[-1] // 2
        end = start + fs * 60

        audio_fname = f"{session}.epoch{epoch:03g}.{start}.{end}.wav"

        if isinstance(noisy, np.ndarray):
            noisy = torch.from_numpy(noisy)
        if isinstance(processed, np.ndarray):
            processed = torch.from_numpy(processed)
        if isinstance(ref, np.ndarray):
            ref = torch.from_numpy(ref)

        if processed.ndim == 1:
            processed = processed.unsqueeze(0)
        torchaudio.save(
            self.dev_sampledir / audio_fname, processed[:, start:end].detach().cpu(), fs
        )

        if epoch == 0:
            for at, audio in zip(["noisy", "ref"], [noisy, ref]):
                audio_fname = f"{session}.{at}.{start}.{end}.wav"
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
                torchaudio.save(
                    self.dev_sampledir / audio_fname,
                    audio[:, start:end].detach().cpu(),
                    fs,
                )

    def get_ckpt_path(self, epoch):
        return self.ckpt_dir / f"epoch{str(epoch).zfill(3)}.pt"
