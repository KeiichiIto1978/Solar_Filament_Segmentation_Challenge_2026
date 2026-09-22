"""The training loop for the semantic baseline.

Kept out of the command line script so that it can be exercised by a test, and
so that the Kaggle notebook can call it directly instead of holding a second
copy of the code.

The best checkpoint is chosen by validation loss, not by PQ. Computing PQ needs
the whole post-processing chain and the full-resolution masks, which is too slow
to do every epoch; the two are correlated well enough to pick a checkpoint, and
PQ is then measured once on the chosen one. Where the two disagree is itself
worth recording.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from filament.data.coco import Dataset, load_annotations
from filament.data.dataset import Augmentation, FilamentSegmentationDataset
from filament.data.split import load_fold
from filament.models.unet import DiceBceLoss, UNetConfig, build_model
from filament.paths import ProjectPaths, load_paths
from filament.training.config import TrainConfig

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "best.pt"
CONFIG_NAME = "config.yaml"
HISTORY_NAME = "history.json"


@dataclass(frozen=True)
class EpochResult:
    """What one epoch cost and how long it took."""

    epoch: int
    train_loss: float
    val_loss: float
    seconds: float


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a run."""

    checkpoint: Path
    config_path: Path
    history: list[EpochResult]

    @property
    def best_val_loss(self) -> float:
        return min(item.val_loss for item in self.history)

    @property
    def best_epoch(self) -> int:
        return min(self.history, key=lambda item: item.val_loss).epoch


def seed_everything(seed: int) -> None:
    """Fix every random source the loop draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick the device to train on, preferring a GPU when one is present."""
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(
    config: TrainConfig,
    dataset: Dataset,
    paths: ProjectPaths,
) -> tuple[DataLoader, DataLoader]:
    """Training and validation loaders for one fold of the frozen split."""
    fold = load_fold(config.fold, paths.splits_dir)
    common = {
        "dataset": dataset,
        "images_dir": paths.train_images,
        "size": config.image_size,
        "vote_targets": config.vote_targets,
    }
    train_set = FilamentSegmentationDataset(
        stems=fold.train,
        augmentation=Augmentation(**vars(config.augmentation)),
        seed=config.seed,
        **common,
    )
    val_set = FilamentSegmentationDataset(stems=fold.val, augmentation=None, **common)
    logger.info(
        "Fold %d: %d training and %d validation annotator-images.",
        config.fold,
        len(train_set),
        len(val_set),
    )
    return (
        DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=len(train_set) > config.batch_size,
        ),
        DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    max_batches: int | None,
) -> float:
    """One pass over ``loader``. Trains when ``optimiser`` is given."""
    training = optimiser is not None
    model.train(training)

    total = 0.0
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            autocast = torch.amp.autocast(device.type, enabled=scaler is not None)
            with autocast:
                loss = criterion(model(image), target)

        if training:
            assert optimiser is not None
            optimiser.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()

        total += loss.detach().item()
        batches += 1

    if batches == 0:
        raise ValueError("The loader yielded no batches; check the split and batch size.")
    return total / batches


def train(
    config: TrainConfig,
    paths: ProjectPaths | None = None,
    device: str | None = None,
) -> TrainResult:
    """Train the baseline on one fold and keep the best checkpoint.

    Args:
        config: Resolved configuration.
        paths: Project paths. Defaults to ``configs/paths.yaml``.
        device: Device override, mainly for tests.

    Returns:
        Where the checkpoint landed and what each epoch cost.
    """
    resolved_paths = (paths or load_paths()).require_dataset()
    seed_everything(config.seed)

    target_device = resolve_device(device)
    use_amp = config.amp and target_device.type == "cuda"
    logger.info("Training on %s (mixed precision: %s).", target_device, use_amp)

    dataset = load_annotations(resolved_paths.train_annotations)
    train_loader, val_loader = build_loaders(config, dataset, resolved_paths)

    model = build_model(
        UNetConfig(encoder_name=config.encoder, encoder_weights=config.encoder_weights)
    ).to(target_device)
    criterion = DiceBceLoss(
        dice_weight=config.loss.dice_weight,
        bce_weight=config.loss.bce_weight,
        dice_majority=config.loss.dice_majority,
        cldice_weight=config.loss.cldice_weight,
        cldice_iterations=config.loss.cldice_iterations,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)
    scaler = torch.amp.GradScaler(target_device.type) if use_amp else None

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.save(config.output_dir / CONFIG_NAME)
    checkpoint_path = config.output_dir / CHECKPOINT_NAME

    history: list[EpochResult] = []
    best = float("inf")
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            target_device,
            optimiser,
            scaler,
            config.max_train_batches,
        )
        val_loss = _run_epoch(
            model, val_loader, criterion, target_device, None, None, config.max_val_batches
        )
        schedule.step()

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            seconds=time.perf_counter() - started,
        )
        history.append(result)

        marker = ""
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            marker = " (best, saved)"
        logger.info(
            "Epoch %d/%d  train %.4f  val %.4f  %.1fs%s",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            result.seconds,
            marker,
        )

    return TrainResult(checkpoint=checkpoint_path, config_path=config_path, history=history)


def load_checkpoint(
    path: Path | str, device: str | None = None
) -> tuple[nn.Module, dict[str, object]]:
    """Rebuild a trained model from a checkpoint.

    Returns:
        The model in evaluation mode, and the configuration it was trained with.
    """
    target_device = resolve_device(device)
    payload = torch.load(Path(path), map_location=target_device, weights_only=False)
    stored = dict(payload["config"])
    model = build_model(
        UNetConfig(
            encoder_name=str(stored["encoder"]),
            # Pretrained weights are irrelevant here: the checkpoint replaces them.
            encoder_weights=None,
        )
    )
    model.load_state_dict(payload["model"])
    model.to(target_device).eval()
    return model, stored
