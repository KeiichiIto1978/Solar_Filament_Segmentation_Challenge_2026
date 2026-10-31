"""The final configuration, from the committed weights to a submission file.

Every piece already exists elsewhere in the package; this module fixes how they
are put together for the submission, so that the full-pipeline notebook and the
command line produce the same file from the same code:

1. the five fold models of HRNet-W32 U-Net at 1024 (``weights/``);
2. each model's probability averaged over the eight flips and quarter turns of
   the frame, then the five models averaged (Phases 7 and 8);
3. the map upsampled to 2048, thresholded at 0.40, fragments rejoined, small
   regions and regions of low mean probability dropped (Phases 9 and 10);
4. run-length encoding, and a check that no two masks of a frame share a pixel.

``FINAL_SETTING`` is the configuration of the main submission (pooled five-fold
PQ 0.4127). ``NO_SCORE_CUT_SETTING`` is the same without the score cut, the
configuration of Phase 9 (0.4065), kept as the second submission.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from filament.data.disk import detect_disk
from filament.data.image import FULL_SIZE, load_grayscale
from filament.evaluation import DIHEDRAL_VIEWS, View, predict_probability_tta
from filament.metrics.overlap import check_no_overlap
from filament.postprocess.search import Setting, predict_from_maps, resampled_builder
from filament.submit.rle import write_submission
from filament.training.loop import load_checkpoint

logger = logging.getLogger(__name__)

FOLDS = (0, 1, 2, 3, 4)
MODEL_SIZE = 1024
WEIGHT_NAME = "hrnet_w32_fold{fold}.pt"

# Distances are in pixels of a 1024 map; resampled_builder doubles them at 2048.
NO_SCORE_CUT_SETTING: Mapping[str, Any] = {
    "resolution": 2048,
    "threshold": 0.4,
    "min_area": 400,
    "join_gap": 16.0,
    "join_offset": 12.0,
}
FINAL_SETTING: Mapping[str, Any] = {**NO_SCORE_CUT_SETTING, "min_score": 0.65}


def load_fold_models(
    weights_dir: Path | str,
    folds: Sequence[int] = FOLDS,
    device: str | None = None,
) -> list[nn.Module]:
    """Load the fold models written by ``scripts/export_weights.py``.

    ``device`` defaults to the GPU when there is one.

    Each file is checked against the configuration stored inside it, so that a
    renamed or misplaced file cannot put the wrong fold or resolution into the
    ensemble without notice.

    Raises:
        FileNotFoundError: If a fold's file is missing.
        ValueError: If a file holds another fold or resolution.
    """
    models = []
    for fold in folds:
        path = Path(weights_dir) / WEIGHT_NAME.format(fold=fold)
        if not path.exists():
            raise FileNotFoundError(
                f"No weights for fold {fold} at {path}. Clone the repository with its "
                "weights/ directory, or regenerate them with scripts/export_weights.py."
            )
        model, stored = load_checkpoint(path, device=device)
        stored_fold, stored_size = int(str(stored.get("fold"))), int(str(stored["image_size"]))
        if stored_fold != fold or stored_size != MODEL_SIZE:
            raise ValueError(
                f"{path} holds fold {stored_fold} at {stored_size} px, "
                f"not fold {fold} at {MODEL_SIZE} px."
            )
        models.append(model)
    return models


def predict_ensemble_tta(
    models: Sequence[nn.Module],
    frame: np.ndarray,
    views: Iterable[View] = DIHEDRAL_VIEWS,
    size: int = MODEL_SIZE,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Mean over the models of each model's mean over the views, at ``size``.

    Raises:
        ValueError: If ``models`` is empty.
    """
    if not models:
        raise ValueError("An ensemble needs at least one model.")
    chosen = tuple(views)
    return np.mean(
        [predict_probability_tta(model, frame, chosen, size, device) for model in models],
        axis=0,
    )


def predict_maps(
    models: Sequence[nn.Module],
    frame_paths: Sequence[Path],
    device: torch.device | str = "cpu",
    log_every: int = 20,
) -> dict[str, np.ndarray]:
    """The ensemble's eight-view probability map for each frame, keyed by file stem."""
    maps = {}
    started = time.perf_counter()
    for position, path in enumerate(frame_paths, start=1):
        maps[path.stem] = predict_ensemble_tta(models, load_grayscale(path), device=device)
        if log_every and position % log_every == 0:
            logger.info(
                "%d of %d frames, %.1f s per frame.",
                position,
                len(frame_paths),
                (time.perf_counter() - started) / position,
            )
    return maps


def submission_from_maps(
    maps: Mapping[str, np.ndarray],
    images_dir: Path | str,
    setting: Mapping[str, Any],
    output: Path | str,
) -> pd.DataFrame:
    """Post-process the maps under ``setting`` and write a checked submission CSV.

    The solar disk of each frame is found in the frame itself, in ``images_dir``,
    and predictions outside it are removed.

    Returns:
        The rows written.

    Raises:
        SubmissionOverlapError: If two masks of one frame share a pixel;
            Kaggle would reject the file and still count the attempt.
    """
    size = next(iter(maps.values())).shape[0]
    disks = {
        stem: detect_disk(load_grayscale(Path(images_dir) / f"{stem}.jpeg")).scaled(
            size / FULL_SIZE
        )
        for stem in maps
    }
    rows = predict_from_maps(maps, Setting(dict(setting)), disks=disks, build=resampled_builder)
    write_submission(rows, output)
    check_no_overlap(output)
    return rows
