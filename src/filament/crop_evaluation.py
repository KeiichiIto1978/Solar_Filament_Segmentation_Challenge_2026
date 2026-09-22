"""Scoring a crop model against the boxes it was given.

Panoptic Quality cannot be computed here and should not be: with the annotated
boxes handed over, finding the filaments is free, and a score that includes
detection would mostly be measuring the gift. What the crop model is being
asked is narrower -- given the right box, how well can it draw? -- and the
answer is the IoU of each mask against the annotation whose box it was given.

That number is directly comparable to the whole-disk model's SQ, which is the
mean IoU over the pairs it matched, and to the best IoU the whole-disk model
reaches on each annotation whether or not it matched. The second comparison is
the fairer one: it asks both models to draw the same filaments.

The share reaching IoU 0.5 is reported alongside, because that is the gate the
metric applies. A run that lifts the mean without moving that share has not
changed what the score would be.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from filament.data.coco import Dataset
from filament.data.crops import (
    DEFAULT_CONTEXT,
    DEFAULT_CROP_SIZE,
    DEFAULT_SEED_PADDING,
    Box,
    build_input,
    crop_window,
    cut,
)
from filament.data.image import load_grayscale

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropScore:
    """One annotation, and how well it was drawn from its own box."""

    stem: str
    image_id: str
    annotation_id: str
    iou: float
    area: int


@dataclass(frozen=True)
class CropResult:
    """What a crop model achieved over a set of annotations."""

    scores: list[CropScore]
    seconds: float

    @property
    def ious(self) -> np.ndarray:
        return np.array([score.iou for score in self.scores])

    @property
    def mean_iou(self) -> float:
        return float(self.ious.mean()) if self.scores else 0.0

    @property
    def median_iou(self) -> float:
        return float(np.median(self.ious)) if self.scores else 0.0

    @property
    def share_over_half(self) -> float:
        """Share reaching the IoU the metric gates on."""
        return float((self.ious > 0.5).mean()) if self.scores else 0.0

    @property
    def mean_iou_over_half(self) -> float:
        """Mean IoU among those that clear the gate, comparable with SQ."""
        clearing = self.ious[self.ious > 0.5]
        return float(clearing.mean()) if clearing.size else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "annotations": len(self.scores),
            "mean_iou": round(self.mean_iou, 4),
            "median_iou": round(self.median_iou, 4),
            "share_over_half": round(self.share_over_half, 4),
            "mean_iou_over_half": round(self.mean_iou_over_half, 4),
            "seconds": round(self.seconds, 1),
        }

    def __str__(self) -> str:
        return (
            f"mean IoU {self.mean_iou:.4f}  median {self.median_iou:.4f}  "
            f"over 0.5 {self.share_over_half:.1%}  "
            f"mean of those {self.mean_iou_over_half:.4f}  "
            f"n={len(self.scores)}"
        )


@torch.no_grad()
def draw_from_box(
    model: nn.Module,
    frame: np.ndarray,
    box: Box,
    size: int = DEFAULT_CROP_SIZE,
    context: float = DEFAULT_CONTEXT,
    seed_padding: float = DEFAULT_SEED_PADDING,
    threshold: float = 0.5,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, Box]:
    """Predict one filament's mask from its box.

    Returns:
        The mask in the crop's coordinates, and the window it came from, which
        is what places it back in the frame.
    """
    window = crop_window(box, frame.shape[0], context)
    prepared = build_input(frame, box, window, size, seed_padding=seed_padding)
    batch = torch.from_numpy(prepared)[None].to(device)
    probability = torch.sigmoid(model(batch))[0, 0].detach().cpu().numpy()
    return probability >= threshold, window


def score_crops(
    model: nn.Module,
    dataset: Dataset,
    images_dir: Path | str,
    stems: Iterable[str],
    size: int = DEFAULT_CROP_SIZE,
    context: float = DEFAULT_CONTEXT,
    seed_padding: float = DEFAULT_SEED_PADDING,
    threshold: float = 0.5,
    device: torch.device | str = "cpu",
    log_every: int = 200,
) -> CropResult:
    """Draw every annotation of ``stems`` from its own box and score it.

    The mask is compared in the crop's own coordinates rather than being
    pasted back to 2048. Both masks are cut the same way, so the comparison is
    fair, and it avoids charging the model for the resampling that putting it
    back would add -- this is an upper bound and should not carry avoidable
    losses.
    """
    subset = dataset.subset(set(stems))
    started = time.perf_counter()
    scores: list[CropScore] = []

    for position, entry in enumerate(subset.annotator_images, start=1):
        frame = load_grayscale(Path(images_dir) / entry.file_name)
        for annotation in entry.annotations:
            truth = annotation.to_mask(entry.height, entry.width)
            if not truth.any():
                continue
            rows = np.flatnonzero(truth.any(axis=1))
            columns = np.flatnonzero(truth.any(axis=0))
            box = Box(int(rows[0]), int(columns[0]), int(rows[-1]) + 1, int(columns[-1]) + 1)

            predicted, window = draw_from_box(
                model, frame, box, size, context, seed_padding, threshold, device
            )
            reference = cut(truth.astype(np.uint8), window, size, mask=True).astype(bool)
            union = np.count_nonzero(predicted | reference)
            scores.append(
                CropScore(
                    stem=entry.stem,
                    image_id=entry.image_id,
                    annotation_id=str(annotation.annotation_id),
                    iou=float(np.count_nonzero(predicted & reference) / union) if union else 0.0,
                    area=int(truth.sum()),
                )
            )
        if log_every and position % log_every == 0:
            logger.info("%d/%d frames scored.", position, len(subset.annotator_images))

    result = CropResult(scores=scores, seconds=time.perf_counter() - started)
    logger.info("%s", result)
    return result
