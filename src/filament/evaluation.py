"""Running the baseline over a set of frames and scoring the result.

Reports more than PQ. PQ alone says a run is worse without saying what to fix,
and the failures behind it call for opposite remedies:

- many false negatives mean filaments are being missed, so the threshold or the
  model's sensitivity is wrong;
- many false positives mean the cut-offs are too loose;
- a healthy count of matches with SQ stuck near 0.55 means the masks themselves
  are too rough.

On top of the PQ breakdown, fusion and splitting are counted separately.
Connected regions merge neighbouring filaments and break thin ones, and PQ
charges for both without distinguishing them, yet fusion needs instances pulled
apart while splitting needs them joined. Those two counts are what decide
whether the two-stage design of Phase 2 earns its cost.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from filament.data.coco import AnnotatorImage, Dataset
from filament.data.dataset import DEFAULT_IMAGE_SIZE
from filament.data.disk import DEFAULT_MASK_MARGIN, detect_disk
from filament.data.image import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    load_grayscale,
    to_model_input,
)
from filament.metrics.pq import PQResult, compute_pq
from filament.postprocess.instances import (
    DEFAULT_MIN_AREA,
    DEFAULT_THRESHOLD,
    Instance,
    extract_instances,
    fusion_and_splitting,
    instances_to_rows,
)
from filament.postprocess.join import DEFAULT_MAX_ANGLE, DEFAULT_MAX_OFFSET
from filament.postprocess.search import save_map
from filament.submit.rle import FULL_HEIGHT, SUBMISSION_COLUMNS, masks_to_gt_df

logger = logging.getLogger(__name__)

# Fusion and splitting measure coarse overlap, so they are counted on
# quarter-size masks. At 2048 the pairwise comparisons would cost about a
# minute per hundred frames without changing the counts.
COUNT_SIZE = 512


@dataclass(frozen=True)
class EvaluationResult:
    """A score, its breakdown, and the settings that produced it."""

    pq: PQResult
    fused: int
    split: int
    stems: int
    annotator_images: int
    predictions: int
    seconds: float
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-writable summary, for the lab notebook."""
        return {
            "pq": round(self.pq.pq, 4),
            "sq": round(self.pq.sq, 4),
            "rq": round(self.pq.rq, 4),
            "tp": self.pq.tp,
            "fp": self.pq.fp,
            "fn": self.pq.fn,
            "fused": self.fused,
            "split": self.split,
            "stems": self.stems,
            "annotator_images": self.annotator_images,
            "predictions": self.predictions,
            "iou_mean": self._mean(self.pq.iou_scores),
            "dice_mean": self._mean(self.pq.dice_scores),
            "seconds": round(self.seconds, 1),
            "settings": self.settings,
        }

    @staticmethod
    def _mean(values: list[float]) -> float:
        return round(float(np.mean(values)), 4) if values else 0.0

    def __str__(self) -> str:
        return (
            f"{self.pq}  fused={self.fused} split={self.split}  "
            f"predictions={self.predictions}  {self.seconds:.1f}s"
        )


@torch.no_grad()
def predict_probability(
    model: nn.Module,
    frame: np.ndarray,
    size: int = DEFAULT_IMAGE_SIZE,
    device: torch.device | str = "cpu",
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Probability of filament per pixel, at ``size`` resolution."""
    prepared = to_model_input(frame, size, clahe_clip_limit, clahe_tile_grid)
    batch = torch.from_numpy(prepared)[None].to(device)
    return torch.sigmoid(model(batch))[0, 0].detach().cpu().numpy()


def predict_frame(
    model: nn.Module,
    frame_path: Path | str,
    size: int = DEFAULT_IMAGE_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    device: torch.device | str = "cpu",
    mask_disk: bool = True,
    output_size: int = FULL_HEIGHT,
    join_gap: float = 0.0,
    join_angle: float = DEFAULT_MAX_ANGLE,
    join_offset: float = DEFAULT_MAX_OFFSET,
) -> list[Instance]:
    """Predict the filaments of one frame, as non-overlapping masks.

    Args:
        model: Trained network, in evaluation mode.
        frame_path: Image file.
        size: Resolution the model runs at.
        threshold: Probability cut-off.
        min_area: Smallest region to emit, in pixels at ``output_size``.
        device: Device to run on.
        mask_disk: Discard whatever falls outside the solar disk.
        output_size: Side length of the returned masks.
    """
    frame = load_grayscale(frame_path)
    probability = predict_probability(model, frame, size, device)

    disk = None
    scale = size / frame.shape[0]
    if mask_disk:
        disk = detect_disk(frame).scaled(scale)
    return extract_instances(
        probability,
        threshold=threshold,
        min_area=min_area,
        disk=disk,
        disk_margin=DEFAULT_MASK_MARGIN * scale,
        output_size=output_size,
        join_gap=join_gap,
        join_angle=join_angle,
        join_offset=join_offset,
    )


def write_probability_maps(
    model: nn.Module,
    frame_paths: Iterable[Path | str],
    out_dir: Path | str,
    size: int = DEFAULT_IMAGE_SIZE,
    device: torch.device | str = "cpu",
    log_every: int = 40,
) -> list[Path]:
    """Run the model over frames and store each probability map as ``uint8``.

    Everything after the forward pass -- thresholding, joining fragments,
    filtering by area, scoring -- reads only these maps, so writing them once
    is what stops the rest of the work needing a GPU. The files are named after
    the frame, which is the key the sweep and the submission both index by.

    Args:
        model: Trained network, in evaluation mode.
        frame_paths: Frames to run.
        out_dir: Directory to write into. Created if missing.
        size: Resolution the model runs at.
        device: Device to run on.
        log_every: Report progress every this many frames.

    Returns:
        The files written, in the order the frames were given.
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    started = time.perf_counter()
    for position, frame_path in enumerate(frame_paths, start=1):
        path = Path(frame_path)
        probability = predict_probability(model, load_grayscale(path), size, device)
        written.append(save_map(probability, destination / f"{path.stem}.npy"))
        if log_every and position % log_every == 0:
            logger.info("%d frames written.", position)

    logger.info(
        "Wrote %d probability maps to %s in %.1fs.",
        len(written),
        destination,
        time.perf_counter() - started,
    )
    return written


def shrink(mask: np.ndarray, size: int = COUNT_SIZE) -> np.ndarray:
    """Downscale a boolean mask, keeping it boolean."""
    if mask.shape == (size, size):
        return mask
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(
        bool
    )


def annotated_masks(entry: AnnotatorImage, size: int = COUNT_SIZE) -> list[np.ndarray]:
    """Each filament one annotator drew, as its own downscaled mask."""
    return [
        shrink(annotation.to_mask(entry.height, entry.width), size)
        for annotation in entry.annotations
    ]


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    images_dir: Path | str,
    stems: list[str],
    size: int = DEFAULT_IMAGE_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    device: torch.device | str = "cpu",
    mask_disk: bool = True,
    gt_df: pd.DataFrame | None = None,
    count_fusion: bool = True,
) -> tuple[EvaluationResult, pd.DataFrame]:
    """Predict every frame of ``stems`` and score it against the annotations.

    Args:
        model: Trained network, in evaluation mode.
        dataset: Loaded annotations.
        images_dir: Directory holding the frames.
        stems: Image stems to evaluate, normally one fold's validation side.
        size: Resolution the model runs at.
        threshold: Probability cut-off.
        min_area: Smallest region to emit, in pixels at 2048.
        device: Device to run on.
        mask_disk: Discard predictions outside the solar disk.
        gt_df: Ground truth already in submission form. Encoding it costs about
            as long as the inference and never changes, so a tuning sweep
            should build it once and pass it in.
        count_fusion: Count fused and split filaments. Costs a few seconds.

    Returns:
        The result, and the predictions as a submission-shaped DataFrame.
    """
    subset = dataset.subset(set(stems))
    truth = gt_df if gt_df is not None else masks_to_gt_df(dataset, stems)

    started = time.perf_counter()
    rows: list[tuple[str, str]] = []
    predicted: dict[str, list[np.ndarray]] = {}
    for stem in sorted(set(stems)):
        instances = predict_frame(
            model,
            Path(images_dir) / f"{stem}.jpeg",
            size=size,
            threshold=threshold,
            min_area=min_area,
            device=device,
            mask_disk=mask_disk,
        )
        rows.extend(instances_to_rows(stem, instances))
        if count_fusion:
            predicted[stem] = [shrink(instance.mask) for instance in instances]
    elapsed = time.perf_counter() - started

    pred_df = pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))
    pq = compute_pq(truth, pred_df)

    fused = split = 0
    if count_fusion:
        for entry in subset.annotator_images:
            entry_fused, entry_split = fusion_and_splitting(
                predicted.get(entry.stem, []), annotated_masks(entry)
            )
            fused += entry_fused
            split += entry_split

    result = EvaluationResult(
        pq=pq,
        fused=fused,
        split=split,
        stems=len(set(stems)),
        annotator_images=len(subset.annotator_images),
        predictions=len(pred_df),
        seconds=elapsed,
        settings={
            "size": size,
            "threshold": threshold,
            "min_area": min_area,
            "mask_disk": mask_disk,
            "device": str(device),
        },
    )
    logger.info("%s", result)
    return result, pred_df
