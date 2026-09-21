"""Recovering filament instances from a semantic probability map.

The chain is: threshold the probability map, split what remains into connected
regions, drop the sky, drop regions that are too small, and scale back up to
2048. Every step is deliberate.

*Connected regions instead of an instance head.* Two regions that are connected
are one region, so the masks cannot overlap and the submission cannot be
rejected for it. The cost is that two filaments lying close together fuse into
one and a thin filament breaks into two, and Panoptic Quality punishes both.
Counting how often each happens is the main thing Phase 1 is for.

*A minimum area.* Under this metric a prediction that misses is worse than no
prediction: a near miss raises a false positive and a false negative, adding
1.0 to the denominator, while staying silent raises only the false negative and
adds 0.5. Emitting a region is worth it only when its chance of clearing IoU
0.5 times the IoU it would then reach exceeds about half the current PQ -- near
PQ 0.4 and a typical IoU of 0.7, roughly a 29% chance. Tiny regions are nowhere
near that, so they are dropped. ``min_area`` is measured at full resolution and
is one of the parameters Phase 3 tunes against PQ directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from filament.data.disk import DEFAULT_MASK_MARGIN, Disk
from filament.postprocess.join import (
    DEFAULT_MAX_ANGLE,
    DEFAULT_MAX_OFFSET,
    connected_and_joined,
)
from filament.submit.rle import FULL_HEIGHT, mask_to_rle

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
# In pixels of a 2048x2048 frame. The 10th percentile of an annotated filament
# is 410 px, so 200 keeps every plausible detection while removing specks.
DEFAULT_MIN_AREA = 200
# 8-connectivity: a filament one pixel wide that steps diagonally is one
# filament, not a chain of separate ones.
CONNECTIVITY = 8


@dataclass(frozen=True)
class Instance:
    """One predicted filament."""

    mask: np.ndarray
    score: float
    area: int

    def to_rle(self) -> str:
        """The counts string this instance contributes to a submission."""
        return mask_to_rle(self.mask)


def extract_instances(
    probability: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    disk: Disk | None = None,
    disk_margin: float = DEFAULT_MASK_MARGIN,
    output_size: int = FULL_HEIGHT,
    join_gap: float = 0.0,
    join_angle: float = DEFAULT_MAX_ANGLE,
    join_offset: float = DEFAULT_MAX_OFFSET,
) -> list[Instance]:
    """Split a probability map into non-overlapping filament masks.

    Args:
        probability: ``(H, W)`` array of probabilities in ``[0, 1]``, at
            whatever resolution the model ran at.
        threshold: Probability above which a pixel counts as filament.
        min_area: Smallest area to keep, in pixels of the output resolution.
        disk: Solar disk **in the coordinates of** ``probability``. Regions
            outside it are dropped. ``None`` keeps everything.
        disk_margin: Slack around the limb, in the same coordinates.
        output_size: Side length of the returned masks. Predictions are made at
            a reduced size and scored at 2048.
        join_gap: Rejoin regions up to this far apart, in pixels of the map.
            Zero switches the rejoining off, which is the plain behaviour.
        join_angle: Largest angle between two regions' long axes, in degrees.
        join_offset: How far one region may sit off the other's axis.

    Returns:
        Instances ordered by descending score. Their masks never overlap.
    """
    if probability.ndim != 2:
        raise ValueError(f"Expected a 2-D probability map, got shape {probability.shape}.")

    binary = (probability >= threshold).astype(np.uint8)
    if disk is not None:
        binary &= disk.mask(*binary.shape, margin=disk_margin).astype(np.uint8)

    count, labels = connected_and_joined(
        binary,
        connectivity=CONNECTIVITY,
        max_gap=join_gap,
        max_angle=join_angle,
        max_offset=join_offset,
    )
    instances = []
    for label in range(1, count):
        region = labels == label
        # The mean probability over the region is the only score available
        # here; a semantic model gives no per-instance confidence.
        score = float(probability[region].mean())
        mask = _to_output_size(region, output_size)
        area = int(mask.sum())
        if area < min_area:
            continue
        instances.append(Instance(mask=mask, score=score, area=area))

    instances.sort(key=lambda item: -item.score)
    logger.debug(
        "Kept %d of %d connected regions at threshold %.2f and min_area %d.",
        len(instances),
        max(count - 1, 0),
        threshold,
        min_area,
    )
    return instances


def _to_output_size(region: np.ndarray, output_size: int) -> np.ndarray:
    """Scale one region up to the submission resolution.

    Nearest-neighbour keeps the mask binary, and scaling each region on its own
    rather than the label image keeps two neighbours from bleeding into each
    other and producing an overlap.
    """
    if region.shape == (output_size, output_size):
        return region.astype(bool)
    scaled = cv2.resize(
        region.astype(np.uint8),
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return scaled.astype(bool)


def instances_to_rows(image_id: str, instances: list[Instance]) -> list[tuple[str, str]]:
    """Submission rows for one image, numbered from one in score order."""
    return [
        (f"{image_id}_{index}", instance.to_rle())
        for index, instance in enumerate(instances, start=1)
    ]


def fusion_and_splitting(
    predicted: list[np.ndarray],
    ground_truth: list[np.ndarray],
    overlap_fraction: float = 0.25,
) -> tuple[int, int]:
    """Count fused and split filaments in one annotator-image.

    Connected regions fail in two ways that Panoptic Quality punishes without
    distinguishing them, and the fix differs: fusion needs the instances pulled
    apart, splitting needs them joined. Counting them separately is what tells
    Phase 2 which to work on.

    A pair counts as overlapping when the shared area is at least
    ``overlap_fraction`` of the smaller mask, which ignores masks that merely
    touch.

    Args:
        predicted: Predicted masks for one annotator-image.
        ground_truth: That annotator's masks.
        overlap_fraction: How much of the smaller mask must be shared.

    Returns:
        The number of predictions covering more than one ground truth (fusion),
        and the number of ground truths covered by more than one prediction
        (splitting).
    """
    if not predicted or not ground_truth:
        return 0, 0

    hits = np.zeros((len(ground_truth), len(predicted)), dtype=bool)
    for gt_index, gt_mask in enumerate(ground_truth):
        gt_area = int(gt_mask.sum())
        for pred_index, pred_mask in enumerate(predicted):
            shared = int(np.count_nonzero(gt_mask & pred_mask))
            smaller = min(gt_area, int(pred_mask.sum()))
            if smaller and shared >= overlap_fraction * smaller:
                hits[gt_index, pred_index] = True

    fused = int((hits.sum(axis=0) > 1).sum())
    split = int((hits.sum(axis=1) > 1).sum())
    return fused, split
