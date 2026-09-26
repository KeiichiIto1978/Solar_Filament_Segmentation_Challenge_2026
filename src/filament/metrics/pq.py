"""Panoptic Quality, as the competition computes it.

    PQ = sum(IoU over true positives) / (|TP| + 0.5|FP| + 0.5|FN|)

Three properties of this metric drive the design of everything downstream.

1. Only pairs with ``IoU > 0.5`` are true positives. A prediction at IoU 0.49
   raises both a false positive and a false negative, adding 1.0 to the
   denominator and nothing to the numerator: strictly worse than not
   predicting at all, which only raises a false negative (+0.5).
2. Matching is not one-to-one; every pair above the threshold is counted. With
   non-overlapping predictions, two of them cannot both exceed IoU 0.5 against
   the same ground truth, so in practice the matching is one-to-one anyway.
3. The loop runs over annotator-images, not images. An image annotated by
   three people is scored three times against three different ground truths,
   so a filament only one annotator drew is a true positive once and a false
   positive twice.

The result is a :class:`PQResult` rather than a single number: ``PQ = SQ * RQ``
separates mask quality (SQ, the mean IoU of the true positives) from detection
quality (RQ, an F1 over the same matches), and the TP/FP/FN counts say which of
the two to work on.

Two interchangeable backends compute the IoU matrix. ``"rle"``, the default,
lets pycocotools work on the encoded masks and never decodes one. ``"dense"``
is the literal reading of the specification, kept as the reference the fast
path is tested against.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils

from filament.submit.rle import FULL_HEIGHT, FULL_WIDTH, rle_to_mask

logger = logging.getLogger(__name__)

# A pair counts as a true positive only strictly above this value.
IOU_THRESHOLD = 0.5

Backend = Literal["rle", "dense"]
DEFAULT_BACKEND: Backend = "rle"


@dataclass(frozen=True)
class PQResult:
    """Panoptic Quality together with the breakdown needed to act on it."""

    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    iou_scores: list[float] = field(default_factory=list)
    dice_scores: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"PQ={self.pq:.4f} SQ={self.sq:.4f} RQ={self.rq:.4f} "
            f"TP={self.tp} FP={self.fp} FN={self.fn}"
        )


def iou_dice_matrices(
    gt_masks: np.ndarray, pred_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise IoU and Dice between two stacks of binary masks.

    Args:
        gt_masks: Boolean array of shape ``(n_gt, height, width)``.
        pred_masks: Boolean array of shape ``(n_pred, height, width)``.

    Returns:
        Two ``(n_gt, n_pred)`` float arrays. Pairs whose union is empty score 0.
    """
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    if n_gt == 0 or n_pred == 0:
        empty = np.zeros((n_gt, n_pred), dtype=float)
        return empty, empty.copy()

    gt_flat = gt_masks.reshape(n_gt, -1)
    pred_flat = pred_masks.reshape(n_pred, -1)
    gt_areas = gt_flat.sum(axis=1, dtype=np.int64)
    pred_areas = pred_flat.sum(axis=1, dtype=np.int64)

    # One ground-truth row at a time rather than a single matrix product: the
    # product of two uint8 operands would overflow, and widening them would
    # allocate n_gt * n_pred * 4M elements.
    intersection = np.empty((n_gt, n_pred), dtype=np.int64)
    for index, gt_row in enumerate(gt_flat):
        intersection[index] = np.count_nonzero(gt_row & pred_flat, axis=1)

    area_sums = gt_areas[:, None] + pred_areas[None, :]
    union = area_sums - intersection

    iou = np.divide(
        intersection,
        union,
        out=np.zeros(intersection.shape, dtype=float),
        where=union > 0,
    )
    dice = np.divide(
        2 * intersection,
        area_sums,
        out=np.zeros(intersection.shape, dtype=float),
        where=area_sums > 0,
    )
    return iou, dice


def iou_dice_matrices_rle(
    gt_rles: list[str],
    pred_rles: list[str],
    height: int = FULL_HEIGHT,
    width: int = FULL_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise IoU and Dice computed on the encoded masks.

    Equivalent to :func:`iou_dice_matrices` but roughly two orders of magnitude
    faster, because pycocotools intersects the run-length encodings in C
    instead of allocating one 2048x2048 array per mask.

    Dice follows from IoU exactly. With ``iou = I / (A + B - I)``, the
    intersection is ``I = iou * (A + B) / (1 + iou)``, so
    ``dice = 2I / (A + B) = 2 * iou / (1 + iou)`` and the areas cancel.

    Args:
        gt_rles: Ground-truth counts strings.
        pred_rles: Predicted counts strings.
        height: Mask height; 2048 for this dataset.
        width: Mask width; 2048 for this dataset.

    Returns:
        Two ``(n_gt, n_pred)`` float arrays. Pairs whose union is empty score 0.
    """
    if not gt_rles or not pred_rles:
        empty = np.zeros((len(gt_rles), len(pred_rles)), dtype=float)
        return empty, empty.copy()

    def encode(counts: list[str]) -> list[dict[str, object]]:
        return [{"size": [height, width], "counts": item.encode("ascii")} for item in counts]

    # iou(dt, gt, iscrowd) returns a (len(dt), len(gt)) matrix; passing the
    # ground truth as dt keeps the orientation the caller expects.
    iou = np.asarray(
        mask_utils.iou(encode(gt_rles), encode(pred_rles), [0] * len(pred_rles)),
        dtype=float,
    ).reshape(len(gt_rles), len(pred_rles))
    return iou, 2.0 * iou / (1.0 + iou)


def group_by_image(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Group RLE strings by the part of ``filament_id`` before the first ``_``.

    For ground truth that key is the annotator-image, ``<annotator>-<stem>``;
    for predictions it is the image id, ``<stem>``.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    columns = zip(frame["filament_id"], frame["segmentation_rle"], strict=True)
    for filament_id, counts in columns:
        grouped[str(filament_id).split("_", 1)[0]].append(str(counts))
    return dict(grouped)


def annotator_image_to_image(annotator_image: str) -> str:
    """Drop the annotator prefix: ``<annotator>-<stem>`` becomes ``<stem>``."""
    _, separator, stem = annotator_image.partition("-")
    if not separator:
        raise ValueError(
            f"Malformed ground-truth image id {annotator_image!r}: "
            "expected an '<annotator>-<stem>' pair."
        )
    return stem


def compute_pq(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = FULL_HEIGHT,
    width: int = FULL_WIDTH,
    annotator_images: Iterable[str] | None = None,
    backend: Backend = DEFAULT_BACKEND,
) -> PQResult:
    """Score predictions against ground truth.

    Args:
        gt_df: Columns ``filament_id`` (``<annotator>-<stem>_<n>``) and
            ``segmentation_rle``.
        pred_df: The same columns, with ``filament_id`` of the form
            ``<stem>_<n>``.
        height: Mask height; 2048 for this dataset.
        width: Mask width; 2048 for this dataset.
        annotator_images: Annotator-images to score. Defaults to those present
            in ``gt_df``. Pass it explicitly to also score an annotator-image
            that has no ground truth, where every prediction is a false
            positive.
        backend: ``"rle"`` scores the encoded masks directly; ``"dense"``
            decodes them first. They agree to within floating-point error.

    Returns:
        The score and its breakdown.
    """
    if backend not in ("rle", "dense"):
        raise ValueError(f"Unknown backend {backend!r}; expected 'rle' or 'dense'.")
    gt_by_annotator_image = group_by_image(gt_df)
    pred_by_image = group_by_image(pred_df)

    scored = (
        sorted(gt_by_annotator_image) if annotator_images is None else sorted(set(annotator_images))
    )
    scored_images = {annotator_image_to_image(key) for key in scored}

    unscored = set(pred_by_image) - scored_images
    if unscored:
        # Dropping these silently would hide false positives, so say so.
        logger.warning(
            "%d predicted image(s) have no ground truth and are not scored: %s",
            len(unscored),
            ", ".join(sorted(unscored)[:5]),
        )

    tp_ious: list[float] = []
    tp_dices: list[float] = []
    false_positives = 0
    false_negatives = 0

    for annotator_image in scored:
        image_id = annotator_image_to_image(annotator_image)
        gt_rles = gt_by_annotator_image.get(annotator_image, [])
        pred_rles = pred_by_image.get(image_id, [])

        if not gt_rles:
            false_positives += len(pred_rles)
            continue
        if not pred_rles:
            false_negatives += len(gt_rles)
            continue

        if backend == "rle":
            iou, dice = iou_dice_matrices_rle(gt_rles, pred_rles, height, width)
        else:
            gt_masks = np.stack([rle_to_mask(counts, height, width) for counts in gt_rles])
            pred_masks = np.stack([rle_to_mask(counts, height, width) for counts in pred_rles])
            iou, dice = iou_dice_matrices(gt_masks, pred_masks)

        hit = iou > IOU_THRESHOLD
        tp_ious.extend(iou[hit].tolist())
        tp_dices.extend(dice[hit].tolist())
        false_positives += int((hit.sum(axis=0) == 0).sum())
        false_negatives += int((hit.sum(axis=1) == 0).sum())

    return _assemble(tp_ious, tp_dices, false_positives, false_negatives)


def _assemble(tp_ious: list[float], tp_dices: list[float], fp: int, fn: int) -> PQResult:
    """Turn the accumulated matches into PQ, SQ and RQ."""
    tp = len(tp_ious)
    denominator = tp + 0.5 * fp + 0.5 * fn
    if denominator == 0:
        return PQResult(pq=0.0, sq=0.0, rq=0.0, tp=0, fp=fp, fn=fn)

    # PQ = SQ * RQ by construction: (sum(iou) / tp) * (tp / denominator).
    sq = float(np.mean(tp_ious)) if tp else 0.0
    rq = tp / denominator
    return PQResult(
        pq=float(sum(tp_ious) / denominator),
        sq=sq,
        rq=rq,
        tp=tp,
        fp=fp,
        fn=fn,
        iou_scores=tp_ious,
        dice_scores=tp_dices,
    )


def pool_pq(parts: Iterable[PQResult]) -> PQResult:
    """Combine scores of disjoint evaluation sets into one, as if scored together.

    Cross-validation folds cover different frames, so their matches and misses
    simply add up. Averaging the per-fold PQ instead would weight a fold with
    few filaments as heavily as one with many.

    Only the counts and SQ are used: the sum of matched IoUs is recovered as
    ``SQ * TP``, which is exact because SQ is their mean. That lets a fold
    scored in an earlier session, and kept only as a summary, be pooled with
    fresh ones. The returned result carries no per-match lists.

    Args:
        parts: Scores of evaluation sets that share no annotator-image.

    Returns:
        The pooled score.
    """
    parts = list(parts)
    tp = sum(part.tp for part in parts)
    fp = sum(part.fp for part in parts)
    fn = sum(part.fn for part in parts)
    denominator = tp + 0.5 * fp + 0.5 * fn
    if denominator == 0:
        return PQResult(pq=0.0, sq=0.0, rq=0.0, tp=0, fp=fp, fn=fn)
    iou_sum = sum(part.sq * part.tp for part in parts)
    return PQResult(
        pq=iou_sum / denominator,
        sq=iou_sum / tp if tp else 0.0,
        rq=tp / denominator,
        tp=tp,
        fp=fp,
        fn=fn,
    )
