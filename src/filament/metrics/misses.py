"""Why an annotated filament was missed.

PQ counts a false negative wherever no emitted prediction clears IoU 0.5
against an annotated filament, and says nothing about why. The reasons call for
different remedies, and only some of them need a GPU:

- the model put no probability there at all, or too little to cross the
  threshold -- something only training can change;
- a candidate matched it but was then removed by the area or score cut-off --
  the post-processing threw it away;
- the candidate covering it was too large (fused with a neighbour, or spilling
  over), or covered too little of it (cut short, or broken into pieces) -- the
  way pixels are grouped into filaments;
- the shapes overlap substantially but not enough.

:func:`classify_misses` sorts each annotated filament into one of these from
quantities measured once per annotator-image, so that the same rule applies to
every fold and can be tested on cases built by hand.
"""

from __future__ import annotations

import numpy as np

from filament.metrics.pq import IOU_THRESHOLD

# Below this peak probability the model is taken not to have seen the filament
# at all; the maps are close to binary, so almost nothing lies between this and
# any usable threshold.
UNSEEN_PEAK = 0.05
# A candidate covering at least this share of a filament covers "most" of it,
# and one with at least this share inside the filament is "mostly" filament.
MOST = 0.5

MATCHED = "matched"
UNSEEN = "unseen"
BELOW_THRESHOLD = "below_threshold"
DROPPED_AREA = "dropped_area"
DROPPED_SCORE = "dropped_score"
TOO_LARGE = "too_large"
FRAGMENTED = "fragmented"
TOO_SMALL = "too_small"
SHAPE = "shape"

# Every outcome, in the order of the pipeline stage it points at.
CATEGORIES = (
    MATCHED,
    UNSEEN,
    BELOW_THRESHOLD,
    DROPPED_AREA,
    DROPPED_SCORE,
    TOO_LARGE,
    FRAGMENTED,
    TOO_SMALL,
    SHAPE,
)

# Why a candidate was not emitted: not dropped, or the cut-off that dropped it.
KEPT = ""
BY_AREA = "area"
BY_SCORE = "score"


def classify_misses(
    iou: np.ndarray,
    intersection: np.ndarray,
    gt_areas: np.ndarray,
    candidate_areas: np.ndarray,
    dropped_by: list[str],
    peaks: np.ndarray,
    threshold: float,
) -> list[str]:
    """The outcome of each annotated filament of one annotator-image.

    Candidates are every region the post-processing produced *before* the
    area and score cut-offs; ``dropped_by`` says which of them were then
    removed. Since the cut-offs act on each region alone, the emitted
    predictions are exactly the candidates with ``dropped_by == KEPT``.

    Args:
        iou: ``(n_gt, n_candidates)`` IoU of every filament with every candidate.
        intersection: The same pairs' overlap in pixels.
        gt_areas: Area of each filament, in pixels.
        candidate_areas: Area of each candidate, in pixels.
        dropped_by: Per candidate, :data:`KEPT`, :data:`BY_AREA` or
            :data:`BY_SCORE`.
        peaks: Highest probability inside each filament.
        threshold: The probability threshold the candidates were cut at.

    Returns:
        One of :data:`CATEGORIES` per filament. Everything but
        :data:`MATCHED` is a false negative.
    """
    n_gt, n_candidates = iou.shape
    if intersection.shape != iou.shape or len(dropped_by) != n_candidates:
        raise ValueError("The IoU, the overlaps and the candidate list disagree in size.")
    kept = np.array([reason == KEPT for reason in dropped_by], dtype=bool)

    outcomes = []
    for row in range(n_gt):
        hits = iou[row] > IOU_THRESHOLD
        if (hits & kept).any():
            outcomes.append(MATCHED)
        elif peaks[row] < UNSEEN_PEAK:
            outcomes.append(UNSEEN)
        elif peaks[row] < threshold:
            outcomes.append(BELOW_THRESHOLD)
        elif hits.any():
            # A candidate would have matched; at most one can, since
            # candidates do not overlap. Area is the first cut-off applied.
            reason = dropped_by[int(np.argmax(hits))]
            outcomes.append(DROPPED_AREA if reason == BY_AREA else DROPPED_SCORE)
        else:
            outcomes.append(_grouping_failure(intersection[row], gt_areas[row], candidate_areas))
    return outcomes


def _grouping_failure(overlaps: np.ndarray, gt_area: float, candidate_areas: np.ndarray) -> str:
    """Why the pixels that are there did not form a match."""
    if overlaps.size == 0 or overlaps.max() <= 0:
        return TOO_SMALL
    best = int(np.argmax(overlaps))
    coverage = overlaps[best] / gt_area
    purity = overlaps[best] / candidate_areas[best]
    if coverage >= MOST and purity < MOST:
        return TOO_LARGE
    if coverage < MOST:
        return FRAGMENTED if overlaps.sum() / gt_area >= MOST else TOO_SMALL
    return SHAPE
