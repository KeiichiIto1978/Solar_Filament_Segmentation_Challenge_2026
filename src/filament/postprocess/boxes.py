"""Deciding what counts as one filament by box rather than by connectivity.

Instances have so far been whatever the thresholded probability map leaves
connected, which ties two questions together that are not the same. Whether a
pixel is filament, and which filament it belongs to, both come out of one
threshold, so the only thing available to tune is how generously pixels join up
-- and turning that up unbreaks filaments while merging neighbours.

A box separates them. Read at a low threshold the map is generous and its
connected regions tend to cover a whole filament, breaks and all; the box drawn
around one of those regions says where a filament is. Read at a higher
threshold inside that box, the map says which pixels are the filament. One
threshold for finding, another for drawing.

What makes this worth trying is measured: reading the existing maps inside the
*annotated* boxes lifts the share of annotations reaching IoU 0.5 from 53.9% to
68.6%, which is PQ 0.465 against 0.3756 -- without touching the pixels. That is
an upper bound, since the annotated boxes are not something any method
delivers. This module is the cheapest attempt at the same thing: boxes taken
from the map that is already saved.

The gain rests entirely on the boxes being right. Everything the old
post-processing did -- discarding small regions, taking whatever was connected
-- was standing in for the information a box carries, so a box switches those
rules off. A wrong box with the rules off is worse than no box at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from filament.data.crops import Box
from filament.data.disk import DEFAULT_MASK_MARGIN, Disk

logger = logging.getLogger(__name__)

# Threshold for finding filaments. Lower than the one for drawing them: a
# generous reading joins a filament that the drawing threshold breaks.
DEFAULT_FIND_THRESHOLD = 0.2

# Threshold for drawing, inside a box that has already been accepted.
DEFAULT_DRAW_THRESHOLD = 0.5

# Smallest box to accept, as the area of its own region at the finding
# threshold, in pixels of the map. Far below the old minimum area, because a
# box no longer has to guess whether a small region is noise -- that is what
# the confirming threshold is for.
DEFAULT_MIN_BOX_AREA = 12

# Boxes are grown by this fraction before the mask is read, so that a filament
# extending slightly past its generous reading is not clipped.
DEFAULT_BOX_PADDING = 0.15


@dataclass(frozen=True)
class BoxCandidate:
    """One proposed filament: where it is, and how strongly it was proposed."""

    box: Box
    score: float
    found_area: int

    def padded(self, fraction: float, limit: int) -> Box:
        """The box grown by ``fraction`` of its own size, clamped to the frame."""
        grow_rows = max(round(self.box.height * fraction / 2), 1)
        grow_columns = max(round(self.box.width * fraction / 2), 1)
        return Box(
            max(self.box.top - grow_rows, 0),
            max(self.box.left - grow_columns, 0),
            min(self.box.bottom + grow_rows, limit),
            min(self.box.right + grow_columns, limit),
        )


def propose_boxes(
    probability: np.ndarray,
    find_threshold: float = DEFAULT_FIND_THRESHOLD,
    min_box_area: int = DEFAULT_MIN_BOX_AREA,
    disk: Disk | None = None,
    disk_margin: float = DEFAULT_MASK_MARGIN,
) -> list[BoxCandidate]:
    """Boxes around the connected regions of a generously thresholded map.

    Args:
        probability: ``(H, W)`` map in ``[0, 1]``, at the model's resolution.
        find_threshold: Probability above which a pixel may belong to a
            filament being looked for.
        min_box_area: Smallest region, in pixels of the map, to propose at all.
        disk: Solar disk in the map's coordinates. Regions outside are dropped.
        disk_margin: Slack around the limb.

    Returns:
        Candidates ordered by descending score, which is the mean probability
        over the region -- the only confidence a semantic map offers.
    """
    if probability.ndim != 2:
        raise ValueError(f"Expected a 2-D probability map, got shape {probability.shape}.")

    found = (probability >= find_threshold).astype(np.uint8)
    if disk is not None:
        found &= disk.mask(*found.shape, margin=disk_margin).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(found, connectivity=8)
    candidates: list[BoxCandidate] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_box_area:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        box = Box(
            top,
            left,
            top + int(stats[label, cv2.CC_STAT_HEIGHT]),
            left + int(stats[label, cv2.CC_STAT_WIDTH]),
        )
        candidates.append(
            BoxCandidate(
                box=box,
                score=float(probability[labels == label].mean()),
                found_area=area,
            )
        )

    candidates.sort(key=lambda candidate: -candidate.score)
    logger.debug(
        "Proposed %d of %d regions at threshold %.2f.",
        len(candidates),
        max(count - 1, 0),
        find_threshold,
    )
    return candidates


def masks_from_boxes(
    probability: np.ndarray,
    candidates: list[BoxCandidate],
    draw_threshold: float = DEFAULT_DRAW_THRESHOLD,
    padding: float = DEFAULT_BOX_PADDING,
    min_area: int = 1,
    output_size: int | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Read the map inside each box, strongest box first, without overlapping.

    Each box claims only pixels no stronger box has taken. That is what keeps
    the submission valid -- two masks sharing a pixel are rejected -- and it is
    also what settles the boundary between two filaments running close
    together: the more confident one keeps it.

    Args:
        probability: The same map the boxes came from.
        candidates: Boxes, in descending order of score.
        draw_threshold: Probability above which a pixel inside a box is
            filament.
        padding: Fraction by which a box is grown before being read.
        min_area: Smallest mask to emit, in pixels of the output. A box that
            has been eaten by stronger neighbours can fall below it.
        output_size: Side length to scale the masks up to. ``None`` keeps them
            at the map's resolution.

    Returns:
        Each mask and the score of the box it came from, strongest first.
    """
    size = probability.shape[0]
    drawn = probability >= draw_threshold
    claimed = np.zeros_like(drawn)

    masks: list[tuple[np.ndarray, float]] = []
    for candidate in candidates:
        box = candidate.padded(padding, size)
        window = np.zeros_like(drawn)
        window[box.top : box.bottom, box.left : box.right] = True

        mask = drawn & window & ~claimed
        if not mask.any():
            continue
        claimed |= mask

        if output_size is not None and output_size != size:
            scaled = cv2.resize(
                mask.astype(np.uint8), (output_size, output_size), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            scaled = mask
        if int(scaled.sum()) < min_area:
            continue
        masks.append((scaled, candidate.score))

    return masks


def extract_by_box(
    probability: np.ndarray,
    find_threshold: float = DEFAULT_FIND_THRESHOLD,
    draw_threshold: float = DEFAULT_DRAW_THRESHOLD,
    min_box_area: int = DEFAULT_MIN_BOX_AREA,
    padding: float = DEFAULT_BOX_PADDING,
    min_area: int = 1,
    disk: Disk | None = None,
    disk_margin: float = DEFAULT_MASK_MARGIN,
    output_size: int | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Find filaments at one threshold and draw them at another.

    The whole chain, so that a sweep can call it with one setting.
    """
    candidates = propose_boxes(probability, find_threshold, min_box_area, disk, disk_margin)
    return masks_from_boxes(probability, candidates, draw_threshold, padding, min_area, output_size)
