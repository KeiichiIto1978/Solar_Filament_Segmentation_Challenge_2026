"""Describing a predicted filament before it is scored, to judge whether to emit it.

Phase 10 found that most false positives are not near misses but detections no
annotator drew, and that the mean probability over a region separates matches
from misses only moderately. A candidate carries more evidence than that one
number: its shape, how its probability is spread, and above all whether the
frame is actually dark where it lies -- in H-alpha a filament is darker than
the chromosphere around it, and the model's map is only a guess at that.

These features are computed at the submission resolution, from the frame and
the probability map the instance was cut from. What decides from them is a
separate step; nothing here drops anything.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from filament.data.disk import Disk
from filament.postprocess.instances import Instance

# The ring around a candidate that its brightness is compared with, in pixels
# of a 2048 frame. Wide enough to reach past the blur of the filament's edge,
# narrow enough to stay on the same part of the disk, whose brightness falls
# off towards the limb.
RING_WIDTH = 15

# Features of one candidate, in the order a classifier sees them. The count of
# candidates in the frame is added by the caller, which sees the whole frame.
FEATURES = (
    "score",
    "peak",
    "probability_spread",
    "confident_share",
    "area",
    "radial_position",
    "elongation",
    "extent",
    "compactness",
    "brightness_contrast",
    "frame_candidates",
)

# A pixel at or above this probability counts as confidently filament.
CONFIDENT = 0.8


def instance_features(
    instance: Instance,
    probability: np.ndarray,
    frame: np.ndarray,
    disk: Disk,
) -> dict[str, float]:
    """What is known about one candidate before it is scored.

    Args:
        instance: The candidate; its mask is at the submission resolution.
        probability: The probability map at the same resolution as the mask.
        frame: The grayscale frame at the same resolution.
        disk: The solar disk in the same coordinates.

    Returns:
        Every entry of :data:`FEATURES` except ``frame_candidates``:

        - ``score``, ``peak``, ``probability_spread``, ``confident_share``: the
          mean, maximum and standard deviation of the probability over the
          region, and the share of its pixels at or above :data:`CONFIDENT`;
        - ``area`` in pixels and ``radial_position``, the distance of its
          centroid from the disk's centre as a fraction of the radius;
        - ``elongation``, the ratio of the long to the short axis of the
          region's second moments (1 for a disk, large for a thin line);
          ``extent``, its share of its bounding box; ``compactness``, its
          perimeter squared over ``4 pi`` times its area (1 for a disk);
        - ``brightness_contrast``, the mean brightness inside the region minus
          that of a ring around it, over the ring's: negative where the frame
          is darker than its surroundings, as a filament is.

    Raises:
        ValueError: If the mask is empty or the arrays disagree in shape.
    """
    mask = instance.mask
    if mask.shape != probability.shape or mask.shape != frame.shape:
        raise ValueError(
            f"Mask {mask.shape}, probability {probability.shape} and frame {frame.shape} "
            "must be at the same resolution."
        )
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("An empty mask has no features.")

    # Work inside the bounding box, padded for the ring: a full frame is
    # four million pixels and a candidate a few thousand.
    pad = RING_WIDTH + 2
    top = max(int(rows.min()) - pad, 0)
    bottom = min(int(rows.max()) + pad + 1, mask.shape[0])
    left = max(int(columns.min()) - pad, 0)
    right = min(int(columns.max()) + pad + 1, mask.shape[1])
    region = mask[top:bottom, left:right]
    values = probability[top:bottom, left:right][region]

    centre = math.hypot(columns.mean() - disk.center_x, rows.mean() - disk.center_y)
    return {
        "score": float(values.mean()),
        "peak": float(values.max()),
        "probability_spread": float(values.std()),
        "confident_share": float((values >= CONFIDENT).mean()),
        "area": float(rows.size),
        "radial_position": float(centre / disk.radius),
        "elongation": _elongation(rows, columns),
        "extent": float(rows.size / ((np.ptp(rows) + 1) * (np.ptp(columns) + 1))),
        "compactness": _compactness(region),
        "brightness_contrast": _brightness_contrast(
            region, frame[top:bottom, left:right], disk, top, left
        ),
    }


def _elongation(rows: np.ndarray, columns: np.ndarray) -> float:
    """Ratio of the long to the short axis of the second moments."""
    if rows.size < 2:
        return 1.0
    covariance = np.cov(np.stack([rows, columns]).astype(np.float64))
    small, large = np.linalg.eigvalsh(covariance)
    # A region one pixel wide has no spread across it; a twelfth of a pixel
    # squared is the variance of a single pixel's extent, which keeps the
    # ratio finite and comparable for the thinnest filaments.
    return float(math.sqrt(large / max(small, 1.0 / 12.0)))


def _compactness(region: np.ndarray) -> float:
    """Perimeter squared over ``4 pi`` times the area; 1 for a disk."""
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = sum(cv2.arcLength(contour, closed=True) for contour in contours)
    return float(perimeter**2 / (4.0 * math.pi * region.sum()))


def _brightness_contrast(
    region: np.ndarray, frame: np.ndarray, disk: Disk, top: int, left: int
) -> float:
    """Inside brightness minus that of a surrounding ring, over the ring's."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_WIDTH + 1,) * 2)
    grown = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)
    ring = grown & ~region
    # Off the disk is sky, far darker than any filament; comparing with it
    # would make every candidate near the limb look bright.
    rows, columns = np.mgrid[top : top + region.shape[0], left : left + region.shape[1]]
    ring &= np.hypot(columns - disk.center_x, rows - disk.center_y) <= disk.radius
    if not ring.any():
        return 0.0
    inside = float(frame[region].mean())
    outside = float(frame[ring].mean())
    return (inside - outside) / max(outside, 1.0)


def frame_features(candidates: list[dict[str, Any]]) -> None:
    """Add the features that depend on the whole frame, in place."""
    for candidate in candidates:
        candidate["frame_candidates"] = float(len(candidates))
