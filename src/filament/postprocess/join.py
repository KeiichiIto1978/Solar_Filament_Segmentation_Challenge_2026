"""Rejoining the fragments of one filament.

Recovering instances from connected regions splits a filament wherever the
probability dips below the threshold along its length. On fold 0 that happens
156 times against 19 fusions, and the count barely moved between a 40-epoch
and a 15-epoch model, so it is a property of the representation rather than of
how long the network trains.

Splitting is expensive under Panoptic Quality. Two fragments each cover part of
one ground truth, so neither clears IoU 0.5: the ground truth becomes a false
negative and both fragments become false positives. Rejoining them turns three
errors into one match.

What makes rejoining possible without a detector is the shape of the object. A
filament is long, thin and smoothly curved, so two fragments of the same one
are close together, point the same way, and lie roughly along each other's
extension. A pair of genuinely different filaments rarely satisfies all three
at once. The three tests below are exactly those conditions, and each has a
tolerance the sweep can tune.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Defaults in pixels of the working resolution, i.e. 1024 for this project.
DEFAULT_MAX_GAP = 24.0
DEFAULT_MAX_ANGLE = 35.0
DEFAULT_MAX_OFFSET = 12.0
# Below this length a region has no reliable direction, so the angle test is
# skipped for it and the gap alone decides.
MIN_ORIENTED_LENGTH = 12.0


@dataclass(frozen=True)
class Fragment:
    """One connected region, described by the geometry the joining tests need."""

    label: int
    mask: np.ndarray
    centroid: np.ndarray
    direction: np.ndarray
    length: float
    points: np.ndarray

    @property
    def is_oriented(self) -> bool:
        """Whether the region is long enough for its direction to mean anything."""
        return self.length >= MIN_ORIENTED_LENGTH


def describe(mask: np.ndarray, label: int = 0) -> Fragment:
    """Measure one region's centroid, principal direction and extent.

    The direction is the first principal component of the region's pixel
    coordinates, which for a thin elongated shape is its long axis. The length
    is the extent along that axis.
    """
    rows, columns = np.nonzero(mask)
    points = np.column_stack([columns, rows]).astype(np.float32)
    centroid = points.mean(axis=0)

    centred = points - centroid
    if len(points) < 2:
        return Fragment(
            label=label,
            mask=mask,
            centroid=centroid,
            direction=np.array([1.0, 0.0], dtype=np.float32),
            length=0.0,
            points=points,
        )

    # Principal axis from the covariance of the coordinates.
    covariance = centred.T @ centred / len(points)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))].astype(np.float32)

    projection = centred @ direction
    return Fragment(
        label=label,
        mask=mask,
        centroid=centroid,
        direction=direction,
        length=float(projection.max() - projection.min()),
        points=points,
    )


def _gap(first: Fragment, second: Fragment) -> float:
    """Smallest distance between the two regions, in pixels.

    Measured on a subsample of each outline: the exact minimum over every pair
    of pixels costs more than it is worth, and a few pixels of error do not
    change a decision made against a tolerance of tens of pixels.
    """
    stride = max(1, len(first.points) // 200)
    other_stride = max(1, len(second.points) // 200)
    left = first.points[::stride]
    right = second.points[::other_stride]
    differences = left[:, None, :] - right[None, :, :]
    return float(np.sqrt((differences**2).sum(axis=2)).min())


def _angle_between(first: Fragment, second: Fragment) -> float:
    """Angle between the two principal axes, in degrees, folded to 0-90.

    Folded because a principal axis has no sign: a direction and its opposite
    describe the same alignment.
    """
    cosine = abs(float(np.dot(first.direction, second.direction)))
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def _offset(first: Fragment, second: Fragment) -> float:
    """How far the second centroid sits off the first's axis, in pixels.

    Two fragments of one filament lie along each other's extension, so this is
    small. Two parallel filaments side by side have a small angle but a large
    offset, which is what this test catches.
    """
    separation = second.centroid - first.centroid
    along = np.dot(separation, first.direction) * first.direction
    return float(np.linalg.norm(separation - along))


def should_join(
    first: Fragment,
    second: Fragment,
    max_gap: float = DEFAULT_MAX_GAP,
    max_angle: float = DEFAULT_MAX_ANGLE,
    max_offset: float = DEFAULT_MAX_OFFSET,
) -> bool:
    """Whether two regions look like fragments of one filament.

    Args:
        first: One region.
        second: The other.
        max_gap: Largest distance between the regions that still joins them.
        max_angle: Largest angle between their long axes, in degrees.
        max_offset: How far the second may sit off the first's axis.

    Returns:
        True when all applicable tests pass.
    """
    if _gap(first, second) > max_gap:
        return False
    # A short region has no meaningful direction, so proximity is all there is.
    if not (first.is_oriented and second.is_oriented):
        return True
    if _angle_between(first, second) > max_angle:
        return False
    # Symmetric, so that a long fragment and a short one are treated alike.
    return min(_offset(first, second), _offset(second, first)) <= max_offset


def join_fragments(
    labels: np.ndarray,
    count: int,
    max_gap: float = DEFAULT_MAX_GAP,
    max_angle: float = DEFAULT_MAX_ANGLE,
    max_offset: float = DEFAULT_MAX_OFFSET,
) -> np.ndarray:
    """Merge the regions of a labelled image that look like one filament.

    Args:
        labels: Label image from ``cv2.connectedComponents``; 0 is background.
        count: Number of labels including the background.
        max_gap: Largest distance between two regions that still joins them.
        max_angle: Largest angle between their long axes, in degrees.
        max_offset: How far one may sit off the other's axis.

    Returns:
        A new label image with joined regions sharing a label. Labels are
        renumbered from one and stay contiguous.
    """
    regions = count - 1
    if regions < 2:
        return labels

    fragments = [describe(labels == label, label) for label in range(1, count)]

    # Union-find over the fragments: joining is transitive, so a chain of three
    # fragments becomes one filament rather than two overlapping pairs.
    parent = list(range(regions))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    joins = 0
    for left in range(regions):
        for right in range(left + 1, regions):
            if find(left) == find(right):
                continue
            if should_join(fragments[left], fragments[right], max_gap, max_angle, max_offset):
                parent[find(right)] = find(left)
                joins += 1

    if joins == 0:
        return labels

    groups: dict[int, int] = {}
    joined = np.zeros_like(labels)
    for index, fragment in enumerate(fragments):
        root = find(index)
        if root not in groups:
            groups[root] = len(groups) + 1
        joined[fragment.mask] = groups[root]

    logger.debug("Joined %d pair(s), %d regions -> %d.", joins, regions, len(groups))
    return joined


def connected_and_joined(
    binary: np.ndarray,
    connectivity: int = 8,
    max_gap: float = DEFAULT_MAX_GAP,
    max_angle: float = DEFAULT_MAX_ANGLE,
    max_offset: float = DEFAULT_MAX_OFFSET,
) -> tuple[int, np.ndarray]:
    """Label a binary mask, then rejoin the fragments of one filament.

    A drop-in replacement for ``cv2.connectedComponents`` that returns the same
    pair, so the instance extraction around it does not change.
    """
    count, labels = cv2.connectedComponents(binary, connectivity=connectivity)
    if max_gap <= 0:
        return count, labels
    joined = join_fragments(labels, count, max_gap, max_angle, max_offset)
    return int(joined.max()) + 1, joined
