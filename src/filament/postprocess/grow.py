"""Extending emitted filaments into a second map's support, without adding new ones.

Phase 12 found the largest group of misses to be long filaments the working
configuration covers only in part: the candidate lies inside the annotated
filament and stops at about a third of it. A second network trained differently
-- on the union of a frame's annotators, which asks for every pixel anyone drew
-- may mark more of each filament. Averaging the two maps would also let it add
detections of its own, which on ResNet-34 cost more false positives than it
found filaments. Used here it only extends: each filament already emitted grows
into the pixels the second map supports and that connect to it, and nothing
the first configuration did not find is added.

Growth respects two limits. A pixel is only reachable through supported pixels
(connectivity), and no further than ``max_distance`` from where the filament
was (reach). Where two filaments' growth meets, each pixel goes to the nearer
one, so the masks still never overlap.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage

from filament.postprocess.instances import Instance

# 8-connectivity, as for the instances themselves.
_STRUCTURE = np.ones((3, 3), dtype=bool)


def grow_instances(
    instances: list[Instance],
    support: np.ndarray,
    threshold: float,
    max_distance: float,
) -> list[Instance]:
    """Grow each instance into connected pixels where ``support`` is high enough.

    Args:
        instances: Emitted filaments; their masks never overlap and share the
            shape of ``support``.
        support: Probability map of a second model at the same resolution.
        threshold: A pixel can be added where ``support`` is at least this.
        max_distance: Pixels further than this from an instance's original
            mask are not added to it.

    Returns:
        The instances in the same order, with their masks grown and their areas
        updated. Scores are unchanged. The masks still never overlap.

    Raises:
        ValueError: If a mask and ``support`` differ in shape.
    """
    if not instances:
        return []
    shape = support.shape
    labels = np.zeros(shape, dtype=np.int32)
    for index, instance in enumerate(instances, start=1):
        if instance.mask.shape != shape:
            raise ValueError(
                f"Mask {instance.mask.shape} and support {shape} must have the same shape."
            )
        labels[instance.mask] = index

    seeds = labels > 0
    # The nearest seed pixel of every pixel, and how far it is: growth goes to
    # the nearer filament where two meet, and stops at max_distance.
    distance, (nearest_rows, nearest_columns) = ndimage.distance_transform_edt(
        ~seeds, return_indices=True
    )
    nearest = labels[nearest_rows, nearest_columns]
    allowed = seeds | ((support >= threshold) & (distance <= max_distance))
    # Only pixels connected to some filament through allowed pixels.
    components, _ = ndimage.label(allowed, structure=_STRUCTURE)
    reached = np.isin(components, np.unique(components[seeds]))
    grown_labels = np.where(reached & allowed, nearest, 0)
    grown_labels[seeds] = labels[seeds]

    grown = []
    for index, instance in enumerate(instances, start=1):
        mask = grown_labels == index
        grown.append(replace(instance, mask=mask, area=int(mask.sum())))
    return grown
