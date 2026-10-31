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
(connectivity), and no further than ``max_distance`` from the original mask of
the filament that takes it (reach). Each filament spreads outwards from its own
mask one pixel ring at a time, in turn, until none can spread further, so a
pixel goes to the filament whose supported path reaches it first (the earlier
filament on a tie): the masks never overlap, and every pixel a filament gains
is connected to it through supported pixels.
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

    supported = support >= threshold
    margin = int(np.ceil(max_distance)) + 1
    # Each filament's reach, measured from its own mask in a window around it:
    # a distance from all the masks together would let one filament claim
    # pixels within reach only of a neighbour.
    reaches = []
    for index, window in enumerate(ndimage.find_objects(labels), start=1):
        assert window is not None
        window = tuple(
            slice(max(part.start - margin, 0), min(part.stop + margin, size))
            for part, size in zip(window, shape, strict=True)
        )
        distance = ndimage.distance_transform_edt(labels[window] != index)
        reaches.append((index, window, supported[window] & (distance <= max_distance)))

    grown_labels = labels.copy()
    spreading = True
    while spreading:
        spreading = False
        for index, window, reach in reaches:
            owners = grown_labels[window]
            ring = (
                ndimage.binary_dilation(owners == index, structure=_STRUCTURE)
                & reach
                & (owners == 0)
            )
            if ring.any():
                owners[ring] = index
                spreading = True

    grown = []
    for index, instance in enumerate(instances, start=1):
        mask = grown_labels == index
        grown.append(replace(instance, mask=mask, area=int(mask.sum())))
    return grown
