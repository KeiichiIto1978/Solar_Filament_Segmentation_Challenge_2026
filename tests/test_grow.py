"""Tests for growing emitted filaments into a second map's support.

The cases are built so that each limit -- connectivity, reach, the threshold,
and the split between two filaments -- is the one that decides.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.postprocess.grow import grow_instances
from filament.postprocess.instances import Instance

SIDE = 60


def _instance(rows: slice, columns: slice, score: float = 0.9) -> Instance:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    mask[rows, columns] = True
    return Instance(mask=mask, score=score, area=int(mask.sum()))


def test_a_filament_grows_along_supported_pixels_it_touches() -> None:
    """Found: columns 10-19. The second map supports columns 10-39 of the same
    row band, so the filament extends to column 39."""
    seed = _instance(slice(28, 32), slice(10, 20))
    support = np.zeros((SIDE, SIDE))
    support[28:32, 10:40] = 0.8

    (grown,) = grow_instances([seed], support, threshold=0.5, max_distance=100)

    assert grown.mask[28:32, 10:40].all()
    assert grown.area == 4 * 30
    assert grown.score == seed.score


def test_supported_pixels_not_connected_to_a_filament_are_not_added() -> None:
    """A separate blob of support is a detection of its own, which growing must
    never create."""
    seed = _instance(slice(10, 14), slice(10, 20))
    support = np.zeros((SIDE, SIDE))
    support[40:44, 40:50] = 0.9

    (grown,) = grow_instances([seed], support, threshold=0.5, max_distance=100)

    assert np.array_equal(grown.mask, seed.mask)


def test_growth_stops_at_the_reach() -> None:
    seed = _instance(slice(28, 32), slice(10, 20))
    support = np.zeros((SIDE, SIDE))
    support[28:32, 10:50] = 0.8

    (grown,) = grow_instances([seed], support, threshold=0.5, max_distance=5)

    columns = np.nonzero(grown.mask.any(axis=0))[0]
    assert columns.max() == 19 + 5


def test_support_below_the_threshold_adds_nothing() -> None:
    seed = _instance(slice(28, 32), slice(10, 20))
    support = np.zeros((SIDE, SIDE))
    support[28:32, 10:40] = 0.3

    (grown,) = grow_instances([seed], support, threshold=0.5, max_distance=100)

    assert np.array_equal(grown.mask, seed.mask)


def test_two_filaments_growing_towards_each_other_split_the_gap() -> None:
    """Columns 0-9 and 50-59 found; support everywhere between. Each takes the
    pixels nearer to it, and the two masks still never overlap."""
    left = _instance(slice(28, 32), slice(0, 10))
    right = _instance(slice(28, 32), slice(50, 60))
    support = np.zeros((SIDE, SIDE))
    support[28:32, :] = 0.9

    grown_left, grown_right = grow_instances([left, right], support, 0.5, max_distance=100)

    assert not (grown_left.mask & grown_right.mask).any()
    assert grown_left.mask[28:32, :29].all() and grown_right.mask[28:32, 31:].all()
    assert (grown_left.mask | grown_right.mask)[28:32, :].all()


def test_a_filament_gains_nothing_it_has_no_supported_path_to() -> None:
    """Support runs from filament A along row 30 and passes two pixels from
    filament B, with an unsupported gap between. The pixels near B are reached
    from A only, so they go to A and B stays as it was."""
    first = _instance(slice(28, 32), slice(0, 10))
    second = _instance(slice(34, 38), slice(30, 40))
    support = np.zeros((SIDE, SIDE))
    support[28:32, 10:50] = 0.9

    grown_first, grown_second = grow_instances([first, second], support, 0.5, 100)

    assert np.array_equal(grown_second.mask, second.mask)
    assert grown_first.mask[28:32, 10:50].all()


def test_the_reach_is_measured_from_the_filament_that_grows() -> None:
    """Support runs from A along rows 28-31 out to column 59; B sits beside the
    far end across an unsupported gap. With a reach of 10, A stops ten columns
    past its own end, however close B is to the pixels beyond."""
    first = _instance(slice(28, 32), slice(0, 10))
    second = _instance(slice(34, 38), slice(30, 60))
    support = np.zeros((SIDE, SIDE))
    support[28:32, 10:60] = 0.9

    grown_first, grown_second = grow_instances([first, second], support, 0.5, 10)

    columns = np.nonzero(grown_first.mask.any(axis=0))[0]
    assert columns.max() == 9 + 10
    assert np.array_equal(grown_second.mask, second.mask)


def test_a_winding_path_is_followed_to_its_end_within_reach() -> None:
    """A U-shaped band of support whose far arm ends 5 pixels from the seed but
    over 40 steps along the band: every pixel is within reach, so all are taken."""
    seed = _instance(slice(10, 14), slice(10, 14))
    support = np.zeros((SIDE, SIDE))
    support[10:14, 14:30] = 0.9  # out along the top
    support[10:20, 26:30] = 0.9  # down
    support[16:20, 14:30] = 0.9  # back along the bottom, ending under the seed

    (grown,) = grow_instances([seed], support, threshold=0.5, max_distance=20)

    assert grown.mask[16:20, 14:18].all()


def test_nothing_in_nothing_out() -> None:
    assert grow_instances([], np.zeros((SIDE, SIDE)), 0.5, 10) == []


def test_a_mask_of_another_shape_is_rejected() -> None:
    seed = _instance(slice(0, 2), slice(0, 2))
    with pytest.raises(ValueError, match="same shape"):
        grow_instances([seed], np.zeros((SIDE // 2, SIDE // 2)), 0.5, 10)
