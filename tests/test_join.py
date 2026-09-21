"""Tests for rejoining the fragments of one filament."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from filament.postprocess.join import (
    connected_and_joined,
    describe,
    join_fragments,
    should_join,
)

SIDE = 200


def _bar(top: int, left: int, length: int, thickness: int = 6) -> np.ndarray:
    """A horizontal bar, standing in for a filament fragment."""
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    mask[top : top + thickness, left : left + length] = True
    return mask


def _vertical_bar(top: int, left: int, length: int, thickness: int = 6) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    mask[top : top + length, left : left + thickness] = True
    return mask


def _labelled(*masks: np.ndarray) -> tuple[int, np.ndarray]:
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    for index, mask in enumerate(masks, start=1):
        labels[mask] = index
    return len(masks) + 1, labels


def test_describe_finds_the_long_axis_of_a_horizontal_bar() -> None:
    fragment = describe(_bar(50, 20, 80))

    assert abs(fragment.direction[0]) > 0.99  # pointing along x
    assert fragment.length == pytest.approx(79, abs=2)
    assert fragment.is_oriented


def test_describe_marks_a_tiny_region_as_unoriented() -> None:
    """A few pixels have no meaningful direction."""
    fragment = describe(_bar(50, 20, 4, thickness=2))

    assert not fragment.is_oriented


def test_two_collinear_fragments_are_joined() -> None:
    first = describe(_bar(50, 20, 60))
    second = describe(_bar(50, 90, 60))  # 10 px gap, same axis

    assert should_join(first, second, max_gap=24)


def test_fragments_further_apart_than_the_gap_are_left_alone() -> None:
    first = describe(_bar(50, 20, 40))
    second = describe(_bar(50, 100, 40))  # 40 px gap

    assert not should_join(first, second, max_gap=24)


def test_perpendicular_fragments_are_not_joined() -> None:
    first = describe(_bar(50, 20, 60))
    second = describe(_vertical_bar(60, 90, 60))

    assert not should_join(first, second, max_gap=40, max_angle=35)


def test_parallel_fragments_side_by_side_are_not_joined() -> None:
    """Same direction, small gap, but offset across the axis."""
    first = describe(_bar(50, 20, 60))
    second = describe(_bar(66, 20, 60))  # 10 px below, fully overlapping in x

    assert not should_join(first, second, max_gap=24, max_angle=35, max_offset=8)


def test_a_wide_angle_tolerance_admits_perpendicular_fragments() -> None:
    first = describe(_bar(50, 20, 60))
    second = describe(_vertical_bar(60, 90, 60))

    assert should_join(first, second, max_gap=40, max_angle=90, max_offset=100)


def test_joining_is_transitive_across_three_fragments() -> None:
    count, labels = _labelled(_bar(50, 10, 40), _bar(50, 60, 40), _bar(50, 110, 40))

    joined = join_fragments(labels, count, max_gap=24)

    assert joined.max() == 1  # all three became one filament


def test_unrelated_fragments_keep_separate_labels() -> None:
    count, labels = _labelled(_bar(20, 10, 40), _bar(150, 120, 40))

    joined = join_fragments(labels, count, max_gap=24)

    assert joined.max() == 2


def test_labels_are_renumbered_contiguously() -> None:
    count, labels = _labelled(
        _bar(20, 10, 40), _bar(20, 60, 40), _bar(150, 10, 40), _bar(180, 150, 20)
    )

    joined = join_fragments(labels, count, max_gap=24)

    present = sorted(set(np.unique(joined)) - {0})
    assert present == list(range(1, len(present) + 1))


def test_a_single_region_is_returned_unchanged() -> None:
    count, labels = _labelled(_bar(50, 20, 60))

    assert np.array_equal(join_fragments(labels, count, max_gap=24), labels)


def test_joining_preserves_every_marked_pixel() -> None:
    count, labels = _labelled(_bar(50, 10, 40), _bar(50, 60, 40))

    joined = join_fragments(labels, count, max_gap=24)

    assert np.count_nonzero(joined) == np.count_nonzero(labels)


def test_connected_and_joined_matches_opencv_when_disabled() -> None:
    """A gap of zero must leave the plain behaviour untouched."""
    binary = (_bar(50, 10, 40) | _bar(50, 60, 40)).astype(np.uint8)

    expected = cv2.connectedComponents(binary, connectivity=8)
    actual = connected_and_joined(binary, max_gap=0.0)

    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])


def test_connected_and_joined_merges_fragments_when_enabled() -> None:
    binary = (_bar(50, 10, 40) | _bar(50, 60, 40)).astype(np.uint8)

    count, labels = connected_and_joined(binary, max_gap=24)

    assert count == 2  # background plus one filament
    assert labels.max() == 1
