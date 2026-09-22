"""Tests for deciding instances by box rather than by connectivity.

The property worth pinning down is the first one: a filament the drawing
threshold breaks in half is still one instance, because the box around it was
drawn at a threshold generous enough to see it whole.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.postprocess.boxes import (
    extract_by_box,
    masks_from_boxes,
    propose_boxes,
)
from filament.postprocess.instances import extract_instances

SIZE = 128


def _bar(rows: tuple[int, int], columns: tuple[int, int], value: float = 0.9) -> np.ndarray:
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    probability[rows[0] : rows[1], columns[0] : columns[1]] = value
    return probability


def test_a_weak_bridge_keeps_a_filament_whole() -> None:
    """Why this exists. Connected regions see a bar with a faint middle as two
    filaments; a box drawn at the finding threshold sees one."""
    probability = _bar((20, 26), (10, 60))
    probability[20:26, 36:40] = 0.3  # above the finding cut, below the drawing one

    by_connection = extract_instances(probability, threshold=0.5, min_area=1, output_size=SIZE)
    by_box = extract_by_box(probability, find_threshold=0.2, draw_threshold=0.5)

    assert len(by_connection) == 2
    assert len(by_box) == 1
    # The gap is not painted in: the drawing threshold still decides the pixels.
    mask, _ = by_box[0]
    assert not mask[22, 38]


def test_two_filaments_stay_two() -> None:
    probability = _bar((20, 26), (10, 60)) + _bar((80, 86), (10, 60))

    masks = extract_by_box(probability, find_threshold=0.2, draw_threshold=0.5)

    assert len(masks) == 2


def test_masks_never_overlap() -> None:
    """A submission whose masks share a pixel is rejected outright."""
    probability = _bar((20, 30), (10, 60))
    # A second, weaker filament running alongside and touching.
    probability[28:38, 30:80] = np.maximum(probability[28:38, 30:80], 0.6)

    masks = extract_by_box(probability, find_threshold=0.2, draw_threshold=0.5)

    claimed = np.zeros((SIZE, SIZE), dtype=int)
    for mask, _ in masks:
        claimed += mask.astype(int)
    assert claimed.max() <= 1


def test_the_stronger_box_keeps_the_pixels_they_share() -> None:
    """Two filaments crossing: the boundary between them goes to whichever the
    map is more confident about, which is the only ordering available here."""
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    probability[20:30, 10:60] = 0.95  # the confident one
    probability[25:35, 40:90] = np.maximum(probability[25:35, 40:90], 0.6)

    masks = extract_by_box(probability, find_threshold=0.2, draw_threshold=0.5)

    assert len(masks) >= 1
    strongest = masks[0][0]
    # The overlap belongs to the first mask, and to no other.
    assert strongest[27, 45]
    for mask, _ in masks[1:]:
        assert not mask[27, 45]


def test_boxes_come_back_strongest_first() -> None:
    probability = _bar((20, 26), (10, 60), value=0.6) + _bar((80, 86), (10, 60), value=0.95)

    candidates = propose_boxes(probability, find_threshold=0.2)

    assert len(candidates) == 2
    assert candidates[0].score > candidates[1].score
    assert candidates[0].box.top == 80


def test_a_region_too_small_to_propose_is_skipped() -> None:
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    probability[10:12, 10:12] = 0.9  # four pixels

    assert propose_boxes(probability, find_threshold=0.2, min_box_area=12) == []
    assert len(propose_boxes(probability, find_threshold=0.2, min_box_area=2)) == 1


def test_padding_lets_a_mask_reach_past_its_box() -> None:
    """The finding threshold can clip a filament that fades at the ends; the
    box is grown so the drawing threshold is not held to that."""
    probability = _bar((20, 26), (20, 50))
    candidates = propose_boxes(probability, find_threshold=0.2)

    tight = masks_from_boxes(probability, candidates, draw_threshold=0.5, padding=0.0)
    loose = masks_from_boxes(probability, candidates, draw_threshold=0.5, padding=0.5)

    assert float(loose[0][0].sum()) >= float(tight[0][0].sum())


def test_masks_can_be_returned_at_the_submission_size() -> None:
    probability = _bar((20, 26), (10, 60))

    masks = extract_by_box(probability, find_threshold=0.2, draw_threshold=0.5, output_size=256)

    assert masks[0][0].shape == (256, 256)


def test_a_map_that_is_not_two_dimensional_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-D probability map"):
        propose_boxes(np.zeros((2, SIZE, SIZE), dtype=np.float32))
