"""Tests for sorting missed filaments by cause.

Each case is one annotated filament of 100 pixels and a handful of candidates
whose overlaps are chosen so that exactly one rule applies.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.metrics.misses import (
    BELOW_THRESHOLD,
    BY_AREA,
    BY_SCORE,
    DROPPED_AREA,
    DROPPED_SCORE,
    FRAGMENTED,
    KEPT,
    MATCHED,
    SHAPE,
    TOO_LARGE,
    TOO_SMALL,
    UNSEEN,
    classify_misses,
)

THRESHOLD = 0.4
GT_AREA = 100.0


def _classify(
    overlaps: list[float],
    candidate_areas: list[float],
    dropped_by: list[str] | None = None,
    peak: float = 0.9,
) -> str:
    """One filament of 100 px against candidates with the given overlaps."""
    overlap = np.array([overlaps], dtype=float)
    areas = np.array(candidate_areas, dtype=float)
    iou = overlap / (GT_AREA + areas - overlap)
    return classify_misses(
        iou,
        overlap,
        np.array([GT_AREA]),
        areas,
        dropped_by if dropped_by is not None else [KEPT] * len(overlaps),
        np.array([peak]),
        THRESHOLD,
    )[0]


def test_a_kept_candidate_above_iou_one_half_is_a_match() -> None:
    # 80 of 100 shared, candidate of 90: IoU 80 / 110.
    assert _classify([80], [90]) == MATCHED


def test_no_probability_at_all_is_unseen() -> None:
    assert _classify([], [], peak=0.01) == UNSEEN


def test_probability_short_of_the_threshold_is_below_threshold() -> None:
    assert _classify([], [], peak=0.3) == BELOW_THRESHOLD


@pytest.mark.parametrize(
    ("reason", "expected"), [(BY_AREA, DROPPED_AREA), (BY_SCORE, DROPPED_SCORE)]
)
def test_a_match_removed_by_a_cut_off_names_the_cut_off(reason: str, expected: str) -> None:
    assert _classify([80], [90], dropped_by=[reason]) == expected


def test_a_candidate_spilling_far_beyond_the_filament_is_too_large() -> None:
    """Covers 90 of the 100, but is 300 px: under a third of it is filament."""
    assert _classify([90], [300]) == TOO_LARGE


def test_pieces_that_together_cover_most_of_it_are_fragments() -> None:
    """Three pieces of 25 px each, all inside: 75% covered, none above IoU 0.5."""
    assert _classify([25, 25, 25], [25, 25, 25]) == FRAGMENTED


def test_one_piece_covering_little_of_it_is_too_small() -> None:
    assert _classify([30], [30]) == TOO_SMALL


def test_a_substantial_overlap_short_of_a_match_is_shape() -> None:
    """60 of 100 covered and 60 of 110 inside: IoU 60 / 150 = 0.4."""
    assert _classify([60], [110]) == SHAPE


def test_a_kept_match_wins_over_a_dropped_one() -> None:
    """Candidates do not overlap, so this cannot arise from the pipeline; the
    rule still has to prefer what was emitted."""
    assert _classify([80, 80], [90, 90], dropped_by=[BY_SCORE, KEPT]) == MATCHED


def test_sizes_that_disagree_are_rejected() -> None:
    with pytest.raises(ValueError, match="disagree"):
        classify_misses(
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.array([GT_AREA]),
            np.array([1.0, 1.0]),
            [KEPT],
            np.array([0.9]),
            THRESHOLD,
        )
