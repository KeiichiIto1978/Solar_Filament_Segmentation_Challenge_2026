"""Tests for the calibration check.

The check exists to answer one question about a run: did the output end up
ordered by how many annotators drew a pixel? These tests pin down what the
three summary numbers say when the answer is clearly yes and clearly no.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.data.coco import Annotation, AnnotatorImage
from filament.metrics.calibration import calibration

SIZE = 32


def _annotator_image(stem: str, annotator: str, polygons: list[list[float]]) -> AnnotatorImage:
    image_id = f"{annotator}-{stem}"
    return AnnotatorImage(
        image_id=image_id,
        annotator=annotator,
        stem=stem,
        file_name=f"{stem}.jpeg",
        height=SIZE,
        width=SIZE,
        annotations=[
            Annotation(
                annotation_id=f"{image_id}_{index}",
                annotator_image_id=image_id,
                category_id=1,
                segmentation=[polygon],
                bbox=(0.0, 0.0, 0.0, 0.0),
                area=0.0,
                spine=[],
            )
            for index, polygon in enumerate(polygons)
        ],
    )


# Two squares: one every annotator drew, one only some of them did.
AGREED = [4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0]
DISPUTED = [20.0, 20.0, 28.0, 20.0, 28.0, 28.0, 20.0, 28.0]


def _three_annotators() -> dict[str, list[AnnotatorImage]]:
    """A frame three people saw, where two of them drew the second square."""
    return {
        "frame": [
            _annotator_image("frame", "a", [AGREED, DISPUTED]),
            _annotator_image("frame", "b", [AGREED, DISPUTED]),
            _annotator_image("frame", "c", [AGREED]),
        ]
    }


def _map_matching_the_votes() -> np.ndarray:
    """What a perfectly calibrated model would have produced."""
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    probability[4:12, 4:12] = 1.0
    probability[20:28, 20:28] = 2.0 / 3.0
    return probability


def test_a_map_that_matches_the_votes_lands_on_the_diagonal() -> None:
    result = calibration({"frame": _map_matching_the_votes()}, _three_annotators())

    assert result.is_monotonic
    assert result.mean_absolute_error == pytest.approx(0.0, abs=0.02)
    assert result.spread == pytest.approx(1.0, abs=0.02)


def test_a_two_valued_map_shows_up_as_no_spread() -> None:
    """The failure this check exists to catch: the model pushed everything to
    the extremes, so the threshold has nothing to act on in between."""
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    # Both squares asserted with full confidence, minority opinion included.
    probability[4:12, 4:12] = 1.0
    probability[20:28, 20:28] = 1.0

    result = calibration({"frame": probability}, _three_annotators())

    populated = [item for item in result.populated if item.low >= 0.05]
    assert len(populated) == 1
    # The one confident bin holds both squares, so it averages their shares.
    assert populated[0].observed == pytest.approx((1.0 + 2 / 3) / 2, abs=0.02)


def test_order_survives_even_when_the_values_are_pulled_up() -> None:
    """Dice pulls the output above the share it was asked for. That costs
    accuracy against the diagonal but must not cost the ordering."""
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    probability[4:12, 4:12] = 0.95
    probability[20:28, 20:28] = 0.75

    result = calibration({"frame": probability}, _three_annotators())

    assert result.is_monotonic
    assert result.mean_absolute_error > 0.0


def test_a_frame_without_annotations_is_skipped_loudly(caplog) -> None:
    """Counting it as all background would quietly flatter the curve."""
    maps = {"frame": _map_matching_the_votes(), "unknown": _map_matching_the_votes()}

    with caplog.at_level("WARNING"):
        result = calibration(maps, _three_annotators())

    assert "unknown" in caplog.text
    assert result.mean_absolute_error == pytest.approx(0.0, abs=0.02)


def test_at_least_two_edges_are_needed() -> None:
    with pytest.raises(ValueError, match="at least two bin edges"):
        calibration({"frame": _map_matching_the_votes()}, _three_annotators(), edges=(0.5,))


def test_the_middle_numbers_separate_a_share_from_a_two_valued_map() -> None:
    """The summary that matters. A background bin holding 99.6% of the frame
    makes the distance from the diagonal look tiny either way, so the check
    reports how much of the marked area sits in the middle of the range and
    how far the observed share moves across it."""
    carries_a_share = _map_matching_the_votes()
    two_valued = np.zeros((SIZE, SIZE), dtype=np.float32)
    two_valued[4:12, 4:12] = 1.0
    two_valued[20:28, 20:28] = 1.0

    shared = calibration({"frame": carries_a_share}, _three_annotators())
    extreme = calibration({"frame": two_valued}, _three_annotators())

    # Both look fine by the distance from the diagonal; only one uses the middle.
    assert extreme.mean_absolute_error < 0.05
    assert shared.middle_share > 0.3
    assert extreme.middle_share == pytest.approx(0.0)


def test_a_middle_that_says_nothing_shows_up_as_no_span() -> None:
    """Values spread through the range, but every one of them corresponds to
    the same share of annotators: moving the threshold changes which pixels
    are kept without changing who drew them."""
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    # Both squares given middling values, so the middle bins are populated...
    probability[4:12, 4:12] = 0.35
    probability[20:28, 20:28] = 0.65

    result = calibration({"frame": probability}, _three_annotators())

    assert result.middle_share > 0.9
    # ...but the two squares were drawn by different numbers of people, so a
    # map that had learned anything would not have put the more agreed one
    # lower. Here the span is negative, which is worse than flat.
    assert result.middle_span < 0.0
    assert not result.is_monotonic
