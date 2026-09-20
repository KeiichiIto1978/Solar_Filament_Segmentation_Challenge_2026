"""Tests for the Panoptic Quality evaluator.

The metric decides every later design choice, so the cases below pin down its
behaviour on hand-built masks whose IoU is known exactly. Two of them sit on
either side of the 0.5 threshold, which is where the metric is discontinuous.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from filament.metrics.pq import PQResult, compute_pq, iou_dice_matrices
from filament.submit.rle import mask_to_rle

# A narrow frame is enough for one-dimensional bars and keeps the tests fast.
HEIGHT = 4
WIDTH = 1400
BAR_LENGTH = 1000

STEM = "20140609195854Bh"
ANNOTATOR_IMAGE = f"040301-{STEM}"


def _bar(start: int, length: int = BAR_LENGTH) -> np.ndarray:
    """A horizontal bar of ``length`` pixels starting at column ``start``.

    Two bars of equal length shifted by ``k`` intersect in ``length - k``
    pixels and cover ``length + k``, so their IoU is exactly
    ``(length - k) / (length + k)``.
    """
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[1, start : start + length] = True
    return mask


def _frame(ids: list[str], masks: list[np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filament_id": ids,
            "segmentation_rle": [mask_to_rle(mask) for mask in masks],
        }
    )


def _gt(*masks: np.ndarray, annotator_image: str = ANNOTATOR_IMAGE) -> pd.DataFrame:
    ids = [f"{annotator_image}_{index}" for index in range(1, len(masks) + 1)]
    return _frame(ids, list(masks))


def _pred(*masks: np.ndarray, stem: str = STEM) -> pd.DataFrame:
    ids = [f"{stem}_{index}" for index in range(1, len(masks) + 1)]
    return _frame(ids, list(masks))


def _score(gt: pd.DataFrame, pred: pd.DataFrame, **kwargs: object) -> PQResult:
    return compute_pq(gt, pred, height=HEIGHT, width=WIDTH, **kwargs)  # type: ignore[arg-type]


def _assert_pq_factorizes(result: PQResult) -> None:
    assert result.pq == pytest.approx(result.sq * result.rq)


def test_iou_dice_matrices_are_exact_for_known_overlaps() -> None:
    gt = np.stack([_bar(0)])
    pred = np.stack([_bar(324), _bar(0)])

    iou, dice = iou_dice_matrices(gt, pred)

    # 1000-pixel bars shifted by 324: intersection 676, union 1324.
    assert iou[0, 0] == pytest.approx(676 / 1324)
    assert dice[0, 0] == pytest.approx(2 * 676 / 2000)
    assert iou[0, 1] == 1.0


def test_iou_of_disjoint_masks_is_zero() -> None:
    iou, dice = iou_dice_matrices(np.stack([_bar(0, 10)]), np.stack([_bar(500, 10)]))

    assert iou[0, 0] == 0.0
    assert dice[0, 0] == 0.0


def test_iou_of_two_empty_masks_is_zero_rather_than_undefined() -> None:
    empty = np.zeros((1, HEIGHT, WIDTH), dtype=bool)

    iou, _ = iou_dice_matrices(empty, empty)

    assert iou[0, 0] == 0.0


def test_a_prediction_just_below_the_threshold_costs_a_false_positive_and_a_negative() -> None:
    # Shift 342: IoU = 658 / 1342 = 0.4903, just under the 0.5 gate.
    result = _score(_gt(_bar(0)), _pred(_bar(342)))

    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert result.pq == 0.0
    _assert_pq_factorizes(result)


def test_a_prediction_just_above_the_threshold_is_a_true_positive() -> None:
    # Shift 324: IoU = 676 / 1324 = 0.5106, just over the gate.
    result = _score(_gt(_bar(0)), _pred(_bar(324)))

    assert (result.tp, result.fp, result.fn) == (1, 0, 0)
    assert result.pq == pytest.approx(676 / 1324)
    assert result.sq == pytest.approx(676 / 1324)
    assert result.rq == 1.0
    assert result.dice_scores == pytest.approx([2 * 676 / 2000])
    _assert_pq_factorizes(result)


def test_missing_a_filament_costs_only_half_of_what_a_near_miss_costs() -> None:
    near_miss = _score(_gt(_bar(0)), _pred(_bar(342)))
    no_prediction = _score(_gt(_bar(0)), _pred())

    # Both score zero here, but the near miss adds 1.0 to the denominator
    # against 0.5 for the silent miss; with other matches present that gap is
    # what makes a low-confidence prediction a bad bet.
    assert near_miss.fp + near_miss.fn == 2
    assert no_prediction.fp + no_prediction.fn == 1


def test_predicting_nothing_turns_every_ground_truth_into_a_false_negative() -> None:
    result = _score(_gt(_bar(0, 10), _bar(100, 10), _bar(200, 10)), _pred())

    assert (result.tp, result.fp, result.fn) == (0, 0, 3)
    assert result.pq == 0.0
    _assert_pq_factorizes(result)


def test_predictions_on_an_image_without_ground_truth_are_all_false_positives() -> None:
    empty_gt = pd.DataFrame({"filament_id": [], "segmentation_rle": []}, dtype=str)

    result = _score(
        empty_gt,
        _pred(_bar(0, 10), _bar(100, 10)),
        annotator_images=[ANNOTATOR_IMAGE],
    )

    assert (result.tp, result.fp, result.fn) == (0, 2, 0)
    assert result.pq == 0.0


def test_an_empty_comparison_scores_zero_instead_of_dividing_by_zero() -> None:
    empty = pd.DataFrame({"filament_id": [], "segmentation_rle": []}, dtype=str)

    result = _score(empty, empty)

    assert result == PQResult(pq=0.0, sq=0.0, rq=0.0, tp=0, fp=0, fn=0)


def test_an_extra_prediction_is_counted_as_a_false_positive() -> None:
    result = _score(_gt(_bar(0)), _pred(_bar(0), _bar(1200, 100)))

    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    assert result.pq == pytest.approx(1.0 / 1.5)
    assert result.sq == 1.0
    _assert_pq_factorizes(result)


def test_identical_masks_score_a_perfect_one() -> None:
    result = _score(_gt(_bar(0), _bar(1100, 100)), _pred(_bar(0), _bar(1100, 100)))

    assert (result.tp, result.fp, result.fn) == (2, 0, 0)
    assert result.pq == 1.0
    _assert_pq_factorizes(result)


def test_the_same_prediction_is_scored_once_per_annotator() -> None:
    # One annotator drew one filament, the other drew two. The prediction
    # matches the shared one only, so it is a true positive twice and leaves a
    # false negative against the second annotator.
    first = _gt(_bar(0), annotator_image=f"040301-{STEM}")
    second = _gt(_bar(0), _bar(1200, 100), annotator_image=f"010401-{STEM}")
    gt = pd.concat([first, second], ignore_index=True)

    result = _score(gt, _pred(_bar(0)))

    assert (result.tp, result.fp, result.fn) == (2, 0, 1)
    assert result.pq == pytest.approx(2.0 / 2.5)
    _assert_pq_factorizes(result)


def test_predictions_for_an_unknown_image_are_reported_rather_than_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    other = _pred(_bar(0), stem="19990101000000Zz")

    with caplog.at_level(logging.WARNING, logger="filament.metrics.pq"):
        result = _score(_gt(_bar(0)), pd.concat([_pred(_bar(0)), other], ignore_index=True))

    assert (result.tp, result.fp, result.fn) == (1, 0, 0)
    assert "19990101000000Zz" in caplog.text


def test_ground_truth_ids_without_an_annotator_prefix_are_rejected() -> None:
    gt = _frame([f"{STEM}_1"], [_bar(0)])

    with pytest.raises(ValueError, match="Malformed ground-truth image id"):
        _score(gt, _pred(_bar(0)))
