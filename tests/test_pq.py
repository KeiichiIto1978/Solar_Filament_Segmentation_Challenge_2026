"""Tests for the Panoptic Quality evaluator.

The metric decides every later design choice, so the cases below pin down its
behaviour on hand-built masks whose IoU is known exactly. Two of them sit on
either side of the 0.5 threshold, which is where the metric is discontinuous.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from filament.metrics.pq import (
    PQResult,
    compute_pq,
    iou_dice_matrices,
    iou_dice_matrices_rle,
    pool_pq,
)
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


Scorer = Callable[..., PQResult]


@pytest.fixture(params=("rle", "dense"))
def score(request: pytest.FixtureRequest) -> Scorer:
    """Score a case on both backends: they must agree on every one of them."""

    def _score(gt: pd.DataFrame, pred: pd.DataFrame, **kwargs: object) -> PQResult:
        return compute_pq(
            gt,
            pred,
            height=HEIGHT,
            width=WIDTH,
            backend=request.param,
            **kwargs,  # type: ignore[arg-type]
        )

    return _score


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


def test_a_prediction_just_below_the_threshold_costs_a_false_positive_and_a_negative(
    score: Scorer,
) -> None:
    # Shift 342: IoU = 658 / 1342 = 0.4903, just under the 0.5 gate.
    result = score(_gt(_bar(0)), _pred(_bar(342)))

    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert result.pq == 0.0
    _assert_pq_factorizes(result)


def test_a_prediction_just_above_the_threshold_is_a_true_positive(score: Scorer) -> None:
    # Shift 324: IoU = 676 / 1324 = 0.5106, just over the gate.
    result = score(_gt(_bar(0)), _pred(_bar(324)))

    assert (result.tp, result.fp, result.fn) == (1, 0, 0)
    assert result.pq == pytest.approx(676 / 1324)
    assert result.sq == pytest.approx(676 / 1324)
    assert result.rq == 1.0
    assert result.dice_scores == pytest.approx([2 * 676 / 2000])
    _assert_pq_factorizes(result)


def test_missing_a_filament_costs_only_half_of_what_a_near_miss_costs(score: Scorer) -> None:
    near_miss = score(_gt(_bar(0)), _pred(_bar(342)))
    no_prediction = score(_gt(_bar(0)), _pred())

    # Both score zero here, but the near miss adds 1.0 to the denominator
    # against 0.5 for the silent miss; with other matches present that gap is
    # what makes a low-confidence prediction a bad bet.
    assert near_miss.fp + near_miss.fn == 2
    assert no_prediction.fp + no_prediction.fn == 1


def test_predicting_nothing_turns_every_ground_truth_into_a_false_negative(score: Scorer) -> None:
    result = score(_gt(_bar(0, 10), _bar(100, 10), _bar(200, 10)), _pred())

    assert (result.tp, result.fp, result.fn) == (0, 0, 3)
    assert result.pq == 0.0
    _assert_pq_factorizes(result)


def test_predictions_on_an_image_without_ground_truth_are_all_false_positives(
    score: Scorer,
) -> None:
    empty_gt = pd.DataFrame({"filament_id": [], "segmentation_rle": []}, dtype=str)

    result = score(
        empty_gt,
        _pred(_bar(0, 10), _bar(100, 10)),
        annotator_images=[ANNOTATOR_IMAGE],
    )

    assert (result.tp, result.fp, result.fn) == (0, 2, 0)
    assert result.pq == 0.0


def test_an_empty_comparison_scores_zero_instead_of_dividing_by_zero(score: Scorer) -> None:
    empty = pd.DataFrame({"filament_id": [], "segmentation_rle": []}, dtype=str)

    result = score(empty, empty)

    assert result == PQResult(pq=0.0, sq=0.0, rq=0.0, tp=0, fp=0, fn=0)


def test_an_extra_prediction_is_counted_as_a_false_positive(score: Scorer) -> None:
    result = score(_gt(_bar(0)), _pred(_bar(0), _bar(1200, 100)))

    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    assert result.pq == pytest.approx(1.0 / 1.5)
    assert result.sq == 1.0
    _assert_pq_factorizes(result)


def test_identical_masks_score_a_perfect_one(score: Scorer) -> None:
    result = score(_gt(_bar(0), _bar(1100, 100)), _pred(_bar(0), _bar(1100, 100)))

    assert (result.tp, result.fp, result.fn) == (2, 0, 0)
    assert result.pq == 1.0
    _assert_pq_factorizes(result)


def test_the_same_prediction_is_scored_once_per_annotator(score: Scorer) -> None:
    # One annotator drew one filament, the other drew two. The prediction
    # matches the shared one only, so it is a true positive twice and leaves a
    # false negative against the second annotator.
    first = _gt(_bar(0), annotator_image=f"040301-{STEM}")
    second = _gt(_bar(0), _bar(1200, 100), annotator_image=f"010401-{STEM}")
    gt = pd.concat([first, second], ignore_index=True)

    result = score(gt, _pred(_bar(0)))

    assert (result.tp, result.fp, result.fn) == (2, 0, 1)
    assert result.pq == pytest.approx(2.0 / 2.5)
    _assert_pq_factorizes(result)


def test_predictions_for_an_unknown_image_are_reported_rather_than_ignored(
    score: Scorer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    other = _pred(_bar(0), stem="19990101000000Zz")

    with caplog.at_level(logging.WARNING, logger="filament.metrics.pq"):
        result = score(_gt(_bar(0)), pd.concat([_pred(_bar(0)), other], ignore_index=True))

    assert (result.tp, result.fp, result.fn) == (1, 0, 0)
    assert "19990101000000Zz" in caplog.text


def test_ground_truth_ids_without_an_annotator_prefix_are_rejected(score: Scorer) -> None:
    gt = _frame([f"{STEM}_1"], [_bar(0)])

    with pytest.raises(ValueError, match="Malformed ground-truth image id"):
        score(gt, _pred(_bar(0)))


def test_the_rle_backend_reproduces_the_dense_matrices() -> None:
    """The fast path must be a pure optimization, not a different metric."""
    gt_masks = [_bar(0), _bar(600, 200), np.zeros((HEIGHT, WIDTH), dtype=bool)]
    pred_masks = [_bar(324), _bar(0), _bar(610, 180), _bar(1300, 90)]

    dense_iou, dense_dice = iou_dice_matrices(np.stack(gt_masks), np.stack(pred_masks))
    rle_iou, rle_dice = iou_dice_matrices_rle(
        [mask_to_rle(mask) for mask in gt_masks],
        [mask_to_rle(mask) for mask in pred_masks],
        HEIGHT,
        WIDTH,
    )

    assert rle_iou == pytest.approx(dense_iou)
    assert rle_dice == pytest.approx(dense_dice)


def test_the_rle_backend_handles_an_empty_side() -> None:
    iou, dice = iou_dice_matrices_rle([], [mask_to_rle(_bar(0))], HEIGHT, WIDTH)

    assert iou.shape == (0, 1)
    assert dice.shape == (0, 1)


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        compute_pq(_gt(_bar(0)), _pred(_bar(0)), backend="numpy")  # type: ignore[arg-type]


def test_pooling_two_folds_equals_scoring_their_matches_together() -> None:
    """Hand-computed: fold A has IoUs 0.6 and 0.8 with 1 FP and 2 FN; fold B
    has one IoU of 0.7 and 3 FP. Together: IoU sum 2.1 over
    3 + 0.5 * 4 + 0.5 * 2 = 6, so PQ 0.35, SQ 0.7, RQ 0.5."""
    fold_a = PQResult(pq=0.0, sq=0.7, rq=0.0, tp=2, fp=1, fn=2)
    fold_b = PQResult(pq=0.0, sq=0.7, rq=0.0, tp=1, fp=3, fn=0)

    pooled = pool_pq([fold_a, fold_b])

    assert (pooled.tp, pooled.fp, pooled.fn) == (3, 4, 2)
    assert pooled.pq == pytest.approx(0.35)
    assert pooled.sq == pytest.approx(0.7)
    assert pooled.rq == pytest.approx(0.5)


def test_pooling_weights_folds_by_size_not_equally() -> None:
    """A small perfect fold must not pull the pooled score up as much as a
    mean of per-fold PQ would: 1 match at IoU 1.0 beside 9 matches at 0.6 with
    9 misses pools to 6.4 / 14.5, not to the mean of 1.0 and 0.4."""
    small = PQResult(pq=1.0, sq=1.0, rq=1.0, tp=1, fp=0, fn=0)
    large = PQResult(pq=0.0, sq=0.6, rq=0.0, tp=9, fp=0, fn=9)

    pooled = pool_pq([small, large])

    assert pooled.pq == pytest.approx(6.4 / 14.5)
    assert pooled.pq < (1.0 + 5.4 / 13.5) / 2


def test_pooling_nothing_scores_zero() -> None:
    assert pool_pq([]).pq == 0.0
