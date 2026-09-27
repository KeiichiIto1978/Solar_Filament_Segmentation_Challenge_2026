"""Tests for prediction and scoring.

A tiny randomly initialised network stands in for a trained one: what is under
test is the chain from probability map to score, not the quality of the masks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from filament.data.coco import load_annotations
from filament.data.split import load_fold
from filament.evaluation import (
    COUNT_SIZE,
    DIHEDRAL_VIEWS,
    FLIP_VIEWS,
    EvaluationResult,
    annotated_masks,
    evaluate,
    predict_frame,
    predict_probability,
    predict_probability_ensemble,
    predict_probability_tta,
    predict_probability_views,
    shrink,
)
from filament.metrics.overlap import find_overlaps
from filament.metrics.pq import PQResult
from filament.paths import ProjectPaths

SIZE = 128


class _AlwaysFilament(nn.Module):
    """A stand-in model that marks one fixed rectangle, whatever the input."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.full((batch.shape[0], 1, *batch.shape[2:]), -10.0)
        side = batch.shape[2]
        logits[:, :, side // 2 : side // 2 + 8, side // 2 : side // 2 + 20] = 10.0
        return logits


def _frame(side: int = 512) -> np.ndarray:
    """A frame with a disk, so that disk detection has something to find."""
    rows = np.arange(side)[:, None]
    columns = np.arange(side)[None, :]
    distance = np.hypot(columns - (side - 1) / 2, rows - (side - 1) / 2)
    frame = np.full((side, side), 5, dtype=np.uint8)
    frame[distance <= 0.44 * side] = 140
    return frame


def test_predict_probability_returns_values_in_the_unit_interval() -> None:
    probability = predict_probability(_AlwaysFilament(), _frame(), size=SIZE)

    assert probability.shape == (SIZE, SIZE)
    assert 0.0 <= probability.min() <= probability.max() <= 1.0


def test_predict_frame_returns_masks_at_the_submission_resolution(tmp_path) -> None:
    import cv2

    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), _frame())

    instances = predict_frame(_AlwaysFilament(), path, size=SIZE, min_area=1, output_size=1024)

    assert len(instances) == 1
    assert instances[0].mask.shape == (1024, 1024)


def test_predict_frame_can_skip_the_disk_mask(tmp_path) -> None:
    import cv2

    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), _frame())

    with_disk = predict_frame(_AlwaysFilament(), path, size=SIZE, min_area=1, mask_disk=True)
    without = predict_frame(_AlwaysFilament(), path, size=SIZE, min_area=1, mask_disk=False)

    # The rectangle sits in the middle, so both keep it.
    assert len(with_disk) == len(without) == 1


def test_shrink_keeps_a_mask_boolean() -> None:
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[100:200, 100:200] = True

    smaller = shrink(mask, 256)

    assert smaller.dtype == bool
    assert smaller.shape == (256, 256)
    assert smaller.sum() == pytest.approx(100 * 100 / 16, rel=0.05)


def test_shrink_of_an_already_sized_mask_is_a_no_op() -> None:
    mask = np.zeros((COUNT_SIZE, COUNT_SIZE), dtype=bool)

    assert shrink(mask) is mask


def test_the_summary_includes_every_component_of_the_score() -> None:
    result = EvaluationResult(
        pq=PQResult(
            pq=0.3,
            sq=0.6,
            rq=0.5,
            tp=10,
            fp=4,
            fn=6,
            iou_scores=[0.6] * 10,
            dice_scores=[0.7] * 10,
        ),
        fused=3,
        split=2,
        stems=100,
        annotator_images=140,
        predictions=700,
        seconds=42.0,
    )

    summary = result.to_dict()

    assert summary["pq"] == 0.3
    for key in ("sq", "rq", "tp", "fp", "fn", "fused", "split", "iou_mean", "dice_mean"):
        assert key in summary
    assert summary["iou_mean"] == 0.6


def test_the_summary_of_an_empty_run_reports_zero_not_nan() -> None:
    result = EvaluationResult(
        pq=PQResult(pq=0.0, sq=0.0, rq=0.0, tp=0, fp=0, fn=9),
        fused=0,
        split=0,
        stems=1,
        annotator_images=1,
        predictions=0,
        seconds=1.0,
    )

    assert result.to_dict()["iou_mean"] == 0.0


@pytest.mark.dataset
def test_annotated_masks_returns_one_mask_per_filament(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    entry = dataset.annotator_images[0]

    masks = annotated_masks(entry)

    assert len(masks) == len(entry.annotations)
    assert all(mask.shape == (COUNT_SIZE, COUNT_SIZE) for mask in masks)


@pytest.mark.dataset
def test_evaluating_a_stand_in_model_produces_a_scorable_submission(
    paths: ProjectPaths,
) -> None:
    """End to end: frames in, PQ and non-overlapping predictions out."""
    dataset = load_annotations(paths.train_annotations)
    stems = load_fold(0, paths.splits_dir).val[:2]

    result, predictions = evaluate(
        _AlwaysFilament(),
        dataset,
        paths.train_images,
        stems,
        size=SIZE,
        min_area=1,
        device="cpu",
    )

    assert result.stems == 2
    assert result.annotator_images >= 2
    assert result.predictions == len(predictions)
    assert result.seconds > 0
    # The stand-in marks one rectangle per frame and never matches a filament.
    assert result.pq.tp == 0
    assert result.pq.fn > 0
    assert find_overlaps(predictions) == []


@pytest.mark.dataset
def test_a_cached_ground_truth_gives_the_same_score(paths: ProjectPaths) -> None:
    from filament.submit.rle import masks_to_gt_df

    dataset = load_annotations(paths.train_annotations)
    stems = load_fold(0, paths.splits_dir).val[:2]
    cached = masks_to_gt_df(dataset, stems)

    without, _ = evaluate(
        _AlwaysFilament(), dataset, paths.train_images, stems, size=SIZE, min_area=1
    )
    with_cache, _ = evaluate(
        _AlwaysFilament(),
        dataset,
        paths.train_images,
        stems,
        size=SIZE,
        min_area=1,
        gt_df=cached,
    )

    assert without.pq.pq == with_cache.pq.pq
    assert (without.pq.tp, without.pq.fp, without.pq.fn) == (
        with_cache.pq.tp,
        with_cache.pq.fp,
        with_cache.pq.fn,
    )


class _ConstantLogit(nn.Module):
    """Answers the same logit everywhere, so the expected mean is known."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.full((batch.shape[0], 1, *batch.shape[2:]), self.logit)


def test_an_ensemble_averages_probabilities_not_logits() -> None:
    """Logits 0 and 10 give probabilities 0.5 and ~1.0. Their mean is ~0.75;
    averaging the logits first would give sigmoid(5) ~0.993 instead."""
    frame = np.full((64, 64), 128, dtype=np.uint8)

    mean = predict_probability_ensemble([_ConstantLogit(0.0), _ConstantLogit(10.0)], frame, size=32)

    expected = (0.5 + 1 / (1 + np.exp(-10.0))) / 2
    assert mean.shape == (32, 32)
    assert np.allclose(mean, expected, atol=1e-6)


def test_an_ensemble_of_one_is_that_model() -> None:
    frame = np.full((64, 64), 128, dtype=np.uint8)
    model = _ConstantLogit(-1.5)

    assert np.array_equal(
        predict_probability_ensemble([model], frame, size=32),
        predict_probability(model, frame, size=32),
    )


def test_an_empty_ensemble_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        predict_probability_ensemble([], np.zeros((64, 64), dtype=np.uint8), size=32)


def test_every_view_is_undone_by_its_inverse() -> None:
    """A random array has no symmetry, so any mistake in an inverse shows."""
    array = torch.rand(1, 2, 16, 16, generator=torch.Generator().manual_seed(0))

    for view in DIHEDRAL_VIEWS:
        assert torch.equal(view.invert(view.apply(array)), array), view.name


def test_the_dihedral_views_are_eight_different_transforms() -> None:
    """Guards against a set with duplicates, which would weight some views
    twice and leave others out while still counting eight."""
    array = torch.arange(16.0).reshape(1, 1, 4, 4)

    transformed = {tuple(view.apply(array).flatten().tolist()) for view in DIHEDRAL_VIEWS}

    assert len(DIHEDRAL_VIEWS) == 8
    assert len(transformed) == 8


class _ReadsTheInput(nn.Module):
    """Answers from the pixel it sits on, so turning the input turns the answer
    with it: the kind of model for which every view agrees."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return 10.0 * (batch[:, :1] - 0.5)


def test_views_of_an_equivariant_model_agree_with_the_plain_prediction() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    model = _ReadsTheInput()

    plain = predict_probability(model, frame, size=32)
    averaged = predict_probability_tta(model, frame, DIHEDRAL_VIEWS, size=32)

    assert np.allclose(averaged, plain, atol=1e-6)


class _FixedPixel(nn.Module):
    """Marks one fixed pixel whatever it is shown, so its answer does not turn
    with the input. Averaged over views, the pixel lands at each of its images
    under the views with an equal share, which is known exactly."""

    ROW, COLUMN = 2, 5

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.full((batch.shape[0], 1, *batch.shape[2:]), -30.0)
        logits[:, :, self.ROW, self.COLUMN] = 30.0
        return logits


def test_views_of_a_fixed_answer_spread_it_evenly_over_its_images() -> None:
    side = 16
    frame = np.full((side, side), 128, dtype=np.uint8)
    row, column, last = _FixedPixel.ROW, _FixedPixel.COLUMN, side - 1

    averaged = predict_probability_tta(_FixedPixel(), frame, DIHEDRAL_VIEWS, size=side)

    # The eight images of (2, 5) under flips, turns and transposes.
    expected = np.zeros((side, side))
    for image_row, image_column in [
        (row, column),
        (column, row),
        (row, last - column),
        (column, last - row),
        (last - row, column),
        (last - column, row),
        (last - row, last - column),
        (last - column, last - row),
    ]:
        expected[image_row, image_column] = 1 / 8
    assert np.allclose(averaged, expected, atol=1e-6)


def test_the_flip_pair_splits_a_fixed_answer_between_it_and_its_mirror() -> None:
    side = 16
    frame = np.full((side, side), 128, dtype=np.uint8)
    row, column = _FixedPixel.ROW, _FixedPixel.COLUMN

    averaged = predict_probability_tta(_FixedPixel(), frame, FLIP_VIEWS, size=side)

    expected = np.zeros((side, side))
    expected[row, column] = expected[row, side - 1 - column] = 0.5
    assert np.allclose(averaged, expected, atol=1e-6)


def test_the_views_come_back_one_map_per_view() -> None:
    frame = np.full((32, 32), 128, dtype=np.uint8)

    maps = predict_probability_views(_ConstantLogit(0.0), frame, DIHEDRAL_VIEWS, size=16)

    assert list(maps) == list(DIHEDRAL_VIEWS)
    assert all(probability.shape == (16, 16) for probability in maps.values())


def test_an_empty_set_of_views_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one view"):
        predict_probability_tta(_ConstantLogit(0.0), np.zeros((32, 32), dtype=np.uint8), [])
