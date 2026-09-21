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
    EvaluationResult,
    annotated_masks,
    evaluate,
    predict_frame,
    predict_probability,
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
