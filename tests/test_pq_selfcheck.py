"""Self-validation of the evaluator against the real annotations.

Feeding the ground truth back in as the prediction has to score exactly 1.0.
If it does not, every later experiment is measured with a broken ruler, so
this check gates the whole project.

It is restricted to images annotated by a single annotator. Predictions carry
no annotator prefix, so for an image annotated by several people the same
prediction is scored against each of their annotations, and disagreement
between annotators keeps the score below 1.0 by construction.
"""

from __future__ import annotations

import pytest

from filament.data.coco import load_annotations
from filament.metrics.pq import compute_pq
from filament.paths import ProjectPaths
from filament.submit.rle import masks_to_gt_df, to_prediction_df

# Scoring a full-resolution annotator-image takes about half a second, so the
# check runs on a sample rather than on all 411 single-annotator images.
SAMPLE_SIZE = 12


@pytest.mark.dataset
def test_ground_truth_scored_against_itself_is_a_perfect_one(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    stems = dataset.single_annotator_stems()[:SAMPLE_SIZE]

    gt_df = masks_to_gt_df(dataset, stems)
    result = compute_pq(gt_df, to_prediction_df(gt_df))

    assert result.tp == len(gt_df)
    assert result.fp == 0
    assert result.fn == 0
    assert result.pq == pytest.approx(1.0)
    assert result.sq == pytest.approx(1.0)
    assert result.rq == pytest.approx(1.0)
    assert min(result.iou_scores) == pytest.approx(1.0)
    assert min(result.dice_scores) == pytest.approx(1.0)


@pytest.mark.dataset
def test_one_annotators_drawing_does_not_score_one_against_another(
    paths: ProjectPaths,
) -> None:
    """Annotator disagreement is large enough to show up as FP and FN.

    This is why the self-check above is limited to single-annotator images,
    and why the split has to group by image stem.
    """
    dataset = load_annotations(paths.train_annotations)
    stem = next(stem for stem, entries in dataset.by_stem().items() if len(entries) == 3)

    gt_df = masks_to_gt_df(dataset, [stem])
    first_annotator = dataset.by_stem()[stem][0].image_id
    prediction = to_prediction_df(gt_df[gt_df["filament_id"].str.startswith(first_annotator)])

    result = compute_pq(gt_df, prediction)

    assert result.pq < 1.0
    assert result.fp + result.fn > 0
