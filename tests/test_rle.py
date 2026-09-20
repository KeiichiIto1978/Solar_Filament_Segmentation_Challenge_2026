"""Tests for RLE encoding and submission CSV writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filament.data.coco import Dataset, load_annotations
from filament.paths import ProjectPaths
from filament.submit.rle import (
    make_filament_id,
    mask_to_rle,
    masks_to_gt_df,
    read_submission,
    rle_to_mask,
    to_prediction_df,
    write_submission,
)


def _blob(height: int = 64, width: int = 64) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[10:30, 12:41] = True
    mask[35, 5] = True  # a detached pixel, to exercise multiple runs
    return mask


def test_mask_to_rle_round_trips() -> None:
    mask = _blob()

    restored = rle_to_mask(mask_to_rle(mask), 64, 64)

    assert np.array_equal(restored, mask)


def test_mask_to_rle_round_trips_a_full_frame() -> None:
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[1000:1100, 500:520] = True

    assert np.array_equal(rle_to_mask(mask_to_rle(mask)), mask)


def test_mask_to_rle_accepts_an_empty_mask() -> None:
    counts = mask_to_rle(np.zeros((2048, 2048), dtype=bool))

    assert counts
    assert rle_to_mask(counts).sum() == 0


def test_mask_to_rle_treats_any_non_zero_value_as_foreground() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:4, 2:4] = 7

    assert rle_to_mask(mask_to_rle(mask), 16, 16).sum() == 4


def test_mask_to_rle_rejects_a_stack_of_masks() -> None:
    with pytest.raises(ValueError, match="Expected a 2-D mask"):
        mask_to_rle(np.zeros((3, 16, 16), dtype=bool))


def test_make_filament_id_appends_the_running_number() -> None:
    assert make_filament_id("20150125172714Mh", 1) == "20150125172714Mh_1"


def test_write_submission_emits_unquoted_two_column_csv(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    counts = mask_to_rle(_blob())

    write_submission([("20150125172714Mh_1", counts)], path)

    text = path.read_text(encoding="utf-8")
    assert '"' not in text
    assert text.splitlines()[0] == "filament_id,segmentation_rle"
    assert text.splitlines()[1] == f"20150125172714Mh_1,{counts}"


def test_write_submission_accepts_a_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    frame = pd.DataFrame({"filament_id": ["a_1", "a_2"], "segmentation_rle": ["PPY1", "QQZ2"]})

    write_submission(frame, path)

    assert read_submission(path).equals(frame)


def test_write_submission_refuses_a_value_containing_a_comma(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Comma in submission row"):
        write_submission([("a_1", "PP,Y1")], tmp_path / "submission.csv")


def test_to_prediction_df_drops_the_annotator_prefix() -> None:
    gt = pd.DataFrame(
        {
            "filament_id": ["040301-20140609195854Bh_1"],
            "segmentation_rle": ["PPY1"],
        }
    )

    assert to_prediction_df(gt)["filament_id"].tolist() == ["20140609195854Bh_1"]


@pytest.mark.dataset
def test_masks_to_gt_df_encodes_every_annotation_of_the_selected_stems(
    paths: ProjectPaths,
) -> None:
    dataset: Dataset = load_annotations(paths.train_annotations)
    stems = dataset.single_annotator_stems()[:5]

    frame = masks_to_gt_df(dataset, stems)

    expected = sum(len(entry) for entry in dataset.subset(stems).annotator_images)
    assert len(frame) == expected
    assert frame["filament_id"].is_unique
    # Ground-truth ids keep the annotator prefix; predictions would not.
    assert all("-" in value for value in frame["filament_id"])


@pytest.mark.dataset
def test_encoded_ground_truth_decodes_back_to_the_annotated_area(
    paths: ProjectPaths,
) -> None:
    dataset = load_annotations(paths.train_annotations)
    stem = dataset.single_annotator_stems()[0]
    entry = dataset.subset([stem]).annotator_images[0]

    frame = masks_to_gt_df(dataset, [stem])

    decoded = [rle_to_mask(counts).sum() for counts in frame["segmentation_rle"]]
    assert decoded == [annotation.area for annotation in entry.annotations]
