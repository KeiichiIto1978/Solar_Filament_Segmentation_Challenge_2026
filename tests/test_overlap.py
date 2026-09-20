"""Tests for the pre-submission overlap check."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filament.metrics.overlap import (
    SubmissionOverlapError,
    check_no_overlap,
    find_overlaps,
)
from filament.submit.rle import mask_to_rle, write_submission

HEIGHT = 256
WIDTH = 256


def _square(top: int, left: int, size: int = 40) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[top : top + size, left : left + size] = True
    return mask


def _frame(rows: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filament_id": [filament_id for filament_id, _ in rows],
            "segmentation_rle": [mask_to_rle(mask) for _, mask in rows],
        }
    )


def test_disjoint_masks_report_no_overlap() -> None:
    frame = _frame(
        [
            ("20140609195854Bh_1", _square(0, 0)),
            ("20140609195854Bh_2", _square(100, 100)),
        ]
    )

    assert find_overlaps(frame, HEIGHT, WIDTH) == []


def test_touching_but_not_shared_pixels_are_allowed() -> None:
    # Adjacent squares share an edge but no pixel.
    frame = _frame(
        [
            ("20140609195854Bh_1", _square(0, 0)),
            ("20140609195854Bh_2", _square(0, 40)),
        ]
    )

    assert find_overlaps(frame, HEIGHT, WIDTH) == []


def test_a_single_shared_pixel_is_reported() -> None:
    frame = _frame(
        [
            ("20140609195854Bh_1", _square(0, 0)),
            ("20140609195854Bh_2", _square(39, 39)),
        ]
    )

    reports = find_overlaps(frame, HEIGHT, WIDTH)

    assert len(reports) == 1
    assert reports[0].image_id == "20140609195854Bh"
    assert reports[0].overlapping_pixels == 1
    assert reports[0].filament_ids == ["20140609195854Bh_1", "20140609195854Bh_2"]


def test_only_the_rows_touching_contested_pixels_are_named() -> None:
    frame = _frame(
        [
            ("20140609195854Bh_1", _square(0, 0)),
            ("20140609195854Bh_2", _square(20, 20)),
            ("20140609195854Bh_3", _square(150, 150)),
        ]
    )

    reports = find_overlaps(frame, HEIGHT, WIDTH)

    assert reports[0].filament_ids == ["20140609195854Bh_1", "20140609195854Bh_2"]


def test_masks_of_different_images_may_share_pixels() -> None:
    frame = _frame(
        [
            ("20140609195854Bh_1", _square(0, 0)),
            ("20150125172714Mh_1", _square(0, 0)),
        ]
    )

    assert find_overlaps(frame, HEIGHT, WIDTH) == []


def test_empty_masks_do_not_overlap_each_other() -> None:
    empty = np.zeros((HEIGHT, WIDTH), dtype=bool)
    frame = _frame([("20140609195854Bh_1", empty), ("20140609195854Bh_2", empty)])

    assert find_overlaps(frame, HEIGHT, WIDTH) == []


def test_every_overlapping_image_is_listed() -> None:
    frame = _frame(
        [
            ("20140609195854Bh_1", _square(0, 0)),
            ("20140609195854Bh_2", _square(10, 10)),
            ("20150125172714Mh_1", _square(0, 0)),
            ("20150125172714Mh_2", _square(5, 5)),
        ]
    )

    reports = find_overlaps(frame, HEIGHT, WIDTH)

    assert [report.image_id for report in reports] == [
        "20140609195854Bh",
        "20150125172714Mh",
    ]


def test_a_frame_without_the_submission_columns_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing the column"):
        find_overlaps(pd.DataFrame({"filament_id": ["a_1"]}), HEIGHT, WIDTH)


def test_check_no_overlap_accepts_a_clean_file(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    write_submission(
        _frame(
            [
                ("20140609195854Bh_1", _square(0, 0)),
                ("20140609195854Bh_2", _square(100, 100)),
            ]
        ),
        path,
    )

    check_no_overlap(path, HEIGHT, WIDTH)


def test_check_no_overlap_names_the_offending_image(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    write_submission(
        _frame(
            [
                ("20140609195854Bh_1", _square(0, 0)),
                ("20140609195854Bh_2", _square(20, 20)),
            ]
        ),
        path,
    )

    with pytest.raises(SubmissionOverlapError, match="20140609195854Bh"):
        check_no_overlap(path, HEIGHT, WIDTH)


def _full_size_submission(images: int, stride: int) -> pd.DataFrame:
    """A submission of ``images`` frames with seven 2048x2048 masks each.

    A ``stride`` of 200 keeps the bars apart; 20 makes every image overlap,
    which forces the slow path that decodes the masks.
    """
    counts = []
    for index in range(1, 8):
        mask = np.zeros((2048, 2048), dtype=bool)
        top = 100 + index * stride
        mask[top : top + 60, 300:340] = True
        counts.append(mask_to_rle(mask))
    rows = [
        (f"2014060919585{image:03d}Bh_{index + 1}", count)
        for image in range(images)
        for index, count in enumerate(counts)
    ]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def test_checking_a_clean_test_set_sized_submission_is_effectively_free() -> None:
    """The whole test set must be checkable before every submission."""
    frame = _full_size_submission(images=180, stride=200)

    started = time.perf_counter()
    reports = find_overlaps(frame)
    elapsed = time.perf_counter() - started

    assert reports == []
    # Comparing encoded areas against the encoded union never decodes a mask.
    assert elapsed < 10


def test_checking_a_submission_where_every_image_overlaps_stays_bounded() -> None:
    """The reporting path decodes masks, so it is timed on a smaller sample.

    Thirty images take about a sixth of the full test set; the cost is linear
    in the number of images, putting 180 at well under two minutes.
    """
    frame = _full_size_submission(images=30, stride=20)

    started = time.perf_counter()
    reports = find_overlaps(frame)
    elapsed = time.perf_counter() - started

    assert len(reports) == 30
    assert elapsed < 20
