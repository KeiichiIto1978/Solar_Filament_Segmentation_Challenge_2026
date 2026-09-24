"""Tests for recovering instances from a probability map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filament.data.disk import Disk
from filament.metrics.overlap import check_no_overlap, find_overlaps
from filament.postprocess.instances import (
    extract_instances,
    fusion_and_splitting,
    instances_to_rows,
)
from filament.submit.rle import write_submission

SIZE = 128


def _map_with_blobs(*boxes: tuple[int, int, int, int], value: float = 0.9) -> np.ndarray:
    """A probability map with a rectangle of ``value`` per box."""
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    for top, left, height, width in boxes:
        probability[top : top + height, left : left + width] = value
    return probability


def test_separate_blobs_become_separate_instances() -> None:
    probability = _map_with_blobs((10, 10, 12, 12), (60, 60, 12, 12))

    instances = extract_instances(probability, min_area=1, output_size=SIZE)

    assert len(instances) == 2
    assert all(instance.area == 144 for instance in instances)


def test_touching_blobs_become_one_instance() -> None:
    # Adjacent rectangles are connected, so they cannot be told apart. This is
    # the fusion failure the phase is meant to measure, not a bug.
    probability = _map_with_blobs((10, 10, 12, 12), (10, 22, 12, 12))

    instances = extract_instances(probability, min_area=1, output_size=SIZE)

    assert len(instances) == 1
    assert instances[0].area == 288


def test_a_diagonal_step_stays_one_instance() -> None:
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    for offset in range(10):
        probability[20 + offset, 20 + offset] = 0.9

    instances = extract_instances(probability, min_area=1, output_size=SIZE)

    assert len(instances) == 1


def test_pixels_below_the_threshold_are_ignored() -> None:
    probability = _map_with_blobs((10, 10, 12, 12), value=0.4)

    assert extract_instances(probability, threshold=0.5, min_area=1, output_size=SIZE) == []
    assert len(extract_instances(probability, threshold=0.3, min_area=1, output_size=SIZE)) == 1


def test_regions_below_the_minimum_area_are_dropped() -> None:
    probability = _map_with_blobs((10, 10, 12, 12), (60, 60, 2, 2))

    instances = extract_instances(probability, min_area=100, output_size=SIZE)

    assert len(instances) == 1
    assert instances[0].area == 144


def test_regions_outside_the_disk_are_dropped() -> None:
    disk = Disk(center_x=63.5, center_y=63.5, radius=30.0)
    probability = _map_with_blobs((55, 55, 10, 10), (2, 2, 10, 10))

    instances = extract_instances(
        probability, min_area=1, disk=disk, disk_margin=0.0, output_size=SIZE
    )

    assert len(instances) == 1  # only the one near the centre survives


def test_instances_are_ordered_by_descending_score() -> None:
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    probability[10:22, 10:22] = 0.6
    probability[60:72, 60:72] = 0.95

    instances = extract_instances(probability, min_area=1, output_size=SIZE)

    assert [round(instance.score, 2) for instance in instances] == [0.95, 0.6]


def test_the_score_is_the_mean_probability_of_the_region() -> None:
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    probability[10:12, 10:12] = [[0.6, 0.8], [1.0, 0.6]]

    instances = extract_instances(probability, min_area=1, output_size=SIZE)

    assert instances[0].score == pytest.approx(0.75)


def test_masks_are_scaled_up_to_the_submission_resolution() -> None:
    probability = _map_with_blobs((10, 10, 12, 12))

    instances = extract_instances(probability, min_area=1, output_size=2048)

    assert instances[0].mask.shape == (2048, 2048)
    # A 12x12 region at 128 becomes 192x192 at 2048.
    assert instances[0].area == pytest.approx(192 * 192, rel=0.02)


def test_scaling_up_preserves_the_position() -> None:
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    probability[0:8, 0:8] = 0.9  # top-left corner

    mask = extract_instances(probability, min_area=1, output_size=1024)[0].mask

    assert mask[0, 0]
    assert not mask[-1, -1]
    # 8 of 128 scales to 64 of 1024.
    assert mask[:64, :64].all()
    assert not mask[65:, :].any()


def test_a_three_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="Expected a 2-D probability map"):
        extract_instances(np.zeros((1, SIZE, SIZE), dtype=np.float32))


def test_extracted_instances_never_overlap(tmp_path: Path) -> None:
    """The property the whole design exists for."""
    rng = np.random.default_rng(0)
    probability = rng.random((SIZE, SIZE)).astype(np.float32)

    instances = extract_instances(probability, threshold=0.55, min_area=1, output_size=256)

    rows = instances_to_rows("20140609195854Bh", instances)
    assert len(rows) > 5  # a random map fragments into many regions
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert find_overlaps(frame, 256, 256) == []

    path = tmp_path / "submission.csv"
    write_submission(rows, path)
    check_no_overlap(path, 256, 256)


def test_instances_to_rows_numbers_from_one() -> None:
    probability = _map_with_blobs((10, 10, 12, 12), (60, 60, 12, 12))
    instances = extract_instances(probability, min_area=1, output_size=SIZE)

    rows = instances_to_rows("20140609195854Bh", instances)

    assert [row[0] for row in rows] == [
        "20140609195854Bh_1",
        "20140609195854Bh_2",
    ]


def _square(top: int, left: int, size: int) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top : top + size, left : left + size] = True
    return mask


def test_one_prediction_covering_two_ground_truths_counts_as_fusion() -> None:
    prediction = _square(10, 10, 40)
    ground_truth = [_square(12, 12, 15), _square(30, 30, 15)]

    fused, split = fusion_and_splitting([prediction], ground_truth)

    assert (fused, split) == (1, 0)


def test_two_predictions_covering_one_ground_truth_count_as_splitting() -> None:
    ground_truth = _square(10, 10, 40)
    predictions = [_square(12, 12, 15), _square(30, 30, 15)]

    fused, split = fusion_and_splitting(predictions, [ground_truth])

    assert (fused, split) == (0, 1)


def test_a_clean_one_to_one_match_counts_as_neither() -> None:
    fused, split = fusion_and_splitting([_square(10, 10, 20)], [_square(10, 10, 20)])

    assert (fused, split) == (0, 0)


def test_masks_that_barely_touch_are_not_counted() -> None:
    ground_truth = _square(10, 10, 40)
    # Overlaps the ground truth in 2 of its 400 pixels, far below the fraction.
    grazing = _square(48, 48, 20)

    fused, split = fusion_and_splitting([_square(10, 10, 40), grazing], [ground_truth])

    assert split == 0


def test_counting_with_nothing_on_one_side_is_zero() -> None:
    assert fusion_and_splitting([], [_square(10, 10, 20)]) == (0, 0)
    assert fusion_and_splitting([_square(10, 10, 20)], []) == (0, 0)


def test_the_limb_margin_is_measured_in_frame_pixels_not_map_pixels() -> None:
    """The margin has to mean the same distance whatever resolution it is read at.

    Two pieces of code produce instances from a map: the one that scores a
    checkpoint and the one that sweeps post-processing settings. Only the first
    used to convert the margin into the map's coordinates, so the sweep kept a
    ring twice as wide and the two could not be compared. The conversion now
    happens once, inside this function, and this pins it there.
    """
    disk = Disk(center_x=63.5, center_y=63.5, radius=30.0)
    # A blob whose nearest pixel sits about 6 pixels of the map past the limb,
    # which is 12 pixels of a frame twice the map's size.
    probability = _map_with_blobs((63, 99, 4, 4))
    at_frame_scale = {"min_area": 1, "disk": disk, "output_size": SIZE * 2}

    inside = extract_instances(probability, disk_margin=16.0, **at_frame_scale)
    outside = extract_instances(probability, disk_margin=8.0, **at_frame_scale)

    # 16 frame pixels is 8 map pixels, which reaches the blob; 8 is 4, which
    # does not. Read as map pixels both would have reached it.
    assert len(inside) == 1
    assert len(outside) == 0
