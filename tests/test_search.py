"""Tests for the post-processing parameter sweep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filament.metrics.pq import PQResult
from filament.postprocess.search import (
    Setting,
    SweepPoint,
    SweepResult,
    grid,
    load_maps,
    predict_from_maps,
    save_map,
    sweep,
)
from filament.submit.rle import SUBMISSION_COLUMNS, mask_to_rle

SIZE = 128
OUTPUT_SIZE = 256
STEM = "20140609195854Bh"
ANNOTATOR_IMAGE = f"040301-{STEM}"


def _point(pq: float, **values: object) -> SweepPoint:
    """A sweep point with only its PQ filled in."""
    return SweepPoint(
        setting=Setting(dict(values)),
        pq=PQResult(pq=pq, sq=0.7, rq=pq / 0.7, tp=10, fp=2, fn=3),
        predictions=12,
        seconds=0.1,
    )


def _probability_map(*boxes: tuple[int, int, int, int], value: float = 0.9) -> np.ndarray:
    probability = np.zeros((SIZE, SIZE), dtype=np.float32)
    for top, left, height, width in boxes:
        probability[top : top + height, left : left + width] = value
    return probability


def _ground_truth(*boxes: tuple[int, int, int, int]) -> pd.DataFrame:
    """Ground truth at the output resolution, from boxes in map coordinates."""
    scale = OUTPUT_SIZE // SIZE
    records = []
    for index, (top, left, height, width) in enumerate(boxes, start=1):
        mask = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
        mask[
            top * scale : (top + height) * scale,
            left * scale : (left + width) * scale,
        ] = True
        records.append((f"{ANNOTATOR_IMAGE}_{index}", mask_to_rle(mask)))
    return pd.DataFrame(records, columns=list(SUBMISSION_COLUMNS))


def test_grid_is_the_cartesian_product() -> None:
    settings = grid(threshold=[0.4, 0.5], min_area=[100, 200, 300])

    assert len(settings) == 6
    assert settings[0].values == {"threshold": 0.4, "min_area": 100}
    assert {tuple(sorted(item.values.items())) for item in settings} == {
        (("min_area", area), ("threshold", threshold))
        for threshold in (0.4, 0.5)
        for area in (100, 200, 300)
    }


def test_a_setting_prints_its_parameters() -> None:
    assert str(Setting({"threshold": 0.5, "min_area": 200})) == "min_area=200 threshold=0.5"


def test_a_sweep_scores_every_setting() -> None:
    box = (20, 20, 10, 40)
    maps = {STEM: _probability_map(box)}
    gt = _ground_truth(box)

    result = sweep(maps, gt, grid(threshold=[0.5, 0.95], min_area=[1]), output_size=OUTPUT_SIZE)

    assert len(result.points) == 2
    # 0.9 clears a threshold of 0.5 and not one of 0.95.
    by_threshold = {point.setting.values["threshold"]: point for point in result.points}
    assert by_threshold[0.5].pq.tp == 1
    assert by_threshold[0.95].pq.tp == 0
    assert by_threshold[0.95].predictions == 0


def test_a_sweep_records_the_breakdown_not_only_pq() -> None:
    box = (20, 20, 10, 40)
    result = sweep(
        {STEM: _probability_map(box)},
        _ground_truth(box),
        grid(threshold=[0.5]),
        output_size=OUTPUT_SIZE,
    )

    row = result.table.iloc[0]
    for column in ("pq", "sq", "rq", "tp", "fp", "fn", "predictions", "seconds"):
        assert column in row


def test_the_table_is_sorted_by_descending_pq() -> None:
    result = SweepResult([_point(0.1, threshold=0.3), _point(0.4, threshold=0.5)])

    assert result.table["pq"].tolist() == [0.4, 0.1]


def test_best_returns_the_highest_scoring_point() -> None:
    result = SweepResult([_point(0.1, threshold=0.3), _point(0.4, threshold=0.5)])

    assert result.best.setting.values["threshold"] == 0.5


def test_best_of_an_empty_sweep_is_rejected() -> None:
    with pytest.raises(ValueError, match="sweep is empty"):
        _ = SweepResult().best


def test_the_plateau_is_the_middle_of_the_near_best_run() -> None:
    # Near-best at 0.4, 0.5 and 0.6; the single best is 0.5 but 0.6 ties it
    # within tolerance, so the middle of the run is what transfers.
    result = SweepResult(
        [
            _point(0.20, threshold=0.2),
            _point(0.25, threshold=0.3),
            _point(0.299, threshold=0.4),
            _point(0.300, threshold=0.5),
            _point(0.298, threshold=0.6),
            _point(0.24, threshold=0.7),
        ]
    )

    value, plateau = result.plateau("threshold")

    assert plateau == [0.4, 0.5, 0.6]
    assert value == 0.5


def test_the_plateau_prefers_the_wider_run_over_the_single_best() -> None:
    # 0.2 is the outright best but stands alone; 0.5-0.7 is a wide near-best
    # stretch. The wide stretch is the safer choice.
    result = SweepResult(
        [
            _point(0.300, threshold=0.2),
            _point(0.10, threshold=0.3),
            _point(0.10, threshold=0.4),
            _point(0.297, threshold=0.5),
            _point(0.296, threshold=0.6),
            _point(0.298, threshold=0.7),
        ]
    )

    value, plateau = result.plateau("threshold")

    assert plateau == [0.5, 0.6, 0.7]
    assert value == 0.6


def test_the_plateau_marginalises_the_other_parameters() -> None:
    """Each value keeps its best score over the other axes, not its average."""
    result = SweepResult(
        [
            _point(0.10, threshold=0.4, min_area=100),
            _point(0.30, threshold=0.4, min_area=200),
            _point(0.29, threshold=0.5, min_area=100),
            _point(0.05, threshold=0.5, min_area=200),
        ]
    )

    value, plateau = result.plateau("threshold", tolerance=0.02)

    assert plateau == [0.4, 0.5]
    assert value == 0.5


def test_the_plateau_of_an_empty_sweep_is_rejected() -> None:
    with pytest.raises(ValueError, match="sweep is empty"):
        SweepResult().plateau("threshold")


def test_a_custom_builder_replaces_the_default_chain() -> None:
    """The hook a joining step will use."""
    calls: list[float] = []

    def builder(probability: object, disk: object, values: dict[str, object]) -> list[object]:
        calls.append(values["threshold"])
        return []

    sweep(
        {STEM: _probability_map((20, 20, 10, 10))},
        _ground_truth((20, 20, 10, 10)),
        grid(threshold=[0.3, 0.6]),
        output_size=OUTPUT_SIZE,
        build=builder,
    )

    assert sorted(calls) == [0.3, 0.6]


def test_load_maps_reads_npy_files_and_widens_them(tmp_path: Path) -> None:
    np.save(tmp_path / f"{STEM}.npy", _probability_map((10, 10, 5, 5)).astype(np.float16))
    np.save(tmp_path / "20150125172714Mh.npy", _probability_map((1, 1, 2, 2)).astype(np.float16))

    maps = load_maps(tmp_path)

    assert set(maps) == {STEM, "20150125172714Mh"}
    assert maps[STEM].dtype == np.float32


def test_load_maps_can_be_restricted_to_some_stems(tmp_path: Path) -> None:
    np.save(tmp_path / f"{STEM}.npy", _probability_map((10, 10, 5, 5)).astype(np.float16))
    np.save(tmp_path / "20150125172714Mh.npy", _probability_map((1, 1, 2, 2)).astype(np.float16))

    assert set(load_maps(tmp_path, stems=[STEM])) == {STEM}


def test_load_maps_reports_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No probability maps"):
        load_maps(tmp_path)


def test_a_map_survives_the_round_trip_through_a_byte(tmp_path: Path) -> None:
    """A byte per pixel resolves to 1/255, far finer than any threshold in use."""
    rng = np.random.default_rng(0)
    probability = rng.random((64, 64)).astype(np.float32)

    save_map(probability, tmp_path / "frame.npy")
    restored = load_maps(tmp_path)["frame"]

    assert restored.dtype == np.float32
    assert np.abs(restored - probability).max() <= 1.0 / 255.0


def test_the_stored_map_is_a_byte_per_pixel(tmp_path: Path) -> None:
    save_map(np.zeros((128, 128), dtype=np.float32), tmp_path / "frame.npy")

    stored = np.load(tmp_path / "frame.npy")

    assert stored.dtype == np.uint8
    # Header aside, one byte per pixel rather than the two float16 costs.
    assert (tmp_path / "frame.npy").stat().st_size < 128 * 128 * 1.1


def test_float_maps_from_the_earlier_phases_still_load(tmp_path: Path) -> None:
    probability = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    np.save(tmp_path / "frame.npy", probability.astype(np.float16))

    restored = load_maps(tmp_path)["frame"]

    assert restored.dtype == np.float32
    assert np.abs(restored - probability).max() < 1e-3


def test_saving_rejects_anything_that_is_not_a_map(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2-D probability map"):
        save_map(np.zeros((2, 8, 8), dtype=np.float32), tmp_path / "frame.npy")


def test_one_setting_can_be_applied_outside_a_sweep() -> None:
    """The setting a sweep settles on has to reach the test frames, which have
    no ground truth to sweep against."""
    maps = {STEM: _probability_map((10, 10, 40, 20))}
    setting = Setting({"threshold": 0.5, "min_area": 10})

    frame = predict_from_maps(maps, setting, output_size=OUTPUT_SIZE)

    assert list(frame.columns) == list(SUBMISSION_COLUMNS)
    assert len(frame) == 1
    assert frame["filament_id"].iloc[0] == f"{STEM}_1"


def test_a_sweep_point_and_a_direct_call_agree() -> None:
    maps = {STEM: _probability_map((10, 10, 40, 20), (60, 60, 90, 70))}
    setting = Setting({"threshold": 0.5, "min_area": 10})
    gt_df = pd.DataFrame(
        [(f"{ANNOTATOR_IMAGE}_1", mask_to_rle(np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)))],
        columns=list(SUBMISSION_COLUMNS),
    )

    direct = predict_from_maps(maps, setting, output_size=OUTPUT_SIZE)
    swept = sweep(maps, gt_df, [setting], output_size=OUTPUT_SIZE)

    assert swept.points[0].predictions == len(direct)
