"""Tests for the features a candidate filament is judged by.

Each test builds a frame where the expected value is known by construction: a
bar is long and thin, a disk is round, and a region painted darker than its
surroundings has a known negative contrast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from filament.data.disk import Disk
from filament.postprocess.candidates import FEATURES, frame_features, instance_features
from filament.postprocess.instances import Instance

SIDE = 400
DISK = Disk(center_x=200.0, center_y=200.0, radius=190.0)


def _candidate(mask: np.ndarray, probability: float = 0.9) -> tuple[Instance, np.ndarray]:
    probability_map = np.where(mask, probability, 0.0).astype(np.float32)
    instance = Instance(mask=mask, score=probability, area=int(mask.sum()))
    return instance, probability_map


def _bar(top: int = 195, left: int = 100, height: int = 4, width: int = 200) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


def _round(radius: int = 30, centre: tuple[int, int] = (200, 200)) -> np.ndarray:
    rows, columns = np.mgrid[:SIDE, :SIDE]
    return np.hypot(columns - centre[0], rows - centre[1]) <= radius


def _frame(dark: np.ndarray | None = None, inside: int = 100, outside: int = 150) -> np.ndarray:
    frame = np.full((SIDE, SIDE), outside, dtype=np.uint8)
    if dark is not None:
        frame[dark] = inside
    return frame


def test_every_feature_but_the_frame_count_is_computed() -> None:
    instance, probability = _candidate(_bar())

    features = instance_features(instance, probability, _frame(), DISK)

    assert set(features) == set(FEATURES) - {"frame_candidates"}


def test_a_uniform_region_has_its_probability_as_score_and_no_spread() -> None:
    instance, probability = _candidate(_bar(), probability=0.9)

    features = instance_features(instance, probability, _frame(), DISK)

    assert features["score"] == pytest.approx(0.9)
    assert features["peak"] == pytest.approx(0.9)
    assert features["probability_spread"] == pytest.approx(0.0, abs=1e-6)
    assert features["confident_share"] == 1.0
    assert features["area"] == 800


def test_a_thin_bar_is_elongated_and_a_disk_is_not() -> None:
    """A 4 x 200 bar has second moments in the ratio of its sides' squares, so
    its axes stand about 50 to 1; a disk's are equal."""
    bar_instance, bar_probability = _candidate(_bar())
    round_instance, round_probability = _candidate(_round())

    bar = instance_features(bar_instance, bar_probability, _frame(), DISK)
    disk = instance_features(round_instance, round_probability, _frame(), DISK)

    assert bar["elongation"] == pytest.approx(50.0, rel=0.05)
    assert disk["elongation"] == pytest.approx(1.0, abs=0.02)
    assert bar["extent"] == pytest.approx(1.0)
    # pi r^2 over the (2r + 1)^2 pixels of its bounding box, at r = 30.
    assert disk["extent"] == pytest.approx(math.pi * 30**2 / 61**2, rel=0.01)
    # A digital circle's traced outline is a little longer than 2 pi r.
    assert disk["compactness"] == pytest.approx(1.0, abs=0.15)
    assert bar["compactness"] > 10


def test_a_region_darker_than_its_surroundings_has_a_negative_contrast() -> None:
    """Painted at 100 on a background of 150: (100 - 150) / 150."""
    mask = _bar()
    instance, probability = _candidate(mask)

    features = instance_features(instance, probability, _frame(dark=mask), DISK)

    assert features["brightness_contrast"] == pytest.approx(-50 / 150)


def test_a_region_no_darker_than_its_surroundings_has_no_contrast() -> None:
    instance, probability = _candidate(_bar())

    features = instance_features(instance, probability, _frame(), DISK)

    assert features["brightness_contrast"] == pytest.approx(0.0)


def test_the_sky_beyond_the_limb_is_left_out_of_the_comparison() -> None:
    """A candidate touching the limb: the sky beside it is black, and counting
    it would make a filament look bright. Only the disk side should count."""
    rows, columns = np.mgrid[:SIDE, :SIDE]
    on_disk = np.hypot(columns - DISK.center_x, rows - DISK.center_y) <= DISK.radius
    frame = np.where(on_disk, 150, 5).astype(np.uint8)
    mask = _bar(top=195, left=370, height=4, width=19) & on_disk
    frame[mask] = 100
    instance, probability = _candidate(mask)

    features = instance_features(instance, probability, frame, DISK)

    assert features["brightness_contrast"] == pytest.approx(-50 / 150)


def test_the_position_is_a_fraction_of_the_radius() -> None:
    instance, probability = _candidate(_round(radius=5, centre=(295, 200)))

    features = instance_features(instance, probability, _frame(), DISK)

    assert features["radial_position"] == pytest.approx(95 / 190, abs=0.01)


def test_mismatched_resolutions_are_rejected() -> None:
    instance, probability = _candidate(_bar())

    with pytest.raises(ValueError, match="same resolution"):
        instance_features(instance, probability[:200, :200], _frame(), DISK)


def test_the_frame_count_is_the_number_of_candidates_in_the_frame() -> None:
    candidates: list[dict[str, object]] = [{}, {}, {}]

    frame_features(candidates)

    assert [candidate["frame_candidates"] for candidate in candidates] == [3.0, 3.0, 3.0]
