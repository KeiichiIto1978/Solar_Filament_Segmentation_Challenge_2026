"""Tests for solar disk detection."""

from __future__ import annotations

import numpy as np
import pytest

from filament.data.coco import load_annotations
from filament.data.disk import (
    DEFAULT_MASK_MARGIN,
    Disk,
    DiskNotFoundError,
    detect_disk,
    radial_profile,
)
from filament.data.image import load_grayscale
from filament.paths import ProjectPaths

SIDE = 512
TRUE_RADIUS = 225
DISK_LEVEL = 140
SKY_LEVEL = 5


def _synthetic_frame(
    radius: int = TRUE_RADIUS,
    side: int = SIDE,
    halo: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """A frame with a flat disk on a dark sky, optionally inside a bright ring.

    Args:
        radius: Radius of the disk.
        side: Frame side length.
        halo: ``(inner, outer, level)`` of a ring outside the disk, which is
            what breaks threshold-based detection on the real frames.
    """
    rows = np.arange(side)[:, None]
    columns = np.arange(side)[None, :]
    distance = np.hypot(columns - (side - 1) / 2, rows - (side - 1) / 2)

    frame = np.full((side, side), SKY_LEVEL, dtype=np.uint8)
    if halo is not None:
        inner, outer, level = halo
        frame[(distance >= inner) & (distance < outer)] = level
    # Limb darkening: the disk is dimmer towards its edge, as in a real frame.
    inside = distance <= radius
    falloff = 1.0 - 0.35 * (distance / max(radius, 1)) ** 2
    frame[inside] = (DISK_LEVEL * falloff[inside]).astype(np.uint8)
    return frame


def test_detect_disk_finds_a_synthetic_limb() -> None:
    disk = detect_disk(_synthetic_frame(), step=1)

    assert disk.radius == pytest.approx(TRUE_RADIUS, abs=2)
    assert disk.center_x == pytest.approx((SIDE - 1) / 2)
    assert disk.center_y == pytest.approx((SIDE - 1) / 2)


def test_a_bright_halo_outside_the_limb_does_not_move_the_radius() -> None:
    # The ring is brighter than the outer part of the disk, which is exactly
    # the case that defeats thresholding.
    frame = _synthetic_frame(halo=(TRUE_RADIUS + 5, 250, 110))

    disk = detect_disk(frame, step=1)

    assert disk.radius == pytest.approx(TRUE_RADIUS, abs=3)


def test_a_dark_streak_near_the_centre_does_not_move_the_radius() -> None:
    # A filament produces a steep local edge; averaging over the annulus
    # dilutes it, whereas a single scanline would be fooled.
    frame = _synthetic_frame()
    frame[250:262, 140:370] = 45

    disk = detect_disk(frame, step=1)

    assert disk.radius == pytest.approx(TRUE_RADIUS, abs=3)


def test_a_frame_without_a_limb_is_rejected() -> None:
    with pytest.raises(DiskNotFoundError, match="does not look like a full disk"):
        detect_disk(np.full((SIDE, SIDE), DISK_LEVEL, dtype=np.uint8), step=1)


def test_a_frame_too_small_to_search_is_rejected() -> None:
    with pytest.raises(DiskNotFoundError, match="no room to search"):
        detect_disk(np.zeros((16, 16), dtype=np.uint8))


def test_detect_disk_rejects_a_stack() -> None:
    with pytest.raises(ValueError, match="Expected a 2-D frame"):
        detect_disk(np.zeros((3, SIDE, SIDE), dtype=np.uint8))


def test_radial_profile_falls_off_at_the_limb() -> None:
    inner_radii, profile = radial_profile(_synthetic_frame(), low=180, high=250, step=1)

    inside = profile[inner_radii < TRUE_RADIUS - 5]
    outside = profile[inner_radii > TRUE_RADIUS + 5]
    assert inside.min() > outside.max()


def test_the_mask_covers_the_disk_and_nothing_beyond_the_margin() -> None:
    disk = Disk(center_x=127.5, center_y=127.5, radius=100.0)

    mask = disk.mask(256, 256, margin=0.0)

    assert mask[127, 127]
    # The left rim sits at x = 127.5 - 100 = 27.5, between two pixel centres.
    assert mask[127, 28]
    assert not mask[127, 27]
    assert mask.sum() == pytest.approx(disk.area, rel=0.01)


def test_the_default_margin_grows_the_mask() -> None:
    disk = Disk(center_x=127.5, center_y=127.5, radius=100.0)

    assert disk.mask(256, 256).sum() > disk.mask(256, 256, margin=0.0).sum()
    assert disk.contains(127.5, 127.5 + 100 + DEFAULT_MASK_MARGIN - 1)
    assert not disk.contains(127.5, 127.5 + 100 + DEFAULT_MASK_MARGIN + 1)


def test_scaling_a_disk_scales_centre_and_radius() -> None:
    disk = Disk(center_x=1023.5, center_y=1023.5, radius=904.0)

    half = disk.scaled(0.5)

    assert half.center_x == pytest.approx(511.75)
    assert half.radius == pytest.approx(452.0)


@pytest.mark.dataset
def test_the_radius_is_consistent_across_the_training_set(paths: ProjectPaths) -> None:
    """Every frame in this dataset is scaled to the same disk size.

    A radius outside this band would mean the detector, not the data, changed.
    """
    dataset = load_annotations(paths.train_annotations)
    sample = dataset.stems[::30]

    radii = [
        detect_disk(load_grayscale(paths.train_images / f"{stem}.jpeg")).radius for stem in sample
    ]

    assert min(radii) >= 896
    assert max(radii) <= 912


@pytest.mark.dataset
def test_every_annotated_filament_lies_on_the_detected_disk(paths: ProjectPaths) -> None:
    """The mask must not clip ground truth, margin included."""
    dataset = load_annotations(paths.train_annotations)
    by_stem = dataset.by_stem()

    for stem in dataset.stems[::30]:
        disk = detect_disk(load_grayscale(paths.train_images / f"{stem}.jpeg"))
        for entry in by_stem[stem]:
            for annotation in entry.annotations:
                points = np.asarray(annotation.segmentation[0], dtype=np.float32).reshape(-1, 2)
                furthest = np.hypot(
                    points[:, 0] - disk.center_x, points[:, 1] - disk.center_y
                ).max()
                assert furthest <= disk.radius + DEFAULT_MASK_MARGIN, (
                    f"{stem} {annotation.annotation_id}"
                )
