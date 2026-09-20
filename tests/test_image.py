"""Tests for frame reading and preparation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from filament.data.image import (
    apply_clahe,
    load_grayscale,
    resize,
    to_model_input,
)
from filament.paths import ProjectPaths


@pytest.fixture
def frame() -> np.ndarray:
    """A synthetic frame with a bright disk and a dark streak across it."""
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(image, (128, 128), 100, 180, thickness=-1)
    image[120:126, 60:190] = 40
    return image


def test_load_grayscale_reads_a_single_channel(tmp_path: Path, frame: np.ndarray) -> None:
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), frame)

    loaded = load_grayscale(path)

    assert loaded.ndim == 2
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, frame)


def test_load_grayscale_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not read an image"):
        load_grayscale(tmp_path / "absent.jpeg")


def test_resize_changes_the_side_length(frame: np.ndarray) -> None:
    assert resize(frame, 128).shape == (128, 128)


def test_resize_of_an_already_sized_image_is_a_no_op(frame: np.ndarray) -> None:
    assert resize(frame, 256) is frame


def test_resizing_a_mask_invents_no_intermediate_labels() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:150, 100:150] = 1

    resized = resize(mask, 128, mask=True)

    # Linear interpolation would produce values between 0 and 1; nearest
    # neighbour keeps the label set intact.
    assert set(np.unique(resized)) == {0, 1}


def test_resize_rejects_a_stack(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Expected a 2-D image"):
        resize(np.stack([frame, frame]), 128)


def test_clahe_raises_the_local_contrast(frame: np.ndarray) -> None:
    equalised = apply_clahe(frame)

    assert equalised.dtype == np.uint8
    assert equalised.shape == frame.shape
    assert equalised.std() > frame.std()


def test_clahe_rejects_non_uint8_input(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="expects uint8"):
        apply_clahe(frame.astype(np.float32))


def test_model_input_has_two_scaled_channels(frame: np.ndarray) -> None:
    prepared = to_model_input(frame, size=128)

    assert prepared.shape == (2, 128, 128)
    assert prepared.dtype == np.float32
    assert prepared.min() >= 0.0
    assert prepared.max() <= 1.0


def test_the_first_model_channel_is_the_untouched_frame(frame: np.ndarray) -> None:
    prepared = to_model_input(frame, size=256)

    assert prepared[0] == pytest.approx(frame.astype(np.float32) / 255.0)
    assert not np.array_equal(prepared[0], prepared[1])


@pytest.mark.dataset
def test_a_real_frame_is_a_2048_square_of_grey_levels(paths: ProjectPaths) -> None:
    path = next(iter(sorted(paths.train_images.glob("*.jpeg"))))

    image = load_grayscale(path)

    assert image.shape == (2048, 2048)
    assert image.dtype == np.uint8
    # A full-disk frame is mostly empty sky with a bright disk in the middle.
    assert image.min() < 20
    assert image.max() > 200
