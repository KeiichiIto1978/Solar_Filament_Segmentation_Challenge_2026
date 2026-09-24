"""Tests for cutting one filament out of a frame.

The geometry is where this can go wrong quietly: a window that is not square
stretches a filament along one axis, a seed drawn in the wrong coordinates
points at the wrong place, and either would show up as a disappointing score
rather than as an error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from filament.data.coco import Annotation, AnnotatorImage, load_annotations
from filament.data.crops import (
    Box,
    FilamentCropDataset,
    box_of,
    build_input,
    crop_window,
    cut,
    list_crops,
    seed_mask,
)
from filament.data.split import load_fold
from filament.paths import ProjectPaths

FRAME = 2048
SIZE = 128


def test_a_box_is_the_tightest_fit_around_the_mask() -> None:
    mask = np.zeros((FRAME, FRAME), dtype=bool)
    mask[900:930, 800:1000] = True

    box = box_of(mask)

    assert box == Box(900, 800, 930, 1000)
    assert (box.height, box.width, box.side) == (30, 200, 200)


def test_an_empty_mask_has_no_box() -> None:
    with pytest.raises(ValueError, match="no box"):
        box_of(np.zeros((16, 16), dtype=bool))


def test_the_window_is_square_and_wider_than_the_box() -> None:
    """Square because resampling a long box to a square crop would stretch the
    filament, making its apparent thickness depend on which way it lies."""
    box = Box(900, 800, 930, 1000)

    window = crop_window(box, FRAME, context=1.5)

    assert window.height == window.width
    assert window.height == pytest.approx(box.side * 1.5, abs=1)
    # Still centred on the filament.
    assert window.centre == pytest.approx(box.centre, abs=1)


def test_a_filament_at_the_edge_keeps_its_magnification() -> None:
    """Shrinking the window at the limb would show that filament at a
    different scale from every other one."""
    box = Box(0, 0, 30, 200)

    window = crop_window(box, FRAME, context=1.5)

    assert window.top >= 0
    assert window.left >= 0
    assert window.height == window.width == pytest.approx(box.side * 1.5, abs=1)


def test_the_seed_marks_where_the_box_is_in_the_crop() -> None:
    box = Box(900, 800, 930, 1000)
    window = crop_window(box, FRAME)

    seed = seed_mask(box, window, SIZE, padding=0.0)

    rows = np.flatnonzero(seed.any(axis=1))
    columns = np.flatnonzero(seed.any(axis=0))
    scale = SIZE / window.height
    assert rows[0] == pytest.approx((box.top - window.top) * scale, abs=2)
    assert columns[0] == pytest.approx((box.left - window.left) * scale, abs=2)
    assert set(np.unique(seed)) <= {0.0, 1.0}


def test_padding_grows_the_seed() -> None:
    """A detector's box does not sit tightly, so the seed should not either."""
    box = Box(900, 800, 930, 1000)
    window = crop_window(box, FRAME)

    tight = seed_mask(box, window, SIZE, padding=0.0)
    loose = seed_mask(box, window, SIZE, padding=0.4)

    assert float(loose.sum()) > float(tight.sum())


def test_the_input_has_three_channels_in_range() -> None:
    rng = np.random.default_rng(0)
    frame = (rng.random((FRAME, FRAME)) * 255).astype(np.uint8)
    box = Box(900, 800, 930, 1000)

    prepared = build_input(frame, box, crop_window(box, FRAME), SIZE)

    assert prepared.shape == (3, SIZE, SIZE)
    assert prepared.dtype == np.float32
    assert 0.0 <= float(prepared.min()) <= float(prepared.max()) <= 1.0
    # The third channel is the seed and nothing else.
    assert set(np.unique(prepared[2])) <= {0.0, 1.0}


def test_a_mask_stays_binary_through_the_cut() -> None:
    mask = np.zeros((FRAME, FRAME), dtype=np.uint8)
    mask[900:930, 800:1000] = 1
    window = crop_window(box_of(mask.astype(bool)), FRAME)

    taken = cut(mask, window, SIZE, mask=True)

    assert set(np.unique(taken)) <= {0, 1}
    assert taken.any()


def _entry(stem: str, polygons: list[list[float]]) -> AnnotatorImage:
    return AnnotatorImage(
        image_id=f"a-{stem}",
        annotator="a",
        stem=stem,
        file_name=f"{stem}.jpeg",
        height=64,
        width=64,
        annotations=[
            Annotation(
                annotation_id=f"a-{stem}_{index}",
                annotator_image_id=f"a-{stem}",
                category_id=1,
                segmentation=[polygon],
                bbox=(0.0, 0.0, 0.0, 0.0),
                area=0.0,
                spine=[],
            )
            for index, polygon in enumerate(polygons)
        ],
    )


def test_one_sample_per_annotation_not_per_frame() -> None:
    """A frame with seven filaments gives seven crops, which is also how the
    metric counts -- and it makes a small filament weigh as much as a large
    one, which no whole-frame loss does."""
    from filament.data.coco import Dataset

    square = [4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0]
    other = [40.0, 40.0, 52.0, 40.0, 52.0, 52.0, 40.0, 52.0]
    dataset = Dataset(annotator_images=[_entry("frame", [square, other])], categories={})

    samples = list_crops(dataset)

    assert len(samples) == 2
    assert {sample.annotation_id for sample in samples} == {"a-frame_0", "a-frame_1"}


@pytest.mark.dataset
def test_a_crop_sample_is_three_channels_and_a_mask(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    fold = load_fold(0, paths.splits_dir)

    crops = FilamentCropDataset(
        dataset, paths.train_images, stems=fold.val[:2], size=SIZE, flips=False
    )
    sample = crops[0]

    assert sample["image"].shape == (3, SIZE, SIZE)
    assert sample["target"].shape == (1, SIZE, SIZE)
    assert set(torch.unique(sample["target"]).tolist()) <= {0.0, 1.0}
    # The filament the box points at must actually be in the crop.
    assert float(sample["target"].sum()) > 0.0


@pytest.mark.dataset
def test_the_crop_holds_the_filament_it_was_asked_about(paths: ProjectPaths) -> None:
    """The seed and the target have to agree, or the network is shown one
    filament and scored on another."""
    dataset = load_annotations(paths.train_annotations)
    fold = load_fold(0, paths.splits_dir)

    crops = FilamentCropDataset(
        dataset, paths.train_images, stems=fold.val[:3], size=SIZE, flips=False
    )

    for index in range(min(8, len(crops))):
        sample = crops[index]
        seed = sample["image"][2] > 0.5
        target = sample["target"][0] > 0.5
        overlap = float((seed & target).sum()) / max(float(target.sum()), 1.0)
        assert overlap > 0.9, f"crop {index}: the seed misses the filament it points at"


@pytest.mark.dataset
def test_an_empty_stem_selection_is_rejected(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)

    with pytest.raises(ValueError, match="No annotations left"):
        FilamentCropDataset(dataset, paths.train_images, stems=["nope"])
