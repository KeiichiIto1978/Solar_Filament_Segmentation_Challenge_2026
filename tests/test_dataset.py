"""Tests for the training dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from filament.data.coco import Annotation, AnnotatorImage, load_annotations
from filament.data.dataset import (
    Augmentation,
    FilamentSegmentationDataset,
    build_target,
    build_union_target,
    build_vote_target,
)
from filament.data.split import load_fold
from filament.paths import ProjectPaths

SIZE = 256


def test_augmentation_keeps_image_and_target_aligned() -> None:
    image = np.zeros((2, 8, 8), dtype=np.float32)
    target = np.zeros((8, 8), dtype=np.uint8)
    # One marked pixel in both: after any flip or turn they must still agree.
    image[:, 1, 2] = 1.0
    target[1, 2] = 1

    for seed in range(12):
        moved_image, moved_target = Augmentation().apply(
            image.copy(), target.copy(), np.random.default_rng(seed)
        )
        assert np.array_equal(moved_image[0] > 0, moved_target > 0)
        assert moved_target.sum() == 1


def test_augmentation_can_be_switched_off() -> None:
    image = np.zeros((2, 8, 8), dtype=np.float32)
    image[:, 0, 0] = 1.0
    target = np.zeros((8, 8), dtype=np.uint8)
    off = Augmentation(horizontal_flip=False, vertical_flip=False, quarter_turns=False)

    moved_image, moved_target = off.apply(image, target, np.random.default_rng(0))

    assert np.array_equal(moved_image, image)
    assert np.array_equal(moved_target, target)


def test_the_same_seed_reproduces_the_same_transform() -> None:
    image = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    target = np.arange(64, dtype=np.uint8).reshape(8, 8)

    first = Augmentation().apply(image.copy(), target.copy(), np.random.default_rng(3))
    second = Augmentation().apply(image.copy(), target.copy(), np.random.default_rng(3))

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


@pytest.mark.dataset
def test_build_target_marks_every_filament_of_one_annotator(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    entry = dataset.annotator_images[0]

    target = build_target(entry, size=entry.height)

    assert target.shape == (entry.height, entry.width)
    assert set(np.unique(target)) <= {0, 1}
    # Built at full resolution, the mask area is the sum of the annotated
    # areas, minus whatever two filaments of one annotator happen to share.
    total = sum(annotation.area for annotation in entry.annotations)
    assert target.sum() == pytest.approx(total, rel=0.02)


@pytest.mark.dataset
def test_a_downscaled_target_keeps_roughly_the_scaled_area(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    entry = dataset.annotator_images[0]

    half = build_target(entry, size=entry.height // 2)
    full = build_target(entry, size=entry.height)

    # Nearest-neighbour downscaling keeps a quarter of the pixels, give or
    # take the thin parts it drops.
    assert half.sum() == pytest.approx(full.sum() / 4, rel=0.25)


@pytest.mark.dataset
def test_a_sample_is_a_two_channel_frame_and_a_binary_mask(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    fold = load_fold(0, paths.splits_dir)

    subset = FilamentSegmentationDataset(dataset, paths.train_images, stems=fold.val[:3], size=SIZE)
    sample = subset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (2, SIZE, SIZE)
    assert sample["image"].dtype == torch.float32
    assert 0.0 <= float(sample["image"].min()) <= float(sample["image"].max()) <= 1.0
    assert sample["target"].shape == (1, SIZE, SIZE)
    assert set(torch.unique(sample["target"]).tolist()) <= {0.0, 1.0}
    assert sample["stem"] in fold.val[:3]


@pytest.mark.dataset
def test_one_sample_per_annotator_not_per_image(paths: ProjectPaths) -> None:
    """A frame three people annotated must appear three times."""
    dataset = load_annotations(paths.train_annotations)
    stem = next(s for s, entries in dataset.by_stem().items() if len(entries) == 3)

    subset = FilamentSegmentationDataset(dataset, paths.train_images, stems=[stem], size=SIZE)

    assert len(subset) == 3
    assert len({subset[index]["image_id"] for index in range(3)}) == 3
    # Same frame, different targets: that disagreement is real and is kept.
    targets = [subset[index]["target"] for index in range(3)]
    assert not torch.equal(targets[0], targets[1]) or not torch.equal(targets[1], targets[2])


@pytest.mark.dataset
def test_the_target_is_empty_outside_the_disk(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    fold = load_fold(0, paths.splits_dir)

    subset = FilamentSegmentationDataset(dataset, paths.train_images, stems=fold.val[:1], size=SIZE)
    sample = subset[0]

    rows = torch.arange(SIZE)[:, None]
    columns = torch.arange(SIZE)[None, :]
    centre = (SIZE - 1) / 2
    # The detected radius is 904 of 2048, scaled to this size, plus the margin.
    outside = ((columns - centre) ** 2 + (rows - centre) ** 2) > ((0.45 * SIZE) ** 2)
    assert float(sample["target"][0][outside].sum()) == 0.0


@pytest.mark.dataset
def test_an_empty_stem_selection_is_rejected(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)

    with pytest.raises(ValueError, match="No annotator-images left"):
        FilamentSegmentationDataset(dataset, paths.train_images, stems=["nope"])


@pytest.mark.dataset
def test_the_dataset_works_through_a_dataloader(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    fold = load_fold(0, paths.splits_dir)

    subset = FilamentSegmentationDataset(
        dataset,
        paths.train_images,
        stems=fold.train[:4],
        size=SIZE,
        augmentation=Augmentation(),
    )
    # num_workers stays 0: on Windows each worker would re-import and re-read
    # the annotation file, which is 48 MB.
    loader = torch.utils.data.DataLoader(subset, batch_size=2, num_workers=0)

    batch = next(iter(loader))

    assert batch["image"].shape == (2, 2, SIZE, SIZE)
    assert batch["target"].shape == (2, 1, SIZE, SIZE)
    assert len(batch["image_id"]) == 2


def _annotator_image(
    stem: str, annotator: str, polygons: list[list[float]], size: int
) -> AnnotatorImage:
    """One annotator's view of one frame, built by hand."""
    image_id = f"{annotator}-{stem}"
    return AnnotatorImage(
        image_id=image_id,
        annotator=annotator,
        stem=stem,
        file_name=f"{stem}.jpeg",
        height=size,
        width=size,
        annotations=[
            Annotation(
                annotation_id=f"{image_id}_{index}",
                annotator_image_id=image_id,
                category_id=1,
                segmentation=[polygon],
                bbox=(0.0, 0.0, 0.0, 0.0),
                area=0.0,
                spine=[],
            )
            for index, polygon in enumerate(polygons)
        ],
    )


def test_a_vote_share_counts_how_many_annotators_drew_each_pixel() -> None:
    """Three annotators, a square all three drew and one only two drew."""
    size = 32
    shared = [4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0]
    extra = [20.0, 20.0, 28.0, 20.0, 28.0, 28.0, 20.0, 28.0]
    entries = [
        _annotator_image("frame", "a", [shared, extra], size),
        _annotator_image("frame", "b", [shared, extra], size),
        _annotator_image("frame", "c", [shared], size),
    ]

    votes = build_vote_target(entries, size=size)

    assert votes.dtype == np.float32
    assert votes[8, 8] == pytest.approx(1.0)
    assert votes[24, 24] == pytest.approx(2.0 / 3.0)
    assert votes[0, 0] == pytest.approx(0.0)


def test_a_vote_share_of_one_annotator_is_their_own_tracing() -> None:
    """With nobody to disagree, the share is the same zeros and ones as before."""
    size = 32
    square = [4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0]
    entry = _annotator_image("frame", "a", [square], size)

    votes = build_vote_target([entry], size=size)

    assert np.array_equal(votes, build_target(entry, size=size).astype(np.float32))


def test_a_vote_share_needs_annotators_of_one_frame() -> None:
    size = 16
    square = [2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]

    with pytest.raises(ValueError, match="at least one annotator"):
        build_vote_target([], size=size)
    with pytest.raises(ValueError, match="Expected one frame"):
        build_vote_target(
            [
                _annotator_image("one", "a", [square], size),
                _annotator_image("two", "a", [square], size),
            ],
            size=size,
        )


@pytest.mark.dataset
def test_vote_targets_keep_one_sample_per_annotator(paths: ProjectPaths) -> None:
    """The weighting the metric applies must survive: three annotators, three
    samples. Only what they ask for changes -- now all three agree."""
    dataset = load_annotations(paths.train_annotations)
    stem = next(s for s, entries in dataset.by_stem().items() if len(entries) == 3)

    subset = FilamentSegmentationDataset(
        dataset, paths.train_images, stems=[stem], size=SIZE, vote_targets=True
    )

    assert len(subset) == 3
    targets = [subset[index]["target"] for index in range(3)]
    assert torch.equal(targets[0], targets[1])
    assert torch.equal(targets[1], targets[2])
    # A third, two thirds and one are all a frame three people saw can hold.
    values = sorted(torch.unique(targets[0]).tolist())
    assert values == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0], abs=1e-6)


@pytest.mark.dataset
def test_vote_targets_change_nothing_on_a_single_annotator_frame(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    stem = dataset.single_annotator_stems()[0]

    common = {"dataset": dataset, "images_dir": paths.train_images, "stems": [stem], "size": SIZE}
    per_annotator = FilamentSegmentationDataset(**common)
    shares = FilamentSegmentationDataset(**common, vote_targets=True)

    assert torch.equal(per_annotator[0]["target"], shares[0]["target"])


def test_a_union_keeps_every_pixel_any_annotator_drew() -> None:
    """Three annotators: a square all three drew, one only two drew, and one
    only the third drew. All three survive, as plain ones."""
    size = 32
    shared = [4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0]
    extra = [20.0, 20.0, 28.0, 20.0, 28.0, 28.0, 20.0, 28.0]
    lone = [20.0, 4.0, 28.0, 4.0, 28.0, 12.0, 20.0, 12.0]
    entries = [
        _annotator_image("frame", "a", [shared, extra], size),
        _annotator_image("frame", "b", [shared, extra], size),
        _annotator_image("frame", "c", [shared, lone], size),
    ]

    union = build_union_target(entries, size=size)

    assert union.dtype == np.uint8
    assert union[8, 8] == 1
    assert union[24, 24] == 1
    assert union[8, 24] == 1
    assert union[0, 0] == 0
    # Nothing beyond what someone drew: the union of the three masks, exactly.
    expected = np.zeros_like(union)
    for entry in entries:
        expected |= build_target(entry, size=size)
    assert np.array_equal(union, expected)


def test_a_union_of_one_annotator_is_their_own_tracing() -> None:
    size = 32
    square = [4.0, 4.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0]
    entry = _annotator_image("frame", "a", [square], size)

    assert np.array_equal(build_union_target([entry], size=size), build_target(entry, size=size))


def test_a_union_needs_annotators_of_one_frame() -> None:
    size = 16
    square = [2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]

    with pytest.raises(ValueError, match="at least one annotator"):
        build_union_target([], size=size)
    with pytest.raises(ValueError, match="Expected one frame"):
        build_union_target(
            [
                _annotator_image("one", "a", [square], size),
                _annotator_image("two", "a", [square], size),
            ],
            size=size,
        )


@pytest.mark.dataset
def test_votes_and_union_cannot_both_be_asked_for(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    with pytest.raises(ValueError, match="not both"):
        FilamentSegmentationDataset(
            dataset, paths.train_images, size=SIZE, vote_targets=True, union_targets=True
        )


@pytest.mark.dataset
def test_union_targets_keep_one_sample_per_annotator(paths: ProjectPaths) -> None:
    """Three annotators, three samples, one shared binary answer that covers
    each annotator's own tracing."""
    dataset = load_annotations(paths.train_annotations)
    stem = next(s for s, entries in dataset.by_stem().items() if len(entries) == 3)

    common = {"dataset": dataset, "images_dir": paths.train_images, "stems": [stem], "size": SIZE}
    union = FilamentSegmentationDataset(**common, union_targets=True)
    own = FilamentSegmentationDataset(**common)

    assert len(union) == 3
    targets = [union[index]["target"] for index in range(3)]
    assert torch.equal(targets[0], targets[1])
    assert torch.equal(targets[1], targets[2])
    assert sorted(torch.unique(targets[0]).tolist()) == [0.0, 1.0]
    for index in range(3):
        assert bool((own[index]["target"] <= targets[0]).all())
