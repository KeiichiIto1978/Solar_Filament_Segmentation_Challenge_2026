"""The training dataset for the semantic baseline.

One sample is **one annotator's view of one image**, not one image. That is the
unit the metric scores: a frame three people annotated is scored three times
against three different ground truths, so it appears three times here, once per
annotator. Collapsing the three into one sample would also change how much a
frame weighs in the gradient, and the metric weighs it three times.

Three kinds of target can sit on that sample.

*Per annotator* (the default): the filaments that one person drew, as zeros and
ones. A frame three people annotated then carries three conflicting targets,
and the network is left to average them itself.

*Vote share*: the fraction of the people who looked at that frame who drew each
pixel -- 0, 1/3, 2/3 or 1 for a frame three people annotated. The sample count
and the weighting are unchanged; only what the sample is asked to predict
differs. The point is that a minority filament survives as 1/3 instead of being
averaged into something uncalibrated, which is what lets the decision to emit it
be made afterwards, at the threshold, where Panoptic Quality can be reasoned
about: a filament k of n people drew is worth emitting at IoU u when
``k * u > 0.5 * n * PQ``.

*Union*: every pixel anyone who looked at that frame drew, as zeros and ones.
The same inequality is the reason: at PQ 0.375 and u 0.66 even a filament one
of three people drew is worth emitting (0.66 > 0.56), so the network is asked to
find it outright rather than to learn a share and leave the decision to a
threshold. Measured on annotations alone, the union of a frame's annotators is
by far the best single prediction of any one of them. It is only asked of the
training side; validation keeps each annotator's own tracing, because that is
what the metric scores.

The target is semantic, not per-instance: every filament that annotator drew is
burned into one binary mask, and instances are recovered afterwards by splitting
the prediction into connected regions. That is what makes overlapping masks
impossible, and a rejected submission with them.

Frames are resized to 1024 rather than kept at 2048. At full resolution a batch
of even two frames does not fit a T4, and the median filament is 1,228 pixels,
which survives halving comfortably.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from filament.data.coco import AnnotatorImage, Dataset, polygon_to_mask
from filament.data.disk import DEFAULT_MASK_MARGIN, detect_disk
from filament.data.image import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    load_grayscale,
    resize,
    to_model_input,
)

DEFAULT_IMAGE_SIZE = 1024


@dataclass(frozen=True)
class Augmentation:
    """Flips and quarter turns, which a full-disk frame is invariant under.

    The Sun has no preferred orientation in these frames, so these leave the
    target valid. Anything that resamples the image (rotation by an arbitrary
    angle, scaling) would blur a filament only a few pixels wide, so it is left
    for later phases to evaluate.
    """

    horizontal_flip: bool = True
    vertical_flip: bool = True
    quarter_turns: bool = True

    def apply(
        self, image: np.ndarray, target: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the same random transform to a ``(C, H, W)`` image and a mask."""
        if self.horizontal_flip and rng.random() < 0.5:
            image, target = image[:, :, ::-1], target[:, ::-1]
        if self.vertical_flip and rng.random() < 0.5:
            image, target = image[:, ::-1, :], target[::-1, :]
        if self.quarter_turns:
            turns = int(rng.integers(0, 4))
            if turns:
                image = np.rot90(image, turns, axes=(1, 2))
                target = np.rot90(target, turns, axes=(0, 1))
        return np.ascontiguousarray(image), np.ascontiguousarray(target)


def build_target(
    entry: AnnotatorImage,
    size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Burn every filament of one annotator into a single binary mask.

    The mask is built at full resolution and then resized, so that a filament
    thinner than the downscaling factor still leaves a mark.

    Args:
        entry: One annotator's annotations of one frame.
        size: Side length of the returned mask.

    Returns:
        A ``(size, size)`` ``uint8`` array of zeros and ones.
    """
    full = np.zeros((entry.height, entry.width), dtype=np.uint8)
    for annotation in entry.annotations:
        full |= polygon_to_mask(annotation.segmentation, entry.height, entry.width)
    return resize(full, size, mask=True)


def build_vote_target(
    entries: Sequence[AnnotatorImage],
    size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Share of the annotators of one frame who drew each pixel.

    Every annotator's filaments are burned into their own mask at full
    resolution; the masks are averaged, and the average is resized. A frame one
    person annotated gives back exactly what :func:`build_target` would.

    The resize stays nearest-neighbour, matching :func:`build_target`. Averaging
    the pixels while downscaling would raise the best IoU the pipeline can reach
    from 0.880 to 0.902, measured over fold 0's annotations, but it is a second
    change and bundling it in would make the two indistinguishable in the
    result.

    Args:
        entries: Every annotator's view of the same frame. Must not be empty.
        size: Side length of the returned map.

    Returns:
        A ``(size, size)`` ``float32`` array in ``[0, 1]``.

    Raises:
        ValueError: If ``entries`` is empty or they are not all the same frame.
    """
    if not entries:
        raise ValueError("A vote share needs at least one annotator.")
    stems = {entry.stem for entry in entries}
    if len(stems) != 1:
        raise ValueError(f"Expected one frame, got {sorted(stems)}.")

    first = entries[0]
    votes = np.zeros((first.height, first.width), dtype=np.float32)
    for entry in entries:
        drawn = np.zeros((entry.height, entry.width), dtype=np.uint8)
        for annotation in entry.annotations:
            drawn |= polygon_to_mask(annotation.segmentation, entry.height, entry.width)
        votes += drawn
    votes /= float(len(entries))
    return resize(votes, size, mask=True)


def build_union_target(
    entries: Sequence[AnnotatorImage],
    size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Every pixel any annotator of one frame drew, as a binary mask.

    Built at full resolution and resized nearest-neighbour, like
    :func:`build_target`, so that the only difference between the two is whose
    filaments are burned in. A frame one person annotated gives back exactly
    what :func:`build_target` would.

    Args:
        entries: Every annotator's view of the same frame. Must not be empty.
        size: Side length of the returned mask.

    Returns:
        A ``(size, size)`` ``uint8`` array of zeros and ones.

    Raises:
        ValueError: If ``entries`` is empty or they are not all the same frame.
    """
    if not entries:
        raise ValueError("A union needs at least one annotator.")
    stems = {entry.stem for entry in entries}
    if len(stems) != 1:
        raise ValueError(f"Expected one frame, got {sorted(stems)}.")

    first = entries[0]
    full = np.zeros((first.height, first.width), dtype=np.uint8)
    for entry in entries:
        for annotation in entry.annotations:
            full |= polygon_to_mask(annotation.segmentation, entry.height, entry.width)
    return resize(full, size, mask=True)


class FilamentSegmentationDataset(TorchDataset):
    """Annotator-images as (2-channel frame, target map) pairs.

    Args:
        dataset: Loaded annotations.
        images_dir: Directory holding the frames.
        stems: Image stems to include, normally one side of a fold.
        size: Side length the frames and masks are resized to.
        augmentation: Random transforms to apply, or ``None`` for validation.
        seed: Seed of the augmentation. Fixed so that a run is reproducible.
        mask_outside_disk: Force the sky to background in the target. A
            filament cannot be there, so a positive label outside the disk
            would only ever be an annotation slip.
        disk_margin: Pixels of slack around the detected limb.
        vote_targets: Ask for the share of annotators who drew each pixel
            instead of what this sample's own annotator drew. The samples and
            their weighting are untouched: every annotator-image is still one
            sample, so a frame three people annotated still counts three times,
            and all three now carry the same target.
        union_targets: Ask for every pixel anyone who annotated the frame drew.
            Weighted like ``vote_targets``: one sample per annotator, all of a
            frame's samples carrying the same target.

    Raises:
        ValueError: If both kinds of shared target are asked for, or no
            annotator-image is left after filtering.
    """

    def __init__(
        self,
        dataset: Dataset,
        images_dir: Path | str,
        stems: Sequence[str] | None = None,
        size: int = DEFAULT_IMAGE_SIZE,
        augmentation: Augmentation | None = None,
        seed: int = 0,
        mask_outside_disk: bool = True,
        disk_margin: float = DEFAULT_MASK_MARGIN,
        clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
        clahe_tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
        vote_targets: bool = False,
        union_targets: bool = False,
    ) -> None:
        if vote_targets and union_targets:
            raise ValueError("Choose one of vote_targets and union_targets, not both.")
        selected = dataset if stems is None else dataset.subset(set(stems))
        self.entries: list[AnnotatorImage] = list(selected.annotator_images)
        if not self.entries:
            raise ValueError("No annotator-images left after filtering by stem.")
        self.vote_targets = vote_targets
        self.union_targets = union_targets
        self._by_stem = selected.by_stem()

        self.images_dir = Path(images_dir)
        self.size = size
        self.augmentation = augmentation
        self.seed = seed
        self.mask_outside_disk = mask_outside_disk
        self.disk_margin = disk_margin
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid

    def __len__(self) -> int:
        return len(self.entries)

    def frame_path(self, index: int) -> Path:
        """Location of the frame behind one sample."""
        return self.images_dir / self.entries[index].file_name

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        frame = load_grayscale(self.frame_path(index))
        image = to_model_input(frame, self.size, self.clahe_clip_limit, self.clahe_tile_grid)
        if self.vote_targets:
            target = build_vote_target(self._by_stem[entry.stem], self.size)
        elif self.union_targets:
            target = build_union_target(self._by_stem[entry.stem], self.size)
        else:
            target = build_target(entry, self.size)

        if self.mask_outside_disk:
            disk = detect_disk(frame).scaled(self.size / frame.shape[0])
            target = target * disk.mask(self.size, self.size, self.disk_margin)

        if self.augmentation is not None:
            # Seeded per sample and per epoch-independent index, so that a run
            # is reproducible whatever order the loader visits samples in.
            rng = np.random.default_rng((self.seed, index))
            image, target = self.augmentation.apply(image, target, rng)

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32))[None],
            "image_id": entry.image_id,
            "stem": entry.stem,
        }
