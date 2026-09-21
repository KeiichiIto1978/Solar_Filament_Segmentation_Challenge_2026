"""The training dataset for the semantic baseline.

One sample is **one annotator's view of one image**, not one image. That is the
unit the metric scores: a frame three people annotated is scored three times
against three different ground truths, so it appears three times here, once per
annotator. Averaging the three into a single target would train the model on a
consensus that the metric never rewards, and disagreement between annotators is
large -- they differ by 2.6 filaments per frame on average.

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


class FilamentSegmentationDataset(TorchDataset):
    """Annotator-images as (2-channel frame, binary mask) pairs.

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
    ) -> None:
        selected = dataset if stems is None else dataset.subset(set(stems))
        self.entries: list[AnnotatorImage] = list(selected.annotator_images)
        if not self.entries:
            raise ValueError("No annotator-images left after filtering by stem.")

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
