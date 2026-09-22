"""One filament per sample, cut out of the frame at full resolution.

The whole-disk model works at 1024 because 2048 does not fit on the GPU, and
that halving caps how well any mask it draws can score: sending fold 0's
annotations down to 1024 and back costs them a mean IoU of 0.880 before a
network is involved at all. The model reaches 0.6555, so the cap is not what
binds -- but a cap of 0.880 is still there, and nothing in the whole-disk
setup can lift it.

A crop lifts it. Taking the box around one filament and resampling that to 512
gives the network four or five times the detail of the whole-disk view, and
asks it a narrower question: not *where are the filaments* but *draw this one*.

Whether that is worth building a detector for is what these crops exist to
measure. Cut around the annotated boxes and evaluated against them, they say
how good the masks could be **if the boxes were perfect** -- an upper bound,
since no detector delivers that. If the bound is close to what the whole-disk
model already achieves, the two-stage design is not worth its cost, and that
is settled by one training run rather than by a fortnight of building.

The third channel is the box itself, filled in. Without it a crop containing
two filaments is ambiguous -- the network would be scored for drawing one and
shown another. The box says which one. It has to be the box rather than the
annotated mask: that would hand over the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from filament.data.coco import Annotation, AnnotatorImage, Dataset
from filament.data.image import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, apply_clahe, load_grayscale

# Side length a crop is resampled to. A filament's box is 250 pixels across at
# the median, so this magnifies rather than shrinks in most cases.
DEFAULT_CROP_SIZE = 512

# The crop is this multiple of the box's longer side, which leaves room for a
# box that sits slightly off and shows the network what surrounds the filament.
DEFAULT_CONTEXT = 1.5

# The seed is the box grown by this much before being filled, matching how a
# detector's box would sit loosely around the structure.
DEFAULT_SEED_PADDING = 0.4

INPUT_CHANNELS = 3


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in frame coordinates, as ``(top, left, bottom, right)``."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.top + self.bottom) / 2.0, (self.left + self.right) / 2.0)

    @property
    def side(self) -> int:
        """The longer side, which is what the crop is scaled from."""
        return max(self.height, self.width)

    def is_empty(self) -> bool:
        return self.height <= 0 or self.width <= 0


def box_of(mask: np.ndarray) -> Box:
    """The tightest box around the set pixels of a mask.

    Raises:
        ValueError: If the mask is empty, which has no box.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("An empty mask has no box.")
    return Box(int(rows[0]), int(columns[0]), int(rows[-1]) + 1, int(columns[-1]) + 1)


def crop_window(
    box: Box,
    frame_size: int,
    context: float = DEFAULT_CONTEXT,
    shift: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> Box:
    """The square region to cut, given the box to centre it on.

    Square rather than the box's own shape, so that resampling to a square
    does not stretch a filament along one axis -- the network would then see a
    thickness that depends on which way the filament happens to lie.

    Args:
        box: The box to centre on, in frame coordinates.
        frame_size: Side length of the frame, for clamping.
        context: Multiple of the box's longer side to take.
        shift: Offset of the centre, as a fraction of the window's side.
        scale: Multiplier on the window's side.

    Returns:
        A square window, clamped to the frame. It stays square unless the box
        sits within half a window of the edge.
    """
    centre_row, centre_column = box.centre
    side = max(box.side * context * scale, 8.0)
    centre_row += shift[0] * side
    centre_column += shift[1] * side

    half = side / 2.0
    top = round(centre_row - half)
    left = round(centre_column - half)
    bottom, right = top + round(side), left + round(side)

    # Slide the window back inside the frame rather than shrinking it, so that
    # a filament near the limb is still seen at the same magnification.
    if top < 0:
        top, bottom = 0, min(round(side), frame_size)
    if left < 0:
        left, right = 0, min(round(side), frame_size)
    if bottom > frame_size:
        bottom, top = frame_size, max(frame_size - round(side), 0)
    if right > frame_size:
        right, left = frame_size, max(frame_size - round(side), 0)
    return Box(top, left, bottom, right)


def seed_mask(
    box: Box, window: Box, size: int, padding: float = DEFAULT_SEED_PADDING
) -> np.ndarray:
    """The box, grown and filled, drawn in the crop's coordinates.

    This is what tells the network which filament in the crop it is being
    asked about. Grown because a detector's box does not sit tightly, and the
    network should not learn to trust its edges.
    """
    scale = size / max(window.height, 1)
    grow_rows = box.height * padding / 2.0
    grow_columns = box.width * padding / 2.0

    top = (box.top - window.top - grow_rows) * scale
    bottom = (box.bottom - window.top + grow_rows) * scale
    left = (box.left - window.left - grow_columns) * scale
    right = (box.right - window.left + grow_columns) * scale

    seed = np.zeros((size, size), dtype=np.float32)
    seed[
        max(round(top), 0) : max(round(bottom), 0),
        max(round(left), 0) : max(round(right), 0),
    ] = 1.0
    return seed


def cut(image: np.ndarray, window: Box, size: int, mask: bool = False) -> np.ndarray:
    """Take a window out of a frame and resample it to ``size``."""
    patch = image[window.top : window.bottom, window.left : window.right]
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interpolation)


def build_input(
    frame: np.ndarray,
    box: Box,
    window: Box,
    size: int = DEFAULT_CROP_SIZE,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
    seed_padding: float = DEFAULT_SEED_PADDING,
) -> np.ndarray:
    """The three channels the crop model expects, in ``[0, 1]``.

    The frame, its contrast-equalised version, and the filled box. Contrast is
    equalised after cutting rather than before, so that it adapts to what is
    inside the crop rather than to the whole disk.
    """
    patch = cut(frame, window, size)
    equalised = apply_clahe(patch, clip_limit, tile_grid)
    seed = seed_mask(box, window, size, seed_padding)
    return np.stack([patch / 255.0, equalised / 255.0, seed]).astype(np.float32)


@dataclass(frozen=True)
class CropSample:
    """One annotated filament, located in its frame."""

    stem: str
    image_id: str
    annotation_id: str
    box: Box


def list_crops(dataset: Dataset, stems: Sequence[str] | None = None) -> list[CropSample]:
    """Every annotated filament of the given frames, with its box.

    Annotations whose rasterised mask is empty are skipped, with their ids kept
    out rather than silently becoming an empty target.
    """
    selected = dataset if stems is None else dataset.subset(set(stems))
    samples: list[CropSample] = []
    for entry in selected.annotator_images:
        for annotation in entry.annotations:
            mask = annotation.to_mask(entry.height, entry.width)
            try:
                box = box_of(mask)
            except ValueError:
                continue
            samples.append(
                CropSample(
                    stem=entry.stem,
                    image_id=entry.image_id,
                    annotation_id=str(annotation.annotation_id),
                    box=box,
                )
            )
    return samples


class FilamentCropDataset(TorchDataset):
    """Annotated filaments as (three-channel crop, mask) pairs.

    One sample is one annotation, not one frame: a frame carrying seven
    filaments yields seven crops. That is also how the metric counts, and it
    makes a small filament weigh as much as a large one, which no whole-disk
    loss does.

    Args:
        dataset: Loaded annotations.
        images_dir: Directory holding the frames.
        stems: Image stems to include, normally one side of a fold.
        size: Side length a crop is resampled to.
        context: Multiple of the box's longer side the crop covers.
        seed_padding: How much the box grows before being drawn as the seed.
        jitter: Fraction of the window the centre may be shifted by, and the
            range the side may be scaled by, when training. Zero matches the
            crops to their boxes exactly, which is what an upper bound wants;
            a system fed by a real detector needs it non-zero so that the seed
            it trains on looks as loose as the one it will be given.
        flips: Mirror the crop at random, as the whole-disk training does.
        seed: Seed of the random transforms.
    """

    def __init__(
        self,
        dataset: Dataset,
        images_dir: Path | str,
        stems: Sequence[str] | None = None,
        size: int = DEFAULT_CROP_SIZE,
        context: float = DEFAULT_CONTEXT,
        seed_padding: float = DEFAULT_SEED_PADDING,
        jitter: float = 0.0,
        flips: bool = True,
        seed: int = 0,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
    ) -> None:
        self.samples = list_crops(dataset, stems)
        if not self.samples:
            raise ValueError("No annotations left after filtering by stem.")

        self.annotations: dict[str, Annotation] = {}
        self.entries: dict[str, AnnotatorImage] = {}
        selected = dataset if stems is None else dataset.subset(set(stems))
        for entry in selected.annotator_images:
            self.entries[entry.image_id] = entry
            for annotation in entry.annotations:
                self.annotations[str(annotation.annotation_id)] = annotation

        self.images_dir = Path(images_dir)
        self.size = size
        self.context = context
        self.seed_padding = seed_padding
        self.jitter = jitter
        self.flips = flips
        self.seed = seed
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        entry = self.entries[sample.image_id]
        frame = load_grayscale(self.images_dir / entry.file_name)

        # Drawn per access rather than per sample, so that a filament is seen
        # from a different offset each epoch.
        rng = np.random.default_rng((self.seed, index, torch.initial_seed() % (2**31)))
        shift = (0.0, 0.0)
        scale = 1.0
        if self.jitter:
            shift = tuple(rng.uniform(-self.jitter, self.jitter, size=2))
            scale = float(rng.uniform(1.0 - self.jitter, 1.0 + self.jitter))

        window = crop_window(sample.box, frame.shape[0], self.context, shift, scale)
        image = build_input(
            frame,
            sample.box,
            window,
            self.size,
            self.clip_limit,
            self.tile_grid,
            self.seed_padding,
        )
        annotation = self.annotations[sample.annotation_id]
        target = cut(
            annotation.to_mask(entry.height, entry.width).astype(np.uint8),
            window,
            self.size,
            mask=True,
        ).astype(np.float32)

        if self.flips:
            if rng.random() < 0.5:
                image, target = image[:, :, ::-1], target[:, ::-1]
            if rng.random() < 0.5:
                image, target = image[:, ::-1, :], target[::-1, :]
            image, target = np.ascontiguousarray(image), np.ascontiguousarray(target)

        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target)[None],
            "stem": sample.stem,
            "image_id": sample.image_id,
            "annotation_id": sample.annotation_id,
        }
