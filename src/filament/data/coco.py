"""Reading of the MAGFiLO COCO annotation file.

The competition annotations deviate from plain COCO in one way that drives the
whole project: the same image is annotated independently by up to three
annotators, and ``image_id`` encodes both of them as
``<annotator batch>-<image stem>`` (for example ``040301-20140609195854Bh``).

Evaluation loops over *annotator-images*, not over images, so a prediction for
an image annotated by three people is scored three times against three
different ground truths.  This module therefore keeps the annotator-image as
the primary unit and exposes the unique image stems separately, which is what
grouped train/validation splitting has to be based on.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils

from filament.paths import load_paths

# GONG observes from six stations; the two trailing characters of a file stem
# identify the station and are used to stratify the cross-validation folds.
OBSERVATORY_CODES = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")

Polygon = list[list[float]]


@dataclass(frozen=True)
class Annotation:
    """A single filament outlined by a single annotator."""

    annotation_id: str
    annotator_image_id: str
    category_id: int
    segmentation: Polygon
    bbox: tuple[float, float, float, float]
    area: float
    spine: list[float]

    def to_mask(self, height: int, width: int) -> np.ndarray:
        """Rasterize this filament into a boolean mask."""
        return polygon_to_mask(self.segmentation, height, width)


@dataclass(frozen=True)
class AnnotatorImage:
    """One image as seen by one annotator: the unit the metric iterates over."""

    image_id: str
    annotator: str
    stem: str
    file_name: str
    height: int
    width: int
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def observatory(self) -> str:
        """Two-letter GONG station code taken from the file stem."""
        return observatory_code(self.stem)

    def __len__(self) -> int:
        return len(self.annotations)


@dataclass(frozen=True)
class Dataset:
    """Normalized view of the annotation file."""

    annotator_images: list[AnnotatorImage]
    categories: dict[int, str]

    @property
    def stems(self) -> list[str]:
        """Unique image stems, sorted. Shorter than ``annotator_images``."""
        return sorted({entry.stem for entry in self.annotator_images})

    @property
    def annotations(self) -> list[Annotation]:
        """Every annotation in the file, flattened."""
        return [ann for entry in self.annotator_images for ann in entry.annotations]

    def by_stem(self) -> dict[str, list[AnnotatorImage]]:
        """Group the annotator-images by image stem."""
        grouped: dict[str, list[AnnotatorImage]] = defaultdict(list)
        for entry in self.annotator_images:
            grouped[entry.stem].append(entry)
        return dict(grouped)

    def single_annotator_stems(self) -> list[str]:
        """Stems annotated by exactly one annotator, sorted.

        Only for these is a prediction equal to the ground truth guaranteed to
        score PQ 1.0, which makes them the basis of the evaluator self-check.
        """
        return sorted(stem for stem, entries in self.by_stem().items() if len(entries) == 1)

    def subset(self, stems: list[str] | set[str]) -> Dataset:
        """Restrict the dataset to the given image stems."""
        wanted = set(stems)
        return Dataset(
            annotator_images=[e for e in self.annotator_images if e.stem in wanted],
            categories=self.categories,
        )

    def __len__(self) -> int:
        return len(self.annotator_images)


def split_image_id(image_id: str) -> tuple[str, str]:
    """Split ``<annotator batch>-<image stem>`` into its two parts.

    The stem never contains a hyphen, so a single split from the left is safe.
    """
    annotator, separator, stem = image_id.partition("-")
    if not separator:
        raise ValueError(f"Malformed image_id {image_id!r}: expected '<annotator>-<stem>'.")
    return annotator, stem


def observatory_code(stem: str) -> str:
    """Return the two-character GONG station code of an image stem."""
    code = stem[-2:]
    if code not in OBSERVATORY_CODES:
        raise ValueError(
            f"Unknown observatory code {code!r} in stem {stem!r}; "
            f"expected one of {', '.join(OBSERVATORY_CODES)}."
        )
    return code


def polygon_to_mask(segmentation: Polygon, height: int, width: int) -> np.ndarray:
    """Rasterize a COCO polygon into a boolean mask of shape ``(height, width)``.

    MAGFiLO filaments are single polygons without holes, but merging the parts
    keeps the function correct for multi-part input as well.
    """
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(bool)


def load_annotations(path: Path | str | None = None) -> Dataset:
    """Read the MAGFiLO COCO annotation file into a :class:`Dataset`.

    Args:
        path: Annotation file. Defaults to the training annotations declared in
            ``configs/paths.yaml``.

    Raises:
        ValueError: If an annotation refers to an image that the file does not
            declare, which would silently drop ground truth.
    """
    annotation_path = Path(path) if path is not None else load_paths().train_annotations
    with annotation_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    grouped: dict[str, list[Annotation]] = defaultdict(list)
    for item in raw["annotations"]:
        bbox = item["bbox"]
        grouped[item["image_id"]].append(
            Annotation(
                annotation_id=str(item["id"]),
                annotator_image_id=str(item["image_id"]),
                category_id=int(item["category_id"]),
                segmentation=item["segmentation"],
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                area=float(item["area"]),
                spine=list(item.get("spine", [])),
            )
        )

    annotator_images = []
    for item in raw["images"]:
        image_id = str(item["id"])
        annotator, stem = split_image_id(image_id)
        annotator_images.append(
            AnnotatorImage(
                image_id=image_id,
                annotator=annotator,
                stem=stem,
                file_name=str(item["file_name"]),
                height=int(item["height"]),
                width=int(item["width"]),
                annotations=grouped.pop(image_id, []),
            )
        )

    if grouped:
        orphans = ", ".join(sorted(grouped)[:5])
        raise ValueError(
            f"{len(grouped)} image_id(s) appear in 'annotations' but not in 'images' "
            f"of {annotation_path}: {orphans}"
        )

    categories = {int(c["id"]): str(c["name"]) for c in raw.get("categories", [])}
    return Dataset(annotator_images=annotator_images, categories=categories)
