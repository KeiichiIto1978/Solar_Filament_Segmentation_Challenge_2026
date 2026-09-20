"""Pre-submission check that predicted masks never overlap.

Kaggle rejects a submission in which two masks of the same image share even a
single pixel, and a rejected submission still consumes one of the five daily
slots. Raw instance-segmentation output routinely violates this, because
box-level NMS leaves the masks themselves free to overlap.

Running this check locally costs nothing and protects the submission budget.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils

from filament.submit.rle import (
    FULL_HEIGHT,
    FULL_WIDTH,
    SUBMISSION_COLUMNS,
    read_submission,
    rle_to_mask,
)

logger = logging.getLogger(__name__)


class SubmissionOverlapError(ValueError):
    """Raised when masks of the same image overlap."""


@dataclass(frozen=True)
class OverlapReport:
    """Which image overlaps, by how much, and which rows are involved."""

    image_id: str
    overlapping_pixels: int
    filament_ids: list[str]

    def __str__(self) -> str:
        involved = ", ".join(self.filament_ids[:6])
        if len(self.filament_ids) > 6:
            involved += f", ... ({len(self.filament_ids)} rows)"
        return f"{self.image_id}: {self.overlapping_pixels} pixel(s) claimed twice [{involved}]"


def _image_id(filament_id: str) -> str:
    """``<image id>_<n>`` without its running number."""
    return str(filament_id).rsplit("_", 1)[0]


def find_overlaps(
    frame: pd.DataFrame,
    height: int = FULL_HEIGHT,
    width: int = FULL_WIDTH,
) -> list[OverlapReport]:
    """Return one report per image whose masks overlap.

    Detection compares the summed area of the masks with the area of their
    union, which pycocotools computes on the encoded form without allocating a
    single full-size array. Only an image that fails that comparison is decoded,
    to name the rows involved.
    """
    missing = [column for column in SUBMISSION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Submission is missing the column(s) {', '.join(missing)}; "
            f"expected exactly {', '.join(SUBMISSION_COLUMNS)}."
        )

    by_image: dict[str, list[tuple[str, str]]] = defaultdict(list)
    rows = zip(frame["filament_id"], frame["segmentation_rle"], strict=True)
    for filament_id, counts in rows:
        by_image[_image_id(filament_id)].append((str(filament_id), str(counts)))

    reports = []
    for image_id, rows_of_image in sorted(by_image.items()):
        if len(rows_of_image) < 2:
            continue
        encoded = [
            {"size": [height, width], "counts": counts.encode("ascii")}
            for _, counts in rows_of_image
        ]
        total_area = int(np.sum(mask_utils.area(encoded), dtype=np.int64))
        union_area = int(mask_utils.area(mask_utils.merge(encoded, intersect=False)))
        if total_area == union_area:
            continue
        reports.append(_describe_overlap(image_id, rows_of_image, height, width))
    return reports


def _describe_overlap(
    image_id: str,
    rows_of_image: list[tuple[str, str]],
    height: int,
    width: int,
) -> OverlapReport:
    """Decode one image's masks to count shared pixels and name the rows."""
    claims = np.zeros((height, width), dtype=np.int16)
    masks = []
    for _, counts in rows_of_image:
        mask = rle_to_mask(counts, height, width)
        masks.append(mask)
        claims += mask
    contested = claims > 1
    involved = [
        filament_id
        for (filament_id, _), mask in zip(rows_of_image, masks, strict=True)
        if np.any(mask & contested)
    ]
    return OverlapReport(
        image_id=image_id,
        overlapping_pixels=int(contested.sum()),
        filament_ids=involved,
    )


def check_no_overlap(
    csv_path: Path | str,
    height: int = FULL_HEIGHT,
    width: int = FULL_WIDTH,
) -> None:
    """Verify that a submission CSV has no overlapping masks.

    Args:
        csv_path: Submission file to check.
        height: Mask height; 2048 for this dataset.
        width: Mask width; 2048 for this dataset.

    Raises:
        SubmissionOverlapError: If any image has overlapping masks. The message
            names every affected image.
    """
    frame = read_submission(csv_path)
    reports = find_overlaps(frame, height, width)
    if reports:
        listed = "\n".join(f"  {report}" for report in reports)
        raise SubmissionOverlapError(
            f"{len(reports)} image(s) in {csv_path} have overlapping masks. "
            "Kaggle rejects such a submission and the attempt still counts "
            f"against the daily limit.\n{listed}"
        )
    logger.info("No overlapping masks in %s (%d rows).", csv_path, len(frame))
