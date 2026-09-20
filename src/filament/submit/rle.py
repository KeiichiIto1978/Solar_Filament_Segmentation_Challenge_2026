"""Run-length encoding and submission CSV writing.

The competition expects a single CSV with two columns, one row per predicted
filament::

    filament_id,segmentation_rle
    20150125172714Mh_1,fNo0...

``filament_id`` is ``<image id>_<running number>`` with the image id carrying no
file extension, and ``segmentation_rle`` is the *counts* field of a pycocotools
compressed RLE on its own: the size is a constant 2048x2048 and is not written,
and the value is never quoted.

Ground-truth rows follow the same shape but keep the annotator prefix of the
image id, because the metric is evaluated per annotator-image.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils

from filament.data.coco import Dataset

# Every frame in this dataset is a full-disk 2048x2048 image.
FULL_HEIGHT = 2048
FULL_WIDTH = 2048

SUBMISSION_COLUMNS = ("filament_id", "segmentation_rle")


def make_filament_id(image_id: str, index: int) -> str:
    """Build ``<image id>_<index>``. The index only makes rows unique."""
    return f"{image_id}_{index}"


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask as the *counts* string of a compressed RLE.

    Args:
        mask: Two-dimensional array; any non-zero value counts as foreground.

    Returns:
        The counts field as ASCII text, without the accompanying size.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}.")
    # pycocotools expects Fortran-ordered uint8 data.
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return encoded["counts"].decode("ascii")


def rle_to_mask(counts: str, height: int = FULL_HEIGHT, width: int = FULL_WIDTH) -> np.ndarray:
    """Decode a counts string back into a boolean mask."""
    rle = {"size": [height, width], "counts": counts.encode("ascii")}
    return mask_utils.decode(rle).astype(bool)


def write_submission(
    rows: Iterable[Sequence[str]] | pd.DataFrame,
    path: Path | str,
) -> Path:
    """Write submission rows to ``path`` as an unquoted two-column CSV.

    Args:
        rows: A DataFrame with the submission columns, or an iterable of
            ``(filament_id, segmentation_rle)`` pairs.
        path: Destination file; parent directories are created.

    Raises:
        ValueError: If a value contains a comma, which would silently corrupt
            the file. Compressed RLE never does, so this signals bad input.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    pairs = (
        list(rows[list(SUBMISSION_COLUMNS)].itertuples(index=False, name=None))
        if isinstance(rows, pd.DataFrame)
        else [tuple(row) for row in rows]
    )

    with destination.open("w", encoding="utf-8", newline="") as handle:
        # QUOTE_NONE never emits quotes; commas are rejected below instead of escaped.
        writer = csv.writer(handle, quoting=csv.QUOTE_NONE)
        writer.writerow(SUBMISSION_COLUMNS)
        for filament_id, rle in pairs:
            if "," in filament_id or "," in rle:
                raise ValueError(f"Comma in submission row {filament_id!r}; refusing to write.")
            writer.writerow((filament_id, rle))
    return destination


def read_submission(path: Path | str) -> pd.DataFrame:
    """Read a submission CSV, keeping the RLE column as text."""
    return pd.read_csv(path, dtype=str).fillna("")


def masks_to_gt_df(dataset: Dataset, stems: Iterable[str] | None = None) -> pd.DataFrame:
    """Convert ground-truth annotations into a submission-shaped DataFrame.

    The ``filament_id`` keeps the annotator prefix (``<annotator>-<stem>_<n>``),
    which is the form the PQ evaluator expects for ground truth.

    Args:
        dataset: Loaded annotations.
        stems: Image stems to include. Defaults to every stem in ``dataset``.
    """
    selected = dataset if stems is None else dataset.subset(set(stems))
    records: list[tuple[str, str]] = []
    for entry in selected.annotator_images:
        for index, annotation in enumerate(entry.annotations, start=1):
            mask = annotation.to_mask(entry.height, entry.width)
            records.append((make_filament_id(entry.image_id, index), mask_to_rle(mask)))
    return pd.DataFrame(records, columns=list(SUBMISSION_COLUMNS))


def to_prediction_df(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Strip the annotator prefix from ground-truth ids to get prediction ids.

    Only meaningful for images annotated by a single annotator: with several
    annotators the resulting ids would collide across their annotations.
    """
    predictions = gt_df.copy()
    predictions["filament_id"] = predictions["filament_id"].str.split("-", n=1).str[1]
    return predictions
