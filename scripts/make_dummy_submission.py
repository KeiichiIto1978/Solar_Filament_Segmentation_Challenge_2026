"""Write a submission of empty masks, to check the format end to end.

This produces one row per test image whose mask is empty, runs the overlap
check on the result, and stops there. It is a format rehearsal: nothing here
is worth submitting, and a submission attempt costs one of five daily slots.

Usage::

    uv run python scripts/make_dummy_submission.py --out submission.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from filament.metrics.overlap import check_no_overlap
from filament.paths import load_paths
from filament.submit.rle import make_filament_id, mask_to_rle, write_submission

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpeg", ".jpg")


def list_test_stems(test_images: Path) -> list[str]:
    """Image ids of the test set: file names without their extension."""
    stems = sorted(
        path.stem for path in test_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not stems:
        raise FileNotFoundError(f"No test images found in {test_images}.")
    return stems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--test-images",
        type=Path,
        default=None,
        help="Directory of test images. Defaults to the one in configs/paths.yaml.",
    )
    parser.add_argument("--out", type=Path, default=Path("submission.csv"), help="Output CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    paths = load_paths()
    test_images = args.test_images or paths.test_images
    stems = list_test_stems(test_images)

    # One empty mask per image: the smallest file that still has the real shape.
    empty = mask_to_rle(np.zeros(paths.image_shape, dtype=bool))
    rows = [(make_filament_id(stem, 1), empty) for stem in stems]

    write_submission(rows, args.out)
    check_no_overlap(args.out, paths.image_height, paths.image_width)

    print(f"Wrote {len(rows)} empty-mask rows for {len(stems)} images to {args.out}")
    print("Format check passed. Do not submit this file; it scores zero by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
