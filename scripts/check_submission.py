"""Verify a submission CSV before spending one of the five daily attempts.

Usage::

    uv run python scripts/check_submission.py --csv submission.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from filament.metrics.overlap import SubmissionOverlapError, check_no_overlap
from filament.submit.rle import FULL_HEIGHT, FULL_WIDTH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True, help="Submission CSV to check.")
    parser.add_argument("--height", type=int, default=FULL_HEIGHT, help="Mask height in pixels.")
    parser.add_argument("--width", type=int, default=FULL_WIDTH, help="Mask width in pixels.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        check_no_overlap(args.csv, args.height, args.width)
    except SubmissionOverlapError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"OK: no overlapping masks in {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
