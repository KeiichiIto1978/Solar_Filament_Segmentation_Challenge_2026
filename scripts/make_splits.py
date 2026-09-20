"""Generate the frozen cross-validation split.

Run once. The fold files are committed and must not be regenerated, because
scores recorded against an older split are not comparable with newer ones.

Usage::

    uv run python scripts/make_splits.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from filament.data.coco import load_annotations
from filament.data.split import (
    DEFAULT_FOLDS,
    DEFAULT_SEED,
    build_folds,
    station_counts,
    write_folds,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Annotation JSON. Defaults to the training file in configs/paths.yaml.",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to the splits directory in configs/paths.yaml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    dataset = load_annotations(args.annotations)
    folds = build_folds(dataset.stems, n_folds=args.folds, seed=args.seed)
    write_folds(folds, args.out)

    print(f"{len(dataset.stems)} image stems, {args.folds} folds, seed {args.seed}")
    for fold in folds:
        stations = station_counts(fold.val)
        mix = " ".join(f"{code}={stations.get(code, 0)}" for code in sorted(stations))
        print(f"  fold{fold.index}: {len(fold.train)} train / {len(fold.val)} val  [{mix}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
