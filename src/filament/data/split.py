"""Frozen cross-validation splits, grouped by image and stratified by station.

Two properties matter here.

*Grouping.* Up to three annotators describe the same image, and the metric
scores a prediction once per annotator. If two annotators of one image landed
on opposite sides of a split, validation would be scoring an image the model
was trained on. Splitting therefore happens on image stems, never on
annotator-images.

*Stratification.* GONG observes from six stations whose images differ in
brightness, contrast and seeing. Keeping their proportions equal across folds
stops a fold from being easier or harder than the others by accident.

The folds are generated once, written to ``configs/splits/`` and then left
alone: comparing two experiments only means something if both were validated
on the same images.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from filament.data.coco import observatory_code
from filament.paths import load_paths

logger = logging.getLogger(__name__)

DEFAULT_FOLDS = 5
# Fixed once and never changed, so that folds stay comparable across experiments.
DEFAULT_SEED = 20260921


@dataclass(frozen=True)
class Fold:
    """One train/validation partition of the image stems."""

    index: int
    train: list[str]
    val: list[str]
    seed: int = DEFAULT_SEED
    n_folds: int = DEFAULT_FOLDS

    @property
    def stems(self) -> list[str]:
        return sorted(self.train + self.val)

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.index,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "train": self.train,
            "val": self.val,
        }


def assign_folds(
    stems: Iterable[str],
    n_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Map every image stem to a fold index.

    Stems are shuffled within each observatory and then dealt round-robin, so
    each fold receives the same station mix and the fold sizes differ by at
    most one.

    Args:
        stems: Unique image stems.
        n_folds: Number of folds.
        seed: Seed of the shuffle. Fixed, so the split is reproducible.
    """
    if n_folds < 2:
        raise ValueError(f"A split needs at least 2 folds, got {n_folds}.")

    unique = sorted(set(stems))
    if not unique:
        raise ValueError("No image stems to split.")

    by_station: dict[str, list[str]] = defaultdict(list)
    for stem in unique:
        by_station[observatory_code(stem)].append(stem)

    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    dealt = 0
    # Sorting the stations keeps the deal order independent of dict ordering.
    for station in sorted(by_station):
        shuffled = rng.permutation(by_station[station])
        for stem in shuffled:
            assignment[str(stem)] = dealt % n_folds
            dealt += 1
    return assignment


def build_folds(
    stems: Iterable[str],
    n_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[Fold]:
    """Build every fold of the split."""
    assignment = assign_folds(stems, n_folds=n_folds, seed=seed)
    folds = []
    for index in range(n_folds):
        val = sorted(stem for stem, fold in assignment.items() if fold == index)
        train = sorted(stem for stem, fold in assignment.items() if fold != index)
        folds.append(Fold(index=index, train=train, val=val, seed=seed, n_folds=n_folds))
    return folds


def fold_path(index: int, splits_dir: Path | str | None = None) -> Path:
    """Location of one fold file."""
    directory = Path(splits_dir) if splits_dir is not None else load_paths().splits_dir
    return directory / f"fold{index}.json"


def write_folds(folds: list[Fold], splits_dir: Path | str | None = None) -> list[Path]:
    """Write one JSON file per fold.

    Raises:
        FileExistsError: If a fold file is already there. The split is meant to
            be decided once; overwriting it would silently invalidate every
            score recorded so far.
    """
    written = []
    for fold in folds:
        path = fold_path(fold.index, splits_dir)
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. The split is frozen on purpose: delete the "
                "existing fold files deliberately if you really mean to replace them, "
                "and note in the lab notebook that earlier scores are no longer "
                "comparable."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(fold.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote %s: %d train / %d val stems.", path, len(fold.train), len(fold.val))
        written.append(path)
    return written


def load_fold(index: int, splits_dir: Path | str | None = None) -> Fold:
    """Read one fold back from ``configs/splits/``."""
    path = fold_path(index, splits_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Fold {index} not found at {path}. Generate the split first with "
            "'uv run python scripts/make_splits.py'."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Fold(
        index=int(payload["fold"]),
        train=list(payload["train"]),
        val=list(payload["val"]),
        seed=int(payload["seed"]),
        n_folds=int(payload["n_folds"]),
    )


def station_counts(stems: Iterable[str]) -> dict[str, int]:
    """Number of stems per observatory, for checking the stratification."""
    counts: dict[str, int] = defaultdict(int)
    for stem in stems:
        counts[observatory_code(stem)] += 1
    return dict(counts)
