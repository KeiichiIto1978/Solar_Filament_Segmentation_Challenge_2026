"""Searching the post-processing parameters against PQ.

Separated from the model on purpose. Once a probability map exists, nothing
here needs a GPU or a forward pass, so a sweep over a few hundred settings
costs seconds rather than an hour. The same sweep runs over a U-Net's maps and
over a filter bank's.

Two rules are built in.

*Report the whole curve, not the best point.* One fold's validation set is 142
frames, and the highest PQ among a few hundred settings is partly an accident
of those frames. :meth:`SweepResult.plateau` therefore returns the middle of
the widest region that is within a tolerance of the best, which is the value
that survives being applied to other folds.

*Report the breakdown.* PQ alone does not say whether a setting helped by
finding more filaments or by discarding bad guesses. SQ, RQ and the TP/FP/FN
counts do, and they are what decide the next move.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from filament.data.disk import Disk
from filament.metrics.pq import PQResult, compute_pq
from filament.postprocess.instances import (
    DEFAULT_MIN_AREA,
    DEFAULT_THRESHOLD,
    extract_instances,
    instances_to_rows,
)
from filament.postprocess.join import DEFAULT_MAX_ANGLE, DEFAULT_MAX_OFFSET
from filament.submit.rle import FULL_HEIGHT, SUBMISSION_COLUMNS

logger = logging.getLogger(__name__)

# How far below the best PQ a setting may sit and still count as part of the
# plateau. Below the noise of a 142-frame fold, and above rounding.
DEFAULT_PLATEAU_TOLERANCE = 0.005

ProbabilityMaps = Mapping[str, np.ndarray]
InstanceBuilder = Callable[[np.ndarray, Disk | None, Mapping[str, Any]], list[Any]]


@dataclass(frozen=True)
class Setting:
    """One point of the search space."""

    values: dict[str, Any]

    def __str__(self) -> str:
        return " ".join(f"{key}={value}" for key, value in sorted(self.values.items()))


@dataclass(frozen=True)
class SweepPoint:
    """The score of one setting."""

    setting: Setting
    pq: PQResult
    predictions: int
    seconds: float

    def to_row(self) -> dict[str, Any]:
        """A flat record, for a DataFrame."""
        return {
            **self.setting.values,
            "pq": self.pq.pq,
            "sq": self.pq.sq,
            "rq": self.pq.rq,
            "tp": self.pq.tp,
            "fp": self.pq.fp,
            "fn": self.pq.fn,
            "predictions": self.predictions,
            "seconds": round(self.seconds, 2),
        }


@dataclass(frozen=True)
class SweepResult:
    """Every point of a sweep, in the order they were evaluated."""

    points: list[SweepPoint] = field(default_factory=list)

    @property
    def table(self) -> pd.DataFrame:
        """The sweep as a DataFrame, best PQ first."""
        frame = pd.DataFrame([point.to_row() for point in self.points])
        return frame.sort_values("pq", ascending=False).reset_index(drop=True)

    @property
    def best(self) -> SweepPoint:
        """The highest-scoring point. Prefer :meth:`plateau` for a decision."""
        if not self.points:
            raise ValueError("The sweep is empty.")
        return max(self.points, key=lambda point: point.pq.pq)

    def plateau(
        self, parameter: str, tolerance: float = DEFAULT_PLATEAU_TOLERANCE
    ) -> tuple[Any, list[Any]]:
        """The middle of the widest near-best run of one parameter.

        Picking the single best value of a parameter on 142 frames chases that
        fold's accidents. Taking the middle of the widest stretch that stays
        within ``tolerance`` of the best gives a value whose neighbours are
        also good, which is what transfers to another fold.

        Args:
            parameter: Which parameter to read.
            tolerance: How far below the best PQ still counts as near-best.

        Returns:
            The chosen value, and every value in the plateau.
        """
        if not self.points:
            raise ValueError("The sweep is empty.")

        # Best PQ attained at each value of this parameter, marginalising the rest.
        best_by_value: dict[Any, float] = {}
        for point in self.points:
            value = point.setting.values[parameter]
            best_by_value[value] = max(best_by_value.get(value, -1.0), point.pq.pq)

        ordered = sorted(best_by_value)
        ceiling = max(best_by_value.values())
        near_best = [value for value in ordered if best_by_value[value] >= ceiling - tolerance]

        # Longest run of consecutive near-best values, in the sorted order.
        chosen = set(near_best)
        runs: list[list[Any]] = []
        previous_was_near_best = False
        for value in ordered:
            if value not in chosen:
                previous_was_near_best = False
                continue
            if previous_was_near_best:
                runs[-1].append(value)
            else:
                runs.append([value])
            previous_was_near_best = True

        widest = max(runs, key=len)
        return widest[len(widest) // 2], widest


def grid(**axes: Sequence[Any]) -> list[Setting]:
    """Every combination of the given parameter values.

    ``grid(threshold=[0.4, 0.5], min_area=[100, 200])`` gives four settings.
    """
    names = list(axes)
    return [
        Setting(dict(zip(names, combination, strict=True)))
        for combination in itertools.product(*(axes[name] for name in names))
    ]


def sweep(
    maps: ProbabilityMaps,
    gt_df: pd.DataFrame,
    settings: Iterable[Setting],
    disks: Mapping[str, Disk] | None = None,
    output_size: int = FULL_HEIGHT,
    build: InstanceBuilder | None = None,
) -> SweepResult:
    """Score every setting against the same probability maps.

    Args:
        maps: Probability map per image stem, at the working resolution.
        gt_df: Ground truth in submission form, with annotator prefixes.
        settings: Settings to try, from :func:`grid` or built by hand.
        disks: Solar disk per stem, in the maps' coordinates. Omit when the
            maps are already masked.
        output_size: Side length the masks are scaled up to before scoring.
        build: Override how instances are produced from a map. Defaults to
            :func:`~filament.postprocess.instances.extract_instances`, and is
            the hook for a chain that also joins fragments.

    Returns:
        Every point, with its PQ breakdown.
    """
    points: list[SweepPoint] = []

    for setting in settings:
        started = time.perf_counter()
        pred_df = predict_from_maps(maps, setting, disks, output_size, build)
        result = compute_pq(gt_df, pred_df)
        point = SweepPoint(
            setting=setting,
            pq=result,
            predictions=len(pred_df),
            seconds=time.perf_counter() - started,
        )
        points.append(point)
        logger.info("%s -> %s", setting, result)

    return SweepResult(points=points)


def predict_from_maps(
    maps: ProbabilityMaps,
    setting: Setting,
    disks: Mapping[str, Disk] | None = None,
    output_size: int = FULL_HEIGHT,
    build: InstanceBuilder | None = None,
) -> pd.DataFrame:
    """Turn probability maps into submission rows under one setting.

    The same path a sweep point takes, exposed on its own so that the setting
    a sweep settles on can be applied to another set of maps -- the test
    frames -- without a second copy of the chain.

    Args:
        maps: Probability map per image stem, at the working resolution.
        setting: The post-processing parameters to apply.
        disks: Solar disk per stem, in the maps' coordinates. Omit when the
            maps are already masked.
        output_size: Side length the masks are scaled up to.
        build: Override how instances are produced from a map.

    Returns:
        A submission-shaped frame: ``filament_id`` and ``segmentation_rle``.
    """
    builder = build or _default_builder
    rows: list[tuple[str, str]] = []
    for stem, probability in maps.items():
        disk = None if disks is None else disks.get(stem)
        instances = builder(probability, disk, setting.values | {"output_size": output_size})
        rows.extend(instances_to_rows(stem, instances))
    return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))


def _default_builder(
    probability: np.ndarray, disk: Disk | None, values: Mapping[str, Any]
) -> list[Any]:
    """Instances from one map, with the sweep's parameters applied."""
    return extract_instances(
        probability,
        threshold=float(values.get("threshold", DEFAULT_THRESHOLD)),
        min_area=int(values.get("min_area", DEFAULT_MIN_AREA)),
        disk=disk,
        output_size=int(values["output_size"]),
        join_gap=float(values.get("join_gap", 0.0)),
        join_angle=float(values.get("join_angle", DEFAULT_MAX_ANGLE)),
        join_offset=float(values.get("join_offset", DEFAULT_MAX_OFFSET)),
    )


def save_map(probability: np.ndarray, path: Path | str) -> Path:
    """Write one probability map as ``uint8``, and return where it landed.

    A byte per pixel resolves the probability to 1/255, which is far finer
    than anything downstream distinguishes: every threshold from 0.3 to 0.7
    scored within 0.001 of the others on fold 0. It halves what float16 costs,
    and a map that is mostly zeros compresses well on top of that, which is
    what makes a fold's worth small enough to carry off the machine that
    produced it.
    """
    if probability.ndim != 2:
        raise ValueError(f"Expected a 2-D probability map, got shape {probability.shape}.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantised = np.clip(np.rint(np.asarray(probability, dtype=np.float32) * 255.0), 0, 255)
    np.save(destination, quantised.astype(np.uint8))
    return destination.with_suffix(".npy")


def load_maps(directory: Path | str, stems: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Read probability maps written as ``<stem>.npy``.

    Both storage formats this project has used are accepted: ``uint8``, where
    a byte holds the probability in 1/255 steps, and the float16 the earlier
    phases wrote. Either way the caller gets float32 in ``[0, 1]``, because the
    comparisons downstream are done in float32.
    """
    folder = Path(directory)
    wanted = None if stems is None else set(stems)
    maps: dict[str, np.ndarray] = {}
    for path in sorted(folder.glob("*.npy")):
        if wanted is not None and path.stem not in wanted:
            continue
        stored = np.load(path)
        scale = 255.0 if stored.dtype == np.uint8 else 1.0
        maps[path.stem] = stored.astype(np.float32) / scale
    if not maps:
        raise FileNotFoundError(f"No probability maps found in {folder}.")
    return maps
