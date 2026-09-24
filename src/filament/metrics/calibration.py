"""Checking that a probability map means what it was trained to mean.

Training against the share of annotators who drew a pixel is only worth doing
if the output ends up ordered by that share: the threshold is then a decision
about how much of a minority opinion to keep, and the point at which keeping it
pays can be worked out from the metric. A filament k of n people drew, emitted
at IoU u, adds ``k * u`` to the numerator of Panoptic Quality and ``0.5 * n``
to its denominator against staying silent, so it is worth emitting when
``k * u > 0.5 * n * PQ``.

That argument needs the map to be ordered by vote share. Whether it is cannot
be assumed -- the Dice term pulls towards the extremes while cross-entropy
pulls towards the share -- so it is measured: pixels are grouped by what the
model said, and each group is scored against the share that actually drew it.

Perfect calibration is the diagonal, but landing on it is not the point and
the distance to it is a poor summary: the background fills the bottom bin with
99.6% of every frame and drags any pixel-weighted average to nothing. The two
numbers that separate a map carrying a share from a two-valued one are how
much of the marked area sits in the middle of the range at all, and how far
the observed share moves across it. The model this replaces put 0.08% of its
pixels between 0.05 and 0.90, and across that middle the share of annotators
went from 0.30 to 0.44 -- a threshold placed anywhere in there was choosing
between pixels that people had agreed about equally.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from filament.data.coco import AnnotatorImage
from filament.data.dataset import build_vote_target

logger = logging.getLogger(__name__)

DEFAULT_EDGES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class CalibrationBin:
    """One group of pixels, and what share of annotators actually drew them."""

    low: float
    high: float
    pixels: int
    predicted: float
    observed: float

    @property
    def label(self) -> str:
        return f"{self.low:.2f}-{self.high:.2f}"


@dataclass(frozen=True)
class Calibration:
    """The curve, and two numbers that summarise it."""

    bins: list[CalibrationBin]

    @property
    def populated(self) -> list[CalibrationBin]:
        """Bins that hold at least one pixel."""
        return [item for item in self.bins if item.pixels]

    @property
    def is_monotonic(self) -> bool:
        """Does the observed share rise with the predicted one?

        The property the threshold depends on. Calibration can be off by a
        constant and the threshold still works; an output that does not rise
        cannot be thresholded into a decision at all.
        """
        observed = [item.observed for item in self.populated]
        return all(later >= earlier for earlier, later in pairwise(observed))

    @property
    def spread(self) -> float:
        """Observed share in the top populated bin minus the bottom one.

        Dominated by the two ends, which are far apart in any model that works
        at all: an output of 0.98 does mean more annotators than one of 0.01.
        Read :attr:`middle_share` and :attr:`middle_span` instead to tell a map
        that carries a share from one that is still two-valued.
        """
        populated = self.populated
        return populated[-1].observed - populated[0].observed if populated else 0.0

    @property
    def middle(self) -> list[CalibrationBin]:
        """Populated bins away from both ends of the range.

        Fixed by the edges rather than by which bins happen to be populated:
        the bottom bin is where the background sits and the top is where the
        model is certain, and it is everything between them that a threshold
        would be choosing among.
        """
        return [item for item in self.bins[1:-1] if item.pixels]

    @property
    def middle_share(self) -> float:
        """Pixels in the middle bins, over the pixels outside the bottom one.

        Whether the model uses the middle of its range at all. The background
        fills the bottom bin in any run -- filaments occupy under half a
        percent of a frame -- so it is excluded from the denominator, leaving
        the question of how much of what the model marked it marked with
        something other than certainty. A two-valued map scores near zero here
        and there is nothing for a threshold to separate.
        """
        above_background = sum(item.pixels for item in self.bins[1:])
        if not above_background:
            return 0.0
        return sum(item.pixels for item in self.middle) / above_background

    @property
    def middle_span(self) -> float:
        """How far the observed share moves across the middle bins.

        What the threshold has to work with. If every middle value corresponds
        to the same share of annotators, moving the threshold through them
        changes which pixels are kept without changing who would have drawn
        them, which is the state the previous model was in.
        """
        middle = self.middle
        return middle[-1].observed - middle[0].observed if len(middle) >= 2 else 0.0

    @property
    def mean_absolute_error(self) -> float:
        """Pixel-weighted distance from the diagonal."""
        populated = self.populated
        if not populated:
            return 0.0
        weights = np.array([item.pixels for item in populated], dtype=float)
        errors = np.array([abs(item.observed - item.predicted) for item in populated])
        return float((weights * errors).sum() / weights.sum())

    def to_rows(self) -> list[dict[str, float | int | str]]:
        """A flat record per bin, for a table or a plot."""
        return [
            {
                "bin": item.label,
                "pixels": item.pixels,
                "predicted": round(item.predicted, 4),
                "observed": round(item.observed, 4),
            }
            for item in self.bins
        ]

    def __str__(self) -> str:
        return (
            f"monotonic={self.is_monotonic} middle_share={self.middle_share:.3f} "
            f"middle_span={self.middle_span:.3f} spread={self.spread:.3f} "
            f"mae={self.mean_absolute_error:.3f} bins={len(self.populated)}"
        )


def calibration(
    maps: Mapping[str, np.ndarray],
    by_stem: Mapping[str, Sequence[AnnotatorImage]],
    edges: Sequence[float] = DEFAULT_EDGES,
) -> Calibration:
    """Group predicted pixels and score each group against the vote share.

    Args:
        maps: Probability map per image stem, at the working resolution.
        by_stem: Every annotator's view of each of those frames, from
            :meth:`~filament.data.coco.Dataset.by_stem`.
        edges: Bin boundaries over the predicted probability, ascending. The
            last bin is closed so that a prediction of exactly 1 is counted.

    Returns:
        The curve. Frames with no annotations on record are skipped, with a
        warning, rather than being counted as all-background.
    """
    if len(edges) < 2:
        raise ValueError("Calibration needs at least two bin edges.")

    pixels = np.zeros(len(edges) - 1, dtype=np.int64)
    predicted = np.zeros(len(edges) - 1, dtype=np.float64)
    observed = np.zeros(len(edges) - 1, dtype=np.float64)

    missing = []
    for stem, probability in maps.items():
        entries = by_stem.get(stem)
        if not entries:
            missing.append(stem)
            continue

        votes = build_vote_target(list(entries), probability.shape[0])
        flat_probability = probability.reshape(-1)
        flat_votes = votes.reshape(-1)
        # Right-closed bins, so a probability of exactly 1 lands in the top one.
        index = np.clip(np.digitize(flat_probability, edges[1:-1], right=True), 0, len(pixels) - 1)
        pixels += np.bincount(index, minlength=len(pixels))
        predicted += np.bincount(index, weights=flat_probability, minlength=len(pixels))
        observed += np.bincount(index, weights=flat_votes, minlength=len(pixels))

    if missing:
        logger.warning(
            "%d frame(s) have a probability map but no annotations and were skipped: %s",
            len(missing),
            ", ".join(sorted(missing)[:5]),
        )

    bins = []
    for position, count in enumerate(pixels):
        bins.append(
            CalibrationBin(
                low=float(edges[position]),
                high=float(edges[position + 1]),
                pixels=int(count),
                predicted=float(predicted[position] / count) if count else 0.0,
                observed=float(observed[position] / count) if count else 0.0,
            )
        )
    return Calibration(bins=bins)
