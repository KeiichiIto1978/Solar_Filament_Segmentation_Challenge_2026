"""Locating the solar disk, so that predictions outside it can be discarded.

Filaments only exist on the disk. Anything a model marks in the surrounding sky
is a false positive, and a false positive costs twice what a missed filament
costs under Panoptic Quality, so masking the sky away is free score.

Three methods were tried on the training set before settling on this one.

*Thresholding and taking the largest bright region* fails on the frames that
carry a bright halo just outside the limb: on some of them the halo is brighter
than the dim outer part of the disk, so the two merge. Four frames out of 24
sampled came out at radius 1000-1020 instead of 904.

*Scanning single rows and columns for the sharpest edge* fixes those frames but
breaks others, because a dark filament lying near the middle of the frame can
produce a local edge as steep as the limb.

*Averaging the intensity over each annulus around the frame centre and taking
the steepest drop* works on all of them. Averaging over a whole ring dilutes
both the halo and any individual filament, while the limb survives because it
is at the same radius all the way round. On all 707 training frames this gives
a radius of 900-908 pixels, 904 for 681 of them, with no failures.

The frame centre is used as the disk centre rather than being estimated. The
annulus average is insensitive to a centre error of a few pixels, and the
containment check below is measured the same way: the furthest annotated vertex
in the training set sits at 0.999 of the detected radius from the frame centre.
Because that leaves no room at all, :data:`DEFAULT_MASK_MARGIN` grows the mask
slightly rather than clipping filaments drawn right up to the limb.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# A full-disk GONG frame has a radius of about 0.44 of its side length. The
# search is restricted to this band, which is what keeps the halo's outer edge
# (0.497 of the side on the frames where it appears) from being mistaken for
# the limb.
MIN_RADIUS_FRACTION = 0.35
MAX_RADIUS_FRACTION = 0.49

# Annotated filaments reach 0.999 of the radius, so the mask is grown by a
# little before anything is cut away.
DEFAULT_MASK_MARGIN = 16.0


class DiskNotFoundError(ValueError):
    """Raised when no plausible solar limb is found in a frame."""


@dataclass(frozen=True)
class Disk:
    """The solar disk in image coordinates."""

    center_x: float
    center_y: float
    radius: float

    @property
    def area(self) -> float:
        """Area of the disk in pixels."""
        return float(np.pi * self.radius**2)

    def scaled(self, factor: float) -> Disk:
        """The same disk in an image resized by ``factor``."""
        return Disk(
            center_x=self.center_x * factor,
            center_y=self.center_y * factor,
            radius=self.radius * factor,
        )

    def mask(
        self,
        height: int,
        width: int,
        margin: float = DEFAULT_MASK_MARGIN,
    ) -> np.ndarray:
        """A boolean mask that is true on the disk.

        Args:
            height: Mask height.
            width: Mask width.
            margin: Pixels to grow the radius by, so that a filament drawn up
                to the limb is not clipped.

        Returns:
            A ``(height, width)`` boolean array.
        """
        rows = np.arange(height, dtype=np.float32)[:, None]
        columns = np.arange(width, dtype=np.float32)[None, :]
        squared = (columns - self.center_x) ** 2 + (rows - self.center_y) ** 2
        return squared <= (self.radius + margin) ** 2

    def distance_from_center(self, x: float, y: float) -> float:
        """Distance of a point from the disk centre, in pixels."""
        return float(np.hypot(x - self.center_x, y - self.center_y))

    def contains(self, x: float, y: float, margin: float = DEFAULT_MASK_MARGIN) -> bool:
        """Whether a point lies on the disk, allowing for ``margin``."""
        return self.distance_from_center(x, y) <= self.radius + margin


@lru_cache(maxsize=8)
def _annulus_bins(
    height: int, width: int, low: int, high: int, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Index, for one frame size, which annulus each pixel belongs to.

    Cached because it costs more than the radial profile itself and depends
    only on the frame size.

    Returns:
        The selection mask, the annulus index of each selected pixel, the
        number of pixels per annulus, and the inner radius of each annulus.
    """
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    distance = np.hypot(columns - (width - 1) / 2, rows - (height - 1) / 2)

    inner_radii = np.arange(low, high, step)
    selected = (distance >= low) & (distance < high)
    index = ((distance[selected] - low) // step).astype(np.int32)
    counts = np.bincount(index, minlength=len(inner_radii))
    return selected, index, counts, inner_radii


def radial_profile(
    image: np.ndarray, low: int, high: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity per annulus around the frame centre.

    Args:
        image: Two-dimensional frame.
        low: Inner radius of the first annulus.
        high: Outer radius of the last annulus.
        step: Annulus width in pixels.

    Returns:
        The inner radius of each annulus and its mean intensity.
    """
    height, width = image.shape
    selected, index, counts, inner_radii = _annulus_bins(height, width, low, high, step)
    totals = np.bincount(
        index, weights=image[selected].astype(np.float64), minlength=len(inner_radii)
    )
    return inner_radii, totals / np.maximum(counts, 1)


def detect_disk(image: np.ndarray, step: int | None = None) -> Disk:
    """Locate the solar disk in a full-disk frame.

    Args:
        image: Two-dimensional grayscale frame.
        step: Annulus width in pixels. Defaults to roughly one part in 500 of
            the frame side, which is 4 pixels for a 2048 frame.

    Returns:
        The detected disk, centred on the frame centre.

    Raises:
        DiskNotFoundError: If the steepest drop falls at the edge of the
            search band, which means no limb was found inside it. Failing
            loudly matters: a wrong radius would either delete real filaments
            or stop masking the sky at all.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D frame, got shape {image.shape}.")

    height, width = image.shape
    side = min(height, width)
    resolution = step if step is not None else max(1, side // 500)
    low = int(MIN_RADIUS_FRACTION * side)
    high = int(MAX_RADIUS_FRACTION * side)
    if high - low < 4 * resolution:
        raise DiskNotFoundError(
            f"A {height}x{width} frame leaves no room to search for a limb "
            f"between radius {low} and {high}."
        )

    inner_radii, profile = radial_profile(image, low, high, resolution)
    # The limb is where the intensity falls off fastest going outwards.
    drop = np.diff(profile)
    position = int(np.argmin(drop))
    if position in (0, len(drop) - 1):
        raise DiskNotFoundError(
            f"The steepest intensity drop is at radius "
            f"{inner_radii[position] + resolution}, at the edge of the "
            f"{low}-{high} search band; this frame does not look like a full disk."
        )

    radius = float(inner_radii[position] + resolution)
    return Disk(center_x=(width - 1) / 2, center_y=(height - 1) / 2, radius=radius)
