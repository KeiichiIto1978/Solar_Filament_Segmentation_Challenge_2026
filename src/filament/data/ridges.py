"""How strongly a frame looks like a dark ridge at each pixel, without any training.

A filament in H-alpha is a thin structure darker than its surroundings. Across
it the brightness curves upwards on both sides; along it, it barely changes.
The Hessian of the smoothed frame measures exactly that: one large positive
eigenvalue (curvature across a dark line) and one small one (along it).
Frangi's vesselness turns the pair into a single number, high on dark lines of
the scale being looked at and near zero on flat ground, bright structures and
blobs.

This is evidence a trained network does not provide. The network reports what
it learnt from the annotators; this reports what the frame shows. Phase 12
found the network covering about a third of many long filaments and putting a
probability of 0.013 on the rest, which leaves open whether the rest is visible
in the frame at all. Comparing this response on those uncovered stretches with
the surrounding background answers that.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

# Widths of dark line to respond to, as Gaussian scales in pixels of the frame
# being filtered. A line about 2.5 sigma wide responds most at that sigma.
DEFAULT_SIGMAS = (1.0, 2.0, 3.0, 4.0, 6.0)
# Frangi's sensitivity to the ratio of the two eigenvalues: lower makes the
# response more selective for lines over blobs.
DEFAULT_BETA = 0.5


def dark_ridge_response(
    image: np.ndarray,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    """Frangi vesselness for dark lines, the maximum over ``sigmas``.

    Args:
        image: Two-dimensional frame, any numeric type; brighter is larger.
        sigmas: Gaussian scales to look at, in pixels of ``image``.
        beta: Selectivity for lines over blobs.

    Returns:
        A ``float32`` array of the same shape in ``[0, 1]``: high on dark
        ridges of one of the scales, zero where the cross-ridge curvature is not
        positive (bright structures) and near zero on flat ground.

    Raises:
        ValueError: If ``image`` is not two-dimensional or ``sigmas`` is empty.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a two-dimensional image, got shape {image.shape}.")
    if not sigmas:
        raise ValueError("At least one scale is needed.")
    frame = image.astype(np.float64)

    best = np.zeros(frame.shape, dtype=np.float64)
    for sigma in sigmas:
        # Scale-normalised second derivatives, so responses at different
        # scales are comparable.
        scale = sigma**2
        hxx = ndimage.gaussian_filter(frame, sigma, order=(0, 2)) * scale
        hyy = ndimage.gaussian_filter(frame, sigma, order=(2, 0)) * scale
        hxy = ndimage.gaussian_filter(frame, sigma, order=(1, 1)) * scale

        # Eigenvalues of the 2x2 Hessian, ordered by magnitude: small is along
        # the line, large across it.
        half_trace = (hxx + hyy) / 2.0
        root = np.sqrt(((hxx - hyy) / 2.0) ** 2 + hxy**2)
        first, second = half_trace + root, half_trace - root
        swap = np.abs(first) > np.abs(second)
        small = np.where(swap, second, first)
        large = np.where(swap, first, second)

        blobness = (small / np.where(large == 0, 1.0, large)) ** 2
        structure = small**2 + large**2
        # Frangi's own choice: half the strongest structure in the frame, so
        # the response adapts to each frame's contrast.
        c = 0.5 * np.sqrt(structure.max()) if structure.max() > 0 else 1.0
        response = np.exp(-blobness / (2 * beta**2)) * (1.0 - np.exp(-structure / (2 * c**2)))
        # Only dark lines: the brightness curves upwards across them.
        response[large <= 0] = 0.0
        best = np.maximum(best, response)
    return best.astype(np.float32)
