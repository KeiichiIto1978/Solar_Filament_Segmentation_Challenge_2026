"""Reading and preparing the H-alpha frames.

The frames are 8-bit grayscale JPEG, 2048x2048. They are read as a single
channel: decoding them as RGB would triple the memory for three identical
copies of the same data.

Two channels are handed to the model. The first is the frame itself, the
second the same frame after contrast-limited adaptive histogram equalisation
(CLAHE). Exploratory analysis showed that single frames vary far more in
brightness and contrast than the six telescope sites do, so the correction
that matters is per-frame rather than per-site. Keeping the original alongside
the corrected version lets the model fall back on the raw values where the
equalisation overshoots.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# The competition frames are all this size.
FULL_SIZE = 2048

# Defaults follow the public notebooks, which makes their results comparable.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def load_grayscale(path: Path | str) -> np.ndarray:
    """Read one frame as an 8-bit single-channel array.

    Args:
        path: Image file.

    Returns:
        A ``(height, width)`` array of ``uint8``.

    Raises:
        FileNotFoundError: If the file is missing or cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read an image from {path}.")
    return image


def resize(image: np.ndarray, size: int, mask: bool = False) -> np.ndarray:
    """Resize a square image to ``size`` x ``size``.

    Args:
        image: Two-dimensional array.
        size: Target side length.
        mask: True for label data. Masks are resized by nearest neighbour so
            that no interpolated value between two labels is invented; images
            use area/linear interpolation, which is smoother.

    Returns:
        The resized array, with the input dtype preserved.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}.")
    if image.shape[0] == size and image.shape[1] == size:
        return image

    if mask:
        interpolation = cv2.INTER_NEAREST
    elif size < image.shape[0]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Equalise local contrast, so that faint filaments stand out.

    Args:
        image: ``uint8`` single-channel array.
        clip_limit: Ceiling on local contrast amplification. Higher values
            bring out more detail and more noise with it.
        tile_grid: Number of tiles the frame is divided into.

    Returns:
        A ``uint8`` array of the same shape.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"CLAHE expects uint8 input, got {image.dtype}.")
    equaliser = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return equaliser.apply(image)


def to_model_input(
    image: np.ndarray,
    size: int,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Build the two-channel input the model expects.

    Args:
        image: ``uint8`` frame, any square size.
        size: Side length to resize to.
        clip_limit: CLAHE clip limit.
        tile_grid: CLAHE tile grid.

    Returns:
        A ``(2, size, size)`` ``float32`` array scaled to ``[0, 1]``: the
        resized frame, then its CLAHE version.
    """
    resized = resize(image, size)
    equalised = apply_clahe(resized, clip_limit, tile_grid)
    stacked = np.stack([resized, equalised]).astype(np.float32)
    return stacked / 255.0
