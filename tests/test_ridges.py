"""Tests for the dark-ridge response.

A synthetic frame -- flat brightness with a dark line, a bright line and a dark
blob painted in -- has a known answer: the dark line lights up, the bright line
does not, and the blob responds much less than the line.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.data.ridges import dark_ridge_response

SIDE = 200


def _frame() -> np.ndarray:
    """Background 150; a dark line 4 px wide; a bright line; a dark round blob."""
    frame = np.full((SIDE, SIDE), 150.0)
    frame[48:52, 20:180] = 110.0  # dark horizontal line
    frame[148:152, 20:180] = 190.0  # bright horizontal line
    rows, columns = np.mgrid[:SIDE, :SIDE]
    frame[np.hypot(rows - 100, columns - 100) <= 8] = 110.0  # dark blob
    return frame


def test_a_dark_line_responds_and_flat_ground_does_not() -> None:
    response = dark_ridge_response(_frame())

    on_line = response[50, 60:140].mean()
    flat = response[20, 60:140].mean()
    assert on_line > 0.5
    assert flat < 0.01


def test_a_bright_line_does_not_respond() -> None:
    response = dark_ridge_response(_frame())

    assert response[150, 60:140].max() == 0.0


def test_a_dark_blob_responds_far_less_than_a_dark_line() -> None:
    response = dark_ridge_response(_frame())

    assert response[100, 100] < 0.25 * response[50, 60:140].mean()


def test_the_response_stays_on_the_line() -> None:
    """Beside a filament the frame is background, and the response has to say
    so; otherwise uncovered stretches and background could not be told apart."""
    response = dark_ridge_response(_frame())

    on_line = response[50, 60:140].mean()
    beside = response[62, 60:140].mean()  # ten pixels from the line's edge
    assert beside < 0.1 * on_line


def test_the_response_lies_in_the_unit_interval() -> None:
    response = dark_ridge_response(_frame())

    assert response.dtype == np.float32
    assert response.min() >= 0.0
    assert response.max() <= 1.0


@pytest.mark.parametrize("bad", [np.zeros((4, 4, 3)), np.zeros(4)])
def test_only_two_dimensional_frames_are_accepted(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        dark_ridge_response(bad)
