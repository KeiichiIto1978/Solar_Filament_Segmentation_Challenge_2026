"""Tests for the centreline loss.

The property the whole term exists for is the last one here: two predictions
that Dice cannot tell apart, one of which has a filament broken in half.
"""

from __future__ import annotations

import pytest
import torch

from filament.models.cldice import (
    SoftClDice,
    soft_dilate,
    soft_erode,
    soft_open,
    soft_skeleton,
)
from filament.models.unet import DiceBceLoss

SIZE = 96


def _bar(rows: tuple[int, int] = (14, 19), columns: tuple[int, int] = (8, 56)) -> torch.Tensor:
    """A horizontal bar: forty-eight pixels long, five thick."""
    mask = torch.zeros(1, 1, SIZE, SIZE)
    mask[0, 0, rows[0] : rows[1], columns[0] : columns[1]] = 1.0
    return mask


def test_eroding_shrinks_and_dilating_grows() -> None:
    bar = _bar()

    assert float(soft_erode(bar).sum()) < float(bar.sum())
    assert float(soft_dilate(bar).sum()) > float(bar.sum())
    # Opening restores what erosion took, except what was too thin to survive.
    assert float(soft_open(bar).sum()) <= float(bar.sum())


def test_a_skeleton_is_thinner_than_the_shape_and_inside_it() -> None:
    bar = _bar()

    skeleton = soft_skeleton(bar, iterations=5)

    assert float(skeleton.sum()) < float(bar.sum())
    # Nothing outside the shape: the skeleton of a bar stays on the bar.
    assert float((skeleton * (1.0 - bar)).sum()) == pytest.approx(0.0, abs=1e-6)


def test_a_line_one_pixel_thick_is_already_its_own_skeleton() -> None:
    line = _bar(rows=(16, 17))

    skeleton = soft_skeleton(line, iterations=5)

    assert torch.allclose(skeleton, line, atol=1e-6)


def test_too_few_iterations_hide_a_break_entirely() -> None:
    """Why the iteration count is 25 and not the reference's 10.

    A solid shape survives being opened, so nothing is added to the skeleton
    until erosion has worn it down to a line. Short of that the skeleton is
    *empty* rather than thick, and an empty skeleton takes both quotients to
    one -- a filament broken in half scores no loss at all.

    Peeling has to reach the middle. Below a bar of radius ten the break here
    is invisible; at ten and above it costs 0.05. Fold 0's thickest filament
    has a radius of 20.6 at 1024 pixels.
    """
    thick = _bar(rows=(22, 42), columns=(8, 88))  # twenty pixels thick
    broken = thick.clone()
    broken[0, 0, 22:42, 44:50] = 0.0

    assert float(soft_skeleton(thick, iterations=8).sum()) == pytest.approx(0.0)
    assert float(soft_skeleton(thick, iterations=12).sum()) > 0.0

    assert float(SoftClDice(iterations=8)(broken, thick)) == pytest.approx(0.0, abs=1e-6)
    assert float(SoftClDice(iterations=12)(broken, thick)) > 0.04


def test_a_perfect_prediction_costs_nothing() -> None:
    bar = _bar()

    assert float(SoftClDice(iterations=5)(bar, bar)) == pytest.approx(0.0, abs=1e-6)


def test_the_loss_falls_as_the_prediction_approaches_the_target() -> None:
    target = _bar()
    close = _bar(columns=(8, 54))
    far = _bar(columns=(8, 30))

    criterion = SoftClDice(iterations=5)

    assert float(criterion(close, target)) < float(criterion(far, target))


def test_breaking_a_filament_costs_more_than_thinning_it_by_the_same_area() -> None:
    """Why this loss is here.

    Both predictions lose thirty pixels of a two-hundred-and-forty pixel bar,
    so Dice scores them identically. One loses them from an edge and stays in
    one piece; the other loses them from the middle and becomes two. Panoptic
    Quality charges heavily for the second -- one annotation covered by two
    predictions is a false negative and two false positives -- and 155 of fold
    0's annotations are split that way.
    """
    target = _bar()
    broken = _bar()
    broken[0, 0, 14:19, 30:36] = 0.0  # a six pixel gap, thirty pixels removed
    thinned = _bar()
    thinned[0, 0, 14, 8:38] = 0.0  # thirty pixels off one edge

    def dice(prediction: torch.Tensor) -> float:
        overlap = float((prediction * target).sum())
        return 2.0 * overlap / float(prediction.sum() + target.sum())

    assert dice(broken) == pytest.approx(dice(thinned))

    criterion = SoftClDice(iterations=5)
    cost_of_breaking = float(criterion(broken, target))
    cost_of_thinning = float(criterion(thinned, target))

    assert cost_of_thinning == pytest.approx(0.0, abs=1e-6)
    assert cost_of_breaking > 0.05


def test_the_term_is_off_unless_it_is_asked_for() -> None:
    """Every number recorded before this term existed still stands."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.9).float()

    without = float(DiceBceLoss()(logits, target))
    explicitly_off = float(DiceBceLoss(cldice_weight=0.0)(logits, target))
    switched_on = float(DiceBceLoss(cldice_weight=0.5, cldice_iterations=5)(logits, target))

    assert without == pytest.approx(explicitly_off)
    assert switched_on > without


def test_the_term_carries_a_gradient() -> None:
    target = _bar()
    logits = torch.zeros(1, 1, SIZE, SIZE, requires_grad=True)

    DiceBceLoss(cldice_weight=1.0, cldice_iterations=5)(logits, target).backward()

    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_malformed_input_is_rejected() -> None:
    bar = _bar()

    with pytest.raises(ValueError, match="N, C, H, W"):
        soft_skeleton(bar[0], iterations=5)
    with pytest.raises(ValueError, match="at least one iteration"):
        soft_skeleton(bar, iterations=0)
    with pytest.raises(ValueError, match="must have the same shape"):
        SoftClDice(iterations=5)(bar, torch.zeros(1, 1, 32, 32))


def test_the_faster_loop_matches_the_published_one() -> None:
    """Carrying the erosion between rounds saves two of five pooling passes.

    Worth having: at 1024 pixels and a batch of four the term costs seconds per
    step on a CPU. Worth checking, because the saving is only legitimate if it
    changes nothing.
    """

    def published(mask: torch.Tensor, iterations: int) -> torch.Tensor:
        """Shit et al.'s loop, written out to compare against."""
        skeleton = torch.relu(mask - soft_open(mask))
        for _ in range(iterations):
            mask = soft_erode(mask)
            delta = torch.relu(mask - soft_open(mask))
            skeleton = skeleton + torch.relu(delta - skeleton * delta)
        return skeleton

    torch.manual_seed(0)
    for shape in ((2, 1, 48, 48), (1, 1, 96, 96)):
        mask = torch.rand(*shape)
        for iterations in (1, 5, 25):
            assert torch.allclose(
                published(mask, iterations), soft_skeleton(mask, iterations), atol=1e-6
            )
