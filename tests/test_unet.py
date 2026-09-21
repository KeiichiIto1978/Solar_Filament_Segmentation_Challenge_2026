"""Tests for the baseline network and its loss.

These run on the CPU: the machine this is developed on has no GPU, and the
point here is that the shapes line up and that the loss can actually be
descended. Training happens on Kaggle.
"""

from __future__ import annotations

import math

import pytest
import torch

from filament.models.unet import DiceBceLoss, UNetConfig, build_model

SIZE = 64


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    # No pretrained weights: downloading them would make the test need network
    # access, and randomly initialised weights exercise the same shapes.
    return build_model(UNetConfig(encoder_weights=None))


def test_the_model_maps_two_channels_to_one_at_the_same_resolution(
    model: torch.nn.Module,
) -> None:
    batch = torch.zeros(2, 2, SIZE, SIZE)

    logits = model(batch)

    assert logits.shape == (2, 1, SIZE, SIZE)


def test_the_output_is_raw_logits_not_probabilities(model: torch.nn.Module) -> None:
    logits = model(torch.rand(1, 2, SIZE, SIZE))

    # A sigmoid would have been applied already if this were bounded.
    assert logits.min() < 0.0 or logits.max() > 1.0


def test_a_backward_pass_reaches_the_first_layer(model: torch.nn.Module) -> None:
    model.zero_grad()
    logits = model(torch.rand(1, 2, SIZE, SIZE))

    logits.mean().backward()

    first = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    assert first.grad is not None
    assert torch.isfinite(first.grad).all()


def test_the_loss_is_zero_for_a_confident_correct_prediction() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1.0
    logits = torch.where(target > 0, 30.0, -30.0)

    loss = DiceBceLoss()(logits, target)

    assert float(loss) == pytest.approx(0.0, abs=1e-3)


def test_the_loss_is_larger_for_an_inverted_prediction() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1.0
    correct = torch.where(target > 0, 30.0, -30.0)

    assert float(DiceBceLoss()(-correct, target)) > float(DiceBceLoss()(correct, target))


def test_an_empty_target_and_an_empty_prediction_cost_nothing() -> None:
    """Dice is undefined there; the smoothing term is what keeps it finite."""
    target = torch.zeros(1, 1, 8, 8)

    loss = DiceBceLoss()(torch.full((1, 1, 8, 8), -30.0), target)

    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0, abs=1e-3)


def test_the_loss_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must have the same shape"):
        DiceBceLoss()(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4))


def test_cross_entropy_alone_would_reward_predicting_nothing() -> None:
    """Why the Dice term is there at all.

    With filaments on under 1% of the pixels, an all-background prediction
    already has a small cross-entropy. Dice sees it for what it is.
    """
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 30:34, 10:50] = 1.0  # about 4% of the frame
    nothing = torch.full((1, 1, 64, 64), -10.0)

    bce_only = DiceBceLoss(dice_weight=0.0, bce_weight=1.0)(nothing, target)
    with_dice = DiceBceLoss(dice_weight=1.0, bce_weight=1.0)(nothing, target)

    assert float(bce_only) < 0.5
    assert float(with_dice) > 0.9


def test_the_model_can_be_driven_down_on_a_single_batch(model: torch.nn.Module) -> None:
    """A network that cannot overfit one batch has a wiring problem."""
    torch.manual_seed(0)
    image = torch.rand(2, 2, SIZE, SIZE)
    target = torch.zeros(2, 1, SIZE, SIZE)
    target[:, :, 20:30, 10:50] = 1.0

    criterion = DiceBceLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        optimiser.zero_grad()
        loss = criterion(model(image), target)
        loss.backward()
        optimiser.step()
        losses.append(loss.detach().item())

    assert losses[-1] < losses[0] * 0.7


def test_a_binary_target_is_scored_exactly_as_before() -> None:
    """The majority vote Dice now applies is a no-op on zeros and ones, so
    every number the earlier phases recorded still stands."""
    torch.manual_seed(0)
    logits = torch.randn(3, 1, 8, 8)
    target = (torch.rand(3, 1, 8, 8) > 0.7).float()

    strict = DiceBceLoss(dice_majority=0.5)(logits, target)
    # Any cut inside (0, 1] splits zeros from ones the same way.
    lenient = DiceBceLoss(dice_majority=1.0)(logits, target)

    assert float(strict) == pytest.approx(float(lenient), abs=1e-6)


def test_cross_entropy_is_lowest_where_the_prediction_equals_the_share() -> None:
    """The term that makes the output mean something a threshold can act on."""
    target = torch.full((1, 1, 4, 4), 2.0 / 3.0)
    bce_only = DiceBceLoss(dice_weight=0.0, bce_weight=1.0)

    def loss_at(probability: float) -> float:
        odds = math.log(probability / (1.0 - probability))
        return float(bce_only(torch.full((1, 1, 4, 4), odds), target))

    assert loss_at(2 / 3) < loss_at(0.5)
    assert loss_at(2 / 3) < loss_at(0.95)


def test_the_two_terms_settle_between_the_share_and_certainty() -> None:
    """The tug-of-war, and the property the threshold actually needs.

    Cross-entropy asks for the share, Dice for one; the loss lands between
    them, so the output is not calibrated outright. What has to survive is the
    order -- a pixel a third of the annotators drew must settle lower than one
    two thirds drew -- because that is what a threshold reads.
    """
    criterion = DiceBceLoss()

    def settles_at(share: float) -> float:
        target = torch.full((1, 1, 16, 16), share)
        candidates = [round(0.05 * step, 2) for step in range(1, 20)]
        return min(
            candidates,
            key=lambda probability: float(
                criterion(
                    torch.full((1, 1, 16, 16), math.log(probability / (1.0 - probability))),
                    target,
                )
            ),
        )

    minority, majority = settles_at(1 / 3), settles_at(2 / 3)

    assert minority < majority < 1.0
    # Pulled above the share it was asked for, but nowhere near certainty.
    assert 2 / 3 < majority <= 0.85


def test_the_majority_vote_keeps_a_pixel_one_of_two_annotators_drew() -> None:
    """Half is a cut-off, not a strict majority: a share of exactly one half
    counts as filament, so a two-annotator frame is not shrunk."""
    half = torch.full((1, 1, 4, 4), 0.5)
    minority = torch.full((1, 1, 4, 4), 1.0 / 3.0)
    confident = torch.full((1, 1, 4, 4), 5.0)
    criterion = DiceBceLoss(dice_weight=1.0, bce_weight=0.0)

    # Predicting filament everywhere is right for the first and wrong for the
    # second, which is only true if one half is inside the target.
    assert float(criterion(confident, half)) < float(criterion(confident, minority))
