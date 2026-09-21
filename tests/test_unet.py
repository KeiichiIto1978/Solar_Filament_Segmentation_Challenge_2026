"""Tests for the baseline network and its loss.

These run on the CPU: the machine this is developed on has no GPU, and the
point here is that the shapes line up and that the loss can actually be
descended. Training happens on Kaggle.
"""

from __future__ import annotations

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
