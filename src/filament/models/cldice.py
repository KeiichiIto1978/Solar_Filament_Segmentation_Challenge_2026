"""A loss that notices when one filament is predicted as two.

Every loss used so far is computed over pixels, and a break in a filament costs
almost nothing there: the few pixels that would have joined two pieces are a
rounding error against the frame. Panoptic Quality charges for it properly --
one annotation covered by two predictions is a false negative and two false
positives -- and 155 of fold 0's 1,795 annotations are split that way, a count
that did not move between a fifteen and a forty epoch schedule. Splitting is
not something more training fixes.

clDice, from Shit et al. (CVPR 2021), is built for exactly this failure on
exactly this shape of structure -- vessels, neurons, roads, all long and thin
and ruined by a gap. It compares each mask against the *skeleton* of the other,
so a break costs the whole branch that the skeleton no longer reaches rather
than the handful of pixels in the gap.

The skeleton is obtained by repeated morphological opening, written with
pooling so that it differentiates. The number of iterations has to reach the
thickest structure present: measured on fold 0's training frames at 1024
pixels, a filament's radius is 4.1 pixels at the median and 20.6 at the most,
so twenty-five iterations skeletonise all of them.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

# Enough iterations to reach the middle of the thickest filament measured on
# fold 0's training split at 1024 pixels (radius 20.6). Going higher costs time
# without changing the result; going lower leaves the thickest cores unpeeled.
DEFAULT_ITERATIONS = 25

# Added to both sides of every quotient, so that a frame with nothing predicted
# in it gives a finite loss rather than a division by zero.
DEFAULT_SMOOTH = 1.0


def soft_erode(mask: torch.Tensor) -> torch.Tensor:
    """Erode a soft mask: the minimum over a plus-shaped neighbourhood.

    Min-pooling is max-pooling of the negative. The two one-dimensional passes
    are the separable form the reference implementation uses; a square
    structuring element would eat the diagonal ends of a filament faster than
    its sides.
    """
    vertical = -functional.max_pool2d(-mask, (3, 1), (1, 1), (1, 0))
    horizontal = -functional.max_pool2d(-mask, (1, 3), (1, 1), (0, 1))
    return torch.min(vertical, horizontal)


def soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    """Dilate a soft mask: the maximum over a three by three neighbourhood."""
    return functional.max_pool2d(mask, (3, 3), (1, 1), (1, 1))


def soft_open(mask: torch.Tensor) -> torch.Tensor:
    """Erode then dilate, which removes whatever is thinner than the kernel."""
    return soft_dilate(soft_erode(mask))


def soft_skeleton(mask: torch.Tensor, iterations: int = DEFAULT_ITERATIONS) -> torch.Tensor:
    """The centreline of a soft mask, differentiably.

    Each round removes one layer from the outside and keeps whatever the
    opening cannot restore -- the parts too thin to survive being eroded and
    dilated, which is what a centreline is. Accumulating those parts over
    rounds peels the shape inwards until only its middle is left.

    Args:
        mask: ``(N, C, H, W)`` in ``[0, 1]``.
        iterations: Rounds of peeling. Must reach the radius of the thickest
            structure, or its core is never reduced to a line.

    Returns:
        A tensor of the same shape, in ``[0, 1]``.
    """
    if mask.dim() != 4:
        raise ValueError(f"Expected an (N, C, H, W) tensor, got shape {tuple(mask.shape)}.")
    if iterations < 1:
        raise ValueError(f"Skeletonising takes at least one iteration, got {iterations}.")

    # Each round needs the erosion of the current mask twice: once to open it,
    # and once to become the next round's mask. Computing it once and carrying
    # it takes a round from five pooling passes to three, and changes nothing.
    eroded = soft_erode(mask)
    skeleton = functional.relu(mask - soft_dilate(eroded))

    for _ in range(iterations):
        mask = eroded
        eroded = soft_erode(mask)
        delta = functional.relu(mask - soft_dilate(eroded))
        # Union rather than sum: a pixel already on the skeleton must not be
        # counted twice when a later round finds it again.
        skeleton = skeleton + functional.relu(delta - skeleton * delta)

    return skeleton


class SoftClDice(nn.Module):
    """One minus the centreline Dice of a prediction against its target.

    Two quantities, each asking whether one mask's centreline lies inside the
    other mask:

    - how much of the predicted centreline is annotated, which falls when the
      prediction runs somewhere it should not;
    - how much of the annotated centreline is predicted, which falls when a
      filament is missed *or broken*, since the piece of centreline crossing
      the gap is no longer covered.

    Their harmonic mean is the score, and the loss is one minus it.

    Computed per sample and averaged, matching how Dice is handled here, so
    that a frame carrying twenty filaments cannot outweigh three frames
    carrying two. The reference implementation sums over the batch instead.

    The target's skeleton is built without gradients: it is a constant, and
    twenty-five rounds of pooling over it are not worth recording.

    Args:
        iterations: Rounds of peeling in the skeletonisation.
        smooth: Added to both sides of each quotient.
    """

    def __init__(
        self,
        iterations: int = DEFAULT_ITERATIONS,
        smooth: float = DEFAULT_SMOOTH,
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss of a ``(N, 1, H, W)`` batch of probabilities against its target."""
        if probability.shape != target.shape:
            raise ValueError(
                f"Probability {tuple(probability.shape)} and target {tuple(target.shape)} "
                "must have the same shape."
            )

        predicted_skeleton = soft_skeleton(probability, self.iterations)
        with torch.no_grad():
            target_skeleton = soft_skeleton(target, self.iterations)

        flat = probability.flatten(start_dim=1)
        flat_target = target.flatten(start_dim=1)
        flat_predicted_skeleton = predicted_skeleton.flatten(start_dim=1)
        flat_target_skeleton = target_skeleton.flatten(start_dim=1)

        precision = ((flat_predicted_skeleton * flat_target).sum(dim=1) + self.smooth) / (
            flat_predicted_skeleton.sum(dim=1) + self.smooth
        )
        sensitivity = ((flat_target_skeleton * flat).sum(dim=1) + self.smooth) / (
            flat_target_skeleton.sum(dim=1) + self.smooth
        )

        score = 2.0 * precision * sensitivity / (precision + sensitivity)
        return (1.0 - score).mean()
