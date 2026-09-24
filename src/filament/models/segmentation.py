"""The segmentation network, and the loss it is trained with.

The architecture comes from ``segmentation_models_pytorch`` rather than being
written out here. Two reasons: the public notebooks for this competition use
the same library, which makes their numbers comparable with ours, and an
encoder pretrained on ImageNet is worth more than a hand-written one on 707
frames.

Which architecture is a setting rather than a decision taken in code. The
baseline is a U-Net because that is where the project started, not because
anything measured said so, and the same library exposes a dozen alternatives
behind one constructor -- so comparing them costs a configuration file each.
The ones worth comparing are those that answer how this data fails: filaments
are thin, they cover 0.35% of a frame, and the ones that vanish from the
probability map entirely are the small ones. An encoder-decoder that halves
its resolution five times has the most to lose there, which is a reason to try
decoders that pool wide context and encoders that see globally from the first
stage -- not merely that they are newer.

The encoder expects three channels and we feed it two (the frame and its CLAHE
version). The library handles that by summing the pretrained weights of the
missing channel into the remaining ones, so the pretrained filters stay useful.

The loss is Dice plus binary cross-entropy, with an optional third term.
Cross-entropy alone is dominated by the background: filaments cover well under
one percent of a frame, so a model that predicts nothing at all already scores
a low loss. Dice measures overlap and is indifferent to how much background
there is, which is what makes it pull the model towards actually marking
something; cross-entropy is kept alongside it because Dice alone gives a weak
gradient early on, when the prediction and the target barely overlap.

Neither notices a filament predicted as two pieces, which is the single
failure the score keeps charging for. :mod:`filament.models.cldice` supplies a
term that does; it is off by default, so every earlier number still stands.
"""

from __future__ import annotations

from dataclasses import dataclass

import segmentation_models_pytorch as smp
import torch
from torch import nn

from filament.models.cldice import DEFAULT_ITERATIONS, SoftClDice

DEFAULT_ARCHITECTURE = "Unet"
DEFAULT_ENCODER = "resnet34"
DEFAULT_ENCODER_WEIGHTS = "imagenet"
INPUT_CHANNELS = 2


@dataclass(frozen=True)
class ModelConfig:
    """How to build the network: which decoder, which encoder, how many inputs."""

    architecture: str = DEFAULT_ARCHITECTURE
    encoder_name: str = DEFAULT_ENCODER
    encoder_weights: str | None = DEFAULT_ENCODER_WEIGHTS
    in_channels: int = INPUT_CHANNELS
    classes: int = 1

    def build(self) -> nn.Module:
        """Instantiate the network.

        Returns a model whose output is one channel of raw logits, at the same
        resolution as the input.

        Raises:
            ValueError: If the architecture is not one the library provides. A
                typo would otherwise surface as an ``AttributeError`` deep in
                the training run, after the data has already been loaded.
        """
        # Looked up case-insensitively through the library's own registry, so
        # that a configuration may name it the way the class is spelled.
        builder = smp.MODEL_ARCHITECTURES_MAPPING.get(self.architecture.lower())
        if builder is None:
            available = ", ".join(
                sorted(item.__name__ for item in smp.MODEL_ARCHITECTURES_MAPPING.values())
            )
            raise ValueError(f"Unknown architecture {self.architecture!r}. Available: {available}.")
        return builder(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.classes,
        )


def build_model(config: ModelConfig | None = None) -> nn.Module:
    """Build the network from ``config``, or from its defaults."""
    return (config or ModelConfig()).build()


class DiceBceLoss(nn.Module):
    """Soft Dice plus binary cross-entropy on logits.

    The two terms read the target differently, which only matters once the
    target carries a share rather than a label.

    Cross-entropy is minimised where the prediction equals the target, so a
    pixel two of three annotators drew pulls the prediction towards 2/3. That
    is the term that makes the output mean something a threshold can act on.

    Dice measures overlap and nothing else, so it is *not* minimised there: on
    a target of 2/3, predicting 1.0 scores 0.802 against 0.670 for predicting
    2/3. Left on the share it would drag the output back to the two extremes
    and undo the calibration. It therefore reads the target through a majority
    vote -- at least half the annotators -- and keeps doing the job it is here
    for, which is to stop a filament occupying under one percent of the frame
    from being drowned by the background.

    On a target that is already zeros and ones the majority vote changes
    nothing, so this is the same loss the earlier phases used.

    The two do pull against each other on a pixel two of three drew, where one
    asks for 2/3 and the other for 1. The output will sit between them rather
    than being calibrated outright; what has to survive is the order, weaker
    where fewer people drew, and that is what the threshold needs.

    Args:
        dice_weight: Weight of the Dice term.
        bce_weight: Weight of the cross-entropy term.
        smooth: Added to both sides of the Dice quotient, which keeps the loss
            finite when a target is empty and the prediction is too.
        dice_majority: Share of annotators above which Dice counts a pixel as
            filament. Half by default, so a pixel one of two people drew is
            kept: the predictions that miss are as often too small as too
            large, which is no reason to shrink the target.
        cldice_weight: Weight of the centreline term, which is what charges for
            a filament predicted as two pieces. Zero by default, leaving the
            loss exactly as the earlier phases had it.
        cldice_iterations: Rounds of peeling in the skeletonisation. Has to
            reach the radius of the thickest filament.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
        dice_majority: float = 0.5,
        cldice_weight: float = 0.0,
        cldice_iterations: int = DEFAULT_ITERATIONS,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.dice_majority = dice_majority
        self.cldice_weight = cldice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.cldice = SoftClDice(iterations=cldice_iterations, smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss of a ``(N, 1, H, W)`` batch of logits against its target."""
        if logits.shape != target.shape:
            raise ValueError(
                f"Logits {tuple(logits.shape)} and target {tuple(target.shape)} "
                "must have the same shape."
            )

        probability = torch.sigmoid(logits)
        # Per sample, so that one frame with many filaments cannot outweigh
        # several frames with few.
        flat_probability = probability.flatten(start_dim=1)
        flat_target = (target >= self.dice_majority).to(probability.dtype).flatten(start_dim=1)
        intersection = (flat_probability * flat_target).sum(dim=1)
        totals = flat_probability.sum(dim=1) + flat_target.sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (totals + self.smooth))

        loss = self.dice_weight * dice.mean() + self.bce_weight * self.bce(logits, target)
        if self.cldice_weight:
            # The centreline term reads the same majority vote as Dice, so that
            # a share of annotators does not blur the skeleton it is built from.
            loss = loss + self.cldice_weight * self.cldice(
                probability, flat_target.view_as(probability)
            )
        return loss
