"""The U-Net of the semantic baseline, and the loss it is trained with.

The architecture comes from ``segmentation_models_pytorch`` rather than being
written out here. Two reasons: the public notebooks for this competition use
the same library, which makes their numbers comparable with ours, and an
encoder pretrained on ImageNet is worth more than a hand-written one on 707
frames.

The encoder expects three channels and we feed it two (the frame and its CLAHE
version). The library handles that by summing the pretrained weights of the
missing channel into the remaining ones, so the pretrained filters stay useful.

The loss is Dice plus binary cross-entropy. Cross-entropy alone is dominated by
the background: filaments cover well under one percent of a frame, so a model
that predicts nothing at all already scores a low loss. Dice measures overlap
and is indifferent to how much background there is, which is what makes it pull
the model towards actually marking something; cross-entropy is kept alongside
it because Dice alone gives a weak gradient early on, when the prediction and
the target barely overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import segmentation_models_pytorch as smp
import torch
from torch import nn

DEFAULT_ENCODER = "resnet34"
DEFAULT_ENCODER_WEIGHTS = "imagenet"
INPUT_CHANNELS = 2


@dataclass(frozen=True)
class UNetConfig:
    """How to build the baseline network."""

    encoder_name: str = DEFAULT_ENCODER
    encoder_weights: str | None = DEFAULT_ENCODER_WEIGHTS
    in_channels: int = INPUT_CHANNELS
    classes: int = 1

    def build(self) -> nn.Module:
        """Instantiate the network.

        Returns a model whose output is one channel of raw logits, at the same
        resolution as the input.
        """
        return smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.classes,
        )


def build_model(config: UNetConfig | None = None) -> nn.Module:
    """Build the baseline network from ``config``, or from its defaults."""
    return (config or UNetConfig()).build()


class DiceBceLoss(nn.Module):
    """Soft Dice plus binary cross-entropy on logits.

    Args:
        dice_weight: Weight of the Dice term.
        bce_weight: Weight of the cross-entropy term.
        smooth: Added to both sides of the Dice quotient, which keeps the loss
            finite when a target is empty and the prediction is too.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

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
        flat_target = target.flatten(start_dim=1)
        intersection = (flat_probability * flat_target).sum(dim=1)
        totals = flat_probability.sum(dim=1) + flat_target.sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (totals + self.smooth))

        return self.dice_weight * dice.mean() + self.bce_weight * self.bce(logits, target)
