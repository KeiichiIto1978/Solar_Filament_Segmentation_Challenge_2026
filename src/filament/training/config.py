"""Training configuration, read from YAML.

Hyperparameters live in ``configs/`` rather than in code so that an experiment
is identified by a file that can be committed next to its result. Every run
writes its resolved configuration beside the checkpoint, which is what makes a
number in the lab notebook traceable months later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from filament.data.crops import (
    DEFAULT_CONTEXT,
    DEFAULT_CROP_SIZE,
    DEFAULT_SEED_PADDING,
)
from filament.data.dataset import DEFAULT_IMAGE_SIZE
from filament.data.split import DEFAULT_SEED
from filament.models.cldice import DEFAULT_ITERATIONS
from filament.models.segmentation import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_ENCODER,
    DEFAULT_ENCODER_WEIGHTS,
)


@dataclass(frozen=True)
class AugmentationConfig:
    """Which random transforms to apply while training."""

    horizontal_flip: bool = True
    vertical_flip: bool = True
    quarter_turns: bool = True


@dataclass(frozen=True)
class LossConfig:
    """Relative weight of the two loss terms, and how Dice reads the target."""

    dice_weight: float = 1.0
    bce_weight: float = 1.0
    dice_majority: float = 0.5
    cldice_weight: float = 0.0
    cldice_iterations: int = DEFAULT_ITERATIONS


@dataclass(frozen=True)
class CropConfig:
    """How a filament is cut out, when training on crops rather than frames."""

    size: int = DEFAULT_CROP_SIZE
    context: float = DEFAULT_CONTEXT
    seed_padding: float = DEFAULT_SEED_PADDING
    # Zero keeps the crop exactly on its box, which is what an upper bound
    # wants. A system fed by a detector needs this non-zero, so that the seed
    # it trains on is as loose as the one it will be handed.
    jitter: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    """Everything one training run needs.

    Attributes:
        fold: Which fold of the frozen split to validate on.
        image_size: Side length frames are resized to.
        architecture: Decoder family, named as segmentation_models_pytorch
            spells the class (``Unet``, ``UPerNet``, ``Segformer``, ...).
        encoder: Encoder name passed to segmentation_models_pytorch.
        encoder_weights: Pretrained weights, or ``None`` for random init.
        epochs: Number of passes over the training split.
        batch_size: Samples per step.
        learning_rate: Initial learning rate of AdamW.
        weight_decay: AdamW weight decay.
        seed: Seed for torch, numpy and the augmentation.
        num_workers: DataLoader worker processes. Zero on Windows, where each
            worker would re-read the 48 MB annotation file.
        amp: Use mixed precision. Ignored without a GPU.
        output_dir: Where checkpoints and the resolved config are written.
        max_train_batches: Stop each epoch early. For smoke runs only.
        max_val_batches: Same, for validation.
        vote_targets: Train against the share of annotators who drew each
            pixel rather than against one annotator's own tracing.
        crops: Train on one filament at a time, cut out of the frame at full
            resolution, instead of on whole frames shrunk to fit. Changes the
            input to three channels: the crop, its contrast-equalised version,
            and the box saying which filament is being asked for.
    """

    fold: int = 0
    image_size: int = DEFAULT_IMAGE_SIZE
    architecture: str = DEFAULT_ARCHITECTURE
    encoder: str = DEFAULT_ENCODER
    encoder_weights: str | None = DEFAULT_ENCODER_WEIGHTS
    epochs: int = 40
    batch_size: int = 4
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    seed: int = DEFAULT_SEED
    num_workers: int = 0
    amp: bool = True
    output_dir: Path = Path("outputs/phase1_unet")
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    vote_targets: bool = False
    crops: CropConfig | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: Any) -> TrainConfig:
        """Read a configuration file, then apply command line overrides.

        Args:
            path: YAML file.
            overrides: Values that win over the file. ``None`` values are
                ignored, so an unset command line flag changes nothing.

        Raises:
            KeyError: If the file names a field that does not exist. A typo in
                a hyperparameter name would otherwise be silently ignored and
                the run would report the wrong settings.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update({key: value for key, value in overrides.items() if value is not None})
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        """Build a configuration from a plain mapping."""
        known = {item.name for item in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(
                f"Unknown configuration field(s): {', '.join(sorted(unknown))}. "
                f"Known fields are {', '.join(sorted(known))}."
            )

        values = dict(raw)
        if "augmentation" in values:
            values["augmentation"] = AugmentationConfig(**values["augmentation"])
        if "loss" in values:
            values["loss"] = LossConfig(**values["loss"])
        if values.get("crops") is not None:
            values["crops"] = CropConfig(**values["crops"])
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """A YAML-writable view of the configuration."""
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def save(self, path: Path | str) -> Path:
        """Write the resolved configuration next to a run's output."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        return destination
