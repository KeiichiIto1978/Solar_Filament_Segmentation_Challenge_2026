"""Tests for the training configuration and loop."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from filament.paths import ProjectPaths
from filament.training.config import TrainConfig
from filament.training.loop import (
    CHECKPOINT_NAME,
    load_checkpoint,
    resolve_device,
    seed_everything,
    train,
)

MINIMAL_CONFIG = """
fold: 1
image_size: 128
epochs: 3
encoder: resnet18
augmentation:
  horizontal_flip: false
  vertical_flip: true
  quarter_turns: false
loss:
  dice_weight: 0.5
  bce_weight: 2.0
"""


def _write(path: Path, text: str = MINIMAL_CONFIG) -> Path:
    config_path = path / "train.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_a_configuration_file_is_read_into_nested_dataclasses(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(_write(tmp_path))

    assert config.fold == 1
    assert config.image_size == 128
    assert config.encoder == "resnet18"
    assert config.augmentation.horizontal_flip is False
    assert config.augmentation.vertical_flip is True
    assert config.loss.dice_weight == 0.5


def test_unset_fields_keep_their_defaults(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(_write(tmp_path))

    assert config.batch_size == TrainConfig().batch_size
    assert config.seed == TrainConfig().seed


def test_command_line_overrides_win_over_the_file(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(_write(tmp_path), fold=4, epochs=2)

    assert config.fold == 4
    assert config.epochs == 2


def test_an_unset_override_changes_nothing(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(_write(tmp_path), fold=None)

    assert config.fold == 1


def test_a_misspelled_field_is_rejected(tmp_path: Path) -> None:
    """A silently ignored typo would make the run report the wrong settings."""
    config_path = _write(tmp_path, "learning_rat: 0.1\n")

    with pytest.raises(KeyError, match="learning_rat"):
        TrainConfig.from_yaml(config_path)


def test_the_resolved_configuration_can_be_written_and_read_back(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(_write(tmp_path))

    saved = config.save(tmp_path / "resolved.yaml")

    restored = TrainConfig.from_dict(yaml.safe_load(saved.read_text(encoding="utf-8")))
    assert restored == config


def test_the_committed_phase_1_configuration_is_valid() -> None:
    config = TrainConfig.from_yaml(Path("configs/phase1_unet.yaml"))

    assert config.image_size == 1024
    assert config.epochs > 0
    assert config.encoder_weights == "imagenet"


def test_seeding_makes_torch_reproducible() -> None:
    seed_everything(7)
    first = torch.rand(4)
    seed_everything(7)

    assert torch.equal(first, torch.rand(4))


def test_the_device_can_be_forced() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_the_default_device_is_a_gpu_only_when_one_exists() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert resolve_device().type == expected


@pytest.mark.dataset
def test_a_short_run_writes_a_checkpoint_that_can_be_reloaded(
    paths: ProjectPaths, tmp_path: Path
) -> None:
    """One tiny epoch on the CPU, to prove the loop and checkpoint round-trip."""
    config = replace(
        TrainConfig.from_yaml(Path("configs/phase1_unet.yaml")),
        epochs=1,
        image_size=128,
        batch_size=2,
        encoder_weights=None,
        max_train_batches=2,
        max_val_batches=1,
        amp=False,
        num_workers=0,
        output_dir=tmp_path / "run",
    )

    result = train(config, paths=paths, device="cpu")

    assert result.checkpoint == tmp_path / "run" / CHECKPOINT_NAME
    assert result.checkpoint.exists()
    assert len(result.history) == 1
    assert result.best_epoch == 1

    model, stored = load_checkpoint(result.checkpoint, device="cpu")
    assert stored["image_size"] == 128
    with torch.no_grad():
        logits = model(torch.zeros(1, 2, 128, 128))
    assert logits.shape == (1, 1, 128, 128)
