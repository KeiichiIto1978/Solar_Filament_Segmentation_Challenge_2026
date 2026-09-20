"""Tests for path resolution and for the MAGFILO_ROOT override."""

from __future__ import annotations

from pathlib import Path

import pytest

from filament.paths import DEFAULT_CONFIG_PATH, REPO_ROOT, load_paths

CONFIG_TEMPLATE = """
dataset:
  root: {root}
  train_images: train/train_images
  train_annotations: train/annotations.json
  test_images: test/test_images
project:
  splits_dir: configs/splits
image:
  height: 2048
  width: 2048
"""


def _write_config(directory: Path, root: str) -> Path:
    config_path = directory / "paths.yaml"
    config_path.write_text(CONFIG_TEMPLATE.format(root=root), encoding="utf-8")
    return config_path


def test_relative_root_is_resolved_against_the_repository_root(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "some_dataset")
    paths = load_paths(config_path, env={})

    assert paths.dataset_root == REPO_ROOT / "some_dataset"
    assert paths.train_images == REPO_ROOT / "some_dataset" / "train" / "train_images"
    assert paths.splits_dir == REPO_ROOT / "configs" / "splits"


def test_environment_variable_overrides_the_configured_root(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "some_dataset")
    elsewhere = tmp_path / "elsewhere"

    paths = load_paths(config_path, env={"MAGFILO_ROOT": str(elsewhere)})

    assert paths.dataset_root == elsewhere
    assert paths.test_images == elsewhere / "test" / "test_images"


def test_absolute_root_in_the_configuration_is_kept_as_is(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, str(tmp_path / "dataset"))

    paths = load_paths(config_path, env={})

    assert paths.dataset_root == tmp_path / "dataset"


def test_require_dataset_names_the_missing_paths(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, str(tmp_path / "absent"))
    paths = load_paths(config_path, env={})

    with pytest.raises(FileNotFoundError, match="train_images"):
        paths.require_dataset()


def test_checked_in_configuration_declares_the_full_frame_size() -> None:
    paths = load_paths(DEFAULT_CONFIG_PATH, env={})

    assert paths.image_shape == (2048, 2048)
