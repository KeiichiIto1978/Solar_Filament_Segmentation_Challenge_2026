"""Resolution of the filesystem paths used across the project.

Paths are declared in ``configs/paths.yaml`` rather than in code, so that the
source tree stays free of machine-specific absolute paths.  The dataset lives
outside version control (it is redistributed under CC BY-NC 4.0), so its
location is also overridable through the ``MAGFILO_ROOT`` environment
variable, which takes precedence over the configuration file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ENV_DATASET_ROOT = "MAGFILO_ROOT"

# paths.py -> filament -> src -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "paths.yaml"


@dataclass(frozen=True)
class ProjectPaths:
    """Absolute paths to the dataset and to project-owned directories."""

    dataset_root: Path
    train_images: Path
    train_annotations: Path
    test_images: Path
    splits_dir: Path
    image_height: int
    image_width: int

    @property
    def image_shape(self) -> tuple[int, int]:
        """Shape of a full frame as ``(height, width)``."""
        return (self.image_height, self.image_width)

    def require_dataset(self) -> ProjectPaths:
        """Fail early, and with an actionable message, if the dataset is absent.

        Returns ``self`` so that the check can be chained onto ``load_paths()``.
        """
        missing = [
            path
            for path in (self.train_images, self.train_annotations, self.test_images)
            if not path.exists()
        ]
        if missing:
            listed = "\n".join(f"  {path}" for path in missing)
            raise FileNotFoundError(
                "The MAGFiLO dataset was not found. The following paths do not exist:\n"
                f"{listed}\n"
                "Download the competition data from "
                "https://www.kaggle.com/competitions/filament-segmentation-2026/data "
                f"and either place it at {self.dataset_root} or point the "
                f"{ENV_DATASET_ROOT} environment variable at your copy."
            )
        return self


def _resolve(base: Path, value: str) -> Path:
    """Interpret ``value`` as an absolute path, or as one relative to ``base``."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def _section(config: Mapping[str, Any], name: str, config_path: Path) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise KeyError(f"Section '{name}' is missing from {config_path}")
    return section


def load_paths(
    config_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> ProjectPaths:
    """Read ``configs/paths.yaml`` and resolve every entry to an absolute path.

    Args:
        config_path: Configuration file to read. Defaults to ``configs/paths.yaml``
            next to the repository root.
        env: Environment mapping to read ``MAGFILO_ROOT`` from. Defaults to
            ``os.environ``. Exposed for testing.

    Returns:
        The resolved paths. Their existence is *not* checked; call
        :meth:`ProjectPaths.require_dataset` when the data is actually needed.
    """
    config_file = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Path configuration not found: {config_file}")

    with config_file.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    environ = os.environ if env is None else env
    dataset = _section(config, "dataset", config_file)
    project = _section(config, "project", config_file)
    image = _section(config, "image", config_file)

    # The environment variable wins over the file so that a checked-in default
    # keeps working on a machine that stores the dataset elsewhere.
    override = environ.get(ENV_DATASET_ROOT)
    root_value = override if override else str(dataset["root"])
    dataset_root = _resolve(REPO_ROOT, root_value)

    return ProjectPaths(
        dataset_root=dataset_root,
        train_images=_resolve(dataset_root, str(dataset["train_images"])),
        train_annotations=_resolve(dataset_root, str(dataset["train_annotations"])),
        test_images=_resolve(dataset_root, str(dataset["test_images"])),
        splits_dir=_resolve(REPO_ROOT, str(project["splits_dir"])),
        image_height=int(image["height"]),
        image_width=int(image["width"]),
    )
