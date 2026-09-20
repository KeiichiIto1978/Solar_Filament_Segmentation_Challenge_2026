"""Fixtures shared by the test suite."""

from __future__ import annotations

import pytest

from filament.paths import ProjectPaths, load_paths


@pytest.fixture(scope="session")
def paths() -> ProjectPaths:
    """Resolved project paths, skipping the test when the dataset is absent.

    The dataset cannot be redistributed, so tests that read it are skipped
    rather than failed on a machine that does not have a copy.
    """
    resolved = load_paths()
    try:
        resolved.require_dataset()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return resolved
