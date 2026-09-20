"""Tests for the empty-mask submission used as a format rehearsal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from filament.metrics.overlap import check_no_overlap
from filament.paths import ProjectPaths
from filament.submit.rle import read_submission


def _load_script() -> ModuleType:
    """Import scripts/make_dummy_submission.py, which is not a package module."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_dummy_submission.py"
    spec = importlib.util.spec_from_file_location("make_dummy_submission", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script()


def _fake_test_images(directory: Path, stems: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (directory / f"{stem}.jpeg").touch()
    return directory


def test_list_test_stems_drops_the_extension(script: ModuleType, tmp_path: Path) -> None:
    images = _fake_test_images(tmp_path / "test_images", ["20150125172714Mh", "20140609195854Bh"])

    assert script.list_test_stems(images) == ["20140609195854Bh", "20150125172714Mh"]


def test_list_test_stems_ignores_other_files(script: ModuleType, tmp_path: Path) -> None:
    images = _fake_test_images(tmp_path / "test_images", ["20150125172714Mh"])
    (images / "notes.txt").touch()

    assert script.list_test_stems(images) == ["20150125172714Mh"]


def test_an_empty_image_directory_is_reported(script: ModuleType, tmp_path: Path) -> None:
    empty = tmp_path / "test_images"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="No test images found"):
        script.list_test_stems(empty)


def test_the_dummy_submission_has_one_row_per_image_and_no_overlap(
    script: ModuleType, tmp_path: Path
) -> None:
    stems = ["20140609195854Bh", "20150125172714Mh", "20160920230134Lh"]
    images = _fake_test_images(tmp_path / "test_images", stems)
    out = tmp_path / "submission.csv"

    assert script.main(["--test-images", str(images), "--out", str(out)]) == 0

    frame = read_submission(out)
    assert list(frame.columns) == ["filament_id", "segmentation_rle"]
    assert frame["filament_id"].tolist() == [f"{stem}_1" for stem in stems]
    assert '"' not in out.read_text(encoding="utf-8")
    check_no_overlap(out)


@pytest.mark.dataset
def test_the_real_test_set_produces_one_row_per_image(
    script: ModuleType, paths: ProjectPaths, tmp_path: Path
) -> None:
    out = tmp_path / "submission.csv"

    assert script.main(["--out", str(out)]) == 0

    frame = read_submission(out)
    assert len(frame) == 180
    check_no_overlap(out)
