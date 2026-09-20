"""Tests for the grouped, station-stratified cross-validation split."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from filament.data.coco import OBSERVATORY_CODES, load_annotations, observatory_code
from filament.data.split import (
    DEFAULT_FOLDS,
    DEFAULT_SEED,
    assign_folds,
    build_folds,
    load_fold,
    station_counts,
    write_folds,
)
from filament.paths import ProjectPaths


def _synthetic_stems(per_station: int = 23) -> list[str]:
    """Stems that differ only in their station suffix and running number."""
    return [
        f"2014060919{index:04d}{code}" for code in OBSERVATORY_CODES for index in range(per_station)
    ]


def test_every_stem_is_assigned_to_exactly_one_fold() -> None:
    stems = _synthetic_stems()

    assignment = assign_folds(stems)

    assert set(assignment) == set(stems)
    assert set(assignment.values()) == set(range(DEFAULT_FOLDS))


def test_fold_sizes_differ_by_at_most_one() -> None:
    sizes = Counter(assign_folds(_synthetic_stems()).values())

    assert max(sizes.values()) - min(sizes.values()) <= 1


def test_each_fold_gets_the_same_station_mix() -> None:
    folds = build_folds(_synthetic_stems(per_station=25))

    for fold in folds:
        per_station = station_counts(fold.val)
        assert set(per_station) == set(OBSERVATORY_CODES)
        # 25 stems per station over 5 folds: exactly 5 of each, every time.
        assert set(per_station.values()) == {5}


def test_the_same_seed_reproduces_the_same_split() -> None:
    stems = _synthetic_stems()

    assert assign_folds(stems, seed=7) == assign_folds(stems, seed=7)


def test_a_different_seed_produces_a_different_split() -> None:
    stems = _synthetic_stems()

    assert assign_folds(stems, seed=7) != assign_folds(stems, seed=8)


def test_train_and_validation_stems_are_disjoint() -> None:
    for fold in build_folds(_synthetic_stems()):
        assert set(fold.train).isdisjoint(fold.val)


def test_the_validation_sets_cover_every_stem_exactly_once() -> None:
    stems = _synthetic_stems()
    folds = build_folds(stems)

    validated = [stem for fold in folds for stem in fold.val]

    assert sorted(validated) == sorted(stems)
    assert len(validated) == len(set(validated))


def test_train_is_everything_outside_the_validation_fold() -> None:
    stems = _synthetic_stems()

    for fold in build_folds(stems):
        assert sorted(fold.train + fold.val) == sorted(stems)


def test_a_split_needs_at_least_two_folds() -> None:
    with pytest.raises(ValueError, match="at least 2 folds"):
        assign_folds(_synthetic_stems(), n_folds=1)


def test_splitting_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="No image stems"):
        assign_folds([])


def test_folds_survive_a_write_and_read_round_trip(tmp_path: Path) -> None:
    folds = build_folds(_synthetic_stems())

    write_folds(folds, tmp_path)

    assert load_fold(0, tmp_path) == folds[0]
    assert load_fold(4, tmp_path).val == folds[4].val


def test_writing_over_an_existing_split_is_refused(tmp_path: Path) -> None:
    folds = build_folds(_synthetic_stems())
    write_folds(folds, tmp_path)

    with pytest.raises(FileExistsError, match="frozen on purpose"):
        write_folds(folds, tmp_path)


def test_loading_a_missing_fold_points_at_the_generator(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_splits.py"):
        load_fold(0, tmp_path)


@pytest.mark.dataset
def test_the_committed_split_covers_the_whole_training_set(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    folds = [load_fold(index, paths.splits_dir) for index in range(DEFAULT_FOLDS)]

    validated = [stem for fold in folds for stem in fold.val]

    assert sorted(validated) == dataset.stems
    assert all(fold.seed == DEFAULT_SEED for fold in folds)


@pytest.mark.dataset
def test_no_annotator_of_a_validation_image_appears_in_training(
    paths: ProjectPaths,
) -> None:
    """The point of grouping: all annotators of an image stay on one side."""
    dataset = load_annotations(paths.train_annotations)
    by_stem = dataset.by_stem()

    for index in range(DEFAULT_FOLDS):
        fold = load_fold(index, paths.splits_dir)
        train_ids = {entry.image_id for stem in fold.train for entry in by_stem[stem]}
        val_ids = {entry.image_id for stem in fold.val for entry in by_stem[stem]}
        assert train_ids.isdisjoint(val_ids)


@pytest.mark.dataset
def test_the_committed_split_keeps_the_station_mix_even(paths: ProjectPaths) -> None:
    dataset = load_annotations(paths.train_annotations)
    overall = Counter(observatory_code(stem) for stem in dataset.stems)

    for index in range(DEFAULT_FOLDS):
        fold = load_fold(index, paths.splits_dir)
        per_station = station_counts(fold.val)
        for code, total in overall.items():
            expected = total / DEFAULT_FOLDS
            # Round-robin dealing leaves at most one stem of slack per station.
            assert abs(per_station.get(code, 0) - expected) <= 1
