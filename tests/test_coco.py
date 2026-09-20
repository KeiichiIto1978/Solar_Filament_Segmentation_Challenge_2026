"""Tests for the MAGFiLO annotation loader.

The expectations on the real file are the counts published on the competition
data page, so a silently changed or truncated download is caught here rather
than halfway through an experiment.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from filament.data.coco import (
    OBSERVATORY_CODES,
    Dataset,
    load_annotations,
    observatory_code,
    polygon_to_mask,
    split_image_id,
)
from filament.paths import ProjectPaths


@pytest.fixture(scope="module")
def dataset(paths: ProjectPaths) -> Dataset:
    return load_annotations(paths.train_annotations)


def test_split_image_id_separates_annotator_from_stem() -> None:
    assert split_image_id("040301-20140609195854Bh") == ("040301", "20140609195854Bh")


def test_split_image_id_rejects_an_id_without_an_annotator_prefix() -> None:
    with pytest.raises(ValueError, match="Malformed image_id"):
        split_image_id("20140609195854Bh")


def test_observatory_code_reads_the_station_suffix() -> None:
    assert observatory_code("20140609195854Bh") == "Bh"


def test_observatory_code_rejects_an_unknown_station() -> None:
    with pytest.raises(ValueError, match="Unknown observatory code"):
        observatory_code("20140609195854Zz")


def test_polygon_to_mask_rasterizes_an_axis_aligned_square() -> None:
    # A 10x10 square placed at the origin covers exactly 100 pixels.
    mask = polygon_to_mask([[0, 0, 10, 0, 10, 10, 0, 10]], 32, 32)

    assert mask.dtype == bool
    assert mask.shape == (32, 32)
    assert mask.sum() == 100
    assert mask[0, 0] and not mask[10, 10]


def test_polygon_to_mask_of_an_empty_segmentation_is_empty() -> None:
    assert polygon_to_mask([], 16, 16).sum() == 0


def test_load_annotations_rejects_an_annotation_without_its_image(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "Left"}],
                "images": [
                    {
                        "id": "01-20140609195854Bh",
                        "file_name": "20140609195854Bh.jpeg",
                        "height": 2048,
                        "width": 2048,
                    }
                ],
                "annotations": [
                    {
                        "id": "a",
                        "image_id": "99-20990101000000Bh",
                        "category_id": 1,
                        "segmentation": [[0, 0, 1, 0, 1, 1]],
                        "bbox": [0, 0, 1, 1],
                        "area": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="appear in 'annotations' but not in 'images'"):
        load_annotations(broken)


@pytest.mark.dataset
def test_dataset_has_the_published_entry_counts(dataset: Dataset) -> None:
    assert len(dataset.annotator_images) == 1154
    assert len(dataset.stems) == 707
    assert len(dataset.annotations) == 8199


@pytest.mark.dataset
def test_every_image_is_annotated_by_one_to_three_annotators(dataset: Dataset) -> None:
    # 411 images were annotated once, 145 twice and 151 three times.
    per_stem = Counter(len(entries) for entries in dataset.by_stem().values())

    assert per_stem == {1: 411, 2: 145, 3: 151}
    assert len(dataset.single_annotator_stems()) == 411


@pytest.mark.dataset
def test_every_stem_belongs_to_a_known_gong_station(dataset: Dataset) -> None:
    per_station = Counter(observatory_code(stem) for stem in dataset.stems)

    assert set(per_station) == set(OBSERVATORY_CODES)
    assert sum(per_station.values()) == 707


@pytest.mark.dataset
def test_every_annotator_image_holds_at_least_one_filament(dataset: Dataset) -> None:
    assert min(len(entry) for entry in dataset.annotator_images) >= 1


@pytest.mark.dataset
def test_rasterized_polygons_reproduce_the_stored_area(dataset: Dataset) -> None:
    # The COCO 'area' field of this dataset is the rasterized pixel count, not
    # the shoelace area of the polygon, so the agreement is exact.  Relying on
    # that lets later stages use 'area' without rasterizing.
    annotations = dataset.annotations
    sample = np.random.default_rng(0).choice(len(annotations), size=300, replace=False)

    for index in sample:
        annotation = annotations[index]
        mask = polygon_to_mask(annotation.segmentation, 2048, 2048)
        assert mask.sum() == annotation.area, annotation.annotation_id


@pytest.mark.dataset
def test_subset_keeps_every_annotator_of_the_requested_stems(dataset: Dataset) -> None:
    stems = dataset.stems[:20]
    subset = dataset.subset(stems)

    assert subset.stems == stems
    assert len(subset) == sum(len(dataset.by_stem()[stem]) for stem in stems)
