"""Write a submission from stored test-frame probability maps.

    uv run python scripts/submit_from_maps.py --maps maps/test.npz \
        --setting setting.json --output submission.csv

The maps are a bundle written by ``save_map_bundle`` -- for this project, the
mean over the five fold models in eight views. The setting is one JSON
dictionary, applied through ``resampled_builder`` as the sweeps apply it, so a
setting chosen by a sweep reaches the test frames unchanged. The overlap check
runs before the file is reported as written: Kaggle rejects a submission whose
masks share a pixel, and the rejected attempt still counts against the day's
five.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from filament.data.disk import detect_disk
from filament.data.image import FULL_SIZE, load_grayscale
from filament.metrics.overlap import check_no_overlap
from filament.paths import load_paths
from filament.postprocess.search import (
    Setting,
    load_map_bundle,
    predict_from_maps,
    resampled_builder,
)
from filament.submit.rle import write_submission

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--maps", type=Path, required=True, help="Map bundle (.npz) of the test frames."
    )
    parser.add_argument(
        "--setting", type=Path, required=True, help="JSON dictionary of the post-processing."
    )
    parser.add_argument("--output", type=Path, required=True, help="Submission CSV to write.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)

    paths = load_paths().require_dataset()
    maps = load_map_bundle(args.maps)
    expected = {path.stem for path in paths.test_images.glob("*.jpeg")}
    if set(maps) != expected:
        raise SystemExit(
            f"{args.maps} holds {len(maps)} maps, but there are {len(expected)} test frames; "
            "a submission must cover the test frames exactly."
        )
    size = next(iter(maps.values())).shape[0]
    disks = {
        stem: detect_disk(load_grayscale(paths.test_images / f"{stem}.jpeg")).scaled(
            size / FULL_SIZE
        )
        for stem in maps
    }
    setting = Setting(json.loads(args.setting.read_text()))

    predicted = predict_from_maps(maps, setting, disks=disks, build=resampled_builder)
    write_submission(predicted, args.output)
    # Raises if any two masks of a frame share a pixel.
    check_no_overlap(args.output)
    frames = predicted["filament_id"].str.rsplit("_", n=1).str[0].nunique()
    logger.info(
        "%s: %d masks over %d of %d frames, setting %s.",
        args.output,
        len(predicted),
        frames,
        len(maps),
        setting,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
