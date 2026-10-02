"""Sweep post-processing settings over one fold's stored probability maps.

    uv run python scripts/sweep_maps.py --maps maps/fold0.npz --fold 0 \
        --settings settings.json --output sweep/stage1_fold0.json

The maps are a bundle written by ``save_map_bundle``; the settings a JSON list
of parameter dictionaries, each scored through ``resampled_builder``, so a
``resolution`` key post-processes at that resolution. The result is written as
JSON records that ``SweepResult.from_records`` reads back.

Run as its own process on purpose. A notebook that has used the GPU cannot
safely fork workers -- Phase 9's first attempt hung for ten hours that way -- so
the notebook starts one fresh interpreter per fold instead. Every setting is
logged as it finishes, which is what shows the sweep is alive.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from filament.data.coco import load_annotations
from filament.data.disk import detect_disk
from filament.data.image import FULL_SIZE, load_grayscale
from filament.data.split import load_fold
from filament.paths import load_paths
from filament.postprocess.search import Setting, load_map_bundle, resampled_builder, sweep
from filament.submit.rle import masks_to_gt_df

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=Path, required=True, help="Map bundle (.npz) of one fold.")
    parser.add_argument("--fold", type=int, required=True, help="Fold the maps were held out of.")
    parser.add_argument(
        "--settings", type=Path, required=True, help="JSON list of parameter dictionaries."
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the result.")
    parser.add_argument(
        "--splits", type=Path, default=None, help="Split directory. Defaults to the configured one."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)
    started = time.perf_counter()

    paths = load_paths().require_dataset()
    maps = load_map_bundle(args.maps)
    held_out = set(load_fold(args.fold, args.splits or paths.splits_dir).val)
    stems = sorted(maps)
    strangers = set(stems) - held_out
    if strangers:
        raise SystemExit(
            f"{len(strangers)} maps in {args.maps} are not held out of fold {args.fold}, "
            f"for example {sorted(strangers)[0]}. Scoring them would score frames the model saw."
        )
    if len(stems) < len(held_out):
        logger.warning(
            "Only %d of the %d frames held out of fold %d have a map; the score covers those only.",
            len(stems),
            len(held_out),
            args.fold,
        )

    settings = [Setting(dict(values)) for values in json.loads(args.settings.read_text())]
    size = next(iter(maps.values())).shape[0]
    disks = {
        stem: detect_disk(load_grayscale(paths.train_images / f"{stem}.jpeg")).scaled(
            size / FULL_SIZE
        )
        for stem in stems
    }
    truth = masks_to_gt_df(load_annotations(paths.train_annotations), stems)
    logger.info(
        "Fold %d: %d maps at %d, %d settings, prepared in %.0fs.",
        args.fold,
        len(maps),
        size,
        len(settings),
        time.perf_counter() - started,
    )

    result = sweep(maps, truth, settings, disks=disks, build=resampled_builder)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "fold": args.fold,
                "maps": str(args.maps),
                "frames": len(stems),
                "settings": [setting.values for setting in settings],
                "points": result.to_records(),
                "seconds": round(time.perf_counter() - started, 1),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Fold %d: wrote %s in %.0fs.", args.fold, args.output, time.perf_counter() - started
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
