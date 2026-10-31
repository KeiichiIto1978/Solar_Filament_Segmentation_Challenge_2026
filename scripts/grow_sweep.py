"""Score growing the emitted filaments into a second source of support, on one fold.

    uv run python scripts/grow_sweep.py --maps maps/fold0.npz --fold 0 \
        --setting setting.json --support ridge --output grow/fold0.json
    uv run python scripts/grow_sweep.py --maps maps/fold0.npz --fold 0 \
        --setting setting.json --support union_fold0.npz --output grow/union_fold0.json

The filaments are those the setting emits from ``--maps``. Each is grown with
``filament.postprocess.grow.grow_instances`` into pixels where the support
reaches a threshold, connected to it and within a reach, for every pair of the
two on a small grid; the ungrown filaments are scored too, as the reference.

Two kinds of support:

- ``ridge``: the dark-ridge response of the frame (``filament.data.ridges``),
  divided by its median over the frame's own emitted filaments, so that 1 means
  "as ridge-like as what was already found" whatever the frame's contrast;
- a map bundle: another model's probabilities, resized to the full frame.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from filament.data.coco import load_annotations
from filament.data.disk import detect_disk
from filament.data.image import FULL_SIZE, load_grayscale
from filament.data.ridges import dark_ridge_response
from filament.data.split import load_fold
from filament.metrics.pq import compute_pq
from filament.paths import load_paths
from filament.postprocess.grow import grow_instances
from filament.postprocess.instances import instances_to_rows
from filament.postprocess.search import load_map_bundle, resampled_builder
from filament.submit.rle import FULL_HEIGHT, SUBMISSION_COLUMNS, masks_to_gt_df

logger = logging.getLogger(__name__)

RIDGE_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
MAP_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
REACHES = (10.0, 25.0, 50.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--maps", type=Path, required=True, help="Map bundle the filaments come from."
    )
    parser.add_argument("--fold", type=int, required=True, help="The fold.")
    parser.add_argument("--setting", type=Path, required=True, help="JSON post-processing.")
    parser.add_argument(
        "--support", type=str, required=True, help="'ridge', or a map bundle of another model."
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON to write.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)
    started = time.perf_counter()
    paths = load_paths().require_dataset()
    maps = load_map_bundle(args.maps)
    held_out = set(load_fold(args.fold, paths.splits_dir).val)
    strangers = set(maps) - held_out
    if strangers:
        raise SystemExit(
            f"{len(strangers)} maps in {args.maps} are not held out of fold {args.fold}, "
            f"for example {sorted(strangers)[0]}. Scoring them would score frames the model saw."
        )
    if len(maps) < len(held_out):
        logger.warning(
            "Only %d of the %d frames held out of fold %d have a map; the score covers those only.",
            len(maps),
            len(held_out),
            args.fold,
        )
    setting = json.loads(args.setting.read_text()) | {"output_size": FULL_HEIGHT}
    ridge = args.support == "ridge"
    others = None if ridge else load_map_bundle(Path(args.support))
    if others is not None and not set(maps) <= set(others):
        missing = sorted(set(maps) - set(others))
        raise SystemExit(
            f"{args.support} has no map for {len(missing)} of the frames in {args.maps}, "
            f"for example {missing[0]}."
        )
    thresholds = RIDGE_THRESHOLDS if ridge else MAP_THRESHOLDS
    grid = list(itertools.product(thresholds, REACHES))

    rows: dict[str, list[tuple[str, str]]] = {"none": []} | {f"{t}_{r}": [] for t, r in grid}
    for stem in sorted(maps):
        frame = load_grayscale(paths.train_images / f"{stem}.jpeg")
        disk = detect_disk(frame)
        probability = maps[stem]
        instances = resampled_builder(
            probability, disk.scaled(probability.shape[0] / FULL_SIZE), setting
        )
        rows["none"].extend(instances_to_rows(stem, instances))
        if not instances:
            continue
        if ridge:
            response = dark_ridge_response(frame)
            emitted = np.zeros(response.shape, dtype=bool)
            for instance in instances:
                emitted |= instance.mask
            reference = float(np.median(response[emitted]))
            support = response / reference if reference > 0 else np.zeros_like(response)
        else:
            assert others is not None
            support = cv2.resize(
                others[stem], (FULL_HEIGHT, FULL_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
        for threshold, reach in grid:
            grown = grow_instances(instances, support, threshold, reach)
            rows[f"{threshold}_{reach}"].extend(instances_to_rows(stem, grown))

    truth = masks_to_gt_df(load_annotations(paths.train_annotations), sorted(maps))
    results = {}
    for name, entries in rows.items():
        result = compute_pq(truth, pd.DataFrame(entries, columns=list(SUBMISSION_COLUMNS)))
        results[name] = {
            "pq": result.pq,
            "sq": result.sq,
            "rq": result.rq,
            "tp": result.tp,
            "fp": result.fp,
            "fn": result.fn,
        }
        logger.info("%s -> %s", name, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "fold": args.fold,
                "frames": len(maps),
                "held_out": len(held_out),
                "support": args.support,
                "results": results,
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
