"""Describe every prediction of one fold and what became of it.

    uv run python scripts/fp_breakdown.py --maps maps/fold0.npz --fold 0 \
        --setting setting.json --output breakdown/fold0.csv

Applies one post-processing setting to a fold's stored probability maps, as
``sweep_maps.py`` does, and writes one row per prediction and annotator-image:
whether it matched, the highest IoU it reached with that annotator's filaments,
and what is known about it before scoring -- its score (the mean probability
over the region), its peak probability, its area and where it sits on the disk.

The rows answer two questions the PQ total cannot. Whether the false positives
are near misses or detections nobody drew decides between improving the masks
and improving the detector; and whether the score separates matches from
misses decides whether a score cut can help.
"""

from __future__ import annotations

import argparse
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
from filament.data.split import load_fold
from filament.metrics.pq import prediction_outcomes
from filament.paths import load_paths
from filament.postprocess.instances import instances_to_rows
from filament.postprocess.search import load_map_bundle, resampled_builder
from filament.submit.rle import FULL_HEIGHT, masks_to_gt_df

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=Path, required=True, help="Map bundle (.npz) of one fold.")
    parser.add_argument("--fold", type=int, required=True, help="Fold the maps were held out of.")
    parser.add_argument(
        "--setting", type=Path, required=True, help="JSON dictionary of one setting."
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the CSV.")
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
    if not set(stems) <= held_out:
        raise SystemExit(
            f"Some maps in {args.maps} are not held out of fold {args.fold}; "
            "scoring them would score frames the model saw."
        )
    values = json.loads(args.setting.read_text()) | {"output_size": FULL_HEIGHT}

    features: list[dict[str, object]] = []
    rows: list[tuple[str, str]] = []
    for stem in stems:
        probability = maps[stem]
        disk = detect_disk(load_grayscale(paths.train_images / f"{stem}.jpeg"))
        map_disk = disk.scaled(probability.shape[0] / FULL_SIZE)
        instances = resampled_builder(probability, map_disk, values)
        # The peak is read from the map at the output resolution, where the
        # masks are; that is the map the builder thresholded when the
        # setting's resolution is the output's.
        full = cv2.resize(probability, (FULL_HEIGHT, FULL_HEIGHT), interpolation=cv2.INTER_LINEAR)
        frame_rows = instances_to_rows(stem, instances)
        for (filament_id, _), instance in zip(frame_rows, instances, strict=True):
            row_index, column_index = np.nonzero(instance.mask)
            centre = np.hypot(column_index.mean() - disk.center_x, row_index.mean() - disk.center_y)
            features.append(
                {
                    "filament_id": filament_id,
                    "score": instance.score,
                    "peak": float(full[instance.mask].max()),
                    "area": instance.area,
                    "radial_position": float(centre / disk.radius),
                }
            )
        rows.extend(frame_rows)

    pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    truth = masks_to_gt_df(load_annotations(paths.train_annotations), stems)
    outcomes = prediction_outcomes(truth, pred_df)
    table = outcomes.merge(pd.DataFrame(features), on="filament_id", how="left")
    table.insert(0, "fold", args.fold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "fold": args.fold,
                "setting": json.loads(args.setting.read_text()),
                "frames": len(stems),
                "predictions": len(pred_df),
                "ground_truth": len(truth),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Fold %d: %d predictions, %d rows, %d matched, %d not, in %.0fs.",
        args.fold,
        len(pred_df),
        len(table),
        int(table["matched"].sum()),
        int((~table["matched"]).sum()),
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
