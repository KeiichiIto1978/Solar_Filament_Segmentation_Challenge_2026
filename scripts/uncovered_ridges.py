"""Is the stretch of a filament the model missed visible in the frame at all?

    uv run python scripts/uncovered_ridges.py --maps maps/fold0.npz --fold 0 \
        --setting setting.json --misses misses/fold0.csv --output ridges/fold0.csv

For every annotated filament the configuration found only in part (Phase 12's
"too small") and, for reference, a sample of those it matched, the filament is
split into the part some emitted prediction covers and the part none does, and
a ring of background around it is taken. The dark-ridge response of the frame
(``filament.data.ridges``) is summarised over each, at the full 2048 and at the
1024 the network sees.

If the uncovered part responds like the covered part and unlike the background,
the frame shows the rest of the filament and the network does not pick it up.
If it responds like the background, the frame at that resolution does not show
it, and no model reading the same frame can be expected to.
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
from filament.data.ridges import dark_ridge_response
from filament.data.split import load_fold
from filament.paths import load_paths
from filament.postprocess.search import load_map_bundle, resampled_builder
from filament.submit.rle import FULL_HEIGHT

logger = logging.getLogger(__name__)

# Background: a ring this far around the filament, outside every annotator's
# filaments, on the disk.
RING = 25
# Filaments the configuration matched, sampled per fold as a reference.
MATCHED_SAMPLE = 200
SIGMAS_FULL = (1.0, 2.0, 3.0, 4.0, 6.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=Path, required=True, help="Map bundle of one fold.")
    parser.add_argument("--fold", type=int, required=True, help="The fold.")
    parser.add_argument("--setting", type=Path, required=True, help="JSON post-processing.")
    parser.add_argument("--misses", type=Path, required=True, help="miss_breakdown.py output.")
    parser.add_argument("--output", type=Path, required=True, help="CSV to write.")
    return parser.parse_args(argv)


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_n": 0, f"{prefix}_median": np.nan, f"{prefix}_p90": np.nan}
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)
    started = time.perf_counter()
    paths = load_paths().require_dataset()
    maps = load_map_bundle(args.maps)
    setting = json.loads(args.setting.read_text()) | {"output_size": FULL_HEIGHT}
    misses = pd.read_csv(args.misses)
    held_out = set(load_fold(args.fold, paths.splits_dir).val)
    if not set(maps) <= held_out:
        raise SystemExit(f"Some maps in {args.maps} are not held out of fold {args.fold}.")
    if set(misses["fold"]) != {args.fold} or not set(misses["stem"]) <= held_out:
        raise SystemExit(f"{args.misses} is not the miss breakdown of fold {args.fold}.")
    if set(misses["stem"]) != set(maps):
        raise SystemExit(
            f"{args.misses} and {args.maps} cover different frames; "
            "the miss breakdown has to be made from the same maps."
        )
    partly = misses[misses["outcome"] == "too_small"]
    matched = misses[misses["outcome"] == "matched"].sample(
        n=min(MATCHED_SAMPLE, int((misses["outcome"] == "matched").sum())), random_state=0
    )
    chosen = pd.concat([partly, matched])
    by_stem = load_annotations(paths.train_annotations).by_stem()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING + 1,) * 2)

    records = []
    for stem, group in chosen.groupby("stem"):
        frame = load_grayscale(paths.train_images / f"{stem}.jpeg")
        disk = detect_disk(frame)
        probability = maps[stem]
        candidates = resampled_builder(
            probability, disk.scaled(probability.shape[0] / FULL_SIZE), setting
        )
        covered = np.zeros((FULL_HEIGHT, FULL_HEIGHT), dtype=bool)
        for candidate in candidates:
            covered |= candidate.mask
        full = dark_ridge_response(frame, SIGMAS_FULL)
        # What the network is given: the frame halved, the same widths at half
        # the scale, and the response brought back to full size to be read.
        half = cv2.resize(frame, (FULL_SIZE // 2,) * 2, interpolation=cv2.INTER_AREA)
        reduced = cv2.resize(
            dark_ridge_response(half, [sigma / 2 for sigma in SIGMAS_FULL]),
            (FULL_SIZE, FULL_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        rows, columns = np.mgrid[:FULL_HEIGHT, :FULL_HEIGHT]
        on_disk = np.hypot(columns - disk.center_x, rows - disk.center_y) <= 0.98 * disk.radius
        entries = {entry.image_id: entry for entry in by_stem[stem]}
        anyone = np.zeros_like(covered)
        for entry in entries.values():
            for annotation in entry.annotations:
                anyone |= annotation.to_mask(entry.height, entry.width).astype(bool)

        for _, row in group.iterrows():
            entry = entries[row["annotator_image"]]
            mask = (
                entry.annotations[int(row["filament"]) - 1]
                .to_mask(entry.height, entry.width)
                .astype(bool)
            )
            ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~anyone & on_disk
            regions = {"covered": mask & covered, "uncovered": mask & ~covered, "ring": ring}
            record = {
                "fold": args.fold,
                "stem": stem,
                "annotator_image": row["annotator_image"],
                "filament": int(row["filament"]),
                "outcome": row["outcome"],
                "area": int(mask.sum()),
            }
            for resolution, response in (("2048", full), ("1024", reduced)):
                for name, region in regions.items():
                    record |= _summary(response[region], f"{resolution}_{name}")
            records.append(record)

    table = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info(
        "Fold %d: %d filaments (%d partly covered) in %.0fs.",
        args.fold,
        len(table),
        int((table["outcome"] == "too_small").sum()),
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
