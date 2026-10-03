"""Sort every annotated filament of one fold by whether, and why, it was missed.

    uv run python scripts/miss_breakdown.py --maps maps/fold0.npz --fold 0 \
        --setting setting.json --output misses/fold0.csv

The setting is the post-processing as submitted, cut-offs included. Candidates
are produced once with the area and score cut-offs switched off; since those
act on each region alone, the submitted predictions are the candidates that
pass both, and every annotated filament can be traced to the stage that lost
it (``filament.metrics.misses``). One row per annotated filament and
annotator, which is the unit PQ counts false negatives in.
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
import pycocotools.mask as mask_utils

from filament.data.coco import load_annotations
from filament.data.disk import detect_disk
from filament.data.image import FULL_SIZE, load_grayscale
from filament.data.split import load_fold
from filament.metrics.misses import BY_AREA, BY_SCORE, KEPT, classify_misses
from filament.paths import load_paths
from filament.postprocess.search import load_map_bundle, resampled_builder
from filament.submit.rle import FULL_HEIGHT, FULL_WIDTH, mask_to_rle

logger = logging.getLogger(__name__)

# Another annotator "drew the same filament" when one of theirs overlaps it at
# least this much. Lower than a match: tracings of one filament by two people
# differ more than a prediction is allowed to differ from either.
SAME_FILAMENT_IOU = 0.3


def _encode(counts: list[str]) -> list[dict[str, object]]:
    return [{"size": [FULL_HEIGHT, FULL_WIDTH], "counts": item.encode("ascii")} for item in counts]


def _iou(first: list[str], second: list[str]) -> np.ndarray:
    if not first or not second:
        return np.zeros((len(first), len(second)))
    return np.asarray(
        mask_utils.iou(_encode(first), _encode(second), [0] * len(second)), dtype=float
    ).reshape(len(first), len(second))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=Path, required=True, help="Map bundle (.npz) of one fold.")
    parser.add_argument("--fold", type=int, required=True, help="Fold the maps were held out of.")
    parser.add_argument(
        "--setting", type=Path, required=True, help="JSON post-processing as submitted."
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)
    started = time.perf_counter()

    paths = load_paths().require_dataset()
    maps = load_map_bundle(args.maps)
    if not set(maps) <= set(load_fold(args.fold, paths.splits_dir).val):
        raise SystemExit(f"Some maps in {args.maps} are not held out of fold {args.fold}.")
    setting = json.loads(args.setting.read_text())
    min_area = int(setting["min_area"])
    min_score = float(setting.get("min_score", 0.0))
    threshold = float(setting["threshold"])
    uncut = setting | {"min_area": 0, "min_score": 0.0, "output_size": FULL_HEIGHT}
    by_stem = load_annotations(paths.train_annotations).by_stem()

    records: list[dict[str, object]] = []
    for stem in sorted(maps):
        probability = maps[stem]
        frame = load_grayscale(paths.train_images / f"{stem}.jpeg")
        disk = detect_disk(frame).scaled(probability.shape[0] / FULL_SIZE)
        candidates = resampled_builder(probability, disk, uncut)
        candidate_rles = [mask_to_rle(candidate.mask) for candidate in candidates]
        candidate_areas = np.array([candidate.area for candidate in candidates], dtype=float)
        dropped_by = [
            BY_AREA
            if candidate.area < min_area
            else BY_SCORE
            if candidate.score < min_score
            else KEPT
            for candidate in candidates
        ]
        full = cv2.resize(probability, (FULL_HEIGHT, FULL_HEIGHT), interpolation=cv2.INTER_LINEAR)

        entries = by_stem.get(stem, [])
        tracings = []
        for entry in entries:
            masks = [
                annotation.to_mask(entry.height, entry.width) for annotation in entry.annotations
            ]
            tracings.append(
                (
                    entry,
                    [mask_to_rle(mask) for mask in masks],
                    np.array([mask.sum() for mask in masks], dtype=float),
                    np.array([full[mask].max() if mask.any() else 0.0 for mask in masks]),
                )
            )
        for position, (entry, rles, areas, peaks) in enumerate(tracings):
            if not rles:
                continue
            iou = _iou(rles, candidate_rles)
            candidate_sizes = candidate_areas[None, :] if len(candidate_rles) else np.zeros((1, 0))
            # The overlap in pixels follows from IoU and the two areas.
            overlap = iou * (areas[:, None] + candidate_sizes) / (1.0 + iou)
            outcomes = classify_misses(
                iou, overlap, areas, candidate_areas, dropped_by, peaks, threshold
            )
            others = [
                rle for index, item in enumerate(tracings) if index != position for rle in item[1]
            ]
            drawn_by_others = (
                _iou(rles, others).max(axis=1) >= SAME_FILAMENT_IOU if others else None
            )
            kept = np.array([reason == KEPT for reason in dropped_by], dtype=bool)
            # The candidate overlapping each filament most, and how the two sit:
            # coverage is the share of the filament it covers, purity the share
            # of itself that lies inside the filament.
            if candidate_rles:
                best = overlap.argmax(axis=1)
                best_overlap = overlap[np.arange(len(rles)), best]
                coverage = best_overlap / areas
                purity = np.where(best_overlap > 0, best_overlap / candidate_areas[best], 0.0)
            else:
                coverage = purity = np.zeros(len(rles))
            for index, outcome in enumerate(outcomes):
                records.append(
                    {
                        "fold": args.fold,
                        "stem": stem,
                        "annotator_image": entry.image_id,
                        "filament": index + 1,
                        "area": float(areas[index]),
                        "peak": float(peaks[index]),
                        "best_iou_emitted": float(iou[index, kept].max()) if kept.any() else 0.0,
                        "coverage": float(coverage[index]),
                        "purity": float(purity[index]),
                        "outcome": outcome,
                        "annotators": len(tracings),
                        "drawn_by_others": None
                        if drawn_by_others is None
                        else bool(drawn_by_others[index]),
                    }
                )

    table = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info(
        "Fold %d: %d filaments, %d matched, %d missed, in %.0fs.",
        args.fold,
        len(table),
        int((table["outcome"] == "matched").sum()),
        int((table["outcome"] != "matched").sum()),
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
