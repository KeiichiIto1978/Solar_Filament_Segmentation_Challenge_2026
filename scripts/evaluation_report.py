"""The quantitative evaluation of the final configuration, over the five folds.

    uv run python scripts/evaluation_report.py --maps <dir with fold0.npz..fold4.npz> \
        --output docs/figures

Reads each fold model's eight-view maps of its held-out frames (as written by
the cross-validation notebooks), applies ``filament.pipeline.FINAL_SETTING``
and reports what the organisers evaluate on, pooled over the 707 frames:

- PQ with SQ, RQ, TP, FP and FN;
- the IoU and Dice of every matched filament, as distributions;
- the Dice of the whole frame, every predicted pixel against every pixel the
  annotator drew, per annotator-image;
- one-to-many and many-to-one relations: annotated filaments covered by more
  than one prediction (split), and predictions covering more than one
  annotated filament (fused).

Writes ``evaluation.json`` and two figures into ``--output``. No GPU is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from filament.data.coco import load_annotations
from filament.data.disk import detect_disk
from filament.data.image import FULL_SIZE, load_grayscale
from filament.data.split import load_fold
from filament.evaluation import COUNT_SIZE, annotated_masks, shrink
from filament.metrics.pq import PQResult, compute_pq
from filament.paths import load_paths
from filament.pipeline import FINAL_SETTING, FOLDS
from filament.postprocess.instances import fusion_and_splitting, instances_to_rows
from filament.postprocess.search import load_map_bundle, resampled_builder
from filament.submit.rle import FULL_HEIGHT, SUBMISSION_COLUMNS, masks_to_gt_df

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--maps", type=Path, required=True, help="Directory with fold0.npz to fold4.npz."
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory to write into.")
    return parser.parse_args(argv)


def frame_dice(predicted: np.ndarray, annotated: np.ndarray) -> float:
    """Dice of two boolean masks; 1 when both are empty, as nothing was there to find."""
    total = int(predicted.sum()) + int(annotated.sum())
    if total == 0:
        return 1.0
    return 2.0 * int((predicted & annotated).sum()) / total


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)
    plt.switch_backend("Agg")
    started = time.perf_counter()
    paths = load_paths().require_dataset()
    dataset = load_annotations(paths.train_annotations)
    setting = dict(FINAL_SETTING) | {"output_size": FULL_HEIGHT}

    scores: list[PQResult] = []
    frame_dices: list[float] = []
    fused = split = annotated_total = predicted_total = 0
    # Masks are compared at COUNT_SIZE, as the fusion counts are.
    empty = np.zeros((COUNT_SIZE, COUNT_SIZE), dtype=bool)
    for fold in FOLDS:
        maps = load_map_bundle(args.maps / f"fold{fold}.npz")
        held_out = set(load_fold(fold, paths.splits_dir).val)
        if set(maps) != held_out:
            raise SystemExit(
                f"fold{fold}.npz covers {len(maps)} frames, but fold {fold} holds out "
                f"{len(held_out)}; the maps must be exactly the held-out frames."
            )
        rows: list[tuple[str, str]] = []
        predicted: dict[str, list[np.ndarray]] = {}
        for stem in sorted(maps):
            probability = maps[stem]
            disk = detect_disk(load_grayscale(paths.train_images / f"{stem}.jpeg"))
            instances = resampled_builder(
                probability, disk.scaled(probability.shape[0] / FULL_SIZE), setting
            )
            rows.extend(instances_to_rows(stem, instances))
            predicted[stem] = [shrink(instance.mask) for instance in instances]
        result = compute_pq(
            masks_to_gt_df(dataset, sorted(maps)),
            pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS)),
        )
        scores.append(result)
        predicted_total += len(rows)
        for entry in dataset.subset(set(maps)).annotator_images:
            truth = annotated_masks(entry)
            entry_fused, entry_split = fusion_and_splitting(predicted[entry.stem], truth)
            fused += entry_fused
            split += entry_split
            annotated_total += len(truth)
            frame_dices.append(
                frame_dice(
                    np.logical_or.reduce([empty, *predicted[entry.stem]]),
                    np.logical_or.reduce([empty, *truth]),
                )
            )
        logger.info("Fold %d: %s", fold, result)

    ious = [value for result in scores for value in result.iou_scores]
    dices = [value for result in scores for value in result.dice_scores]
    tp = sum(result.tp for result in scores)
    fp = sum(result.fp for result in scores)
    fn = sum(result.fn for result in scores)
    sq = float(np.mean(ious))
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    summary = {
        "setting": dict(FINAL_SETTING),
        "pq": sq * rq,
        "sq": sq,
        "rq": rq,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_fold_pq": [result.pq for result in scores],
        "matched_iou": {"mean": sq, "median": float(np.median(ious))},
        "matched_dice": {"mean": float(np.mean(dices)), "median": float(np.median(dices))},
        "frame_dice": {
            "mean": float(np.mean(frame_dices)),
            "median": float(np.median(frame_dices)),
            "annotator_images": len(frame_dices),
        },
        "annotated_filaments": annotated_total,
        "predicted_filaments": predicted_total,
        "split_one_to_many": split,
        "fused_many_to_one": fused,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    panels = (
        (axes[0], ious, "IoU of matched filaments", np.linspace(0.5, 1.0, 26)),
        (axes[1], dices, "Dice of matched filaments", np.linspace(2 / 3, 1.0, 26)),
        (axes[2], frame_dices, "Dice of the whole frame", np.linspace(0.0, 1.0, 26)),
    )
    for axis, values, title, bins in panels:
        axis.hist(values, bins=bins, color="#4C72B0")
        axis.axvline(float(np.mean(values)), color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{title}\nmean {np.mean(values):.3f}, n = {len(values)}")
    axes[0].set_ylabel("count")
    plt.tight_layout()
    figure.savefig(args.output / "distributions.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    labels = [
        "matched\n(TP)",
        "missed\n(FN)",
        "spurious\n(FP)",
        "split\n(1 to many)",
        "fused\n(many to 1)",
    ]
    axis.bar(
        labels,
        [tp, fn, fp, split, fused],
        color=["#55A868", "#C44E52", "#DD8452", "#8172B3", "#937860"],
    )
    axis.set_ylabel("filaments")
    axis.set_title(f"Five folds, {annotated_total} annotated filaments")
    plt.tight_layout()
    figure.savefig(args.output / "outcomes.png", dpi=150)
    plt.close(figure)

    logger.info(
        "Pooled PQ %.4f (SQ %.4f, RQ %.4f, TP %d, FP %d, FN %d); split %d, fused %d; %.0fs.",
        summary["pq"],
        sq,
        rq,
        tp,
        fp,
        fn,
        split,
        fused,
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
