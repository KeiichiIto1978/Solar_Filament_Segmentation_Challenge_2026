"""Judge each candidate filament from several features, and keep the likely ones.

Two steps, both on stored probability maps and without a GPU::

    # 1. Candidates, their features and (for a fold) what became of them.
    uv run python scripts/candidate_filter.py features --maps maps/fold0.npz \
        --fold 0 --setting setting.json --output candidates/fold0
    uv run python scripts/candidate_filter.py features --maps maps/test.npz \
        --setting setting.json --output candidates/test

    # 2. Cross-fitted evaluation against a score cut, the decision, and a submission.
    uv run python scripts/candidate_filter.py evaluate --candidates candidates \
        --baseline-score 0.65 --output filter

The classifier for fold k is trained on the other four folds only. Each fold's
candidates come from a model that never saw that fold's frames, so the
features are those the test frames will have, and fold k's score is held out
from both the segmentation model and the classifier. Only the probability cut
on the classifier is chosen on the same folds it is scored on.

Removing a candidate changes no other candidate's outcome, so every filter is
scored exactly from the outcome rows (``pq_from_outcomes``), not by matching
masks again.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from filament.data.coco import load_annotations
from filament.data.disk import detect_disk
from filament.data.image import FULL_SIZE, load_grayscale
from filament.data.split import load_fold
from filament.metrics.overlap import check_no_overlap
from filament.metrics.pq import PQResult, pool_pq, pq_from_outcomes, prediction_outcomes
from filament.paths import load_paths
from filament.postprocess.candidates import FEATURES, frame_features, instance_features
from filament.postprocess.instances import instances_to_rows
from filament.postprocess.search import (
    Setting,
    SweepPoint,
    SweepResult,
    load_map_bundle,
    pool_sweeps,
    resampled_builder,
)
from filament.submit.rle import FULL_HEIGHT, masks_to_gt_df, write_submission

logger = logging.getLogger(__name__)

# The classifier is not tuned: every setting tried on these folds would be one
# more choice made on the data it is scored on.
CLASSIFIER = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "random_state": 0,
}
CUTS = tuple(round(0.05 * step, 2) for step in range(13))  # 0.00 to 0.60
FOLDS = (0, 1, 2, 3, 4)
MIN_POOLED_GAIN = 0.005
MIN_FOLDS_IMPROVED = 4


def build_features(args: argparse.Namespace) -> int:
    """Candidates of one bundle, with features, and outcomes for a fold."""
    started = time.perf_counter()
    paths = load_paths().require_dataset()
    maps = load_map_bundle(args.maps)
    images_dir = paths.train_images if args.fold is not None else paths.test_images
    if args.fold is not None:
        held_out = set(load_fold(args.fold, paths.splits_dir).val)
        if not set(maps) <= held_out:
            raise SystemExit(f"Some maps in {args.maps} are not held out of fold {args.fold}.")
    values = json.loads(args.setting.read_text()) | {"output_size": FULL_HEIGHT}

    records: list[dict[str, object]] = []
    for stem in sorted(maps):
        probability = maps[stem]
        frame = load_grayscale(images_dir / f"{stem}.jpeg")
        disk = detect_disk(frame)
        instances = resampled_builder(
            probability, disk.scaled(probability.shape[0] / FULL_SIZE), values
        )
        # Features are read where the masks are: at the submission resolution.
        full = cv2.resize(probability, (FULL_HEIGHT, FULL_HEIGHT), interpolation=cv2.INTER_LINEAR)
        frame_full = cv2.resize(frame, (FULL_HEIGHT, FULL_HEIGHT), interpolation=cv2.INTER_LINEAR)
        disk_full = disk.scaled(FULL_HEIGHT / frame.shape[0])
        candidates = []
        for (filament_id, counts), instance in zip(
            instances_to_rows(stem, instances), instances, strict=True
        ):
            candidates.append(
                {"image": stem, "filament_id": filament_id, "segmentation_rle": counts}
                | instance_features(instance, full, frame_full, disk_full)
            )
        frame_features(candidates)
        records.extend(candidates)

    table = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "candidates.csv", index=False)
    summary: dict[str, object] = {"frames": len(maps), "candidates": len(table)}
    if args.fold is not None:
        truth = masks_to_gt_df(load_annotations(paths.train_annotations), sorted(maps))
        outcomes = prediction_outcomes(truth, table[["filament_id", "segmentation_rle"]])
        outcomes.to_csv(args.output / "outcomes.csv", index=False)
        summary |= {"fold": args.fold, "ground_truth": len(truth), "rows": len(outcomes)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    logger.info("%s: %s in %.0fs.", args.output, summary, time.perf_counter() - started)
    return 0


def _fold_rows(directory: Path, fold: int) -> tuple[pd.DataFrame, int]:
    """Outcome rows of one fold joined with their candidates' features."""
    candidates = pd.read_csv(directory / "candidates.csv")
    outcomes = pd.read_csv(directory / "outcomes.csv")
    rows = outcomes.merge(candidates.drop(columns="segmentation_rle"), on="filament_id", how="left")
    if rows[list(FEATURES)].isna().any().any():
        raise SystemExit(f"Outcome rows in {directory} without a candidate.")
    ground_truth = int(json.loads((directory / "summary.json").read_text())["ground_truth"])
    rows.insert(0, "fold", fold)
    return rows, ground_truth


def _point(name: str, value: float, result: PQResult, kept: int) -> SweepPoint:
    return SweepPoint(setting=Setting({name: value}), pq=result, predictions=kept, seconds=0.0)


def _as_dict(result: PQResult) -> dict[str, float | int]:
    return {
        "pq": round(result.pq, 4),
        "sq": round(result.sq, 4),
        "rq": round(result.rq, 4),
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
    }


def evaluate(args: argparse.Namespace) -> int:
    """Cross-fitted classifier against the score cut; decision and submission."""
    folds = {fold: _fold_rows(args.candidates / f"fold{fold}", fold) for fold in FOLDS}

    # Out-of-fold probability for every candidate of every fold.
    importances = []
    for fold in FOLDS:
        train = pd.concat([folds[other][0] for other in FOLDS if other != fold])
        model = HistGradientBoostingClassifier(**CLASSIFIER).fit(
            train[list(FEATURES)], train["matched"].astype(int)
        )
        rows = folds[fold][0]
        rows["match_probability"] = model.predict_proba(rows[list(FEATURES)])[:, 1]
        # Permutation importance on the held-out fold: how much ranking skill
        # (area under the ROC curve) is lost when one feature is shuffled.
        result = permutation_importance(
            model,
            rows[list(FEATURES)],
            rows["matched"].astype(int),
            scoring="roc_auc",
            n_repeats=5,
            random_state=0,
        )
        importances.append(result.importances_mean)

    # Every filter as a sweep, per fold, then pooled exactly as before.
    sweeps: dict[int, SweepResult] = {}
    baselines: dict[int, PQResult] = {}
    unfiltered: dict[int, PQResult] = {}
    for fold, (rows, ground_truth) in folds.items():
        points = []
        for cut in CUTS:
            kept = rows[rows["match_probability"] >= cut]
            points.append(
                _point(
                    "cut", cut, pq_from_outcomes(kept, ground_truth), kept["filament_id"].nunique()
                )
            )
        sweeps[fold] = SweepResult(points)
        baselines[fold] = pq_from_outcomes(rows[rows["score"] >= args.baseline_score], ground_truth)
        unfiltered[fold] = pq_from_outcomes(rows, ground_truth)
    pooled = pool_sweeps(sweeps[fold] for fold in FOLDS)
    baseline = pool_pq(baselines.values())
    cut, plateau = pooled.plateau("cut")
    chosen = next(point for point in pooled.points if point.setting.values["cut"] == cut)
    position = [point.setting.values["cut"] for point in pooled.points].index(cut)
    gains = {fold: sweeps[fold].points[position].pq.pq - baselines[fold].pq for fold in FOLDS}
    gain = chosen.pq.pq - baseline.pq
    improved = sum(value > 0 for value in gains.values())
    adopted = gain >= MIN_POOLED_GAIN and improved >= MIN_FOLDS_IMPROVED

    all_rows = pd.concat([folds[fold][0] for fold in FOLDS])
    auc = _auc(all_rows["match_probability"], all_rows["matched"])
    auc_score = _auc(all_rows["score"], all_rows["matched"])

    # The submission: trained on all five folds, applied to the test candidates.
    test = pd.read_csv(args.candidates / "test" / "candidates.csv")
    final = HistGradientBoostingClassifier(**CLASSIFIER).fit(
        all_rows[list(FEATURES)], all_rows["matched"].astype(int)
    )
    test["match_probability"] = final.predict_proba(test[list(FEATURES)])[:, 1]
    # Validation candidates come from one fold model each, test candidates from
    # the mean of five; the quantiles show how far apart that puts them.
    validation = all_rows.drop_duplicates(["fold", "filament_id"])
    shift = {
        name: {
            "validation": np.round(
                validation[name].quantile([0.1, 0.5, 0.9]).to_numpy(), 3
            ).tolist(),
            "test": np.round(test[name].quantile([0.1, 0.5, 0.9]).to_numpy(), 3).tolist(),
        }
        for name in (*FEATURES, "match_probability")
    }

    args.output.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, kept in (
        ("classifier", test[test["match_probability"] >= cut]),
        ("score_cut", test[test["score"] >= args.baseline_score]),
    ):
        # Renumbered so that every frame's ids run from one without gaps.
        kept = kept.copy()
        kept["filament_id"] = (
            kept["image"] + "_" + (kept.groupby("image").cumcount() + 1).astype(str)
        )
        path = args.output / f"submission_{name}.csv"
        write_submission(kept[["filament_id", "segmentation_rle"]], path)
        check_no_overlap(path)
        written[name] = {
            "masks": len(kept),
            "frames_with_predictions": int(kept["image"].nunique()),
        }

    summary = {
        "classifier": CLASSIFIER,
        "features": list(FEATURES),
        "baseline_score": args.baseline_score,
        "unfiltered": _as_dict(pool_pq(unfiltered.values())),
        "baseline": _as_dict(baseline),
        "baseline_per_fold": {fold: round(result.pq, 4) for fold, result in baselines.items()},
        "sweep": [
            point.to_row()
            | {"per_fold": [round(sweeps[fold].points[index].pq.pq, 4) for fold in FOLDS]}
            for index, point in enumerate(pooled.points)
        ],
        "cut": cut,
        "plateau": plateau,
        "chosen": _as_dict(chosen.pq),
        "gain": round(gain, 4),
        "gain_per_fold": {fold: round(value, 4) for fold, value in gains.items()},
        "folds_improved": improved,
        "adopted": adopted,
        "auc": {"classifier_out_of_fold": round(auc, 3), "score": round(auc_score, 3)},
        "importance": dict(
            zip(FEATURES, np.round(np.mean(importances, axis=0), 4).tolist(), strict=True)
        ),
        "shift": shift,
        "submissions": written,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        json.dumps(
            {key: summary[key] for key in summary if key not in ("shift", "sweep")}, indent=1
        )
    )
    for row in summary["sweep"]:
        print(
            f"cut {row['cut']:.2f} PQ {row['pq']:.4f} SQ {row['sq']:.4f} RQ {row['rq']:.4f} "
            f"TP {row['tp']} FP {row['fp']} FN {row['fn']} kept {row['predictions']} "
            f"| {row['per_fold']}"
        )
    for name, values in shift.items():
        print(f"{name:20s} validation {values['validation']}  test {values['test']}")
    return 0


def _auc(scores: pd.Series, labels: pd.Series) -> float:
    """Area under the ROC curve, from ranks."""
    ranks = scores.rank().to_numpy()
    positive = labels.astype(bool).to_numpy()
    count = positive.sum()
    return float((ranks[positive].sum() - count * (count + 1) / 2) / (count * (~positive).sum()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    features = commands.add_parser("features", help="Candidates and features of one map bundle.")
    features.add_argument("--maps", type=Path, required=True, help="Map bundle (.npz).")
    features.add_argument(
        "--fold", type=int, default=None, help="Fold held out; omit for the test frames."
    )
    features.add_argument("--setting", type=Path, required=True, help="JSON post-processing.")
    features.add_argument("--output", type=Path, required=True, help="Directory to write.")
    scoring = commands.add_parser("evaluate", help="Cross-fitted evaluation and submissions.")
    scoring.add_argument(
        "--candidates", type=Path, required=True, help="Directory with fold0..fold4 and test."
    )
    scoring.add_argument(
        "--baseline-score", type=float, required=True, help="The score cut to compare with."
    )
    scoring.add_argument("--output", type=Path, required=True, help="Directory to write.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)
    return build_features(args) if args.command == "features" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
