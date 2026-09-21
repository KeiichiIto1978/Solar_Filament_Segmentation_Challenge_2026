"""Score a trained checkpoint on one fold's validation split.

    uv run python scripts/evaluate.py --checkpoint outputs/phase1_unet/best.pt --fold 0

Prints PQ with its breakdown, the fusion and splitting counts, and the
inference time, and writes the same as JSON next to the checkpoint. The
ground-truth encoding is cached, because it takes as long as the inference and
does not depend on the model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from filament.data.coco import load_annotations
from filament.data.split import load_fold
from filament.evaluation import evaluate
from filament.paths import load_paths
from filament.postprocess.instances import DEFAULT_MIN_AREA, DEFAULT_THRESHOLD
from filament.submit.rle import masks_to_gt_df, read_submission, write_submission
from filament.training.loop import load_checkpoint

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained weights.")
    parser.add_argument("--fold", type=int, default=None, help="Fold to validate on.")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Probability cut-off."
    )
    parser.add_argument(
        "--min-area", type=int, default=DEFAULT_MIN_AREA, help="Smallest region to emit."
    )
    parser.add_argument("--device", type=str, default=None, help="Force cpu or cuda.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N stems.")
    parser.add_argument(
        "--no-fusion-counts",
        action="store_true",
        help="Skip counting fused and split filaments.",
    )
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=None,
        help="Where to keep the encoded ground truth. Defaults to beside the checkpoint.",
    )
    parser.add_argument(
        "--save-predictions", type=Path, default=None, help="Write the predictions as CSV."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)

    paths = load_paths().require_dataset()
    model, stored = load_checkpoint(args.checkpoint, device=args.device)
    fold_index = args.fold if args.fold is not None else int(stored["fold"])
    size = int(stored["image_size"])

    dataset = load_annotations(paths.train_annotations)
    stems = load_fold(fold_index, paths.splits_dir).val
    if args.limit is not None:
        stems = stems[: args.limit]

    cache = args.gt_cache or args.checkpoint.parent / f"gt_fold{fold_index}_{len(stems)}.csv"
    if cache.exists():
        logger.info("Reusing the encoded ground truth at %s.", cache)
        gt_df = read_submission(cache)
    else:
        logger.info("Encoding the ground truth for %d stems.", len(stems))
        gt_df = masks_to_gt_df(dataset, stems)
        write_submission(gt_df, cache)

    device = args.device or ("cuda" if _cuda_available() else "cpu")
    result, predictions = evaluate(
        model,
        dataset,
        paths.train_images,
        stems,
        size=size,
        threshold=args.threshold,
        min_area=args.min_area,
        device=device,
        gt_df=gt_df,
        count_fusion=not args.no_fusion_counts,
    )

    summary = result.to_dict() | {
        "checkpoint": str(args.checkpoint),
        "fold": fold_index,
    }
    destination = args.checkpoint.parent / f"eval_fold{fold_index}.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_predictions is not None:
        write_submission(predictions, args.save_predictions)
        print(f"Predictions: {args.save_predictions}")

    print(json.dumps(summary, indent=2))
    print(f"Summary: {destination}")
    return 0


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


if __name__ == "__main__":
    raise SystemExit(main())
