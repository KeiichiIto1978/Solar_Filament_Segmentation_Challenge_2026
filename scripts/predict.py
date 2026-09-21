"""Predict the test set and write a submission.

    uv run python scripts/predict.py --checkpoint outputs/phase1_unet/best.pt

The overlap check runs on the result before the script exits. Kaggle rejects a
submission whose masks share a pixel and the rejected attempt still counts
against the five allowed per day, so the check is not optional.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from filament.evaluation import predict_frame
from filament.metrics.overlap import check_no_overlap
from filament.paths import load_paths
from filament.postprocess.instances import (
    DEFAULT_MIN_AREA,
    DEFAULT_THRESHOLD,
    instances_to_rows,
)
from filament.submit.rle import write_submission
from filament.training.loop import load_checkpoint

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpeg", ".jpg")


def list_stems(images_dir: Path) -> list[str]:
    """Image ids in a directory: file names without their extension."""
    stems = sorted(
        path.stem for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not stems:
        raise FileNotFoundError(f"No images found in {images_dir}.")
    return stems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained weights.")
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Directory of frames. Defaults to the test set in configs/paths.yaml.",
    )
    parser.add_argument("--out", type=Path, default=Path("submission.csv"), help="Output CSV.")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Probability cut-off."
    )
    parser.add_argument(
        "--min-area", type=int, default=DEFAULT_MIN_AREA, help="Smallest region to emit."
    )
    parser.add_argument("--device", type=str, default=None, help="Force cpu or cuda.")
    parser.add_argument(
        "--no-disk-mask",
        action="store_true",
        help="Keep predictions outside the solar disk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)

    paths = load_paths()
    images_dir = args.images or paths.test_images
    model, stored = load_checkpoint(args.checkpoint, device=args.device)
    size = int(stored["image_size"])

    stems = list_stems(images_dir)
    logger.info("Predicting %d frames at size %d.", len(stems), size)

    started = time.perf_counter()
    rows: list[tuple[str, str]] = []
    empty = []
    for position, stem in enumerate(stems, start=1):
        instances = predict_frame(
            model,
            images_dir / f"{stem}.jpeg",
            size=size,
            threshold=args.threshold,
            min_area=args.min_area,
            device=args.device or "cpu",
            mask_disk=not args.no_disk_mask,
        )
        if not instances:
            empty.append(stem)
        rows.extend(instances_to_rows(stem, instances))
        if position % 20 == 0:
            logger.info("%d/%d frames done.", position, len(stems))
    elapsed = time.perf_counter() - started

    write_submission(rows, args.out)
    check_no_overlap(args.out, paths.image_height, paths.image_width)

    print(f"{len(rows)} predictions over {len(stems)} frames in {elapsed:.1f}s")
    if empty:
        # Worth knowing rather than discovering from the score: every frame in
        # the training set holds at least one filament, so an empty prediction
        # is a miss, not a legitimate answer.
        print(f"{len(empty)} frame(s) got no prediction at all: {', '.join(empty[:5])}")
    print(f"Wrote {args.out}. Overlap check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
