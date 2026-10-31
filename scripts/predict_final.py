"""Write both final submissions from the committed weights.

    uv run python scripts/predict_final.py --output outputs/final

The command line counterpart of ``notebooks/99_full_pipeline.ipynb``: the five
fold models in ``weights/``, eight-view test-time augmentation, and the
post-processing of ``filament.pipeline``. Writes ``submission.csv`` (the final
configuration) and ``submission_no_score_cut.csv`` (the same without the cut on
mean probability), each checked for overlapping masks. A GPU is strongly
advised: on a CPU the 180 test frames take a few hours.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from filament.paths import load_paths
from filament.pipeline import (
    FINAL_SETTING,
    NO_SCORE_CUT_SETTING,
    load_fold_models,
    predict_maps,
    submission_from_maps,
)
from filament.training.loop import resolve_device

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--weights", type=Path, default=REPO_ROOT / "weights", help="The five fold models."
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "outputs" / "final", help="Directory to write."
    )
    parser.add_argument("--device", type=str, default=None, help="Force cpu or cuda.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)
    paths = load_paths().require_dataset()
    frames = sorted(paths.test_images.glob("*.jpeg"))
    if not frames:
        raise SystemExit(f"No test frames in {paths.test_images}.")

    started = time.perf_counter()
    models = load_fold_models(args.weights, device=args.device)
    maps = predict_maps(models, frames, device=resolve_device(args.device))
    logger.info("Predicted %d frames in %.0fs.", len(maps), time.perf_counter() - started)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, setting in (
        ("submission.csv", FINAL_SETTING),
        ("submission_no_score_cut.csv", NO_SCORE_CUT_SETTING),
    ):
        rows = submission_from_maps(maps, paths.test_images, setting, args.output / name)
        logger.info("%s: %d filaments.", args.output / name, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
