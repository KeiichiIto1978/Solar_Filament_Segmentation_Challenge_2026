"""Convert the five fold checkpoints into the half-precision weights under ``weights/``.

    uv run python scripts/export_weights.py --checkpoints <dir with fold0..fold4/best.pt>

The training checkpoints come from the Kaggle notebook that trained the five
folds of the final configuration (HRNet-W32 U-Net at 1024). Each is written as
``weights/hrnet_w32_fold<k>.pt`` with ``export_for_inference``, which keeps the
training configuration beside the weights so that ``load_checkpoint`` can
rebuild the network from the file alone.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from filament.training.loop import CHECKPOINT_NAME, export_for_inference

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDS = range(5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints",
        type=Path,
        required=True,
        help="Directory holding fold0/best.pt to fold4/best.pt.",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "weights", help="Directory to write into."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    sources = [args.checkpoints / f"fold{fold}" / CHECKPOINT_NAME for fold in FOLDS]
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        raise SystemExit(f"Missing checkpoints: {', '.join(missing)}")
    for fold, source in zip(FOLDS, sources, strict=True):
        target = export_for_inference(source, args.output / f"hrnet_w32_fold{fold}.pt")
        logger.info(
            "Fold %d: %s (%.1f MB) -> %s (%.1f MB)",
            fold,
            source,
            source.stat().st_size / 1e6,
            target,
            target.stat().st_size / 1e6,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
