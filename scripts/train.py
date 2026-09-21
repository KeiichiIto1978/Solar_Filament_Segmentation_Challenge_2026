"""Train the semantic baseline on one fold.

On a GPU (Kaggle, two T4s)::

    uv run python scripts/train.py --config configs/phase1_unet.yaml

To check the wiring on a CPU in about a minute, at a useless quality::

    uv run python scripts/train.py --config configs/phase1_unet.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from filament.training.config import TrainConfig
from filament.training.loop import HISTORY_NAME, train

# Applied after the configuration is built rather than as command line
# overrides: an override of None means "not specified", so it could never
# switch the pretrained weights off.
SMOKE_OVERRIDES = {
    "epochs": 1,
    "image_size": 256,
    "batch_size": 2,
    "encoder_weights": None,
    "max_train_batches": 2,
    "max_val_batches": 2,
    "amp": False,
    "num_workers": 0,
}
SMOKE_OUTPUT_DIR = Path("outputs/phase1_unet_smoke")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML.")
    parser.add_argument("--fold", type=int, default=None, help="Override the fold.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size.")
    parser.add_argument("--encoder", type=str, default=None, help="Override the encoder.")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override where outputs are written."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Force a device, e.g. cpu or cuda."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One tiny epoch on a few batches, to check the wiring without a GPU.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)

    config = TrainConfig.from_yaml(
        args.config,
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder=args.encoder,
        output_dir=args.output_dir,
    )
    if args.smoke:
        config = replace(
            config,
            output_dir=args.output_dir or SMOKE_OUTPUT_DIR,
            **SMOKE_OVERRIDES,
        )
    result = train(config, device=args.device)

    history_path = config.output_dir / HISTORY_NAME
    history_path.write_text(
        json.dumps([asdict(item) for item in result.history], indent=2), encoding="utf-8"
    )

    print(f"Best epoch {result.best_epoch}, validation loss {result.best_val_loss:.4f}")
    print(f"Checkpoint: {result.checkpoint}")
    print(f"History:    {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
