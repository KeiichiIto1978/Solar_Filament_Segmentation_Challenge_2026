# Solar Filament Segmentation Challenge 2026

Instance segmentation of dark filaments in GONG H-alpha full-disk images
(2048x2048, 8-bit grayscale), for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

Submissions are scored with **Panoptic Quality (PQ)** at an IoU threshold of 0.5.

## Status

Phase 0 — evaluation foundation. No model yet: this stage builds the pieces
needed to measure PQ locally, so that experiments are not gated on the five
daily Kaggle submissions.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone git@github.com:KeiichiIto1978/Solar_Filament_Segmentation_Challenge_2026.git
cd Solar_Filament_Segmentation_Challenge_2026
uv sync
```

## Dataset

The competition data is distributed under CC BY-NC 4.0 and is **not** included
in this repository. Download it from the
[competition data page](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
and unpack it so that the layout is:

```
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/                                  707 images
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json  COCO format
└── test/
    └── test_images/                                   180 images
```

Place it at the repository root, or point the `MAGFILO_ROOT` environment
variable at your copy:

```bash
export MAGFILO_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026
```

Paths are configured in [`configs/paths.yaml`](configs/paths.yaml);
`MAGFILO_ROOT` takes precedence over the `dataset.root` entry.

The publicly released version of MAGFiLO is **not** used anywhere in this
repository: it overlaps the competition test set, and the organizers forbid it.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests that read the dataset are skipped when it is not present.

## License

[Apache-2.0](LICENSE).
