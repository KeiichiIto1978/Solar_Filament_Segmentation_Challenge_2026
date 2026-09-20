# Solar Filament Segmentation Challenge 2026

Instance segmentation of dark filaments in GONG H-alpha full-disk images
(2048x2048, 8-bit grayscale), for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

Submissions are scored with **Panoptic Quality** at an IoU threshold of 0.5:

```
PQ = sum(IoU over matched pairs) / (|TP| + 0.5|FP| + 0.5|FN|)
```

Only pairs above IoU 0.5 are true positives, and the score is computed once per
*annotator-image*: an image annotated by three people is scored three times,
against three different ground truths.

## Status

**Phase 0 — evaluation foundation.** There is no model yet. This stage builds
the pieces needed to measure PQ locally, so that experiments are not rationed
by the five daily Kaggle submissions.

What works today:

- the annotation file is read into a form that keeps annotators separate;
- PQ is computed locally, broken down into SQ, RQ, TP, FP and FN;
- a submission can be checked for overlapping masks before it is uploaded;
- the cross-validation split is frozen and committed.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

No GPU is needed for anything in this repository yet.

## Setup

```bash
git clone git@github.com:KeiichiIto1978/Solar_Filament_Segmentation_Challenge_2026.git
cd Solar_Filament_Segmentation_Challenge_2026
uv sync
```

`uv sync` installs the pinned dependency set. `requirements.txt` holds the same
versions for environments without uv.

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
`MAGFILO_ROOT` takes precedence over the `dataset.root` entry, so no absolute
path ever enters the source tree.

The publicly released version of MAGFiLO is **not** used anywhere in this
repository. It overlaps the competition test set, and the organizers forbid it.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests that read the dataset are skipped when it is not present. The suite
includes the two checks the rest of the project depends on: the evaluator
returns exactly 1.0 when the ground truth is fed back as the prediction, and it
reproduces hand-computed scores on either side of the IoU 0.5 threshold.

## Usage

Regenerate the cross-validation split (already committed; the script refuses to
overwrite existing folds, because replacing them would silently invalidate
every score recorded against them):

```bash
uv run python scripts/make_splits.py
```

Write a submission of empty masks, as a format rehearsal:

```bash
uv run python scripts/make_dummy_submission.py --out submission.csv
```

Check any submission for overlapping masks before uploading it. Kaggle rejects
a submission whose masks share a pixel, and the rejected attempt still counts
against the five per day:

```bash
uv run python scripts/check_submission.py --csv submission.csv
```

Score predictions against ground truth in Python:

```python
from filament.data.coco import load_annotations
from filament.metrics.pq import compute_pq
from filament.submit.rle import masks_to_gt_df, read_submission

dataset = load_annotations()
gt = masks_to_gt_df(dataset, stems=["20140609195854Bh"])
result = compute_pq(gt, read_submission("submission.csv"))
print(result)  # PQ=... SQ=... RQ=... TP=... FP=... FN=...
```

## Submission format

A single CSV with two columns and one row per predicted filament:

```
filament_id,segmentation_rle
20150125172714Mh_1,PPYd1...
```

- `filament_id` is `<image id>_<running number>`, the image id without its
  extension; the number only makes rows unique.
- `segmentation_rle` is the *counts* field of a pycocotools compressed RLE on
  its own. The size is a constant 2048x2048 and is not written, and the value
  is never quoted.
- Masks of the same image must not overlap, not even by one pixel.

## Repository layout

```
configs/            paths.yaml and the frozen splits
notebooks/          00_eda.ipynb
scripts/            command line entry points
src/filament/
├── data/           COCO reading, cross-validation splits
├── metrics/        Panoptic Quality, submission overlap check
└── submit/         RLE encoding, submission CSV
tests/
```

## License

[Apache-2.0](LICENSE).
