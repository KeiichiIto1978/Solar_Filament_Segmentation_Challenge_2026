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

**Phase 1 — semantic baseline.** A U-Net predicts one binary mask over the
whole disk and instances are recovered from its connected regions, which makes
overlapping masks impossible by construction. The target for this phase is
local PQ 0.25 on fold 0.

What works today:

- the annotation file is read into a form that keeps annotators separate;
- PQ is computed locally, broken down into SQ, RQ, TP, FP and FN;
- fused and split filaments are counted separately, since PQ charges for both
  without saying which one happened;
- the solar disk is located, so predictions in the sky can be discarded;
- a submission can be checked for overlapping masks before it is uploaded;
- the cross-validation split is frozen and committed;
- training, evaluation and test-set prediction run from the command line.

Scoring one annotator's drawing against the other annotators of the same image
gives PQ 0.73 on fold 0, which is the practical ceiling this pipeline is
measured against.

## Requirements

- Python 3.12, the version Kaggle notebooks run
- [uv](https://docs.astral.sh/uv/)

`torch` is pinned to its **CPU build**: development and evaluation need no GPU,
and training runs on Kaggle, whose notebooks ship their own CUDA build. To
train elsewhere on a GPU, replace the two torch lines with a build matching
your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

## Setup

```bash
git clone git@github.com:KeiichiIto1978/Solar_Filament_Segmentation_Challenge_2026.git
cd Solar_Filament_Segmentation_Challenge_2026
uv sync
```

`uv sync` installs the pinned dependency set. For environments without uv,
`requirements.txt` holds the same versions:

```bash
pip install -r requirements.txt
```

Regenerate that file after changing a dependency:

```bash
uv run python scripts/export_requirements.py
```

### Running on Kaggle

Training runs on Kaggle notebooks (two T4 GPUs);
[`notebooks/10_train_kaggle.ipynb`](notebooks/10_train_kaggle.ipynb) is ready to
open there. Set the accelerator to **GPU T4 x2**, turn **Internet on**, and
attach the competition data. The first cell brings the code in:

```python
!git clone -q https://github.com/KeiichiIto1978/Solar_Filament_Segmentation_Challenge_2026.git /kaggle/working/repo
!pip install -q segmentation-models-pytorch
import sys; sys.path.insert(0, "/kaggle/working/repo/src")
```

The repository is cloned rather than installed from its URL because
`configs/paths.yaml` and the frozen splits sit beside the package rather than
inside it; a wheel would leave them behind and the run would be validated on a
different set of frames. The clone goes on `sys.path` rather than through pip:
this is a pure Python package, so the import works either way, and skipping pip
skips its interpreter-version check, which Kaggle's Python trips as it moves
ahead of the version this is locked against. Kaggle's preinstalled CUDA build
of torch is left alone, since replacing it costs minutes and risks a
mismatch with the driver.

Pushing to `main` and rerunning that cell is the whole update procedure, so the
notebook never holds a second copy of the code. Check out a commit hash instead
of `main` when a run has to be reproducible.

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

Train the baseline (needs a GPU; use `--smoke` to check the wiring on a CPU in
about a minute):

```bash
uv run python scripts/train.py --config configs/phase1_unet.yaml
```

Score a checkpoint on its validation fold. This prints PQ with its breakdown
plus the counts of fused and split filaments, which is what says whether to
work on detection or on mask quality:

```bash
uv run python scripts/evaluate.py --checkpoint outputs/phase1_unet/best.pt --fold 0
```

Predict the test set. The overlap check runs before the script exits:

```bash
uv run python scripts/predict.py --checkpoint outputs/phase1_unet/best.pt --out submission.csv
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

## Experiment records

One record per phase, in [`docs/experiments/`](docs/experiments/): what was
built, what it measured, what was learned, and what came next.

| Phase | Outcome |
|---|---|
| [0 — Evaluation foundation](docs/experiments/phase0-evaluation-foundation.md) | PQ measurable locally; human agreement puts the ceiling at PQ 0.73 |
| [1 — Semantic baseline](docs/experiments/phase1-semantic-baseline.md) | fold 0 PQ 0.3454, leaderboard 0.29 |
| [2 — Choosing the source of the masks](docs/experiments/phase2-mask-source.md) | U-Net kept; a filter bank reached PQ 0.16 at fifteen times the cost |

## Repository layout

```
configs/            paths.yaml, the frozen splits, training settings
notebooks/          00_eda.ipynb, 10_train_kaggle.ipynb
scripts/            command line entry points
src/filament/
├── data/           COCO reading, splits, frames, solar disk, torch dataset
├── models/         the U-Net and its loss
├── training/       configuration and the training loop
├── postprocess/    probability map to non-overlapping instances
├── metrics/        Panoptic Quality, submission overlap check
├── submit/         RLE encoding, submission CSV
└── evaluation.py   inference and scoring
tests/
```

## License

[Apache-2.0](LICENSE).
