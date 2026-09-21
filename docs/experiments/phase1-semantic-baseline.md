# Phase 1 — Semantic baseline

Closed 2026-09-21. Exit target was local PQ 0.25 on fold 0.

## What was done

A U-Net predicts one binary mask over the whole disk; instances are recovered
from its connected regions. Two regions that are connected are one region, so
the masks cannot overlap and the submission cannot be rejected for it.

- ResNet34 encoder pretrained on ImageNet, 1024x1024 input
- Two input channels: the frame, and the same frame after CLAHE
- Dice + binary cross-entropy, AdamW 3e-4, cosine schedule, 40 epochs
- Flips and quarter turns only; the Sun has no preferred orientation in these
  frames, while resampling would blur a filament a few pixels wide
- The solar disk is located per frame and predictions outside it are dropped
- One training sample is one annotator's view of one frame, matching the unit
  the metric scores, rather than a consensus of the annotators

Training ran on one Kaggle T4. Locating the disk took three attempts:
thresholding fails on the frames carrying a bright halo outside the limb, and
scanning single rows mistakes a dark filament near the middle for the limb.
Averaging intensity over each annulus works on all 707 frames, giving a radius
of 900-908 pixels.

## Results

fold 0 validation, 142 frames / 239 annotator-images / 1,795 ground-truth
filaments, post-processing left at its defaults:

| Metric | Value |
|---|---|
| PQ | **0.3454** |
| SQ | 0.6520 |
| RQ | 0.5298 |
| TP / FP / FN | 1001 / 983 / 794 |
| Fragmented / fused | 155 / 19 |
| Predictions | 1167 |

Validation loss bottomed at epoch 8-10 and did not improve over the following
19 epochs, while training loss fell from 0.315 to 0.193. Epoch 10 was kept.
146 s per epoch.

Submitted 1,369 masks over 175 of the 180 test frames: **public leaderboard
0.29**.

## Discussion

**RQ is the bottleneck, not SQ.** Masks that match are reasonable; the counts
are wrong. Work should go to how many instances are emitted, not to how
cleanly they are drawn.

**Fragmentation outnumbers fusion eight to one.** The usual objection to
connected components is that it fuses neighbouring objects. On this data it
splits single filaments instead: 155 ground-truth filaments were covered by
more than one prediction, against 19 predictions covering more than one
filament. Rejoining fragments, not separating fused blobs, is the lever here.
An optimistic estimate puts it at about +0.05 PQ.

**Forty epochs was wasteful.** Twelve to fifteen with early stopping would
reach the same validation loss in a third of the time, which matters more for
the number of experiments possible than any other change identified so far.

**The gap to the leaderboard is 0.055 and in the expected direction.** Fold 0
selected the epoch, so an unseen test set scoring lower is ordinary; the
threshold set beforehand for suspecting the split or the format was 0.1.

Five test frames received no prediction at all. Every frame in the training
set holds at least one filament, so these are misses, not empty frames.

## Next

Phase 2a: compare this U-Net against a multi-scale filter bank with a random
forest on the same 142 frames, the same evaluator and the same post-processing,
and decide the source of the masks by measurement. The filter bank trains in
seconds on a CPU, which would make cross-validation free rather than costing
an hour; whether it segments well enough is the open question.
