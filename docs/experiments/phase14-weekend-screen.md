# Phase 14 — A fold-0 screen: a larger network, a union-trained one, and growing into the faint ends

Closed 2026-10-31. **Nothing adopted; the Phase 10 configuration stays.** The
screen asked for at least +0.02 over run *e* on fold 0, an effect well beyond
what one retraining moves a fold. HRNet-W48 scored 0.4060 (+0.0030 against the
better of the two references), HRNet-W32 trained on the union of the annotators
0.4029 (−0.0001). Growing the working configuration's filaments into their
faint ends, guided either by a classical dark-ridge filter or by the union
model's map, lowered PQ in every setting tried. Averaging the union model's map
into run *e*'s gained +0.0050 at best.

The working configuration (HRNet-W32 U-Net at 1024, five fold models,
eight-view TTA, the Phase 10 post-processing) scores a pooled five-fold PQ of
0.4127 and 0.36 on the public leaderboard. Phase 12 traced the largest group of
its misses to long filaments covered only in part: the prediction lies inside
the annotated filament and stops at about a third of it. This phase looked for a
larger step than the +0.002 the previous three phases each moved, and aimed most
of it at those partly covered filaments.

## What was done

**References.** Fold 0 under eight-view TTA and the Phase 10 post-processing
(2048, threshold 0.40, minimum area 400, rejoining 16, score cut 0.65): run *e*
from Phase 9's maps, 0.4016, and its retraining as Phase 13's control, 0.4030.
No new control was trained.

**A. HRNet-W48.** HRNet won Phase 6b's comparison of four encoders at 1024; W48
widens every branch by half. A memory probe before training found that two
frames fit a T4 (peak 7.1 GiB), so it trained with run *e*'s batch of two.
Gradient accumulation (`accumulation_steps`) was added for the case where they
would not, and tested to give the same update as the larger batch; it was not
needed.

**B. HRNet-W32 trained on the union of a frame's annotators.** The union asks
for every pixel anyone drew, so it should reach further along faint filaments.
It was scored alone, and its map was used afterwards as described below.

Both trained at once on two T4s, fold 0, twenty epochs, in
`notebooks/100_train_kaggle_screen.ipynb`, and were scored the way the
submission is made, with Phase 12's breakdown of the misses.

**C. Are the faint ends in the frame at all?** For every partly covered
filament of the five folds (953) and a sample of matched ones (1,000), the
annotated filament was split into the part some emitted prediction covers and
the part none does, and a ring of background 25 pixels wide was taken around
it, on the disk and outside every annotator's filaments. Each region was
summarised by a Frangi-type dark-ridge response
(`filament.data.ridges.dark_ridge_response`: scale-normalised Hessian
eigenvalues, sigmas 1 to 6 pixels), at 2048 and at the 1024 the network sees
(`scripts/uncovered_ridges.py`).

**Growing the emitted filaments.** `filament.postprocess.grow.grow_instances`
extends each filament the configuration already emits into pixels where a
second source of support reaches a threshold, connected to the filament and no
further than a reach from it. It never creates a filament, and where two grow
into each other each pixel goes to the nearer one, so the masks still never
overlap. Two supports were tried (`scripts/grow_sweep.py`):

- the dark-ridge response, divided per frame by its median over the frame's own
  emitted filaments (thresholds 0.1 to 0.5), on all five folds;
- the union model's map (thresholds 0.2 to 0.5), on fold 0;

each with reaches of 10, 25 and 50 pixels at 2048.

**Averaging with the union model.** Run *e*'s fold-0 map and the union model's
map were averaged with the union weighted 0.25 or 0.5, and post-processed under
thresholds 0.35 to 0.50 and score cuts 0.60 to 0.70
(`scripts/sweep_maps.py`).

## Results

### A and B, fold 0

| run | best epoch | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| run *e* (reference) | 13 | 0.4016 | 0.6670 | 0.6021 | 995 | 515 | 800 |
| run *e* retrained (reference) | 13 | 0.4030 | 0.6647 | 0.6063 | 1,008 | 522 | 787 |
| HRNet-W48 | 17 of 20 | **0.4060** | 0.6679 | 0.6078 | 1,018 | 537 | 777 |
| HRNet-W32, union | 11 of 20 | 0.4029 | 0.6580 | 0.6123 | 1,048 | 580 | 747 |

Without the score cut: W48 0.3997 (TP 1,039, FP 630, FN 756), union 0.3981
(TP 1,088, FP 706, FN 707).

The misses of fold 0, against the retrained run *e*:

| outcome | retrained run *e* | W48 | union |
|---|---|---|---|
| unseen | 137 | 127 (−10) | 90 (**−47**) |
| below threshold | 129 | 124 (−5) | 103 (**−26**) |
| dropped by area | 57 | 63 (+6) | 46 (−11) |
| dropped by score | 34 | 21 (−13) | 40 (+6) |
| too large | 127 | 107 (−20) | 172 (**+45**) |
| fragmented | 52 | 52 (0) | 50 (−2) |
| too small | 179 | 215 (**+36**) | 159 (−20) |
| shape | 72 | 68 (−4) | 87 (+15) |

### C, the dark-ridge response (median over each region)

| filaments | covered part | uncovered part | background ring |
|---|---|---|---|
| partly covered, 2048 | 0.022 | 0.007 | 0.002 |
| partly covered, 1024 | 0.022 | 0.007 | 0.001 |
| matched, 2048 | 0.030 | 0.003 | 0.001 |

For the partly covered filaments, the uncovered part's median exceeds the
ring's median in 98.1% of them (1024: 97.3%) but the ring's 90th percentile in
only 38.2% (1024: 38.7%). The uncovered part responds at about a fifth of the
covered part (0.19; 1024: 0.17).

### Growing along the dark-ridge response, pooled over five folds

| threshold \ reach | 10 | 25 | 50 |
|---|---|---|---|
| 0.1 | 0.2334 | 0.0570 | 0.0222 |
| 0.2 | 0.3148 | 0.2065 | 0.1559 |
| 0.3 | 0.3537 | 0.2960 | 0.2652 |
| 0.5 | 0.3886 | 0.3680 | 0.3562 |
| not grown | **0.4127** | | |

The best, threshold 0.5 and reach 10: SQ 0.6571, RQ 0.5915, TP 4,522, FP 2,570,
FN 3,677; worse on all five folds.

### Growing along the union model's map, fold 0

| threshold \ reach | 10 | 25 | 50 |
|---|---|---|---|
| 0.2 | 0.3527 | 0.3480 | 0.3480 |
| 0.3 | 0.3793 | 0.3797 | 0.3801 |
| 0.4 | 0.3908 | 0.3929 | 0.3931 |
| 0.5 | 0.3972 | 0.3978 | 0.3993 |
| not grown | **0.4016** | | |

At threshold 0.2 and reach 10, TP falls by 88 and FP rises by 88.

### Averaging with the union model, fold 0

| union weight | best setting | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| 0 (run *e*) | Phase 10 | 0.4016 | 0.6670 | 0.6021 | 995 | 515 | 800 |
| 0.25 | threshold 0.35, cut 0.60 | 0.4056 | 0.6642 | 0.6106 | 1,024 | 535 | 771 |
| 0.5 | threshold 0.35, cut 0.65 | 0.4066 | 0.6626 | 0.6137 | 1,019 | 507 | 776 |

Of the 24 settings, all but three lie between 0.4011 and 0.4056.

## Discussion

**A wider HRNet is not a larger step.** W48's +0.0030 sits inside the 0.005 to
0.01 a retraining moves fold 0. Its misses shift from too large to too small:
it draws tighter regions, which help where the configuration overshot and hurt
where it already stopped short.

**The union target trades misses for overshoot.** The union model sees 47 more
filaments that run *e* missed entirely and lifts 26 more above the threshold,
but 45 more of its regions are too large for any one annotator. Scoring is per
annotator, so drawing the union of their filaments is drawing too much for each
of them. The two cancel.

**The faint ends are in the frame, but not distinctly.** The uncovered stretches
respond above the background on average, and not above its upper tail: in most
filaments, a tenth of the background ring is as ridge-like as the missed
stretch. The same holds at 1024 and 2048, so the network's input resolution is
not what hides them, and this gives no reason to train at 2048.

**Growing into the faint ends breaks more matches than it makes.** Along the
ridge filter, PQ falls the further the growth reaches and the lower the
threshold, and returns towards the ungrown value as both tighten. The chromosphere
around a filament is full of dark fibrils that respond as strongly as its faint
ends, and growth runs into them; the grown region crosses IoU 0.5 the wrong way.
Along the union map, every match lost is a false positive gained, for the same
reason the union model alone overshoots. Neither support separates a filament's
continuation from what lies beside it.

**Averaging gains what a second model always gains.** +0.0050 matches the
+0.0054 that mixing a different model gave in Phase 8. A second network type
doubles the inference time, and computational efficiency is part of the
evaluation; a gain of this size does not justify it.

## Next

- The Phase 10 configuration is the final one; the next step is the submission
  version (weights in the repository, the full-pipeline notebook).
- What remains of the partly covered filaments is not reachable from the same
  frame by a wider network, a broader target or growth along a filter. The one
  direction not tried is self-supervised pre-training on unannotated GONG
  frames, which is out of reach before the deadline.

## Reproduction

| | |
|---|---|
| configurations | `configs/phase14/hrnet_w48.yaml`, `configs/phase14/hrnet_w32_union.yaml` (`hrnet_w48_accum.yaml` prepared, not used) |
| notebook | `notebooks/100_train_kaggle_screen.ipynb`, Kaggle `Filament-screen` Version 1 |
| commits | `bbcd1ce` (the Kaggle run), `d3e6b67` (the ridge response and growth) |
| ridge diagnostic | `scripts/uncovered_ridges.py --maps <fold maps> --fold <k> --setting <Phase 10 JSON> --misses <miss_breakdown output>` |
| growth | `scripts/grow_sweep.py --maps <run e fold maps> --fold <k> --setting <Phase 10 JSON> --support ridge` (or `--support maps/union_fold0.npz`) |
| averaging | the two fold-0 bundles read with `load_map_bundle`, averaged pixel-wise, written with `save_map_bundle`, then `scripts/sweep_maps.py` |
| environment | Kaggle, two T4s, one run per card, 10,123 seconds (W48 155.9 minutes, union 114.1 minutes); the rest on a local CPU |
