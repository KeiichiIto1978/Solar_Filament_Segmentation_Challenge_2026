# Phase 8 — Test-time augmentation, and mixing two kinds of model

Closed 2026-09-27. **Averaging each model's predictions over the eight flips and
quarter turns raises the pooled five-fold PQ from 0.3807 to 0.3998 (+0.0191),
and every fold improves.** The five-model ensemble with this augmentation scores
**0.35 on the public leaderboard**, up from 0.34. **Mixing HRNet at 1024 with
ResNet-34 at 2048 does not earn its training cost**: on fold 0 it adds +0.0009
beyond what scoring at 2048 alone gives.

Phase 7 found that averaging five fold models lifted the leaderboard by 0.02,
consistent with the average removing detections the models disagree on. This
phase measured two cheaper ways to get more of the same effect, neither of
which needs any training.

## What was done

**Adoption rule.** From this phase on, a change is adopted when its pooled
five-fold PQ is at least 0.005 above the reference **and** at least four of the
five folds move the same way. A change that can only be measured on fold 0
still needs +0.01 there, because one retraining moves fold 0 by 0.005 to 0.01.

**Test-time augmentation (TTA).** Each of Phase 7's five fold models predicted
the frames it had not seen in all eight orientations of the square: with and
without a left-right flip, each turned by 0, 90, 180 and 270 degrees. This is
the set the training augmentation draws from. Each prediction was turned back
to the original orientation, and probabilities were averaged, as in the
ensemble. Three variants were built from the same eight forward passes:

- *none* — the identity view alone, which is Phase 7's measurement;
- *flip* — the identity and the left-right flip;
- *d4* — all eight.

Since the variants share their forward passes, they differ in the averaging and
in nothing else. The post-processing is Phase 3's throughout: threshold 0.5,
minimum area 400, rejoining within 24 pixels. The transforms are applied to the
prepared input, after contrast equalisation, where the training augmentation
applies them; `tests/test_evaluation.py` checks that each one is undone by its
inverse and that the average over them has the expected value for a model whose
answer does not depend on the orientation of its input.

**Mixing two kinds of model, on fold 0.** In Phase 6b, HRNet at 1024 (run *e*)
had the best detection (RQ) and ResNet-34 at 2048 (run *f*) the best mask shape
(SQ). Their probabilities were averaged at 2048, run *e*'s map being upsampled
bilinearly, and the rejoining distances were doubled (48 pixels) to mean the
same distance on the Sun. Run *f* exists on fold 0 only, so this is a fold 0
measurement, without TTA.

## Results

### TTA, five folds

| variant | PQ | SQ | RQ | TP | FP | FN | predictions |
|---|---|---|---|---|---|---|---|
| none (Phase 7) | 0.3807 | 0.6592 | 0.5776 | 4,538 | 2,977 | 3,661 | 4,619 |
| flip | 0.3870 | 0.6625 | 0.5842 | 4,535 | 2,792 | 3,664 | 4,473 |
| **d4** | **0.3998** | **0.6679** | **0.5986** | **4,663** | **2,718** | **3,536** | 4,521 |

All three are pooled over the 707 frames and 1,154 annotator-images.

| fold | none | flip | d4 | d4 − none | d4 TP / FP / FN |
|---|---|---|---|---|---|
| 0 | 0.3815 | 0.3797 | 0.3941 | +0.0126 | 1,000 / 575 / 795 |
| 1 | 0.3774 | 0.3903 | 0.3986 | +0.0212 | 915 / 566 / 675 |
| 2 | 0.3808 | 0.3923 | 0.3952 | +0.0144 | 858 / 536 / 642 |
| 3 | 0.3875 | 0.3961 | 0.4116 | +0.0241 | 995 / 544 / 705 |
| 4 | 0.3759 | 0.3768 | 0.3991 | +0.0232 | 895 / 497 / 719 |

*d4* gains +0.0191 pooled (SQ +0.0087, RQ +0.0210) with five folds of five
improving, and is adopted. *flip* gains +0.0063 with four of five and also
passes the rule, but falls well short of *d4*. The *none* variant reproduces
Phase 7 on every fold to four decimals.

### Mixing, fold 0

| reading | PQ | SQ | RQ | TP | FP | FN | predictions |
|---|---|---|---|---|---|---|---|
| *e*, post-processed at 1024 | 0.3815 | 0.6542 | 0.5832 | 980 | 586 | 815 | 927 |
| *e*, upsampled and post-processed at 2048 | 0.3860 | 0.6582 | 0.5864 | 984 | 577 | 811 | 923 |
| *f*, at 2048 | 0.3782 | 0.6677 | 0.5664 | 968 | 655 | 827 | 959 |
| mean of *e* and *f*, at 2048 | 0.3869 | 0.6700 | 0.5775 | 958 | 565 | 837 | 904 |

Against run *e* as measured before, the mix gains +0.0054, short of the +0.01
fold 0 needs. Against run *e* post-processed at the same resolution, it gains
+0.0009. Runs *e* and *f* alone reproduce Phase 6b's 0.3815 and 0.3782.

### Submissions

| submission | masks | frames with a prediction | public leaderboard |
|---|---|---|---|
| ensemble of five, none (Phase 7) | 1,135 | 175 | 0.34 |
| **ensemble of five, d4** | 1,141 | 175 | **0.35** |
| ensemble of five, flip | 1,141 | 175 | — |

## Discussion

**TTA helps mainly by finding filaments, not only by dropping bad guesses.**
Pooled, *d4* removes 259 false positives and also adds 125 true positives. The
ensemble of Phase 7 was read as averaging away weak detections; here, in
addition, a faint filament that crosses the threshold in some orientations and
not in others is lifted above it by the average. SQ rises on every fold as well
(0.6640 to 0.6703 against 0.6542 to 0.6649), consistent with the averaged mask
having a smoother edge that overlaps the tracing better.

**Two views are not enough; eight are.** *flip* loses on fold 0 and gains
+0.0009 on fold 4, which is within the 0.005 to 0.01 that one retraining moves a
fold. With eight views the smallest per-fold gain is +0.0126. The model's
predictions evidently vary with orientation by more than one extra view can
average out.

**The mix gains almost nothing that is due to the mixing.** Of its +0.0054 over
run *e*, +0.0045 comes from post-processing at 2048: the same map, upsampled
before thresholding and rejoining, scores that much better on its own. On top of
that, mixing in run *f* raises SQ (+0.0118) but costs as much in RQ (−0.0090),
with 26 fewer matches. Run *f*'s better shape comes with its worse detection.
Training run *f* on the other four folds, about twelve GPU hours, is not
justified by this.

**The leaderboard moved by one display step.** 0.35 against 0.34 is within
rounding on about half the test set, so on its own it would not be readable.
It is in the direction the five-fold measurement predicts, and the decision
rests on that measurement. The gap between the local and leaderboard scores has
widened from about 0.04 to about 0.05.

## Next

- The *d4* ensemble is the submission to keep selected.
- Post-processing at 2048 after upsampling gave +0.0045 on fold 0 with no
  training, raising both SQ and RQ. It is below the fold 0 bar, but combined
  with *d4* it can be measured on all five folds at no training cost.
- The two questions Phase 7 left open still stand: the ResNet-34 reference on
  all five folds, and how far one retraining moves fold 0.

## Reproduction

| | |
|---|---|
| models | Phase 7's five fold models (Kaggle `Use_HRNet` Version 3); runs *e* and *f* of Phase 6b (Kaggle `Filament-architecture-comparison` Version 4) |
| notebook | `notebooks/80_kaggle_tta_and_model_mix.ipynb` |
| commit of the run | `2a2a8fa` |
| environment | Kaggle, one T4 used, 52 minutes (TTA 25 minutes, the mix 5, the test submissions 21) |
