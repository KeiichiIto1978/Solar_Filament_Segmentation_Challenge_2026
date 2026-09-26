# Phase 7 — The working configuration on all five folds, and an ensemble

Closed 2026-09-26. **HRNet-W32 at 1024 scores a pooled PQ of 0.3807 over all
five folds**, and fold 0's 0.3815, on which Phase 6b adopted it, turns out to be
typical rather than lucky. **Averaging the five fold models reaches 0.34 on the
public leaderboard**, the first submission above 0.32.

Phase 6b adopted run *e* on the strength of one fold: 142 frames, +0.0087 over
its reference, from a single training run. Two things were left open — whether
fold 0 was simply an easy fold, and whether the number would survive a rerun.
This phase answers the first. The second is still open; see below.

## What was done

`configs/phase6/e_unet_hrnet.yaml` was trained on folds 1 to 4, unchanged
except for the fold. Fold 0 is run *e* itself. Each fold's model was scored on
the frames it did not see, through the post-processing Phase 3 settled on:
threshold 0.5, minimum area 400, rejoining within 24 pixels.

The five scores are **pooled**: true positives, false positives, false
negatives and the sum of matched IoUs are added across folds, which gives the
PQ that scoring all 707 frames at once would give. A plain mean of the five PQs
would weight a fold with few filaments as heavily as one with many; it is shown
beside the pooled figure for comparison.

Six submissions were written from one pass over the test frames: one per fold
model, and one from the mean of the five models' probabilities. Probabilities
rather than logits are averaged, so that one model's confident extreme cannot
outvote the others.

## Results

| fold | frames | annotator-images | PQ | SQ | RQ | TP | FP | FN | fused | split | best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (run *e*) | 142 | 239 | 0.3815 | 0.6542 | 0.5832 | 980 | 586 | 815 | 44 | 78 | 13 |
| 1 | 142 | 239 | 0.3774 | 0.6554 | 0.5758 | 900 | 636 | 690 | 33 | 90 | 14 |
| 2 | 141 | 221 | 0.3808 | 0.6615 | 0.5757 | 840 | 578 | 660 | 45 | 62 | 9 |
| 3 | 141 | 246 | 0.3875 | 0.6649 | 0.5828 | 950 | 610 | 750 | 49 | 85 | 13 |
| 4 | 141 | 209 | 0.3759 | 0.6603 | 0.5694 | 868 | 567 | 746 | 73 | 77 | 19 |
| **pooled** | 707 | 1,154 | **0.3807** | 0.6592 | 0.5776 | 4,538 | 2,977 | 3,661 | | | |
| mean of folds | | | 0.3806 | | | | | | | | |

Across folds the standard deviation is 0.0045 and the range 0.0116. Folds 1 to
4 average 0.3804.

| submission | masks | frames with a prediction | public leaderboard |
|---|---|---|---|
| fold 0 (run *e*) | 1,138 | 175 | 0.32 |
| fold 3 | 1,142 | 176 | 0.33 |
| ensemble of five | 1,135 | 175 | **0.34** |
| fold 1 | 1,201 | 176 | — |
| fold 2 | 1,227 | 176 | — |
| fold 4 | 1,158 | 176 | — |

The previous best submission, from the Phase 3 configuration (fold 0 PQ
0.3756), also scored 0.32.

## Discussion

**Fold 0 was not an easy fold.** The four folds never used for a decision
average 0.3804 against fold 0's 0.3815. Every choice since Phase 1 was read on
fold 0 alone, so this is the first evidence that those readings describe the
data rather than those 142 frames.

**What this does not show is that HRNet beats ResNet-34 across the folds.** The
reference was measured on fold 0 only, and the margin there, +0.0087, is about
twice the spread seen here between folds. The comparison would need the
reference on all five folds.

**The ensemble's gain is readable; the single folds' differences are not.** The
leaderboard shows two decimals and scores about half the test set. 0.34 against
0.32 is twice the display step. 0.33 against 0.32 is one step and within
rounding. The ensemble emits fewer masks than any single model (1,135 against
1,138 to 1,227), which is what averaging should do if the models disagree
mostly on weak detections — fewer false positives. That cannot be checked
locally: together the five models have seen every training frame, so no frame
is held out from the ensemble.

**Fold 3 scores best on every measure, which does not make its model the
best.** The folds differ in their validation frames as well as their models,
and the five models are trained the same way on the same amount of data.

**The rerun question is still open.** Fold 0 was meant to be trained again, to
measure how far one rerun moves 0.3815. The run that produced these numbers was
the version of the notebook that reuses run *e* instead, with its output
attached as an input. The notebook as committed now retrains fold 0.

## Next

- The ensemble of five is the submission to keep selected.
- Two measurements would each settle an open question: the ResNet-34 reference
  on all five folds, for whether the adoption of HRNet holds beyond fold 0; and
  fold 0 retrained, for how much a single run moves.
- The public leaderboard can no longer separate configurations that differ by
  less than about 0.01; the pooled cross-validation score is the measure from
  here.

## Reproduction

| | |
|---|---|
| configuration | `configs/phase6/e_unet_hrnet.yaml`, fold 0 to 4 |
| notebook | `notebooks/70_train_kaggle_cv_ensemble.ipynb` |
| commit of the run | `4bf9027`, with notebook 50's Version 4 output attached so that fold 0 is run *e* |
| commit of the notebook as it stands | `2a6ba0d`, which retrains fold 0 instead |
| environment | Kaggle, two T4s, two folds trained at a time |
