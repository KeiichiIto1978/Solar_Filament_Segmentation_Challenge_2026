# Phase 10 — Where the false positives come from, and a cut on the score

Closed 2026-10-03. **Dropping every predicted filament whose score — the mean
probability over its region — is below 0.65 raises the pooled five-fold PQ from
0.4065 to 0.4127 (+0.0062), and every fold improves.** The public leaderboard
still shows 0.36, two decimals, but the submission ranks above Phase 9's, which
it can only do with a higher unrounded score. Nothing was trained and no GPU was
used: everything here reads the probability maps stored in Phase 9.

Phase 9 left 2,863 false positives against 4,816 true positives. Each instance
already carried a score, but only to order the instances; the cut-off on it,
which the metric's break-even argument of Phase 0 calls for alongside the
minimum area, had never been tried. And Phase 3 had left open whether the false
positives are near misses — a mask problem — or detections nobody drew — a
detection problem.

## What was done

**One row per prediction and annotator-image.** `prediction_outcomes` applies
the metric's own matching (a pair counts strictly above IoU 0.5) and records,
for every prediction in every annotator-image it is scored in, whether it
matched, the best IoU it reached, how many filaments it matched and the sum of
those IoUs. The last two make the rows add back up to the metric exactly — one
prediction can clear IoU 0.5 against two overlapping tracings by the same
annotator, which the metric counts as two true positives. Rebuilt from the rows,
Phase 9's result comes back unchanged: PQ 0.4065, TP 4,816, FP 2,863, FN 3,383.
`compute_pq` itself is unchanged.

`scripts/fp_breakdown.py` adds what is known about each prediction before
scoring: its score, its peak probability, its area and its distance from the
centre of the disk.

**A score cut.** `extract_instances` takes a `min_score`, default zero (no
cut). Because predictions never overlap and a match needs IoU above 0.5, no
filament is matched by two predictions, and dropping one prediction changes no
other row; PQ after any cut can therefore be computed from the rows alone. The
cut was swept both ways — rebuilt from the rows and re-scored from the maps —
and the two agree at every value.

## Results

### The false positives

Phase 9 setting (2048, threshold 0.40, minimum area 400, rejoining 16), five
folds:

| best IoU with that annotator's filaments | matched by no annotator | matched by another annotator of the frame | total |
|---|---|---|---|
| below 0.1 | 821 | 251 | 1,072 |
| 0.1 to 0.3 | 499 | 131 | 630 |
| 0.3 to 0.4 | 330 | 138 | 468 |
| 0.4 to 0.5 | 406 | 287 | 693 |
| total | 2,056 | 807 | 2,863 |

Of 4,715 predictions, 1,404 match no annotator and 2,675 match every annotator
of their frame.

How well each quantity separates matched from unmatched rows (area under the
ROC curve): score 0.738, area 0.647, peak probability 0.628. Only 1.4% of the
false positives lie beyond 0.9 of the disk radius.

### The cut

Pooled over five folds, re-scored from the maps:

| minimum score | PQ | SQ | RQ | TP | FP | FN | predictions |
|---|---|---|---|---|---|---|---|
| none | 0.4065 | 0.6701 | 0.6066 | 4,816 | 2,863 | 3,383 | 4,715 |
| 0.50 | 0.4080 | 0.6702 | 0.6088 | 4,808 | 2,788 | 3,391 | 4,663 |
| 0.55 | 0.4096 | 0.6705 | 0.6109 | 4,791 | 2,696 | 3,408 | 4,606 |
| 0.60 | 0.4110 | 0.6710 | 0.6126 | 4,749 | 2,556 | 3,450 | 4,489 |
| **0.65** | **0.4127** | 0.6716 | 0.6145 | 4,698 | 2,394 | 3,501 | 4,352 |
| 0.70 | 0.4121 | 0.6721 | 0.6132 | 4,588 | 2,178 | 3,611 | 4,141 |
| 0.75 | 0.4110 | 0.6730 | 0.6107 | 4,466 | 1,961 | 3,733 | 3,931 |
| 0.80 | 0.4069 | 0.6750 | 0.6027 | 4,231 | 1,609 | 3,968 | 3,561 |
| 0.85 | 0.3771 | 0.6826 | 0.5524 | 3,542 | 1,083 | 4,657 | 2,836 |

Cuts of 0.40 and 0.45 change nothing, since every instance's mean is above the
threshold of 0.40. The plateau runs from 0.50 to 0.75, bounded on both sides;
its middle, 0.65, is also the peak.

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| gain | +0.0046 | +0.0081 | +0.0058 | +0.0079 | +0.0046 |

Pooled +0.0062 (SQ +0.0015, RQ +0.0079) with five folds of five: adopted. A cut
on the peak probability instead does best at 0.80, at 0.4111 (+0.0046).

| submission | masks | frames with a prediction | public leaderboard |
|---|---|---|---|
| Phase 9 | 1,184 | 175 | 0.36 |
| **Phase 10 (score cut 0.65)** | 1,084 | 175 | **0.36**, ranked above Phase 9's |

## Discussion

**The cut works on detection, as the metric's break-even predicts.** It removes
469 false positives and 118 true positives. A false positive costs 0.5 in the
denominator; a true positive lost costs its IoU, about 0.67, in the numerator
and adds 0.5 to the denominator. Below a score of about 0.65 the predictions are
unlikely enough to match that dropping them pays; above it they are not.

**More than a quarter of the false positives are out of the model's reach.** 807
of 2,863 are predictions that another annotator of the same frame did draw.
Removing one loses a true positive elsewhere, so no cut and no better mask can
take them away. This is a floor under the false positive count for this data.

**Most of the rest are detections, not masks.** 1,072 false positives overlap
their nearest filament by less than IoU 0.1, against 693 near misses between
0.4 and 0.5, of which 287 are themselves matches for another annotator. A better
mask could rescue at most the 406 near misses no annotator matched. Improving
detection — deciding whether a candidate is a filament at all — has more to
gain than refining shapes.

**The score separates, but not cleanly.** At an area under the curve of 0.74,
many false positives score as high as true ones: the median unmatched row scores
0.819 against 0.879 for a matched one. A cut at 0.65 takes only the low tail. A
classifier on more than the mean probability is the natural next step if
detection is to be improved without retraining.

**Choosing and scoring on the same folds flatters the choice**, as in Phase 9.
The grid here has one parameter, the plateau is bounded on both sides, its
middle coincides with the peak, and all five folds move the same way, so the
flattery is small.

## Next

- The Phase 10 submission is the one to keep selected.
- A filter that decides from several features — score, area, shape, position —
  whether a candidate will match. The rows written by `fp_breakdown.py` are its
  training data, and the five folds give it a held-out estimate.

## Reproduction

| | |
|---|---|
| maps | Kaggle `Filament-TTA-and-mix` Version 2 output (`maps/`), as in Phase 9 |
| breakdown | `scripts/fp_breakdown.py` per fold at the Phase 9 setting, commit `7d4f9a0` |
| sweep | `scripts/sweep_maps.py` per fold over `min_score` 0.40 to 0.85 in steps of 0.05, commit `7d4f9a0`, run locally without a GPU |
| submission | `scripts/submit_from_maps.py` on `maps/test.npz` with the Phase 9 setting and `min_score` 0.65; uploaded directly. The same script reproduces Phase 9's submission byte for byte |
