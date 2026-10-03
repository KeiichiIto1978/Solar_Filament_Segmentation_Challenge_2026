# Phase 11 — Judging each candidate from several features

Closed 2026-10-03. **Not adopted.** A classifier that judges each candidate from
eleven features — its probability, shape, position and how dark the frame is
where it lies — beats the single score cut of Phase 10 by only +0.0018 pooled
(0.4127 → 0.4145), short of the +0.005 the adoption rule requires. The score cut
stays. What the experiment shows is that the evidence a candidate carries
beyond its mean probability is small: the classifier ranks matches above misses
with an area under the ROC curve of 0.758, against 0.738 for the score alone.

Phase 10 found that most false positives are detections no annotator drew
rather than near misses, so deciding whether a candidate is a filament at all
had more to gain than refining its shape. The score separated matches from
misses only moderately. This phase asked whether more of what is known about a
candidate before scoring would separate them better.

## What was done

**Features.** For every candidate of the Phase 9 setting (2048, threshold 0.40,
minimum area 400, rejoining 16, no score cut), `filament.postprocess.candidates`
computes, at the submission resolution:

| group | features |
|---|---|
| probability | mean (the Phase 10 score), peak, standard deviation, share of pixels at 0.8 or above |
| size and place | area; distance of the centroid from the disk centre, as a fraction of the radius |
| shape | ratio of the long to the short axis of the second moments; share of the bounding box; perimeter squared over 4π times the area |
| the frame | mean brightness inside minus that of a 15-pixel ring around it (on the disk only), over the ring's — negative where the frame is darker, as a filament is in H-alpha |
| the frame's other candidates | how many candidates the frame has |

**Classifier.** A gradient-boosted tree (`HistGradientBoostingClassifier`,
scikit-learn, added as a dependency for this) predicts whether a candidate
matches, one training row per candidate and annotator-image. Its settings were
fixed before the run and not tuned, since every setting tried on these folds is
one more choice made on the data it is scored on.

**Cross-fitting.** The classifier that scores fold *k* is trained on the other
four folds only. Each fold's candidates come from a segmentation model that
never saw that fold's frames, so the features are those the test frames will
have, and fold *k* is held out from both models. Only the probability cut on the
classifier's output is chosen on the folds it is scored on, by the plateau rule
of Phases 9 and 10.

**Scoring.** Removing a candidate changes no other candidate's outcome, so every
filter is scored exactly from the outcome rows of Phase 10
(`pq_from_outcomes`, now in `filament.metrics.pq` and tested against
`compute_pq`). The score cut at 0.65 rebuilt this way gives Phase 10's 0.4127
exactly.

## Results

Pooled over five folds, the classifier's probability cut:

| cut | PQ | SQ | RQ | TP | FP | FN | kept |
|---|---|---|---|---|---|---|---|
| none | 0.4065 | 0.6701 | 0.6066 | 4,816 | 2,863 | 3,383 | 4,715 |
| 0.10 | 0.4093 | 0.6701 | 0.6108 | 4,808 | 2,737 | 3,391 | 4,636 |
| 0.15 | 0.4111 | 0.6703 | 0.6133 | 4,784 | 2,619 | 3,415 | 4,545 |
| 0.20 | 0.4133 | 0.6708 | 0.6161 | 4,744 | 2,457 | 3,455 | 4,424 |
| 0.25 | 0.4141 | 0.6714 | 0.6169 | 4,692 | 2,321 | 3,507 | 4,303 |
| **0.30** | **0.4145** | 0.6718 | 0.6169 | 4,612 | 2,140 | 3,587 | 4,142 |
| 0.35 | 0.4126 | 0.6729 | 0.6131 | 4,490 | 1,957 | 3,709 | 3,954 |
| 0.40 | 0.4105 | 0.6733 | 0.6097 | 4,377 | 1,781 | 3,822 | 3,773 |
| 0.45 | 0.4060 | 0.6744 | 0.6020 | 4,219 | 1,599 | 3,980 | 3,564 |
| 0.50 | 0.3991 | 0.6758 | 0.5905 | 4,025 | 1,409 | 4,174 | 3,339 |

The plateau runs from 0.15 to 0.40 and its middle, 0.30, is also the peak.

| | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| Phase 10: score cut 0.65 | 0.4127 | 0.6716 | 0.6145 | 4,698 | 2,394 | 3,501 |
| classifier, cut 0.30 | 0.4145 | 0.6718 | 0.6169 | 4,612 | 2,140 | 3,587 |

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| gain over the score cut | +0.0036 | +0.0018 | +0.0017 | −0.0001 | +0.0017 |

Pooled +0.0018 with four folds of five: below the +0.005 bar. Not adopted.

What the classifier leans on — the loss in area under the ROC curve on the
held-out fold when one feature is shuffled, averaged over the five folds:

| score | compactness | brightness contrast | radial position | spread | elongation | frame count | confident share | area | extent | peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.161 | 0.022 | 0.017 | 0.016 | 0.011 | 0.004 | 0.004 | 0.004 | 0.003 | 0.002 | 0.000 |

Validation candidates come from one fold model each and test candidates from
the mean of five, so their distributions could differ. They barely do: the
10th, 50th and 90th percentiles of the score are 0.679 / 0.866 / 0.913 on
validation and 0.670 / 0.861 / 0.910 on test, and every other feature and the
classifier's output agree as closely. The same holds for Phase 10's score cut,
which rests on the same assumption.

## Discussion

**Almost everything the classifier knows is already in the score.** Shuffling
the score costs 0.161 of area under the curve; every other feature costs 0.022
or less. Together they lift the ranking from 0.738 to 0.758, and the PQ by
0.0018, less than one retraining moves a fold (0.005 to 0.01).

**The frame's darkness says little the model has not already said.** Candidates
are darker than their surroundings almost without exception — the median
contrast is −0.126 and the 90th percentile −0.091 — whether or not an annotator
drew them. The model has learnt to find dark elongated structures; whether a
given one counts as a filament is what the annotators disagree on, and that is
not written in its brightness either.

**What remains is the model.** A filter after the fact can only rearrange the
model's own beliefs. Phase 10 took the low-confidence tail; this phase shows
that the rest of the false positives look, by every measure available here,
like the true positives. Further gains in detection have to come from what the
segmentation model learns, not from judging its output.

## Next

- The Phase 10 submission stays selected.
- The feature module, the cross-fitted evaluation and `pq_from_outcomes` stay
  in the repository: any future candidate-level decision can be scored with them
  in minutes and without a GPU.

## Reproduction

| | |
|---|---|
| maps | Kaggle `Filament-TTA-and-mix` Version 2 output (`maps/`), as in Phases 9 and 10 |
| features | `scripts/candidate_filter.py features` per fold and for the test frames, at the Phase 9 setting |
| evaluation | `scripts/candidate_filter.py evaluate --baseline-score 0.65`, run locally without a GPU |
| commit | the commit that adds this record |
