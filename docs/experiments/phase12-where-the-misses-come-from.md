# Phase 12 — Where the misses come from

Closed 2026-10-03. **Nothing adopted; this phase measures.** Of the 3,501
annotated filaments the current configuration misses, about 60% are
filaments the model does not see, or sees only a third of: 1,146 have no
probability above the threshold anywhere inside them, and 953 more are covered
by a candidate lying almost entirely inside them that reaches about a third of
their area. Only 10% were found and then thrown away by the post-processing,
and re-tuning the threshold together with the score cut gains at most +0.0017.
**What remains to be won is in what the model sees, not in how its output is
cut.**

Phases 10 and 11 took the false positives as far as post-processing could. The
largest term of the score is now the false negatives (3,501 against 2,394 false
positives), and the remedies for them differ entirely by cause: training for
filaments the model cannot see, post-processing for those it sees but loses,
grouping for those it fuses or breaks.

## What was done

Every annotated filament, once per annotator — the unit PQ counts false
negatives in — was traced through the pipeline of Phase 10 (2048, threshold
0.40, minimum area 400, rejoining 16, score cut 0.65) on the stored eight-view
maps of all five folds. Candidates were produced once with the area and score
cut-offs off; since those act on each region alone, the submitted predictions
are the candidates that pass both, and every miss can be attributed
(`filament.metrics.misses.classify_misses`, tested on constructed cases):

| outcome | rule |
|---|---|
| unseen | highest probability inside the filament below 0.05 |
| below threshold | highest probability between 0.05 and the threshold |
| dropped (area / score) | a candidate cleared IoU 0.5 but was removed by that cut-off |
| too large | the most-overlapping candidate covers at least half the filament, but less than half of it is filament |
| fragmented | it covers less than half, but all candidates together cover at least half |
| too small | it covers less than half, and so do all candidates together |
| shape | it covers at least half and is at least half filament, yet IoU stays at or below 0.5 |

The classification reproduces the score exactly: 4,698 matched and 3,501
missed, Phase 10's true positives and false negatives.

## Results

### Why the misses happen

| outcome | missed | share |
|---|---|---|
| unseen | 545 | 15.6% |
| below threshold | 601 | 17.2% |
| dropped by area | 225 | 6.4% |
| dropped by score | 118 | 3.4% |
| too large | 488 | 13.9% |
| fragmented | 229 | 6.5% |
| **too small** | **953** | **27.2%** |
| shape | 342 | 9.8% |

How the most-overlapping candidate sits on the filament (10th / 50th / 90th
percentile):

| outcome | share of the filament it covers | share of it inside the filament |
|---|---|---|
| matched | 0.66 / 0.86 / 0.97 | 0.63 / 0.80 / 0.93 |
| too small | 0.10 / 0.31 / 0.47 | 0.75 / 0.95 / 1.00 |
| too large | 0.70 / 0.92 / 0.99 | 0.17 / 0.36 / 0.48 |
| fragmented | 0.31 / 0.40 / 0.48 | 0.67 / 0.84 / 0.95 |

A "too small" miss is a candidate lying inside the filament and covering about
a third of it. On the rest of such a filament the median probability is 0.013,
and only about a third of those pixels reach 0.1: the model has not seen the
remainder, faintly or otherwise.

### Who drew them

In frames with more than one annotator, 2,033 filaments were missed. 1,184 of
them another annotator also drew (a tracing of theirs overlaps at IoU 0.3 or
more); 849 were drawn by one annotator alone. A filament only one annotator
drew is missed 82.7% of the time, against 29.9% for one that others drew too.

### By size

| annotated area (px) | filaments | missed | miss rate | unseen | below threshold | dropped by area | too large | fragmented | too small | shape |
|---|---|---|---|---|---|---|---|---|---|---|
| under 400 | 781 | 672 | 86.0% | 179 | 154 | 138 | 108 | 0 | 70 | 18 |
| 400–800 | 1,838 | 931 | 50.7% | 175 | 220 | 87 | 173 | 1 | 137 | 79 |
| 800–1,600 | 2,346 | 808 | 34.4% | 131 | 133 | 0 | 132 | 26 | 244 | 99 |
| 1,600–3,200 | 1,779 | 564 | 31.7% | 40 | 60 | 0 | 60 | 69 | 240 | 84 |
| over 3,200 | 1,455 | 526 | 36.2% | 20 | 34 | 0 | 15 | 133 | 262 | 62 |

(Dropped by score is spread thinly across sizes and omitted from the table.)

### Re-tuning the cut-offs together

The threshold was chosen in Phase 9, before the score cut existed. Sweeping both
together (threshold 0.25–0.40 × score cut 0.55–0.75, re-scored from the maps):

| threshold \ score cut | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 |
|---|---|---|---|---|---|
| 0.25 | 0.4071 | 0.4063 | 0.4055 | 0.4006 | 0.3896 |
| 0.30 | 0.4120 | 0.4126 | 0.4121 | 0.4103 | 0.4049 |
| 0.35 | 0.4136 | **0.4143** | 0.4138 | 0.4133 | 0.4088 |
| 0.40 | 0.4096 | 0.4110 | *0.4127* | 0.4121 | 0.4110 |

The best combination gains +0.0017, with two folds of five going down. The
current setting stays.

## Discussion

**Most misses are the model's.** Unseen, below threshold and too small together
are 2,099 of 3,501. In all three the probability is simply not there on most of
the filament. The small filaments are the extreme case — 86% of those under 400
pixels are missed, 333 of them never reaching the threshold — but the largest
single group is long filaments found only in part: "too small" grows with size
and is the main cause of misses above 800 pixels. The model finds the darkest
stretch of a filament and stops where the annotators carry on.

**The post-processing has little left.** Area and score cut-offs remove 343
filaments that would have matched, but Phases 9 and 10 chose those cut-offs on
the trade-off with the false positives they remove, and the joint sweep above
confirms that no nearby setting does better. Grouping failures — too large and
fragmented, 717 together — are real, but rejoining distance was flat from 8 to
32 pixels in Phase 9, so they are not a matter of one dial either.

**A sizable part is out of reach.** 849 misses are filaments only one of
several annotators drew; any prediction there is a false positive for the
others. The same structure appeared on the false-positive side in Phase 10
(807).

**For the next GPU hours** the measurements point at training rather than
post-processing. What to change in training is not decided by this phase; it
shows where the model's maps fall short — on faint stretches of long filaments
and on small ones — which is what a training change would have to move.

## Reproduction

| | |
|---|---|
| maps | Kaggle `Filament-TTA-and-mix` Version 2 output (`maps/`), as in Phases 9 to 11 |
| setting | Phase 10: resolution 2048, threshold 0.40, minimum area 400, rejoining 16, score cut 0.65 |
| breakdown | `scripts/miss_breakdown.py` per fold |
| joint sweep | `scripts/sweep_maps.py` per fold, threshold {0.25, 0.30, 0.35, 0.40} × score cut {0.55 … 0.75} |
| environment | locally, without a GPU |
