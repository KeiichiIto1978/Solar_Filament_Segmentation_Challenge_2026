# Phase 0 — Evaluation foundation

Closed 2026-09-21. No model was trained in this phase.

## What was done

Built the instruments needed to measure Panoptic Quality locally, so that
experiments would not be rationed by the five daily Kaggle submissions:

- a PQ evaluator that returns SQ, RQ, TP, FP, FN and the per-match IoU and
  Dice distributions, not a single number;
- an overlap checker, because Kaggle rejects a submission whose masks of one
  image share a pixel and the rejected attempt still costs a daily slot;
- a frozen 5-fold split, grouped by image stem and stratified by GONG station;
- RLE encoding and submission writing;
- an exploratory analysis of the annotations and the frames.

The split groups by stem because up to three annotators describe the same
image and the metric scores a prediction once per annotator. If two annotators
of one image landed on opposite sides of a split, validation would be scoring
an image the model had trained on.

## Results

| Check | Outcome |
|---|---|
| Ground truth fed back as the prediction | PQ 1.000, FP 0, FN 0 |
| A prediction at IoU 0.4903 | TP 0, FP 1, FN 1 |
| A prediction at IoU 0.5106 | TP 1, FP 0, FN 0 |
| Overlap check, 180 clean frames | 0.006 s |
| One annotator's drawing scored against the others, fold 0 | PQ 0.7295 |
| The same, restricted to multi-annotator frames | PQ 0.5890 |

Filament area: median 1,228 px of 4.2 M, 10th percentile 410, 90th 4,685.
Filaments per annotator-image: median 7, maximum 26, never zero.
Annotators of the same frame differ by 2.58 filaments on average.

## Discussion

**A near miss costs more than silence.** At IoU 0.49 a prediction raises both
a false positive and a false negative, adding 1.0 to the denominator, while
predicting nothing raises only the false negative and adds 0.5. Emitting a
candidate pays only when its chance of clearing IoU 0.5, times the IoU it
would then reach, exceeds about half the current PQ. Near PQ 0.4 and a typical
IoU of 0.7 that is roughly a 29% chance. Confidence and minimum-area cut-offs
are therefore parameters to optimise against PQ directly, not afterthoughts.

**Human agreement is the ceiling.** One annotator's work scores PQ 0.73
against the others, and 0.59 on the frames more than one person annotated.
A target of PQ 0.40 is 55% of that ceiling, which is a different proposition
from 40% of a perfect score.

**Frames vary more than stations do.** Station medians for disk brightness
span 107-115 grey levels, while single frames within one station span about 15.
The correction that matters is therefore per frame, not per station.
Stratifying the folds by station stays as cheap insurance.

## Next

Phase 1: a semantic U-Net over the whole disk, with instances recovered from
connected regions. Target: local PQ 0.25 on fold 0.
