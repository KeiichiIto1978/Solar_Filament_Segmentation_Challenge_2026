# Phase 4 — Training on the share of annotators who drew each pixel

Closed 2026-09-22. **The change was rejected.**

## What was done

Phase 3 left the score at PQ 0.3756 with the post-processing tuned and nothing
obvious left in it. Four measurements, all free once the probability maps had
been saved, then ruled out the mechanical explanations for why the masks are
rough — SQ 0.6555, against 0.833 between two people outlining the same
filament.

| Measured | Result | Ruled out |
|---|---|---|
| Best SQ reachable while working at 1024 | 0.880 | Resolution is not the main cause; the model reaches 74% of its own ceiling |
| SQ against the threshold | 0.655 to 0.664 across 0.3 to 0.7 | The threshold is inert |
| Annotations broken by downscaling | 20 of 1,795 | The target is not being shredded |
| Coverage against precision on matched pairs | 0.829 against 0.807; dilation only hurts | No systematic bias in mask thickness |

What survived was the disagreement itself. A frame three people annotated
produces three samples carrying three conflicting targets, and the network
averages them on its own. What comes out is not a vote share but an
uncalibrated compromise: 0.133% of the pixels inside the solar disk fall
between 0.05 and 0.90, and a region only one of three people drew is asserted
at 1.0 as firmly as one all three drew.

This phase asked for the share instead — 0, 1/3, 2/3 or 1 — leaving the samples
and their weighting untouched, so that a frame three people annotated still
counts three times as the metric counts it.

The two loss terms had to read the target differently. Cross-entropy is
minimised where the prediction equals the target, so it can teach a share.
Dice cannot: it measures overlap, and on a target of 2/3 it prefers predicting
1.0, scoring 0.802 against 0.670. Left on the share it would drag the output
back to the extremes, so it reads the target through a majority vote instead
and goes on doing what it is there for, which is to stop a structure covering
0.35% of a frame from being drowned by the background. On a target that is
already zeros and ones the majority vote changes nothing, so the loss is
identical to the previous phases' on the 411 frames one person annotated.

The point was not calibration for its own sake. If the output rose with the
number of people who drew a pixel, the threshold would stop being inert and
become a decision with an answer: a filament **k of n** people drew, emitted at
IoU **u**, adds `k·u` to the numerator of Panoptic Quality and `0.5·n` to its
denominator against staying silent, so it pays when `k·u > 0.5·n·PQ`. At PQ
0.40 a filament one of three drew is worth emitting only if it can be outlined
to IoU 0.6, while one two of three drew is worth it at 0.3.

## Results

fold 0 validation, 142 frames / 239 annotator-images / 1,795 ground truths.
The comparison that decides is the first two rows: the same post-processing,
so that what moves is the model.

| | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| Previous model | **0.3756** | 0.6555 | 0.5730 | 968 | 616 | 827 |
| Vote shares, same settings | 0.3642 | 0.6512 | 0.5593 | 979 | 727 | 816 |
| Vote shares, post-processing retuned | 0.3680 | 0.6531 | 0.5635 | 998 | 749 | 797 |

Retuning recovered 0.0038 of the 0.0114 lost, and the best single point of the
27-setting sweep reached 0.3671 — no post-processing rescues it. The
leaderboard returned 0.32, unchanged, which is what a public score displayed to
two decimals can say about a local difference of 0.0076.

**The map did not become a vote share.** The two numbers that separate a
calibrated map from a two-valued one barely moved, and one of them moved the
wrong way:

| | Previous | Vote shares |
|---|---|---|
| Share of marked pixels away from the extremes | 0.207 | 0.222 |
| How far the observed vote share moves across that middle | 0.139 | **0.130** |

The threshold's plateau widened from 0.3–0.7 to 0.2–0.9. It is more inert than
before, not less.

## Discussion

**The loss got worse by painting more, not by finding less.** False positives
rose by 111 while true positives rose by 11. Pixels the model asserted above
0.9 grew by 8 to 13% on every group of frames, and the annotators agreed with
those pixels less often than before: the top bin went from 466,078 pixels at an
observed share of 0.762 to 516,760 at 0.726. The model became confident over
more of what only a minority had drawn.

**It is not confined to the frames whose target changed.** Splitting the
validation frames by how many people annotated them, the frames one person
annotated — whose target is identical under both runs, since the share of one
annotator is that annotator's own tracing — lost 0.0067 PQ and gained 44 false
positives. One network carries whatever it learned into every frame.

| Annotators | Frames | Target changed | PQ | False positives | Confident pixels |
|---|---|---|---|---|---|
| 1 | 80 | no | −0.0067 | +44 | ×1.13 |
| 2 | 27 | grew to a union | −0.0137 | +14 | ×1.11 |
| 3 | 35 | shrank to a majority | −0.0140 | +53 | ×1.08 |

**Nor is it that the target grew.** Binarising the share at 0.5 does turn a
two-annotator frame's target into the union of what either person drew — 1.35
times the area — but a three-annotator frame's becomes a majority, 0.85 times.
Over the training split the two nearly cancel: the area Dice is shown grew by
3%, which does not account for 13% more confident pixels. Cross-entropy's
target mass is unchanged exactly, by construction — averaging n tracings and
handing the average to n samples preserves the mean; only the variation between
samples is gone.

**Why it happened is not established.** Two candidate mechanisms were measured
and neither holds. What remains is a conjecture: in a contested region the old
arrangement had one sample insisting the pixel is filament while two insisted
it is not, and the new one has all three saying 0.33 mildly. The mean is the
same and the gradient is weaker, and a network that can no longer resolve those
boundaries may settle for painting over them. This was not measured, and
chasing it further would not change what to try next.

**The guard rail worked as intended.** The decision was made on the symmetric
comparison, at the previous post-processing settings. Had it been made on the
retuned numbers alone the loss would have looked smaller, and had the
post-processing been tuned first it would have been unclear whether the model
or the tuning moved.

## Next

The configuration is unchanged: the Phase 3 model and settings remain the best,
at local PQ 0.3756 and 0.32 on the leaderboard.

Mask quality is still where the score is. Three candidates, none started:

- **A loss that penalises a broken structure.** clDice and the topology-aware
  losses are built for thin tubular structures that fragment, which is what 155
  split ground truths are. Cheap to implement and published evidence exists on
  vessels and roads.
- **Training and predicting at the native 2048.** This lifts the ceiling from
  0.880 to 1.0, but the model reaches only 74% of the ceiling it already has,
  and tiling costs the network its view of the whole disk.
- **A loss shaped like the metric.** Match predictions to annotations without
  gradients, as detection models do, then compute a differentiable score over
  the matched pairs. The most ambitious of the three, and the only one that
  could act on the IoU 0.5 cliff directly.
