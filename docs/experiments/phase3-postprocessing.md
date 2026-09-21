# Phase 3 — Tuning the post-processing

Closed 2026-09-21.

## What was done

Phase 2 left the score sitting at PQ 0.3649 with the post-processing untouched:
threshold 0.5, minimum area 200, and instances taken straight from the
connected regions of the thresholded map. Phase 1 had already established where
the remaining score was — 156 ground-truth filaments were covered by more than
one prediction, against 19 predictions covering more than one filament, and the
count was the same whether the network trained for 15 epochs or 40. Splitting
is a property of recovering instances from connected regions, not of training,
so it had to be addressed after the model.

Two changes were measured, both on the 142 saved probability maps of fold 0's
validation split. Nothing here needs a GPU: one setting scores in 30 seconds on
a CPU, which is what made a few dozen of them affordable.

**A grid over the threshold and the minimum area.** Twenty-five settings.

**A step that rejoins fragments**, inserted between the connected-region
labelling and the area filter. Two regions are merged when they are close, when
their principal axes align, and when one sits near the other's extension —
the three conditions that hold for two pieces of one filament and rarely for
two different ones. Merging is transitive, so a filament broken into three
pieces becomes one instance rather than two overlapping pairs.

## Results

fold 0 validation, 142 frames / 239 annotator-images / 1,795 ground truths:

| Step | PQ | SQ | RQ | TP | FP | FN | Predictions |
|---|---|---|---|---|---|---|---|
| Phase 2 as it stood | 0.3649 | 0.6563 | 0.5561 | 1031 | 882 | 764 | 1121 |
| Minimum area 400 | 0.3699 | 0.6583 | 0.5619 | 981 | 716 | 814 | 996 |
| Plus rejoining, gap 24 | 0.3755 | 0.6550 | 0.5734 | 977 | 636 | 818 | 944 |
| **Adopted** | **0.3756** | 0.6555 | 0.5730 | 968 | 616 | 827 | 928 |

Adopted settings: `threshold 0.5, min_area 400, join_gap 24, join_angle 50,
join_offset 20`.

Each was taken from the middle of its plateau rather than from the peak. The
best single point scored 0.3772; on 142 frames a margin of 0.0016 is not a
result.

| Parameter | Plateau | Adopted |
|---|---|---|
| threshold | 0.3 to 0.7, spread 0.001 | 0.5 |
| min_area | 200, 400 | 400 |
| join_gap | 8 to 32 within 0.003 | 24 |
| join_angle | 20 to 90, spread 0.003 | 50 |
| join_offset | 6 to 32, spread 0.003 | 20 |

## Discussion

**The threshold does not matter.** Every value from 0.3 to 0.7 scores within
0.001. The maps are sharply bimodal — inside the disk the median probability is
0.006 and the 99th percentile is 0.72 — so the same regions survive wherever
the cut is placed. A parameter worth sweeping once and then leaving alone.

**Raising the minimum area is the metric working as designed.** Going from 200
to 400 lost 50 true positives and removed 166 false positives. Under Panoptic
Quality a prediction that misses costs 1.0 in the denominator while silence
costs 0.5, so trading 50 matches for 166 fewer misses is profitable. At 800 it
reverses: 291 true positives go and PQ falls to 0.332.

**Rejoining worked, but not for the reason it was built.** The gap between
fragments is the only condition that matters. Sweeping the angle and off-axis
tolerances over twenty settings left every one of them between 0.3727 and
0.3772, with both plateaus covering the entire range tested — a tolerance of 90
degrees, which admits perpendicular fragments, scores the same as one of 20
degrees. Within 24 pixels there is rarely a second filament to confuse, so
proximity alone is sufficient discrimination. The geometric conditions are kept
in case another fold has frames where two filaments run close together, but on
this one they are inert.

**The area filter has little room left.** Keeping every region and inspecting
what the 400-pixel cut discards: 167 of 1,095 candidates, with a median area of
220 pixels and none above 396. Since 166 of the 1,795 annotated filaments are
themselves under 400 pixels, some of what is discarded is real. A smarter
filter — predicting from shape and confidence whether a candidate will clear
IoU 0.5 — can only act on those 167 regions, which bounds its effect at a few
thousandths of PQ. It was not built.

**What remains is the 616 false positives that survive the filter.** These are
regions of 400 pixels or more that fail to match. Whether they fail because
their boundaries drift past IoU 0.5 or because they mark places no annotator
marked at all decides the next move, and the IoU distribution of the
non-matching predictions has not been looked at yet.

## Next

Freeze this configuration and cross-validate it: train folds 1 to 4 and report
the mean over their validation groups. Fold 0 is excluded from that mean, since
every parameter above was chosen on it.
