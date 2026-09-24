# Phase 5 — Charging for a filament predicted as two, and what it exposed

Closed 2026-09-22. **The change was rejected, and the phase ended somewhere
else entirely.**

## What was done

155 of fold 0's 1,795 annotations were covered by more than one prediction: one
filament, found, but delivered in pieces. Panoptic Quality charges for that
twice over — the annotation is a false negative and each piece a false positive
— and the count had not moved since the first baseline, nor between a fifteen
and a forty epoch schedule. It is not something more training fixes.

Neither loss in use could see it. Both are sums over pixels, and the few pixels
that would have joined two pieces are a rounding error against a frame where
filaments cover 0.35%. Two predictions losing the same area, one from an edge
and one from the middle, score identically under Dice; only one has broken a
filament in half.

**clDice** (Shit et al., CVPR 2021) compares each mask against the *skeleton* of
the other, so a break costs the stretch of skeleton it leaves uncovered rather
than the pixels in the gap. It was built for vessels, neurons and roads —
structures with the same shape and the same failure — and it was chosen over
two alternatives because it is cheap and because someone else had already
demonstrated it.

Two details were settled before training. The skeleton is peeled by repeated
opening, and peeling has to reach the middle of the thickest structure: a solid
shape survives being opened, so the skeleton stays *empty* until erosion wears
it to a line, and an empty skeleton scores a perfect clDice. On a synthetic bar
of radius ten, a break costs exactly zero at eight iterations and 0.050 at ten.
Fold 0's filaments reach a radius of 20.6 at 1024 pixels, so the reference
implementation's default of ten is not enough here and 25 was used.

The term took its weight from Dice rather than adding to it, following the
paper's `(1-a)·Dice + a·clDice`, to leave the total weight on overlap where it
was.

## Results

fold 0 validation, 142 frames / 239 annotator-images / 1,795 ground truths, at
the post-processing the previous run was tuned with.

| | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| Previous model | **0.3756** | 0.6555 | 0.5730 | 968 | 616 | 827 |
| Centreline loss | 0.3537 | 0.6453 | 0.5480 | 973 | 783 | 822 |
| Centreline, post-processing retuned | 0.3534 | 0.6510 | 0.5429 | 905 | 634 | 890 |

The leaderboard returned 0.32, unchanged — a public score shown to two decimals
cannot resolve a local difference of 0.02 in this range.

**The loss did what it was built to do.**

| | Previous | Centreline |
|---|---|---|
| Annotations split across predictions | 155 | **81** |
| Predictions covering several annotations | 19 | **66** |
| Joining tolerance chosen by the sweep | 24 px | anything, including 0 |

Splitting nearly halved — the first time any change moved that count — and the
post-processing step that rejoins fragments became unnecessary: every gap from
0 to 32 pixels scored within 0.003. The work moved from the post-processing
into the model.

It cost more than it gained. False positives rose by 167 while true positives
rose by 11.

## Discussion

**Splitting and fusion are one knob, not two.** Instances are whatever the
thresholded map leaves connected, so the only thing a pixel loss can change is
how generously pixels connect. Turn it up and fewer filaments break; turn it up
and more neighbours merge. clDice turned it up: 74 fewer splits, 47 more
fusions. No pixel-level loss improves both, because the grouping rule cannot
tell the two situations apart.

**The extra false positives are not fusion.** Of the 167, ninety touch no
annotation at all, and the masks are not fatter — the median instance grew 3%
while the number of instances grew 10%. clDice's skeleton is proportional to a
filament's *length* rather than its area, so it weighs filaments more equally
than Dice does, and the model answered that pressure by asserting more
structures. Mostly wrong ones. Halving the Dice weight may share the blame;
that was not tested, because even removing all ninety would leave PQ at 0.363.

**Then the phase turned.** If grouping is the knob, how much is the knob worth?
Merging, for every annotation the pipeline missed, all the predictions that
touch it and rescoring: **24 of 794 recovered, PQ 0.3756 to 0.3851**. Regrouping
the pixels already predicted is worth a hundredth. The masks are wrong
independently of how they are grouped.

**And then it turned again.** The same probability maps, read only inside each
annotation's own box — no new model, no retraining, the decisions the
post-processing was making set aside because the box makes them unnecessary:

| | Pipeline | Inside the box |
|---|---|---|
| Mean IoU per annotation | 0.4382 | **0.5191** |
| Share reaching IoU 0.5 | 53.9% | **68.6%** |
| Mean IoU of those that do | 0.6555 | 0.6774 |

In Panoptic Quality that is **0.465 against 0.3756**. The same pixels.

What moved is which filaments clear the gate, not how well any of them is
drawn — the mean among those clearing rose by 0.022 and the share clearing rose
by fifteen points. The gain is concentrated in the small filaments the
pipeline's minimum-area filter discards: annotations under 400 pixels go from
15.7% clearing to 45.8%.

The filter is not a mistake. Without knowing where a filament is, a small blob
is probably noise, and under this metric emitting one that misses costs twice
what staying silent does. The filter was standing in for information the
project did not have. **A box supplies that information, and every compensating
rule downstream of it becomes unnecessary at once.** That is one mechanism, not
three, which also means the whole 0.09 rests on the boxes being right — a wrong
box with the filters switched off is worse than no box at all.

**What the phase did not settle.** Whether training at a higher resolution
would draw better masks is untouched: every measurement here reads the same
1024 map a different way. The ceiling that resolution imposes is known — sending
the annotations to 1024 and back costs them a mean IoU of 0.880 — and the model
reaches 74% of it, but nothing says what a model trained at native resolution
would reach. That needs a training run, and it did not happen because the box
result made a cheaper question more urgent.

## Next

Where the instances come from, and where the pixels come from, are independent
choices. Only two cells of the grid have numbers in them.

| | Existing whole-disk map | Crop-trained model | Tile-trained model |
|---|---|---|---|
| **Connected regions** | 0.3756 | not possible | — |
| **Boxes from thresholds** | — | — | — |
| **Boxes from a detector** | — | — | — |
| *Perfect boxes* | *0.465* | — | — |

Phase 6 fills it in and submits the best. The cheapest cell needs no training
at all: boxes taken from the connected regions of a lower threshold, masks read
from the map already saved.
