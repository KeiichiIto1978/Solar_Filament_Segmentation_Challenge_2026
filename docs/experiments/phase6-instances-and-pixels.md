# Phase 6 — Where the instances come from, and where the pixels come from

Closed 2026-09-24. **Nothing adopted.** Every configuration tried lost to the
one already in place, and the phase's value is in what the failures located.

Phase 5 ended by separating two choices that had been made together: how a
prediction is divided into filaments, and what draws the pixels. This phase
filled the cells.

| | Existing whole-disk map | Crop-trained model | Tile-trained model |
|---|---|---|---|
| **Connected regions** | 0.3756 | not possible | — |
| **Boxes from thresholds** | 0.3761 | — | — |
| **Boxes from a detector** | ceiling 0.42–0.43 | — | — |
| *Boxes from the annotations* | *0.4141* | — | — |

Alongside it, a second question that turned out to matter more: the network
itself. Four were compared at 1024 pixels and three of them lost badly, which
is where the phase found its one durable result.

## What was done

### Instances: boxes instead of connectivity

An instance is currently whatever the thresholded map leaves connected, which
decides *whether a pixel is filament* and *which filament it belongs to* with
one number. A box separates them: read the map generously to find where a
filament is, then read it strictly inside that box to draw it.

Boxes taken from the connected regions of a lower threshold reach **PQ
0.3761** against 0.3756. Sweeping the finding threshold from 0.5 down to 0.1
moves the number of ground truths a box lands on from 1,301 to 1,322 — 21 more
out of 1,795. The map is bimodal: 0.133% of the pixels inside the solar disk
lie between 0.05 and 0.90, so a lower threshold finds almost nothing new.

**This method turned out to be the rejoining step in another form.** Both turn
one dial — how generously pixels are joined — and differ only in whether the
evidence is distance or probability. With the same material, they reach the
same place.

### The ceiling that motivated it was measured wrongly

Phase 5 reported that reading the same maps inside the *annotated* boxes gives
PQ 0.465, and treated the 0.09 above the pipeline as what boxes were worth.
That figure hands each annotator their own box set. **The submission format
allows one set of predictions per image and says nothing about which annotator
will score it**, so no method can deliver it.

Rebuilt with one box set per image — annotators' boxes merged where they
overlap — the same maps give **0.4141**. Scored through the same painting code,
the per-annotator condition reproduces at 0.4858, so the 0.072 between them is
the merge and not the implementation.

The merge rule matters and the one first used was not the best: grouping at IoU
0.3 rather than 0.5 gives 0.4232, while not merging at all gives 0.3615. How a
group is collapsed into one box does not matter at all (0.4129 to 0.4138 for
the mean, the enclosing box, and the largest member).

**So the reachable ceiling for the instance axis is 0.42 to 0.43, not 0.465.**
Against 0.3756 that is a prize of about 0.05, and it requires boxes that are
both nearly complete and nearly exact:

| Box recall | dropped at random | dropped smallest first |
|---|---|---|
| 1.00 | 0.4141 | 0.4141 |
| 0.90 | 0.3974 | 0.4023 |
| 0.80 | 0.3755 | 0.3932 |
| 0.73 | 0.3601 | 0.3814 |
| 0.60 | 0.3317 | 0.3507 |

Break-even against the incumbent sits between 65% and 80% depending on which
filaments a detector misses; a detector that loses the small ones first pays
less, because those are the ones the map answers with nothing anyway. Box
*accuracy* is the harsher constraint: at full recall, displacing each box edge
by 10% of the box's own size costs the whole prize (0.3778), and 20% costs
twice it (0.2873).

### Why the boxes cannot be better than the pixels

Of 1,339 correct boxes, **266 contain no pixel above the drawing threshold, and
250 of those contain no pixel above 0.05**. The median peak probability inside
an empty box is 0.000. The emptiness is concentrated in the small filaments:

| Annotation area | empty / boxes | |
|---|---|---|
| under 400 px | 64 / 138 | **46.4%** |
| 400–800 | 98 / 319 | 30.7% |
| 800–1,600 | 72 / 397 | 18.1% |
| 1,600–3,200 | 23 / 273 | 8.4% |
| over 3,200 | 9 / 212 | 4.2% |

A box is a way of grouping pixels that exist. Where none exist, no box helps,
and the instance axis is capped by the pixel axis rather than the reverse.

Two properties of the boxes themselves were measured while the question was
open. An axis-aligned box around a filament is 24.5% filament at the median
(42.9% if the box may rotate), falling to 17.4% for the largest filaments —
they curve, and a rectangle around an S is mostly background. Boxes of
different filaments, on the other hand, barely overlap: of 7,990 pairs within
an annotator-image only 55 overlap at all, and the worst pair reaches IoU
0.223, so **non-maximum suppression at any usual threshold would delete
nothing**. The representation is loose but unambiguous.

### Pixels: four networks

The baseline is a U-Net with a ResNet-34 encoder because that is where the
project started, not because anything measured said so. Four configurations
were trained for twenty epochs at 1024 pixels, differing in the network and in
nothing else, and scored through the post-processing Phase 3 settled on.

| | decoder / encoder | parameters | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| incumbent | Unet / resnet34, 15 epochs | 24.4M | **0.3756** | 0.6555 | 0.5730 | 968 | 616 | 827 |
| a | Unet / resnet34 | 24.4M | 0.3728 | 0.6551 | 0.5691 | 970 | 644 | 825 |
| b | Unet / convnext_tiny | 31.9M | 0.3422 | 0.6439 | 0.5315 | 946 | 819 | 849 |
| c | UPerNet / convnext_tiny | 37.0M | 0.3256 | 0.6390 | 0.5095 | 886 | 797 | 909 |
| d | Segformer / mit_b2 | 24.7M | 0.3228 | 0.6244 | 0.5170 | 910 | 815 | 885 |

Run *d* trains on batches of two where the others use four: a transformer
compares every patch with every other, and at 1024 a batch of four does not fit
a T4.

Each configuration was then swept over thirty post-processing settings on its
own probability maps, so that a network could not be rejected for a threshold
that did not suit it.

| | at the shared setting | retuned | difference |
|---|---|---|---|
| a | 0.3728 | 0.3760 | +0.0032 |
| b | 0.3422 | 0.3440 | +0.0018 |
| c | 0.3256 | 0.3297 | +0.0041 |
| d | 0.3228 | 0.3260 | +0.0032 |

**No ranking changed.** A minimum area of 400 pixels and a rejoining distance
of 24 won for every one of the four, which is Phase 3's choice reproducing
across four quite different networks. The threshold remains inert: over the
whole grid *a* spans 0.3565 to 0.3760, and from 0.3 to 0.7 at the winning area
and distance it spans 0.003 — less than the 0.005 the same configuration moves
between runs.

## Results

**The reason three of the four lost is a property that could have been checked
in thirty seconds before choosing them.**

| feature map | resnet34 | convnext_tiny | mit_b2 |
|---|---|---|---|
| 1/2 | **64 channels** | **none** | **none** |
| 1/4 | 64 | 96 | 64 |
| 1/8 | 128 | 192 | 128 |
| 1/16 | 256 | 384 | 320 |
| 1/32 | 512 | 768 | 512 |

ConvNeXt and MiT reduce the input to a quarter in a single stride-four step.
ResNet-34 goes through a half first. At 1024 that missing level is 512 by 512,
which is where a filament a few pixels wide lives, and the U-Net decoder's
finest skip connection has nothing to attach to. SQ follows exactly that order:
0.6551 with the level, 0.6439, 0.6390 and 0.6244 without it.

Patchify stems are a deliberate trade in ConvNeXt and in vision transformers,
and they cost nothing for objects tens of pixels across. This is not one of
those problems. **The conclusion is that these three networks lose at 1024, not
that newer designs are worse**: raising the resolution would make their 1/4 as
fine as ResNet-34's 1/2 is now, and the comparison would have to be redone.

The two failures are not the same failure. Run *b* reaches a *higher* training
loss than *a* despite 31% more parameters, which is a limit on what it can
represent rather than on what it can generalise. Run *c* reaches a lower
training loss and a higher validation loss, which is the ordinary one.

**Validation loss does not order these networks by PQ.** Run *b* reaches a
better best validation loss than *a* (0.3659 against 0.3671) and scores 0.031
lower. Across all four, validation loss spans 5% while PQ spans 15%. Dice and
cross-entropy are sums over pixels dominated by the large filaments; PQ counts
each filament once. What separates these networks is the small ones, and the
loss barely sees them.

## Discussion

**What the labels allow.** The same frame is traced by up to three people, the
network is trained against one tracing and validated against another, so part
of the validation loss is not the network's to remove. Measuring the Dice term
on the 62 fold-0 frames with more than one annotator:

| predictor | mean | median |
|---|---|---|
| union of everyone (has seen the answer) | **0.2222** | 0.1973 |
| intersection of everyone | 0.2780 | 0.2481 |
| **the network** | **0.3470** | 0.3255 |
| union of the other annotators | 0.3719 | 0.3583 |
| share of the other annotators | 0.3716 | 0.3550 |
| one other annotator | 0.3750 | 0.3558 |

Three things follow. The network is already better than anything that can be
built from the *other* annotators' tracings, so it has learned something about
the image that the labels alone do not carry. Nothing built from the other
annotators beats a single one of them, because 26% of a person's pixels are
theirs alone. And a predictor that marks everything anyone marked reaches
0.222, so **a validation loss well below today's is reachable, and the
direction is to mark more rather than less.**

That agrees with the metric. A filament *k* of *n* annotators drew, emitted at
IoU *u*, adds `k·u` to the numerator of Panoptic Quality and `0.5·n` to its
denominator against staying silent, so emitting pays when `k·u > 0.5·n·PQ`. At
PQ 0.375 and IoU 0.66 that holds even for a filament one of three people drew
(0.66 against 0.563) — though it stops holding once PQ passes about 0.44, so
the best target changes as the model improves.

**Both axes point at the same filaments.** The instance axis is capped because
266 correct boxes are empty and the emptiness is concentrated below 400 pixels.
The pixel axis lost three networks that cannot see half-resolution detail. The
loss says the reachable improvement is in marking more. Small filaments are
missing, and everything measured this phase is a different view of that.

**On method.** Two of this phase's measurements had to be redone because the
first version answered a question nobody asked: the box ceiling was computed
under a condition the submission format forbids, and the annotator floor was
computed against a predictor — one person's tracing — that is bad by
construction. Both first answers were quotable and wrong in the same way, by
measuring an idealisation rather than something reachable.

## Next

The architecture axis is not exhausted; it was sampled on reputation instead of
on the property that turned out to decide it. Three changes, one at a time,
against run *a*:

- the training target as the union of all annotators, which puts one consistent
  rule into training instead of three conflicting ones
- 2048 pixels at a batch of one, which is the same number of pixels per step as
  the baseline's 1024 at four
- HRNet-W32, which keeps a high-resolution branch through every stage rather
  than reducing and recovering

Whichever of them works is combined afterwards. A union target may only pay
once the resolution is high enough for the filaments it asks for to be visible,
so a failure at 1024 does not settle it.
