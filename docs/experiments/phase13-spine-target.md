# Phase 13 — The annotators' spines as a second target

Closed 2026-10-04. **Not adopted.** Training HRNet-W32 at 1024 with the
hand-drawn spines as a second output channel scores fold 0 PQ 0.4052 against
0.4030 for the same network retrained without them in the same session
(+0.0021, short of the +0.01 one fold needs). It did not do what it was meant
to: the long filaments found only in part became more common, not less (201
against 179). The spines made the map more conservative — fewer false positives
and fewer matches — rather than carrying it further along faint filaments.

Phase 12 traced the largest group of misses to long filaments the model finds
only the darkest third of, and estimated that recovering two thirds of them
would take the pooled PQ past 0.45. Every annotation carries a centre line drawn
by hand from one end of the filament to the other, which the organisers allow
as auxiliary supervision. The hypothesis was that a target running end to end
would teach the shared features where a filament continues.

## What was done

**The target.** Each annotator's spines are drawn at 2048 as lines five pixels
wide and resized to 1024 like the masks, giving a second target channel; the
flips and quarter turns move both channels together. With `spine_weight` zero —
the default — the network has one output and the loss is unchanged, which the
existing tests confirm.

**The network and the loss.** The U-Net gets a second output channel. The loss
is the mask's Dice plus cross-entropy, plus `spine_weight` times the same on the
spine channel; the run used a weight of 1.0. Inference reads only the first
channel, so TTA and post-processing are unchanged. The validation loss includes
the spine term, so the checkpoint is chosen on both tasks together.

**The comparison.** `notebooks/90_train_kaggle_spine.ipynb` trained two runs at
once on two T4s, fold 0, twenty epochs each: the spine run, and run *e*'s
configuration unchanged as the control. Both were scored the way the
submission is made — eight-view TTA, then the Phase 10 post-processing (2048,
threshold 0.40, minimum area 400, rejoining 16, score cut 0.65) — and their
misses were sorted by Phase 12's rule.

## Results

| run | best epoch | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| spine | 15 of 20 | **0.4052** | 0.6634 | 0.6107 | 972 | 416 | 823 |
| control | 13 of 20 | 0.4030 | 0.6647 | 0.6063 | 1,008 | 522 | 787 |
| run *e* (Phase 10, from Phase 9's maps) | 13 | 0.4016 | | | | | |

Without the score cut: spine 0.3999 (TP 1,007, FP 534, FN 788), control 0.3951
(TP 1,042, FP 660, FN 753).

Spine against control: PQ +0.0021, SQ −0.0013, RQ +0.0044. Below the bar.

The misses of fold 0:

| outcome | spine | control | change |
|---|---|---|---|
| unseen | 144 | 137 | +7 |
| below threshold | 166 | 129 | **+37** |
| dropped by area | 52 | 57 | −5 |
| dropped by score | 35 | 34 | +1 |
| too large | 120 | 127 | −7 |
| fragmented | 30 | 52 | **−22** |
| too small | 201 | 179 | **+22** |
| shape | 75 | 72 | +3 |

The control lands +0.0014 from run *e* on fold 0: one retraining, one data point
on the question Phase 7 left open.

## Discussion

**The spines joined fragments but did not extend filaments.** Fragmented misses
fell by 22 and partly covered ones rose by the same 22: pieces the control
predicted separately along one filament came out joined, but still short. Below
threshold rose by 37. What the auxiliary target changed is continuity between
pieces the model already saw, not how far along a faint filament it sees.

**The map became more conservative.** The spine run emits 106 fewer false
positives and 36 fewer matches. That is the profile of a stricter threshold, and
it is where the small PQ gain comes from (RQ +0.0044) — a trade the score cut
already makes after the fact.

**The hypothesis behind it does not hold in this form.** Phase 12 read the
partly covered filaments as the model stopping where the annotators carry on. A
line drawn end to end did not move that stop. Either the faint ends are not
visible enough in the frame at 1024 for any target to make the network find
them, or a five-pixel line weighted equally with the mask is too weak a signal
to change what the mask channel predicts. This run does not separate the two.

**The predicted spines themselves were not examined.** Inference read only the
mask channel. Whether the spine channel extends further than the mask — which
would make it usable to grow masks along predicted spines — is not known.

## Next

- The Phase 10 configuration stays.
- The trained spine model is in the Kaggle output of `Filament-spine` Version 1.
  Reading its spine channel on fold 0 would show whether the predicted spines run
  further than the masks; if they do, growing masks along them is a
  post-processing step that needs no further training.

## Reproduction

| | |
|---|---|
| configurations | `configs/phase13/e_unet_hrnet_spine.yaml` (spine), `configs/phase6/e_unet_hrnet.yaml` (control) |
| notebook | `notebooks/90_train_kaggle_spine.ipynb`, Kaggle `Filament-spine` Version 1 |
| commit of the run | `953259d` |
| environment | Kaggle, two T4s, one run per card; 7,261 seconds in all (training 110.5 minutes each) |
