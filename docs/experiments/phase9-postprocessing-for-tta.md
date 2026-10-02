# Phase 9 — The post-processing, chosen again for the eight-view maps

Closed 2026-10-02. **Post-processing at 2048 with the threshold lowered to 0.40
raises the pooled five-fold PQ from 0.3998 to 0.4065 (+0.0067), and every fold
improves.** The submission scores **0.36 on the public leaderboard**, up from
0.35. Nothing was trained, and the sweep itself needed no GPU.

Phase 8 adopted test-time augmentation over the eight flips and quarter turns
(*d4*) but kept the post-processing Phase 3 had chosen: threshold 0.5, minimum
area 400, rejoining within 24 pixels, at 1024. That setting was chosen on a
ResNet-34 without TTA, on fold 0 alone. Two observations said it might no longer
fit: on fold 0 in Phase 8, upsampling the map to 2048 before post-processing
gained 0.0045 on its own; and Phase 3's finding that the threshold does not
matter rested on maps that were almost all zeros and ones, which a mean over
eight views need not be.

## What was done

**Maps.** Each fold's model predicted its held-out frames in the eight views,
and the mean was stored a byte per pixel, one file per fold; the test frames got
the mean over the five models. Re-scored on these stored maps, the current
setting reproduces Phase 8's per-fold PQ to four decimals, so the rounding to
1/255 changes nothing.

**Resolution as a parameter.** `resampled_builder` upsamples a map bilinearly to
the setting's resolution before thresholding, and scales the rejoining distances
and the solar disk with it, so that one setting means the same distance on the
Sun at either resolution. Settings are written in pixels at 1024.

**Two-stage sweep, pooled over the five folds.** One setting takes about 45
seconds per fold, so the full grid was split:

1. resolution (1024, 2048) × threshold (0.25 to 0.60), at minimum area 400 and
   rejoining 24;
2. at the stage 1 choice, minimum area (300 to 600) × rejoining distance (8 to
   32).

Each parameter was taken from the middle of its near-best plateau (within 0.005
of the best), not from the peak. The adoption rule is that of Phase 8: pooled PQ
at least +0.005 over the current setting on the same maps, and at least four of
five folds improving.

**The grid was widened once.** The first pass started the threshold at 0.35 and
the rejoining distance at 16, and the best of each sat at the lowest value
tried. A plateau cut off by the edge of the grid has no middle to take, so both
were extended downwards. The numbers below are from the widened grid; the first
pass would have chosen threshold 0.45 and rejoining 24, at +0.0052 with four
folds of five.

## Results

### Stage 1: resolution × threshold

Pooled PQ (minimum area 400, rejoining 24):

| threshold | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.3964 | 0.3994 | 0.4035 | 0.4021 | 0.4005 | 0.3998 | 0.3980 | 0.3949 |
| 2048 | 0.3972 | 0.4042 | **0.4076** | 0.4063 | 0.4050 | 0.4043 | 0.4007 | 0.3979 |

2048 is better at every threshold. At 2048 the plateau runs from 0.30 to 0.50,
bounded on both sides, and its middle, 0.40, is taken.

Resolution alone (2048 at the current threshold 0.50) gains +0.0045 pooled (SQ
+0.0050, RQ +0.0022), on all five folds (+0.0008 to +0.0110).

### Stage 2: minimum area × rejoining distance

Pooled PQ (2048, threshold 0.40):

| minimum area | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| 300 | 0.4044 | 0.4051 | 0.4056 | 0.4058 | 0.4058 |
| **400** | 0.4056 | 0.4061 | **0.4065** | 0.4063 | 0.4063 |
| 500 | 0.4037 | 0.4041 | 0.4046 | 0.4040 | 0.4037 |
| 600 | 0.3966 | 0.3970 | 0.3974 | 0.3968 | 0.3965 |

The minimum area plateau is 300 to 500 and 400 is kept. The rejoining distance
plateau is the whole range, and 16, its middle, is taken.

### The decision

| setting | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| current (1024, 0.50, 400, 24) | 0.3998 | 0.6679 | 0.5986 | 4,663 | 2,718 | 3,536 |
| **selected (2048, 0.40, 400, 16)** | **0.4065** | 0.6701 | 0.6066 | 4,816 | 2,863 | 3,383 |

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| gain | +0.0029 | +0.0047 | +0.0110 | +0.0033 | +0.0124 |

Pooled +0.0067 (SQ +0.0021, RQ +0.0080) with five folds of five: adopted.

| submission | masks | frames with a prediction | public leaderboard |
|---|---|---|---|
| Phase 8 (d4, current setting) | 1,141 | 175 | 0.35 |
| **Phase 9 (d4, selected setting)** | 1,184 | 175 | **0.36** |

## Discussion

**The gain is detection.** RQ rises by 0.0080: 153 more matches and 153 fewer
misses, at the cost of 145 more false positives. A lower threshold lets the
faint ends of filaments in, and slightly more of what it adds clears IoU 0.5
than does not. Post-processing at 2048 adds the smaller SQ part, consistent with
a mask cut from an upsampled map having a smoother edge.

**The threshold matters now.** Phase 3 found every threshold from 0.3 to 0.7
within 0.001. Here the range from 0.25 to 0.60 spans 0.010 at 2048, with a peak
at 0.35. The maps are still mostly zeros and ones — 0.32% of pixels on fold 0
are at 0.5 or above — but averaging eight views leaves an intermediate band at
the edges of filaments, and that band is where the threshold acts.

**The rejoining distance does not.** Anything from 8 to 32 pixels is within
0.001. Within that range there is rarely a second filament to confuse, the same
reading as Phase 3's.

**Choosing and scoring on the same folds flatters the choice.** The best single
point (threshold 0.35, rejoining 24) scores 0.4076; the plateau rule took a
setting 0.0011 below it. Widening the grid from 24 to 36 settings moved the
chosen setting's gain from +0.0052 to +0.0067, and every fold moves the same
way, so the gain is not one fold's accident. The leaderboard's +0.01 is one
display step and in the same direction.

**The first Kaggle run hung for eleven hours.** Version 2 wrote all the maps in
36 minutes, then sat silent in the sweep until it was stopped at 11 hours 15
minutes of GPU time. The sweep forked worker processes from a notebook that had
used the GPU; forked children of a process with live CUDA, OpenCV or OpenMP
threads can deadlock, and a hang raises nothing, so the fallback to sequential
sweeping never ran. The same sweep runs in about 45 seconds per setting and fold
when each fold is a fresh process (`scripts/sweep_maps.py`), so slowness was
ruled out; the deadlock is the remaining explanation, not a confirmed one. The
maps had already been written and survived the stop, so nothing had to be
predicted again.

## Next

- The Phase 9 submission is the one to keep selected.
- The maps of every fold are now stored, so any further post-processing can be
  swept without a GPU.
- Notebook 80 as committed reproduces this phase in a Kaggle session without an
  accelerator, from Version 2's maps; that run has not been made. The submitted
  file came from the same cells run locally, as recorded below.

## Reproduction

| | |
|---|---|
| maps | Kaggle `Filament-TTA-and-mix` Version 2 output (`maps/`), written at commit `8cb6502` from `Use_HRNet` Version 3's fold models |
| sweep and submission | `notebooks/80_kaggle_tta_and_model_mix.ipynb` at commit `e65c73d`, its cells run locally on those maps without a GPU (stage 1 12.8 minutes; stage 2 about 17 minutes, interrupted once by a time limit and resumed from the folds already written; five folds at once) |
| sweep script | `scripts/sweep_maps.py` |
| submitted file | `submission_selected.csv` of that run (1,184 masks), uploaded directly to the leaderboard |
