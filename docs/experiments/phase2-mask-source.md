# Phase 2 — Choosing the source of the masks

Closed 2026-09-21.

## What was done

Phase 1 left one question unanswered: whether the U-Net was worth its training
cost. Fine-tuning a fold took 92 minutes, and with cross-validation that is
several hours before any decision can be validated. A source of masks whose
retraining was cheap would change how many choices in the project could be
measured rather than assumed, so one was built and compared against the U-Net
on the same 142 frames, the same evaluator and the same default
post-processing.

The candidate was a multi-scale filter bank feeding a per-pixel random forest:
background subtraction, Frangi's vesselness on dark ridges, Hessian
eigenvalues, smoothing, gradient magnitude, local deviation and the normalised
distance from the disk centre, at scales of 2 to 16 pixels -- 23 channels. Like
the U-Net it is supervised, so it can learn what the annotators call a
filament, which no fixed rule can; unlike the U-Net it judges each pixel from
its own neighbourhood statistics alone.

Alongside it, the machinery for the next phase was built: a sweep that takes
probability maps and scores a grid of post-processing parameters. It needs no
GPU and no forward pass, so a few hundred settings cost seconds.

## Results

fold 0 validation, 142 frames / 239 annotator-images / 1,795 ground-truth
filaments:

| Source | PQ | SQ | RQ | TP | FP | FN | ms/frame |
|---|---|---|---|---|---|---|---|
| U-Net, 40 epochs | 0.3454 | 0.6520 | 0.5298 | 1001 | 983 | 794 | 316 |
| U-Net, 15 epochs | **0.3649** | 0.6563 | 0.5561 | 1031 | 882 | 764 | 315 |
| Filter bank, threshold 0.5 | 0.0424 | 0.5806 | 0.0730 | 360 | 7708 | 1435 | 4730 |
| Filter bank, threshold 0.9 | 0.1617 | 0.6221 | 0.2600 | 617 | 2335 | 1178 | 4730 |

The filter bank's threshold was swept from 0.3 to 0.9; PQ rose monotonically
across the whole range and never turned over.

Training the forest on 2.4 M sampled pixels took 11 minutes, not the seconds
estimated from a six-frame trial: a tree's fit grows faster than linearly in
the sample size.

## Discussion

**The filter bank lost on both axes**: fifteen times slower per frame and less
than half the PQ. There is nothing to salvage, and its code was discarded.

**The failure was spatial coherence, not weak features.** SQ was 0.622 against
the U-Net's 0.656 — the masks that matched were nearly as good. RQ was 0.260
against 0.556. Judging each pixel from local statistics alone leaves the
probability map speckled rather than smooth: the body of a filament breaks into
fragments and isolated points appear elsewhere, so connected components
returned 2,952 scored predictions against 1,795 ground truths. Raising the
threshold trims the speckle but also thins the bodies below IoU 0.5, so both
ends of the trade-off stay bad. More local features would not change this;
what is missing is a receptive field, which is what a convolutional encoder
supplies and a per-pixel classifier cannot.

**Frangi's measure did not earn its place.** It was included because filaments
are dark elongated ridges, the shape vessel filters were designed for. It
ranked seventeenth of twenty-three by importance (0.020), while the largest
Hessian eigenvalue at three scales accounted for 0.41 between them. Frangi
combines those same eigenvalues into a single score; the forest preferred them
raw.

**Shortening the schedule beat lengthening it.** Fifteen epochs reached a
better validation loss than forty (0.3579 against 0.3661) in 36 minutes
instead of 92, and a better PQ: 0.3649 against 0.3454. The gain is in
detection — 101 fewer false positives and 30 fewer false negatives — while SQ
barely moved.

**Fragmentation is untouched by training.** 156 ground-truth filaments were
covered by more than one prediction at 15 epochs, against 155 at 40. That is a
property of recovering instances from connected regions, not of how long the
network trains, and it is where the next phase has to work.

## Next

Phase 3: tune the post-processing on the U-Net's probability maps — rejoin
fragments first, then sweep the threshold and the minimum area, then cut
candidates unlikely to clear IoU 0.5. None of it needs a GPU once the maps are
written out.
