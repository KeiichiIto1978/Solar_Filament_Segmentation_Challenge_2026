# Phase 6b — The training target, the resolution, and the encoder

Closed 2026-09-26. **HRNet-W32 at 1024 (run *e*) becomes the working
configuration: fold 0 PQ 0.3815**, against 0.3756 for the incumbent and 0.3728
for the reference it was measured against. It does not clear the +0.01 bar on
its own, and why it wins is only partly explained; both are stated below rather
than left out.

Phase 6 closed the network axis at 1024 with nothing adopted, and located why
three of four networks lost: ConvNeXt and MiT expose no feature map at half
resolution. Its last section proposed three changes, one at a time, against
run *a* (U-Net on ResNet-34, per-annotator targets, 1024, batch 4, twenty
epochs). This phase ran them, and then one combination.

## What was done

All runs are scored on fold 0 (142 frames, 239 annotator-images, 1,795
filaments) through the post-processing Phase 3 settled on: threshold 0.5,
minimum area 400, rejoining within 24 pixels. The rejoining distances are
counted in pixels of the probability map, so they were scaled with the map —
36 at 1536, 48 at 2048 — to mean the same distance on the Sun. The minimum
area is already in pixels of the full frame.

| run | changes against *a* | why |
|---|---|---|
| g | training target: the union of every annotator of the frame | one consistent rule instead of up to three conflicting tracings |
| f | 2048 pixels at batch 1 | nothing thrown away before the network sees it; same pixels per step as *a* |
| e | encoder: HRNet-W32, batch 2 | keeps a half-resolution branch through every stage |
| h | *e* at 1536, batch 1 | *e* and *f* won on different parts of PQ, so they were stacked |

Run *g* changes only what the training side is asked for. Validation loss and
PQ are still measured against each annotator's own tracing, so its number sits
beside the others. Runs *e* and *h* use smaller batches because a larger one
does not fit a T4: HRNet at 1024 holds two frames, and at 1536 one.

Runs *g*, *f* and *e* were trained in one session, *f* on one card and *g* then
*e* on the other, so their training times reflect sharing the CPU and are not
comparable with Phase 6's. Run *h* was trained alone.

## Results

| run | configuration | PQ | SQ | RQ | TP | FP | FN | fused | split |
|---|---|---|---|---|---|---|---|---|---|
| incumbent | resnet34, 15 epochs | 0.3756 | 0.6555 | 0.5730 | 968 | 616 | 827 | 19 | 155 |
| a | resnet34, 20 epochs | 0.3728 | 0.6551 | 0.5691 | 970 | 644 | 825 | 54 | 77 |
| **e** | **HRNet-W32, batch 2** | **0.3815** | 0.6542 | **0.5832** | 980 | **586** | 815 | 44 | 78 |
| f | resnet34, 2048, batch 1 | 0.3782 | **0.6677** | 0.5664 | 968 | 655 | 827 | 47 | 93 |
| g | resnet34, union target | 0.3633 | 0.6465 | 0.5620 | 1010 | 789 | 785 | 74 | 60 |
| h | HRNet-W32, 1536, batch 1 | 0.3661 | 0.6530 | 0.5606 | 932 | 598 | 863 | 55 | 75 |

Against run *a*:

| run | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| e | **+0.0087** | −0.0009 | **+0.0141** | +10 | **−58** | −10 |
| f | +0.0054 | **+0.0126** | −0.0027 | −2 | +11 | +2 |
| g | −0.0095 | −0.0086 | −0.0071 | **+40** | **+145** | −40 |
| h | −0.0067 | −0.0021 | −0.0085 | −38 | −46 | +38 |

Each run was also swept over thirty post-processing settings on its own maps.

| run | shared setting | retuned | difference | best setting |
|---|---|---|---|---|
| e | 0.3815 | 0.3820 | +0.0005 | threshold 0.7 |
| f | 0.3782 | 0.3782 | 0 | the shared setting |
| g | 0.3633 | 0.3676 | +0.0043 | threshold 0.7, no rejoining |
| h | 0.3661 | 0.3688 | +0.0027 | threshold 0.6 |

**No ranking changed**, as in Phase 6. The sweep and the scoring code agree to
four decimals at the shared setting for every run.

## Discussion

**Run *f* won where it was expected to.** SQ 0.6677 is the first time any run
has passed the incumbent's 0.6555; every run before it sat between 0.624 and
0.6555. Its matches are the incumbent's
exactly — 968 true positives, 827 false negatives — so the same filaments are
found and drawn better. Resolution was what limited the mask shape. Training at
a batch of one was stable. The gain in PQ, +0.0054, is the size of the 0.005 a
single configuration moved between two runs in Phase 6, so one run cannot
separate it from chance.

**Run *e* won somewhere it was not expected to.** HRNet was chosen for its
half-resolution branch, on the reasoning that it would draw thin filaments
better. Its SQ is run *a*'s. Its gain is 58 fewer false positives. That is a
real improvement, larger than the run-to-run spread, but the reason given for
trying it is not the reason it worked, and no other reason has been measured.

**Run *g* lost the way the metric said it could.** The union asks for more, and
the network delivered: 40 more matches. It also emitted 145 more false
positives, and each one adds 0.5 to the denominator of PQ. SQ fell as well,
because a union mask is wider than any one annotator's tracing and matches each
of them less closely. Phase 6 anticipated that a union target might only pay
once the filaments it asks for are visible; that was conditional on *f*
clearing the bar, which it did not, so it was not tried at 2048.

**Stacking *e* and *f* did not stack.** Run *h* found 48 fewer filaments than
*e* and drew them no better. It changed two things against *e* at once, the
resolution and the batch, so this result does not say that resolution hurts —
*f* says the opposite for ResNet-34. Its validation loss jumped from 0.366 to
0.403 between two epochs and took three to recover while the training loss fell
smoothly, which is what batch statistics taken from a single frame look like;
the HRNet encoder has 321 batch normalisation layers against ResNet-34's 36.
That explanation is not tested.

**Validation loss again failed to order the runs by PQ.** Run *g* reached a
lower best validation loss than *a* (0.3629 against 0.3671) and scored 0.0095
lower. Run *h*'s best checkpoint was chosen on a curve that moved by more
between epochs than the runs differ from each other.

**The adoption decision.** The rule is +0.01 over the reference and an account
of why. Run *e* meets neither in full: +0.0087, and a mechanism that is not the
one it was chosen for. It is adopted as the working configuration anyway,
because it is the best measured by a margin above the run-to-run spread, it
beats the incumbent as well as the reference, and every alternative left on
this axis costs three to four GPU hours for a gain of the size *f* showed. The
decision is recorded as provisional on those grounds.

## Next

- The submission is written from run *e* by
  `notebooks/60_train_kaggle_hrnet_submission.ipynb`, which picks the best
  candidate at the shared setting. Its fold 0 model is trained on four fifths of
  the training frames.
- Run *e* has one run behind it. Before anything is built on it, a second seed
  would say whether +0.0087 survives.
- Two questions stay open and are not worth their GPU time now: whether
  ResNet-34 at 1536 and batch 2 keeps *f*'s SQ without the batch of one, and
  whether *h* lost to its normalisation.

## Reproduction

| run | configuration | notebook | commit |
|---|---|---|---|
| e, f, g | `configs/phase6/{e_unet_hrnet,f_unet_resnet34_2048,g_unet_resnet34_union}.yaml` | `notebooks/50_train_kaggle_architectures.ipynb` (Kaggle Version 4) | `1679966` |
| h | `configs/phase6/h_unet_hrnet_1536.yaml` | `notebooks/60_train_kaggle_hrnet_submission.ipynb` | `ef99943` |
