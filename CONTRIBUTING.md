# Contributing

How this repository is worked on, and the rules followed throughout the
competition (by hand and by the coding agent used along the way).

## Ground rules

- **The publicly released MAGFiLO ground truth is never used, downloaded or
  referenced.** Most of the competition test frames are in it, and the
  organisers forbid it.
- **The competition data is never committed.** It is distributed under
  CC BY-NC 4.0, which does not allow redistribution. Point `MAGFILO_ROOT` at
  your copy (see the README).
- No absolute local paths in code: paths go through `configs/paths.yaml` and
  environment variables.
- No secrets in the repository (`kaggle.json`, `.env`, tokens).
- The licence is Apache-2.0.

## Evaluation discipline

The metric decides most design choices, so it is protected first.

- `src/filament/metrics/` must stay covered by tests. A broken evaluator makes
  every experiment after it meaningless.
- Experiments report PQ **together with SQ, RQ, TP, FP and FN**. Changes that
  raise Dice can lower PQ: a prediction at IoU 0.49 counts as a false positive
  *and* a false negative.
- Train and validation are always split **by frame**, never by
  annotator-image: the same frame drawn by two annotators must not sit on both
  sides. The five folds in `configs/splits/` are frozen; every recorded score
  refers to them.
- A change is adopted only if it clears a margin larger than retraining noise
  (fold 0: +0.01; pooled over five folds: +0.005 with at least four folds
  improving).
- Local PQ decides, not the public leaderboard, which allows five submissions a
  day and shows two decimals.

## Experiment records

Each phase ends with one record in `docs/experiments/`: what was done, the
results, what they mean, and what comes next. A record is written once, when
the phase closes, and not edited afterwards; when a later phase overturns an
earlier conclusion, the later record says so. Freezing them keeps a number from
being updated in one place and left stale in another.

## Code

- Python 3.12, dependencies managed with [uv](https://docs.astral.sh/uv/).
  `requirements.txt` is generated with `scripts/export_requirements.py`.
- `ruff` for lint and format (line length 100) and `pytest` for tests. `mypy`
  is run on new modules; the older ones are not yet clean under it.

  ```bash
  uv run pytest
  uv run ruff check . && uv run ruff format --check .
  ```

- Comments explain *why*, not what the line does.
- Error messages and logs are written for the person running the code: what
  went wrong and what to do about it.
- New dependencies are added with a stated reason.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:` …).

## Heavy jobs

Training and full inference run on Kaggle notebooks (`notebooks/`), each cloning
a fixed commit of this repository so that the run can be reproduced from that
commit. Scoring, sweeps and analysis read the stored probability maps and need
no GPU.
