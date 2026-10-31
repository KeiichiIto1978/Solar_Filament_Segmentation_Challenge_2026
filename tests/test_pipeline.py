"""Tests for assembling the final configuration from the committed weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from filament.models.segmentation import ModelConfig, build_model
from filament.pipeline import (
    FINAL_SETTING,
    NO_SCORE_CUT_SETTING,
    WEIGHT_NAME,
    load_fold_models,
    predict_ensemble_tta,
)
from filament.training.config import TrainConfig


def _write_weights(directory: Path, fold: int, stored_fold: int, image_size: int = 1024) -> None:
    config = TrainConfig(
        fold=stored_fold, encoder="resnet18", encoder_weights=None, image_size=image_size
    )
    model = build_model(ModelConfig(encoder_name="resnet18", encoder_weights=None))
    torch.save(
        {"model": model.state_dict(), "config": config.to_dict()},
        directory / WEIGHT_NAME.format(fold=fold),
    )


class _ConstantLogit(nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.full((batch.shape[0], 1, *batch.shape[2:]), self.logit)


def test_the_fold_models_load_in_fold_order(tmp_path: Path) -> None:
    for fold in (0, 1):
        _write_weights(tmp_path, fold, stored_fold=fold)

    models = load_fold_models(tmp_path, folds=(0, 1))

    assert len(models) == 2
    assert all(not model.training for model in models)


def test_a_file_holding_another_fold_is_refused(tmp_path: Path) -> None:
    """A copy of fold 1 saved under fold 0's name would average one model in
    twice and leave another out, without any visible error."""
    _write_weights(tmp_path, fold=0, stored_fold=1)

    with pytest.raises(ValueError, match="holds fold 1"):
        load_fold_models(tmp_path, folds=(0,))


def test_a_file_trained_at_another_resolution_is_refused(tmp_path: Path) -> None:
    _write_weights(tmp_path, fold=0, stored_fold=0, image_size=2048)

    with pytest.raises(ValueError, match="2048 px"):
        load_fold_models(tmp_path, folds=(0,))


def test_a_missing_fold_says_where_the_weights_come_from(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="export_weights"):
        load_fold_models(tmp_path, folds=(0,))


def test_the_ensemble_averages_the_models_probabilities() -> None:
    """Logits 0 and 10 everywhere: probabilities 0.5 and ~1.0, whatever the
    view, so the mean over views and models is their plain mean."""
    frame = np.full((64, 64), 128, dtype=np.uint8)

    mean = predict_ensemble_tta([_ConstantLogit(0.0), _ConstantLogit(10.0)], frame, size=32)

    assert mean.shape == (32, 32)
    assert np.allclose(mean, (0.5 + 1 / (1 + np.exp(-10.0))) / 2, atol=1e-6)


def test_an_empty_ensemble_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        predict_ensemble_tta([], np.zeros((64, 64), dtype=np.uint8), size=32)


def test_the_two_submissions_differ_only_in_the_score_cut() -> None:
    """The second submission is the first without the cut on mean probability
    (Phase 9 against Phase 10); any other difference would make the pair
    compare something else."""
    assert {key: FINAL_SETTING[key] for key in NO_SCORE_CUT_SETTING} == NO_SCORE_CUT_SETTING
    assert set(FINAL_SETTING) - set(NO_SCORE_CUT_SETTING) == {"min_score"}
