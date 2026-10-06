import pytest
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType, ColumnType, ColumnProfile
from backend.engine.models.registry import ModelRegistry
from backend.engine.training.cv_runner import CrossValidationRunner
from backend.engine.evaluation.evaluator import MetricsEvaluator


def test_model_registry_returns_models():
    clf_models = ModelRegistry.get_models(ProblemType.BINARY_CLASSIFICATION, random_state=42)
    assert "Logistic Regression" in clf_models
    assert "Random Forest" in clf_models
    assert "Extra Trees" in clf_models
    assert "HistGradientBoosting" in clf_models

    reg_models = ModelRegistry.get_models(ProblemType.REGRESSION, random_state=42)
    assert "Linear Regression" in reg_models
    assert "Ridge" in reg_models
    assert "Random Forest" in reg_models


def test_cross_validation_runner_classification():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num": np.random.randn(n),
        "cat": np.random.choice(["X", "Y", "Z"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    profiles = {
        "num": ColumnProfile("num", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
        "cat": ColumnProfile("cat", ColumnType.CATEGORICAL, 0, 0.0, 3, 0.05, False),
    }

    models = {
        "Logistic Regression": ModelRegistry.get_models(ProblemType.BINARY_CLASSIFICATION)["Logistic Regression"]
    }

    results, encoder = CrossValidationRunner.run_cv(
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        models=models,
        column_profiles=profiles,
        n_splits=3,
        random_state=42,
    )

    assert len(results) == 1
    res = results[0]
    assert res.status == "success"
    assert len(res.fold_metrics) == 3
    assert 0.0 <= res.mean_cv_score <= 1.0


def test_leaderboard_ranking_deterministic():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num": np.random.randn(n),
        "target": np.random.randn(n) * 10 + 50,
    })

    profiles = {
        "num": ColumnProfile("num", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
    }

    models = {
        "Linear Regression": ModelRegistry.get_models(ProblemType.REGRESSION)["Linear Regression"],
        "Ridge": ModelRegistry.get_models(ProblemType.REGRESSION)["Ridge"],
    }

    results, _ = CrossValidationRunner.run_cv(
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.REGRESSION,
        models=models,
        column_profiles=profiles,
        n_splits=3,
        random_state=42,
    )

    leaderboard = MetricsEvaluator.build_leaderboard(results, primary_metric="r2")
    assert len(leaderboard) == 2
    assert leaderboard[0].rank == 1
    assert leaderboard[1].rank == 2
    assert leaderboard[0].cv_score_mean >= leaderboard[1].cv_score_mean
