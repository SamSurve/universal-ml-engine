import pytest
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType, ColumnType, ColumnProfile
from backend.engine.automl.flaml_screener import FLAMLScreener
from backend.engine.automl.optuna_tuner import OptunaTuner
from backend.engine.orchestrator import AutoMLEngine
from backend.engine.artifacts.serializer import ArtifactManager


def test_flaml_screening_classification():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.uniform(10, 50, size=n),
        "cat1": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    profiles = {
        "num1": ColumnProfile("num1", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
        "num2": ColumnProfile("num2", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
        "cat1": ColumnProfile("cat1", ColumnType.CATEGORICAL, 0, 0.0, 3, 0.05, False),
    }

    candidates = FLAMLScreener.screen(
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        column_profiles=profiles,
        time_budget_seconds=5,
        n_splits=3,
        random_state=42,
        estimator_list=["rf", "extra_tree"],
    )

    assert len(candidates) >= 1
    top_cand = candidates[0]
    assert top_cand.rank == 1
    assert top_cand.status == "success"
    assert -1.0 <= top_cand.cv_score <= 1.0


def test_flaml_screening_regression():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.uniform(5, 25, size=n),
        "target": np.random.uniform(100, 500, size=n),
    })

    profiles = {
        "feat1": ColumnProfile("feat1", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
        "feat2": ColumnProfile("feat2", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
    }

    candidates = FLAMLScreener.screen(
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.REGRESSION,
        column_profiles=profiles,
        time_budget_seconds=5,
        n_splits=3,
        random_state=42,
        estimator_list=["rf", "extra_tree"],
    )

    assert len(candidates) >= 1
    top_cand = candidates[0]
    assert top_cand.rank == 1
    assert top_cand.status == "success"


def test_optuna_tuning_classification():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "cat1": np.random.choice(["X", "Y"], size=n),
        "target": np.random.choice(["stay", "leave"], size=n),
    })

    profiles = {
        "num1": ColumnProfile("num1", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
        "cat1": ColumnProfile("cat1", ColumnType.CATEGORICAL, 0, 0.0, 2, 0.03, False),
    }

    tuned_cand, pipeline = OptunaTuner.tune_model(
        model_name="Random Forest",
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        column_profiles=profiles,
        baseline_cv_score=0.70,
        n_trials=3,
        timeout_seconds=10,
        n_splits=3,
        random_state=42,
    )

    assert tuned_cand.status == "success"
    assert tuned_cand.n_trials >= 1
    assert "n_estimators" in tuned_cand.best_params or "max_depth" in tuned_cand.best_params
    assert pipeline is not None


def test_optuna_tuning_regression():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "x": np.random.randn(n),
        "target": np.random.uniform(50, 150, size=n),
    })

    profiles = {
        "x": ColumnProfile("x", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
    }

    tuned_cand, pipeline = OptunaTuner.tune_model(
        model_name="Ridge",
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.REGRESSION,
        column_profiles=profiles,
        baseline_cv_score=0.40,
        n_trials=3,
        timeout_seconds=10,
        n_splits=3,
        random_state=42,
    )

    assert tuned_cand.status == "success"
    assert "alpha" in tuned_cand.best_params
    assert pipeline is not None


def test_end_to_end_automl_pipeline_with_tuning():
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n),
        "income": np.random.uniform(30000, 90000, size=n),
        "dept": np.random.choice(["sales", "eng", "hr"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    temp_dir = tempfile.mkdtemp()
    try:
        result = AutoMLEngine.run(
            data_source=df,
            target_column="target",
            output_dir=temp_dir,
            random_state=42,
            n_splits=3,
            enable_screening=True,
            screening_time_budget=5,
            enable_tuning=True,
            top_k_to_tune=1,
            tuning_trials=3,
            tuning_time_budget=10,
        )

        assert result.problem_detection.problem_type == ProblemType.BINARY_CLASSIFICATION
        assert len(result.leaderboard) >= 3
        assert len(result.screened_candidates) >= 1
        assert len(result.tuned_candidates) >= 1
        assert result.artifact_path is not None
        assert Path(result.artifact_path).exists()
        assert Path(result.metadata_path).exists()

        # Verify artifact reloading
        loaded_artifact = ArtifactManager.load_pipeline(result.artifact_path)
        sample = df.drop(columns=["target"]).head(4)
        preds = loaded_artifact.predict(sample)
        assert len(preds) == 4

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
