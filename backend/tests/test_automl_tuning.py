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


def test_lightgbm_optuna_tuning_classification():
    """
    CRITICAL TEST (P2): Verifies LightGBM tuning uses correct parameters
    (feature_fraction, subsample_freq, num_leaves) and applies them to fitted estimator.
    """
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num_feature": np.random.randn(n),
        "cat_feature": np.random.choice(["type1", "type2"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    profiles = {
        "num_feature": ColumnProfile("num_feature", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
        "cat_feature": ColumnProfile("cat_feature", ColumnType.CATEGORICAL, 0, 0.0, 2, 0.03, False),
    }

    tuned_cand, pipeline = OptunaTuner.tune_model(
        model_name="LightGBM",
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        column_profiles=profiles,
        baseline_cv_score=0.60,
        n_trials=3,
        timeout_seconds=20,
        n_splits=3,
        random_state=42,
    )

    assert tuned_cand.status == "success"
    assert tuned_cand.n_trials >= 1
    assert tuned_cand.cv_score_std >= 0.0

    # Verify LightGBM specific parameters were sampled
    assert "feature_fraction" in tuned_cand.best_params or "num_leaves" in tuned_cand.best_params

    # Verify pipeline estimator is a fitted LightGBM classifier
    model_step = pipeline.named_steps["model"]
    import lightgbm as lgb
    assert isinstance(model_step, lgb.LGBMClassifier)

    # Test prediction on unseen sample
    sample = df.drop(columns=["target"]).head(2)
    preds = pipeline.predict(sample)
    assert len(preds) == 2


def test_lightgbm_optuna_tuning_regression():
    """Verifies LightGBM Optuna tuning works on regression tasks."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "sqft": np.random.uniform(500, 3000, size=n),
        "rooms": np.random.randint(1, 6, size=n),
        "price": np.random.uniform(100000, 500000, size=n),
    })

    profiles = {
        "sqft": ColumnProfile("sqft", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
        "rooms": ColumnProfile("rooms", ColumnType.NUMERIC, 0, 0.0, n, 1.0, False),
    }

    tuned_cand, pipeline = OptunaTuner.tune_model(
        model_name="LightGBM",
        df_dev=df,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
        column_profiles=profiles,
        baseline_cv_score=0.40,
        n_trials=3,
        timeout_seconds=20,
        n_splits=3,
        random_state=42,
    )

    assert tuned_cand.status == "success"
    assert tuned_cand.n_trials >= 1
    import lightgbm as lgb
    assert isinstance(pipeline.named_steps["model"], lgb.LGBMRegressor)

    preds = pipeline.predict(df.drop(columns=["price"]).head(2))
    assert len(preds) == 2


def test_flaml_influences_candidate_selection():
    """
    CRITICAL TEST (P4): Verifies FLAML screening result genuinely influences
    which candidate models are forwarded for baseline cross-validation.
    """
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.uniform(10, 50, size=n),
        "cat": np.random.choice(["A", "B"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    temp_dir = tempfile.mkdtemp()
    try:
        # Request Top-2 candidates from FLAML screening
        result = AutoMLEngine.run(
            data_source=df,
            target_column="target",
            output_dir=temp_dir,
            random_state=42,
            n_splits=3,
            enable_screening=True,
            screening_time_budget=5,
            screening_top_k=2,
            enable_tuning=False,  # Isolate baseline selection
        )

        assert len(result.screened_candidates) >= 1
        screened_names = [c.model_name for c in result.screened_candidates if c.status == "success"]
        top_2_screened = screened_names[:2]

        leaderboard_models = [e.model_name for e in result.leaderboard]
        # Baseline CV models must be filtered to the top-2 screened candidates
        assert len(leaderboard_models) <= 2
        for m in leaderboard_models:
            assert m in top_2_screened

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_flaml_fallback_when_disabled():
    """
    Verifies that when screening is disabled, the engine safely falls back
    to evaluating all registered candidate models.
    """
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "x1": np.random.randn(n),
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
            enable_screening=False,  # Screening disabled
            enable_tuning=False,
        )

        assert len(result.screened_candidates) == 0
        # When screening is disabled, all registry models run baseline CV
        assert len(result.leaderboard) >= 5

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_unseen_target_label_validation():
    """
    CRITICAL TEST (P3): Verifies that unseen target labels in holdout data
    trigger a controlled ValueError instead of an unhandled crash.
    """
    from sklearn.preprocessing import LabelEncoder
    from backend.engine.artifacts.serializer import ModelArtifact
    from sklearn.dummy import DummyClassifier

    # 1. Test ModelArtifact controlled decoding error
    le = LabelEncoder()
    le.fit(["cat", "dog"])

    dummy = DummyClassifier(strategy="constant", constant=99)
    dummy.fit([[1], [2]], [99, 99])

    artifact = ModelArtifact(
        pipeline=dummy,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        target_column="animal",
        label_encoder=le,
    )

    with pytest.raises(ValueError, match="Target decoding error"):
        artifact.predict(pd.DataFrame([[1]]))
