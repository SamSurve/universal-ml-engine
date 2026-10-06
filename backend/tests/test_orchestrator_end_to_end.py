import pytest
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from backend.engine.orchestrator import AutoMLEngine
from backend.engine.contracts.schemas import ProblemType
from backend.engine.artifacts.serializer import ArtifactManager


def test_orchestrator_end_to_end_classification():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n),
        "salary": np.random.uniform(30000, 120000, size=n),
        "department": np.random.choice(["sales", "engineering", "hr"], size=n),
        "turnover": np.random.choice(["stay", "leave"], size=n, p=[0.7, 0.3]),
    })

    temp_dir = tempfile.mkdtemp()
    try:
        result = AutoMLEngine.run(
            data_source=df,
            target_column="turnover",
            output_dir=temp_dir,
            random_state=42,
            n_splits=3,
        )

        assert result.problem_detection.problem_type == ProblemType.BINARY_CLASSIFICATION
        assert len(result.leaderboard) >= 3
        assert result.best_model_name is not None
        assert result.artifact_path is not None
        assert Path(result.artifact_path).exists()
        assert Path(result.metadata_path).exists()

        # Test reloading artifact and predicting
        loaded_artifact = ArtifactManager.load_pipeline(result.artifact_path)
        sample = df.drop(columns=["turnover"]).head(5)
        preds = loaded_artifact.predict(sample)
        assert len(preds) == 5
        assert set(preds).issubset({"stay", "leave"})

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_orchestrator_end_to_end_regression():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "sqft": np.random.uniform(500, 3500, size=n),
        "bedrooms": np.random.randint(1, 5, size=n),
        "zipcode": np.random.choice(["94016", "94102", "94103"], size=n),
        "price": np.random.uniform(200000, 1500000, size=n),
    })

    temp_dir = tempfile.mkdtemp()
    try:
        result = AutoMLEngine.run(
            data_source=df,
            target_column="price",
            output_dir=temp_dir,
            random_state=42,
            n_splits=3,
        )

        assert result.problem_detection.problem_type == ProblemType.REGRESSION
        assert len(result.leaderboard) >= 3
        assert result.best_model_name is not None
        assert result.artifact_path is not None
        assert Path(result.artifact_path).exists()

        # Reload and predict
        loaded_artifact = ArtifactManager.load_pipeline(result.artifact_path)
        sample = df.drop(columns=["price"]).head(5)
        preds = loaded_artifact.predict(sample)
        assert len(preds) == 5
        assert isinstance(preds[0], (float, np.floating, np.integer, int))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_orchestrator_reproducibility():
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.choice(["A", "B"], size=n),
        "y": np.random.choice([0, 1], size=n),
    })

    temp_dir1 = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()
    try:
        res1 = AutoMLEngine.run(df, "y", output_dir=temp_dir1, random_state=42, n_splits=3)
        res2 = AutoMLEngine.run(df, "y", output_dir=temp_dir2, random_state=42, n_splits=3)

        assert res1.best_model_name == res2.best_model_name
        assert res1.best_model_cv_score == res2.best_model_cv_score
        assert res1.holdout_metrics == res2.holdout_metrics
    finally:
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)
