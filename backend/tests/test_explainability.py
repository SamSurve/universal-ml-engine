import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from backend.engine.contracts.schemas import ProblemType
from backend.engine.explainability.shap_explainer import ModelExplainer


def test_tree_explainer_with_preprocessing_pipeline():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num1": np.random.normal(0, 1, size=n),
        "num2": np.random.normal(5, 2, size=n),
        "cat1": np.random.choice(["type_a", "type_b"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["num1", "num2"]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["cat1"]),
        ]
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])

    X = df.drop(columns=["target"])
    y = df["target"]
    pipe.fit(X, y)

    res = ModelExplainer.explain_model(
        pipeline=pipe,
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        max_background_samples=30,
        max_eval_samples=20,
        random_state=42,
    )

    assert res.status in ("success", "fallback")
    assert len(res.global_importance) > 0
    assert len(res.raw_feature_importance) > 0
    # Cleaned names should strip 'num__' and 'cat__'
    all_clean_names = [e.feature_name for e in res.global_importance]
    assert any("num1" in name or "num2" in name for name in all_clean_names)
    assert not any(name.startswith("num__") for name in all_clean_names)

    # Raw features should aggregate to original columns
    raw_names = [e.feature_name for e in res.raw_feature_importance]
    assert "num1" in raw_names
    assert "num2" in raw_names
    assert "cat1" in raw_names

    # Check local explanations
    if res.example_explanations:
        ex = res.example_explanations[0]
        assert ex.predicted_value is not None
        assert len(ex.contributions) > 0


def test_linear_explainer_with_preprocessing_pipeline():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "feature_x": np.random.normal(10, 2, size=n),
        "feature_y": np.random.normal(20, 5, size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    pipe = Pipeline([
        ("preprocessor", ColumnTransformer([("num", StandardScaler(), ["feature_x", "feature_y"])])),
        ("model", LogisticRegression(random_state=42)),
    ])

    X = df.drop(columns=["target"])
    y = df["target"]
    pipe.fit(X, y)

    res = ModelExplainer.explain_model(
        pipeline=pipe,
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        max_background_samples=25,
        max_eval_samples=20,
        random_state=42,
    )

    assert res.status in ("success", "fallback")
    assert len(res.global_importance) == 2
    assert res.global_importance[0].relative_importance_pct > 0


def test_explainer_graceful_fallback():
    # If a custom non-scikit object is passed, it should fall back without crashing
    class DummyModel:
        pass

    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    res = ModelExplainer.explain_model(
        pipeline=DummyModel(),
        df_dev=df,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
    )

    assert res.status in ("fallback", "failed")
    assert len(res.global_importance) > 0
