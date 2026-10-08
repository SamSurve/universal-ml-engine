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


def test_shap_preserves_original_feature_names_with_underscores():
    """
    CRITICAL TEST (P1): Verifies that features with underscores like 'Work_Life_Balance'
    are NOT corrupted or truncated to 'Work'.
    """
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "Work_Life_Balance": np.random.uniform(1.0, 5.0, size=n),
        "Years_At_Company": np.random.randint(1, 15, size=n),
        "Department_Name": np.random.choice(["Sales_Dept", "Engineering_Team"], size=n),
        "Job_Role_Type": np.random.choice(["Manager_Lead", "Junior_Dev"], size=n),
        "Turnover_Status": np.random.choice([0, 1], size=n),
    })

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                ["Work_Life_Balance", "Years_At_Company"],
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                ["Department_Name", "Job_Role_Type"],
            ),
        ]
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])

    X = df.drop(columns=["Turnover_Status"])
    y = df["Turnover_Status"]
    pipe.fit(X, y)

    res = ModelExplainer.explain_model(
        pipeline=pipe,
        df_dev=df,
        target_column="Turnover_Status",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        max_background_samples=25,
        max_eval_samples=20,
        random_state=42,
    )

    assert res.status == "success"

    # 1. Verify raw_feature_importance preserves exact original parent names
    raw_feature_names = [e.raw_feature_name for e in res.raw_feature_importance]
    assert "Work_Life_Balance" in raw_feature_names
    assert "Years_At_Company" in raw_feature_names
    assert "Department_Name" in raw_feature_names
    assert "Job_Role_Type" in raw_feature_names

    # MUST NOT contain truncated fragments
    assert "Work" not in raw_feature_names
    assert "Years" not in raw_feature_names
    assert "Department" not in raw_feature_names
    assert "Job" not in raw_feature_names

    # 2. Verify global_importance has exact raw_feature_name mapped
    for entry in res.global_importance:
        assert entry.raw_feature_name in ["Work_Life_Balance", "Years_At_Company", "Department_Name", "Job_Role_Type"]
        # Numeric features must have feature_name equal to raw_feature_name
        if entry.raw_feature_name in ["Work_Life_Balance", "Years_At_Company"]:
            assert entry.feature_name == entry.raw_feature_name

    # 3. Verify local explanations preserve raw_feature_name
    assert len(res.example_explanations) > 0
    first_ex = res.example_explanations[0]
    for contrib in first_ex.contributions:
        assert contrib.raw_feature_name in ["Work_Life_Balance", "Years_At_Company", "Department_Name", "Job_Role_Type"]
        assert contrib.raw_feature_name != "Work"


def test_shap_regression_feature_names():
    """Verifies SHAP explanation feature names for regression tasks."""
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "Total_Living_Area": np.random.uniform(500, 3000, size=n),
        "Garage_Car_Spaces": np.random.randint(0, 4, size=n),
        "Neighborhood_Zone": np.random.choice(["Zone_North", "Zone_South"], size=n),
        "Sale_Price": np.random.uniform(100000, 500000, size=n),
    })

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Total_Living_Area", "Garage_Car_Spaces"]),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["Neighborhood_Zone"]),
        ]
    )
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
    ])
    pipe.fit(df.drop(columns=["Sale_Price"]), df["Sale_Price"])

    res = ModelExplainer.explain_model(
        pipeline=pipe,
        df_dev=df,
        target_column="Sale_Price",
        problem_type=ProblemType.REGRESSION,
        max_background_samples=25,
        max_eval_samples=20,
        random_state=42,
    )

    assert res.status == "success"
    raw_names = [e.raw_feature_name for e in res.raw_feature_importance]
    assert "Total_Living_Area" in raw_names
    assert "Garage_Car_Spaces" in raw_names
    assert "Neighborhood_Zone" in raw_names
    assert "Total" not in raw_names
    assert "Garage" not in raw_names


def test_shap_multiclass_aggregation():
    """Verifies multiclass SHAP aggregates across all classes without crashing or dropping classes."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "Feature_A": np.random.randn(n),
        "Feature_B": np.random.randn(n),
        "Target_Class": np.random.choice([0, 1, 2], size=n),
    })

    pipe = Pipeline([
        ("preprocessor", ColumnTransformer([("num", StandardScaler(), ["Feature_A", "Feature_B"])])),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipe.fit(df.drop(columns=["Target_Class"]), df["Target_Class"])

    res = ModelExplainer.explain_model(
        pipeline=pipe,
        df_dev=df,
        target_column="Target_Class",
        problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
        max_background_samples=25,
        max_eval_samples=20,
        random_state=42,
    )

    assert res.status == "success"
    assert len(res.global_importance) == 2
    raw_names = [e.raw_feature_name for e in res.raw_feature_importance]
    assert "Feature_A" in raw_names
    assert "Feature_B" in raw_names
