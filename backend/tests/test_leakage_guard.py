import pytest
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType
from backend.engine.leakage.leakage_guard import LeakageGuard


def test_leakage_guard_drops_id_columns():
    df = pd.DataFrame({
        "id": range(100),
        "user_uuid": [f"uuid_{i}" for i in range(100)],
        "valid_feature": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100),
    })

    df_safe, exclusions = LeakageGuard.audit_and_filter(
        df, target_column="target", problem_type=ProblemType.BINARY_CLASSIFICATION
    )

    assert "id" not in df_safe.columns
    assert "user_uuid" not in df_safe.columns
    assert "valid_feature" in df_safe.columns
    assert any("identifier" in ex.reason.lower() for ex in exclusions)


def test_leakage_guard_drops_perfectly_correlated_features():
    target = np.linspace(10.0, 100.0, 50)
    df = pd.DataFrame({
        "target": target,
        "leakage_feature": target * 1.000001,  # Correlation 1.0
        "good_feature": np.random.randn(50),
    })

    df_safe, exclusions = LeakageGuard.audit_and_filter(
        df, target_column="target", problem_type=ProblemType.REGRESSION
    )

    assert "leakage_feature" not in df_safe.columns
    assert "good_feature" in df_safe.columns
    assert any("correlation" in ex.reason.lower() for ex in exclusions)
