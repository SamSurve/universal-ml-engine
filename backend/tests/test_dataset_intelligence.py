import pytest
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType, QualitySeverity
from backend.engine.intelligence.data_intelligence import DatasetIntelligenceAnalyzer
from backend.engine.intelligence.health_score import DatasetHealthCalculator


def test_dataset_intelligence_clean_dataset():
    # Clean synthetic dataset
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n),
        "income": np.random.normal(50000, 15000, size=n),
        "department": np.random.choice(["sales", "engineering", "hr"], size=n),
        "turnover": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    })

    result = DatasetIntelligenceAnalyzer.analyze(
        df=df, target_column="turnover", problem_type=ProblemType.BINARY_CLASSIFICATION
    )

    assert result.dataset_shape == (100, 4)
    assert result.target_column == "turnover"
    assert result.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert "age" in result.numeric_columns
    assert "income" in result.numeric_columns
    assert "department" in result.categorical_columns
    assert result.duplicate_rows_count == 0
    assert len(result.constant_columns) == 0
    assert result.health_score.overall_score >= 80.0
    assert result.health_score.grade in ["Excellent", "Good"]


def test_dataset_intelligence_detects_risks():
    # Dataset with intentional flaws: duplicate rows, constant column, suspicious ID, high missingness
    n = 100
    df = pd.DataFrame({
        "user_id": np.arange(1000, 1000 + n),  # Suspicious ID
        "const_feat": [42.0] * n,  # Constant
        "near_const": [1.0] * 96 + [2.0] * 4,  # Near-constant
        "high_null": [np.nan] * 60 + [1.0] * 40,  # 60% null
        "num_a": np.arange(n, dtype=float),
        "num_b": np.arange(n, dtype=float) + np.random.normal(0, 0.01, size=n),  # High collinearity
        "severe_target": [0] * 95 + [1] * 5,  # Severe class imbalance (19:1)
    })
    # Add duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    result = DatasetIntelligenceAnalyzer.analyze(
        df=df, target_column="severe_target", problem_type=ProblemType.BINARY_CLASSIFICATION
    )

    assert result.duplicate_rows_count == 5
    assert "const_feat" in result.constant_columns
    assert "near_const" in result.near_constant_columns
    assert "user_id" in result.suspicious_id_columns
    assert result.missing_percentages["high_null"] > 0.5
    assert len(result.correlated_numeric_pairs) > 0
    assert result.class_imbalance is not None
    assert result.class_imbalance["severity"] == "severe"

    # Health score should reflect penalties
    assert result.health_score.overall_score <= 80.0
    assert len(result.health_score.major_warnings) > 0


def test_dataset_health_score_determinism_and_bounds():
    # Verify score is bounded in [0, 100] and deterministic
    missing_dict = {"feat_a": 0.8, "feat_b": 0.0, "target": 0.0}
    score1 = DatasetHealthCalculator.calculate(
        total_rows=100,
        total_cols=5,
        duplicate_rows_count=20,
        missing_percentages=missing_dict,
        constant_columns=["const_col"],
        near_constant_columns=[],
        high_cardinality_columns=["high_card"],
        suspicious_id_columns=["id"],
        target_leakage_risks=[],
        correlated_pairs_count=2,
        class_imbalance={"severity": "moderate", "imbalance_ratio": 4.0},
        target_column="target",
        warnings=[],
    )

    score2 = DatasetHealthCalculator.calculate(
        total_rows=100,
        total_cols=5,
        duplicate_rows_count=20,
        missing_percentages=missing_dict,
        constant_columns=["const_col"],
        near_constant_columns=[],
        high_cardinality_columns=["high_card"],
        suspicious_id_columns=["id"],
        target_leakage_risks=[],
        correlated_pairs_count=2,
        class_imbalance={"severity": "moderate", "imbalance_ratio": 4.0},
        target_column="target",
        warnings=[],
    )

    assert score1.overall_score == score2.overall_score
    assert 0.0 <= score1.overall_score <= 100.0
    assert len(score1.categories) == 4
    for cat_name, cat in score1.categories.items():
        assert 0.0 <= cat.score <= 100.0
