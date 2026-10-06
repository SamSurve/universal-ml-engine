import pytest
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType
from backend.engine.problem_detection.detector import ProblemDetector


def test_detect_binary_classification_boolean():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [True, False, True, False],
    })
    res = ProblemDetector.detect(df, target_column="target")
    assert res.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert res.confidence == 1.0


def test_detect_binary_classification_strings():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": ["Yes", "No", "Yes", "No"],
    })
    res = ProblemDetector.detect(df, target_column="target")
    assert res.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert res.confidence == 1.0


def test_detect_multiclass_classification():
    df = pd.DataFrame({
        "feature": range(30),
        "target": ["low", "medium", "high"] * 10,
    })
    res = ProblemDetector.detect(df, target_column="target")
    assert res.problem_type == ProblemType.MULTICLASS_CLASSIFICATION
    assert res.confidence >= 0.90


def test_detect_regression_continuous_floats():
    df = pd.DataFrame({
        "feature": range(50),
        "target": np.linspace(100.5, 999.8, 50),
    })
    res = ProblemDetector.detect(df, target_column="target")
    assert res.problem_type == ProblemType.REGRESSION
    assert res.confidence >= 0.90


def test_detect_unknown_on_missing_column():
    df = pd.DataFrame({"feature": [1, 2, 3]})
    res = ProblemDetector.detect(df, target_column="nonexistent")
    assert res.problem_type == ProblemType.UNKNOWN
    assert res.confidence == 0.0
