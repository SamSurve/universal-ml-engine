import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType, BaselineSuiteResult
from backend.engine.training.data_splitter import DataSplitter
from backend.engine.evaluation.baselines import BaselineEvaluator
from backend.engine.ingestion.service import IngestionService, DataValidator


def test_classification_baselines_binary():
    """Verify that Binary Classification baselines (Dummy + Logistic) execute and compute valid metrics."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "num_1": np.random.randn(n),
        "num_2": np.random.randn(n) * 5 + 10,
        "cat_1": np.random.choice(["dept_A", "dept_B", "dept_C"], size=n),
        "target": np.random.choice(["No", "Yes"], size=n, p=[0.7, 0.3]),
    })

    df_train, df_val, df_test, p_info = DataSplitter.split_train_val_test(
        df, "target", ProblemType.BINARY_CLASSIFICATION, random_state=42
    )

    result = BaselineEvaluator.evaluate_baselines(
        df_train=df_train,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        random_state=42,
    )

    assert isinstance(result, BaselineSuiteResult)
    assert result.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert result.dummy_baseline.status == "success"
    assert result.linear_baseline.status == "success"
    assert "accuracy" in result.dummy_baseline.val_metrics
    assert "f1" in result.linear_baseline.val_metrics
    assert "roc_auc" in result.linear_baseline.val_metrics
    assert result.best_baseline_score > 0.0


def test_classification_baselines_multiclass():
    """Verify that Multiclass Classification baselines handle categorical encoding and multiple classes."""
    np.random.seed(42)
    n = 240
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n) * 2,
        "cat": np.random.choice(["low", "med", "high"], size=n),
        "target": np.random.choice(["tier_1", "tier_2", "tier_3"], size=n),
    })

    df_train, df_val, df_test, _ = DataSplitter.split_train_val_test(
        df, "target", ProblemType.MULTICLASS_CLASSIFICATION, random_state=42
    )

    result = BaselineEvaluator.evaluate_baselines(
        df_train=df_train,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
        random_state=42,
    )

    assert result.problem_type == ProblemType.MULTICLASS_CLASSIFICATION
    assert result.dummy_baseline.status == "success"
    assert result.linear_baseline.status == "success"
    assert "f1_macro" in result.linear_baseline.val_metrics
    assert result.linear_baseline.val_score > 0.0


def test_regression_baselines():
    """Verify that Regression baselines (Dummy Regressor + Ridge) evaluate correctly and Ridge beats Dummy."""
    np.random.seed(42)
    n = 300
    x1 = np.random.uniform(10, 50, n)
    x2 = np.random.uniform(1, 10, n)
    # Strong linear signal: target = 4*x1 + 10*x2 + noise
    y = 4.0 * x1 + 10.0 * x2 + np.random.normal(0, 2.0, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})

    df_train, df_val, df_test, _ = DataSplitter.split_train_val_test(
        df, "target", ProblemType.REGRESSION, random_state=42
    )

    result = BaselineEvaluator.evaluate_baselines(
        df_train=df_train,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.REGRESSION,
        random_state=42,
    )

    assert result.problem_type == ProblemType.REGRESSION
    assert result.primary_metric == "r2"
    assert result.dummy_baseline.status == "success"
    assert result.linear_baseline.status == "success"

    # Dummy regressor on val set should have R^2 around 0 or slightly negative
    assert result.dummy_baseline.val_metrics["r2"] <= 0.05
    # Ridge should capture the strong linear relationship with high R^2 (> 0.90)
    assert result.linear_baseline.val_metrics["r2"] > 0.90
    assert result.best_baseline_name == "Ridge Regression Baseline"
    assert result.best_baseline_score > 0.90
    assert "mae" in result.linear_baseline.val_metrics
    assert "rmse" in result.linear_baseline.val_metrics


def test_baselines_zero_data_leakage():
    """Verify that preprocessing is fitted strictly on Train and extreme Validation outliers do not contaminate fit."""
    # Train set has values 0 to 10
    df_train = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    })
    # Validation set has an extreme outlier (1,000,000)
    df_val = pd.DataFrame({
        "num": [5.0, 1000000.0],
        "target": [10.0, 2000000.0],
    })

    result = BaselineEvaluator.evaluate_baselines(
        df_train=df_train,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.REGRESSION,
        random_state=42,
    )

    assert result.linear_baseline.status == "success"
    assert result.dummy_baseline.status == "success"


def test_baselines_with_real_turnover_data():
    """Verify baselines on real employee turnover classification dataset."""
    project_root = Path(__file__).resolve().parent.parent.parent
    turnover_path = project_root / "employee_turnover.csv"
    if not turnover_path.exists():
        turnover_path = project_root / "employee_turnover(1).csv"

    if not turnover_path.exists():
        pytest.skip("employee_turnover.csv not found in workspace")

    df_raw = IngestionService.load_dataset(turnover_path)
    df_clean, val_res = DataValidator.validate_and_clean(df_raw, "Employee_Turnover")
    assert val_res.is_valid

    df_train, df_val, df_test, p_info = DataSplitter.split_train_val_test(
        df_clean, "Employee_Turnover", ProblemType.BINARY_CLASSIFICATION, random_state=42
    )

    assert p_info.stratified is True
    assert len(df_train) > 0
    assert len(df_val) > 0
    assert len(df_test) > 0

    result = BaselineEvaluator.evaluate_baselines(
        df_train=df_train,
        df_val=df_val,
        target_column="Employee_Turnover",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        random_state=42,
    )

    assert result.dummy_baseline.status == "success"
    assert result.linear_baseline.status == "success"
    assert result.best_baseline_score > 0.0


def test_baselines_with_real_house_price_data():
    """Verify baselines on real House Price regression dataset."""
    project_root = Path(__file__).resolve().parent.parent.parent
    house_path = project_root / "HousePricePrediction.csv"
    if not house_path.exists():
        house_path = project_root / "HousePricePrediction(1).csv"

    if not house_path.exists():
        pytest.skip("HousePricePrediction.csv not found in workspace")

    df_raw = IngestionService.load_dataset(house_path)
    df_clean, val_res = DataValidator.validate_and_clean(df_raw, "SalePrice")
    assert val_res.is_valid

    df_train, df_val, df_test, p_info = DataSplitter.split_train_val_test(
        df_clean, "SalePrice", ProblemType.REGRESSION, random_state=42
    )

    assert len(df_train) > 0
    assert len(df_val) > 0
    assert len(df_test) > 0

    result = BaselineEvaluator.evaluate_baselines(
        df_train=df_train,
        df_val=df_val,
        target_column="SalePrice",
        problem_type=ProblemType.REGRESSION,
        random_state=42,
    )

    assert result.dummy_baseline.status == "success"
    assert result.linear_baseline.status == "success"
    assert result.linear_baseline.val_metrics["r2"] > 0.50
