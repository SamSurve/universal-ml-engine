import os
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.engine.contracts.schemas import (
    ProblemType,
    FeatureImportanceEntry,
    ChampionCandidate,
    ChampionSelectionResult,
    FinalTestEvaluationResult,
    PersistedChampionManifest,
    PermutationImportanceResult,
)
from backend.engine.explainability.permutation_explainer import PermutationExplainer
from backend.engine.reporting.visualizer import M5Visualizer
from backend.engine.reporting.m5_report_generator import M5ReportGenerator
from backend.engine.training.m5_orchestrator import M5PipelineRunner


class MockPredictor:
    """Mock predictor wrapping a standard scikit-learn model."""

    def __init__(self, model, is_classifier=False):
        self.model = model
        self.is_classifier = is_classifier

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.is_classifier and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise ValueError("predict_proba not available")


@pytest.fixture
def synthetic_clf_val_data():
    np.random.seed(42)
    n = 100
    feat_signal = np.random.randn(n)
    feat_noise = np.random.randn(n)
    # Target depends strongly on feat_signal
    prob = 1.0 / (1.0 + np.exp(-3.0 * feat_signal))
    target = (prob > 0.5).astype(int)

    df_train = pd.DataFrame({"signal": feat_signal[:60], "noise": feat_noise[:60], "target": target[:60]})
    df_val = pd.DataFrame({"signal": feat_signal[60:], "noise": feat_noise[60:], "target": target[60:]})

    lr = LogisticRegression()
    lr.fit(df_train[["signal", "noise"]], df_train["target"])
    predictor = MockPredictor(lr, is_classifier=True)

    return predictor, df_val


@pytest.fixture
def synthetic_reg_val_data():
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(1, 10, n)
    x2 = np.random.uniform(1, 10, n)
    noise = np.random.randn(n) * 0.1
    # Target depends purely on 10 * x1
    y = 10.0 * x1 + 0.1 * x2 + noise

    df_train = pd.DataFrame({"strong_feat": x1[:60], "weak_feat": x2[:60], "price": y[:60]})
    df_val = pd.DataFrame({"strong_feat": x1[60:], "weak_feat": x2[60:], "price": y[60:]})

    model = Ridge()
    model.fit(df_train[["strong_feat", "weak_feat"]], df_train["price"])
    predictor = MockPredictor(model, is_classifier=False)

    return predictor, df_val


def test_01_permutation_importance_classification(synthetic_clf_val_data):
    """1. Verify permutation importance correctly ranks informative features over noise in classification."""
    predictor, df_val = synthetic_clf_val_data

    result = PermutationExplainer.compute_importance(
        predictor=predictor,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="accuracy",
        n_repeats=5,
        random_state=42,
    )

    assert result.status == "success"
    assert len(result.importance_entries) == 2
    assert result.importance_entries[0].feature_name == "signal"
    assert result.importance_entries[0].importance_score > result.importance_entries[1].importance_score
    assert result.importance_entries[0].rank == 1
    assert result.importance_entries[1].rank == 2
    assert result.importance_entries[0].relative_importance_pct > 50.0


def test_02_permutation_importance_regression(synthetic_reg_val_data):
    """2. Verify permutation importance correctly ranks strong predictor in regression."""
    predictor, df_val = synthetic_reg_val_data

    result = PermutationExplainer.compute_importance(
        predictor=predictor,
        df_val=df_val,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
        primary_metric="r2",
        n_repeats=5,
        random_state=42,
    )

    assert result.status == "success"
    assert len(result.importance_entries) == 2
    assert result.importance_entries[0].feature_name == "strong_feat"
    assert result.importance_entries[0].importance_score > 0.5
    assert result.baseline_score > 0.9


def test_03_no_test_partition_access_during_explainability(synthetic_clf_val_data):
    """3. Verify explainability is strictly evaluated on validation, never test partition."""
    predictor, df_val = synthetic_clf_val_data

    # Create dummy untouched test set
    df_test = pd.DataFrame({
        "signal": [999.0, -999.0],
        "noise": [0.0, 0.0],
        "target": [1, 0],
    })

    # Call importance computation strictly with df_val
    result = PermutationExplainer.compute_importance(
        predictor=predictor,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        n_repeats=3,
    )

    assert result.sample_size == len(df_val)
    assert result.sample_size != len(df_test)


def test_04_unsupported_edge_cases_handling(synthetic_clf_val_data):
    """4. Verify edge cases: empty DataFrame, single column, missing target column, constant feature."""
    predictor, _ = synthetic_clf_val_data

    # Empty DataFrame
    res_empty = PermutationExplainer.compute_importance(
        predictor=predictor,
        df_val=pd.DataFrame(),
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
    )
    assert res_empty.status == "empty_dataset"
    assert len(res_empty.importance_entries) == 0

    # Missing target column
    with pytest.raises(ValueError, match="Target column 'missing_col' is missing"):
        PermutationExplainer.compute_importance(
            predictor=predictor,
            df_val=pd.DataFrame({"a": [1, 2]}),
            target_column="missing_col",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
        )

    # Only target column (no features)
    res_no_features = PermutationExplainer.compute_importance(
        predictor=predictor,
        df_val=pd.DataFrame({"target": [1, 0, 1]}),
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
    )
    assert res_no_features.status == "no_features"

    # Constant feature
    df_constant_train = pd.DataFrame({
        "const_feat": [5.0] * 10,
        "target": [0, 1] * 5,
    })
    lr_const = LogisticRegression()
    lr_const.fit(df_constant_train[["const_feat"]], df_constant_train["target"])
    pred_const = MockPredictor(lr_const, is_classifier=True)

    df_constant_val = pd.DataFrame({
        "const_feat": [5.0, 5.0, 5.0, 5.0],
        "target": [0, 1, 0, 1],
    })
    res_const = PermutationExplainer.compute_importance(
        predictor=pred_const,
        df_val=df_constant_val,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
    )
    assert res_const.status == "success"
    assert res_const.importance_entries[0].importance_score == 0.0


def test_05_classification_plots(tmp_path):
    """5. Verify classification plots generation (confusion matrix and feature importance)."""
    y_true = np.array(["retained", "churned", "retained", "retained", "churned"])
    y_pred = np.array(["retained", "churned", "churned", "retained", "churned"])

    importance_entries = [
        FeatureImportanceEntry("satisfaction", "satisfaction", 0.25, 62.5, 1, 0.02),
        FeatureImportanceEntry("salary", "salary", 0.15, 37.5, 2, 0.01),
    ]

    plot_dir = tmp_path / "clf_plots"
    paths = M5Visualizer.generate_classification_plots(
        y_true_test=y_true,
        y_pred_test=y_pred,
        importance_entries=importance_entries,
        output_dir=plot_dir,
        class_labels=["churned", "retained"],
    )

    assert "confusion_matrix" in paths
    assert "feature_importance" in paths
    assert Path(paths["confusion_matrix"]).exists()
    assert Path(paths["feature_importance"]).exists()
    assert Path(paths["confusion_matrix"]).stat().st_size > 1000
    assert Path(paths["feature_importance"]).stat().st_size > 1000


def test_06_regression_plots(tmp_path):
    """6. Verify regression plots generation (actual vs predicted, residuals, and feature importance)."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([105.0, 195.0, 310.0, 390.0, 505.0])

    importance_entries = [
        FeatureImportanceEntry("sqft", "sqft", 0.65, 80.0, 1, 0.04),
        FeatureImportanceEntry("rooms", "rooms", 0.15, 20.0, 2, 0.02),
    ]

    plot_dir = tmp_path / "reg_plots"
    paths = M5Visualizer.generate_regression_plots(
        y_true_test=y_true,
        y_pred_test=y_pred,
        importance_entries=importance_entries,
        output_dir=plot_dir,
    )

    assert "actual_vs_predicted" in paths
    assert "residuals" in paths
    assert "feature_importance" in paths
    assert Path(paths["actual_vs_predicted"]).exists()
    assert Path(paths["residuals"]).exists()
    assert Path(paths["feature_importance"]).exists()
    assert Path(paths["actual_vs_predicted"]).stat().st_size > 1000


def test_07_m5_report_generator_json_and_markdown(tmp_path):
    """7. Verify automated machine-readable JSON and Markdown report generation with clear labels."""
    manifest = PersistedChampionManifest(
        schema_version="1.0.0",
        champion_id="champ_123",
        backend_name="autogluon",
        model_name="WeightedEnsemble_L2",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        target_column="churn",
        primary_metric="f1",
        feature_names=["f1", "f2"],
        feature_types={"f1": "float64", "f2": "int64"},
        val_metrics={"f1": 0.88, "accuracy": 0.90},
        test_metrics={"f1": 0.86, "accuracy": 0.89},
        selection_rationale="Best validation f1",
    )

    cand = ChampionCandidate(
        candidate_id="c1",
        backend_name="autogluon",
        model_name="WeightedEnsemble_L2",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        val_metrics={"f1": 0.88},
        primary_metric="f1",
        primary_val_score=0.88,
    )

    champ_sel = ChampionSelectionResult(
        champion=cand,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="f1",
        selection_rationale="Highest validation F1 score.",
        all_candidates=[cand],
        validation_comparison={"c1": {"f1": 0.88}},
    )

    test_eval = FinalTestEvaluationResult(
        champion_model_name="WeightedEnsemble_L2",
        champion_backend="autogluon",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        test_metrics={"f1": 0.86, "accuracy": 0.89},
        test_rows=50,
        test_sha256="fake_sha",
    )

    importance_result = PermutationImportanceResult(
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="f1",
        baseline_score=0.88,
        importance_entries=[FeatureImportanceEntry("f1", "f1", 0.20, 80.0, 1, 0.01)],
        n_repeats=5,
        sample_size=50,
    )

    plot_paths = {"confusion_matrix": str(tmp_path / "cm.png")}

    report_artifacts = M5ReportGenerator.generate_report(
        manifest=manifest,
        champion_selection=champ_sel,
        test_evaluation=test_eval,
        importance_result=importance_result,
        plot_paths=plot_paths,
        output_dir=tmp_path,
        dataset_name="test_customer_churn",
    )

    assert Path(report_artifacts.report_json_path).exists()
    assert Path(report_artifacts.report_md_path).exists()

    # Check JSON structure
    with open(report_artifacts.report_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    assert json_data["schema_version"] == "1.0.0"
    assert "dataset_overview" in json_data
    assert "champion_selection" in json_data
    assert "metrics_comparison" in json_data
    assert "explainability" in json_data
    assert json_data["metrics_comparison"]["validation_metrics"]["f1"] == 0.88
    assert json_data["metrics_comparison"]["final_test_metrics"]["f1"] == 0.86

    # Check Markdown labels
    with open(report_artifacts.report_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "VALIDATION METRICS" in md_text
    assert "FINAL TEST METRICS" in md_text
    assert "Permutation Feature Importance" in md_text
    assert "WeightedEnsemble_L2" in md_text


def test_08_end_to_end_m5_pipeline_smoke_classification(tmp_path):
    """8. Complete synthetic end-to-end M5 smoke test for classification."""
    np.random.seed(42)
    n = 120
    feat1 = np.random.randn(n)
    feat2 = np.random.uniform(10, 50, size=n)
    prob = 1.0 / (1.0 + np.exp(-(feat1 * 2.0 + (feat2 - 30) / 10.0)))
    target = np.where(prob > 0.5, "churned", "retained")

    df_full = pd.DataFrame({
        "num_feat1": feat1,
        "num_feat2": feat2,
        "target": target,
    })

    out_dir = tmp_path / "m5_e2e_clf"

    results = M5PipelineRunner.run(
        data_source=df_full,
        target_column="target",
        output_dir=out_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        eval_metric="f1",
        time_limit_seconds=8.0,
        presets="medium_quality",
        include_baselines=True,
        n_importance_repeats=3,
        random_state=42,
        stage_dir=tmp_path / "m5_stage_clf",
    )

    assert results["importance_result"] is not None
    assert len(results["importance_result"].importance_entries) > 0
    assert len(results["plot_paths"]) == 2
    assert Path(results["plot_paths"]["confusion_matrix"]).exists()
    assert Path(results["plot_paths"]["feature_importance"]).exists()
    assert Path(results["report_artifacts"].report_json_path).exists()
    assert Path(results["report_artifacts"].report_md_path).exists()


def test_09_end_to_end_m5_pipeline_smoke_regression(tmp_path):
    """9. Complete synthetic end-to-end M5 smoke test for regression."""
    np.random.seed(42)
    n = 120
    sqft = np.random.uniform(500, 3000, size=n)
    rooms = np.random.randint(1, 6, size=n)
    price = 50000.0 + 150.0 * sqft + 10000.0 * rooms + np.random.normal(0, 1000, size=n)

    df_full = pd.DataFrame({
        "sqft": sqft,
        "rooms": rooms,
        "price": price,
    })

    out_dir = tmp_path / "m5_e2e_reg"

    results = M5PipelineRunner.run(
        data_source=df_full,
        target_column="price",
        output_dir=out_dir,
        problem_type=ProblemType.REGRESSION,
        eval_metric="r2",
        time_limit_seconds=8.0,
        presets="medium_quality",
        include_baselines=True,
        n_importance_repeats=3,
        random_state=42,
        stage_dir=tmp_path / "m5_stage_reg",
    )

    assert results["importance_result"] is not None
    assert len(results["importance_result"].importance_entries) > 0
    assert len(results["plot_paths"]) == 3
    assert Path(results["plot_paths"]["actual_vs_predicted"]).exists()
    assert Path(results["plot_paths"]["residuals"]).exists()
    assert Path(results["plot_paths"]["feature_importance"]).exists()
    assert Path(results["report_artifacts"].report_json_path).exists()
    assert Path(results["report_artifacts"].report_md_path).exists()
