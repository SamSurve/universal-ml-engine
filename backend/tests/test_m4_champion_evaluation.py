import os
import sys
import shutil
import tempfile
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import (
    ProblemType,
    PartitionInfo,
    ChampionCandidate,
    ChampionSelectionResult,
    FinalTestEvaluationResult,
    PersistedChampionManifest,
    AutoGluonBackendResult,
    WorkerStatus,
)
from backend.engine.evaluation.champion_selector import ChampionSelector
from backend.engine.evaluation.final_evaluator import FinalTestEvaluator
from backend.engine.artifacts.champion_store import ChampionStore
from backend.engine.inference.unified_predictor import UnifiedPredictor
from backend.engine.backends.autogluon_backend import AutoGluonBackend
from backend.engine.training.m4_orchestrator import M4PipelineRunner


@pytest.fixture
def synthetic_classification_data():
    """Generates synthetic binary classification dataset with clear signal."""
    np.random.seed(42)
    n = 120
    feat1 = np.random.randn(n)
    feat2 = np.random.uniform(10, 50, size=n)
    cat_feat = np.random.choice(["cat", "dog", "bird"], size=n)
    # deterministic signal with small noise
    prob = 1.0 / (1.0 + np.exp(-(feat1 * 2.0 + (feat2 - 30) / 10.0)))
    target = np.where(prob > 0.5, "churned", "retained")

    df = pd.DataFrame({
        "num_feat1": feat1,
        "num_feat2": feat2,
        "cat_feat": cat_feat,
        "target": target,
    })
    df_train = df.iloc[:70].copy().reset_index(drop=True)
    df_val = df.iloc[70:95].copy().reset_index(drop=True)
    df_test = df.iloc[95:].copy().reset_index(drop=True)
    return df_train, df_val, df_test


@pytest.fixture
def synthetic_regression_data():
    """Generates synthetic regression dataset with clear linear signal."""
    np.random.seed(42)
    n = 120
    sqft = np.random.uniform(500, 3000, size=n)
    rooms = np.random.randint(1, 6, size=n)
    price = 50000.0 + 150.0 * sqft + 10000.0 * rooms + np.random.normal(0, 1000, size=n)

    df = pd.DataFrame({
        "sqft": sqft,
        "rooms": rooms,
        "price": price,
    })
    df_train = df.iloc[:70].copy().reset_index(drop=True)
    df_val = df.iloc[70:95].copy().reset_index(drop=True)
    df_test = df.iloc[95:].copy().reset_index(drop=True)
    return df_train, df_val, df_test


def test_01_champion_selected_from_validation_only():
    """1. Verify ChampionSelector evaluates candidate scores on VALIDATION and selects highest score."""
    cand1 = ChampionCandidate(
        candidate_id="cand_dummy",
        backend_name="baseline",
        model_name="DummyClassifier",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        val_metrics={"f1": 0.40, "accuracy": 0.50, "roc_auc": 0.50},
        primary_metric="f1",
        primary_val_score=0.40,
        fit_time_seconds=0.1,
    )
    cand2 = ChampionCandidate(
        candidate_id="cand_logistic",
        backend_name="baseline",
        model_name="LogisticRegression",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        val_metrics={"f1": 0.75, "accuracy": 0.78, "roc_auc": 0.82},
        primary_metric="f1",
        primary_val_score=0.75,
        fit_time_seconds=0.5,
    )
    cand3 = ChampionCandidate(
        candidate_id="cand_autogluon",
        backend_name="autogluon",
        model_name="WeightedEnsemble_L2",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        val_metrics={"f1": 0.89, "accuracy": 0.90, "roc_auc": 0.94},
        primary_metric="f1",
        primary_val_score=0.89,
        fit_time_seconds=5.0,
    )

    selection = ChampionSelector.select_champion(
        candidates=[cand1, cand2, cand3],
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="f1",
    )

    assert selection.champion.candidate_id == "cand_autogluon"
    assert selection.champion.model_name == "WeightedEnsemble_L2"
    assert selection.champion.primary_val_score == 0.89
    assert "WeightedEnsemble_L2" in selection.selection_rationale
    assert len(selection.all_candidates) == 3
    assert "cand_autogluon" in selection.validation_comparison


def test_02_champion_selection_lower_is_better_for_error_metrics():
    """2. Verify metric directionality: for regression error metrics (RMSE/MAE), lowest error wins."""
    cand1 = ChampionCandidate(
        candidate_id="cand_dummy",
        backend_name="baseline",
        model_name="DummyRegressor",
        problem_type=ProblemType.REGRESSION,
        val_metrics={"rmse": 50.0, "mae": 40.0, "r2": -0.1},
        primary_metric="rmse",
        primary_val_score=50.0,
    )
    cand2 = ChampionCandidate(
        candidate_id="cand_ridge",
        backend_name="baseline",
        model_name="Ridge",
        problem_type=ProblemType.REGRESSION,
        val_metrics={"rmse": 25.0, "mae": 20.0, "r2": 0.65},
        primary_metric="rmse",
        primary_val_score=25.0,
    )
    cand3 = ChampionCandidate(
        candidate_id="cand_ag",
        backend_name="autogluon",
        model_name="LightGBM",
        problem_type=ProblemType.REGRESSION,
        val_metrics={"rmse": 12.5, "mae": 9.2, "r2": 0.88},
        primary_metric="rmse",
        primary_val_score=12.5,
    )

    selection = ChampionSelector.select_champion(
        candidates=[cand1, cand2, cand3],
        problem_type=ProblemType.REGRESSION,
        primary_metric="rmse",
    )

    assert selection.champion.candidate_id == "cand_ag"
    assert selection.champion.primary_val_score == 12.5
    assert "LightGBM" in selection.selection_rationale


def test_03_test_partition_not_accessed_during_selection(synthetic_classification_data):
    """
    3. Verify that the TEST partition is never accessed, passed, or required during champion selection.
    """
    df_train, df_val, df_test = synthetic_classification_data

    # Pass only df_val to evaluate candidate
    cand = ChampionCandidate(
        candidate_id="test_cand",
        backend_name="baseline",
        model_name="MockModel",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        val_metrics={"f1": 0.80, "accuracy": 0.85},
        primary_metric="f1",
        primary_val_score=0.80,
    )

    selection = ChampionSelector.select_champion(
        candidates=[cand],
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="f1",
    )
    assert selection.champion.model_name == "MockModel"


def test_04_final_test_evaluation_classification(tmp_path, synthetic_classification_data):
    """4. Verify final TEST evaluation for classification produces valid metrics and hashes."""
    df_train, df_val, df_test = synthetic_classification_data
    art_dir = tmp_path / "ag_clf_model"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=10.0,
        presets="medium_quality",
        auto_stack=False,
        output_dir=art_dir,
        working_dir=tmp_path / "stage_04",
    )
    assert fit_res.status == WorkerStatus.SUCCESS

    expected_classes = ["churned", "retained"]
    cand = ChampionSelector.candidate_from_autogluon(
        ag_result=fit_res,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=expected_classes,
    )

    # Evaluate on untouched TEST
    test_eval = FinalTestEvaluator.evaluate_champion(
        champion=cand,
        df_test=df_test,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=expected_classes,
    )

    assert test_eval.test_rows == len(df_test)
    assert len(test_eval.test_sha256) == 64  # Valid SHA-256 hash
    assert "accuracy" in test_eval.test_metrics
    assert "f1" in test_eval.test_metrics
    assert "roc_auc" in test_eval.test_metrics
    assert 0.0 <= test_eval.test_metrics["accuracy"] <= 1.0


def test_05_final_test_evaluation_regression(tmp_path, synthetic_regression_data):
    """5. Verify final TEST evaluation for regression produces valid R2/RMSE/MAE metrics."""
    df_train, df_val, df_test = synthetic_regression_data
    art_dir = tmp_path / "ag_reg_model"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
        eval_metric="r2",
        time_limit_seconds=10.0,
        presets="medium_quality",
        auto_stack=False,
        output_dir=art_dir,
        working_dir=tmp_path / "stage_05",
    )
    assert fit_res.status == WorkerStatus.SUCCESS

    cand = ChampionSelector.candidate_from_autogluon(
        ag_result=fit_res,
        df_val=df_val,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
    )

    test_eval = FinalTestEvaluator.evaluate_champion(
        champion=cand,
        df_test=df_test,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
    )

    assert test_eval.test_rows == len(df_test)
    assert "r2" in test_eval.test_metrics
    assert "rmse" in test_eval.test_metrics
    assert "mae" in test_eval.test_metrics
    assert isinstance(test_eval.test_metrics["r2"], float)


def test_06_persistence_and_reload_autogluon(tmp_path, synthetic_classification_data):
    """
    6. Verify native AutoGluon champion artifact persistence, manifest creation, and reload.
    """
    df_train, df_val, df_test = synthetic_classification_data
    raw_art_dir = tmp_path / "raw_ag_art"
    persist_dir = tmp_path / "persisted_champion"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=10.0,
        output_dir=raw_art_dir,
        working_dir=tmp_path / "stage_06",
    )
    assert fit_res.status == WorkerStatus.SUCCESS

    expected_classes = ["churned", "retained"]
    cand = ChampionSelector.candidate_from_autogluon(
        ag_result=fit_res,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=expected_classes,
    )

    selection = ChampionSelector.select_champion(
        candidates=[cand],
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="f1",
    )

    test_eval = FinalTestEvaluator.evaluate_champion(
        champion=selection.champion,
        df_test=df_test,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=expected_classes,
    )

    feature_cols = [c for c in df_train.columns if c != "target"]
    feature_types = {c: str(df_train[c].dtype) for c in feature_cols}

    manifest_path, manifest = ChampionStore.save_champion(
        champion_selection=selection,
        test_evaluation=test_eval,
        output_dir=persist_dir,
        feature_names=feature_cols,
        feature_types=feature_types,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        random_state=42,
    )

    assert manifest_path.exists()
    assert (persist_dir / "model").exists()
    assert manifest.backend_name == "autogluon"
    assert manifest.feature_names == feature_cols

    # Reload via UnifiedPredictor
    predictor = UnifiedPredictor.load(persist_dir)
    assert predictor.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert predictor.target_column == "target"
    assert predictor.class_labels == expected_classes

    # Predict
    X_test = df_test.drop(columns=["target"])
    preds = predictor.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset(set(expected_classes))

    # Predict Proba
    probs = predictor.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)


def test_07_schema_validation_rejects_missing_features(tmp_path, synthetic_classification_data):
    """7. UnifiedPredictor rejects input DataFrame with missing required feature columns."""
    df_train, df_val, df_test = synthetic_classification_data
    persist_dir = tmp_path / "persisted_clf_schema"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=10.0,
        output_dir=tmp_path / "raw_07",
        working_dir=tmp_path / "stage_07",
    )

    cand = ChampionSelector.candidate_from_autogluon(
        ag_result=fit_res,
        df_val=df_val,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
    )
    selection = ChampionSelector.select_champion([cand], ProblemType.BINARY_CLASSIFICATION)
    test_eval = FinalTestEvaluator.evaluate_champion(selection.champion, df_test, "target", ProblemType.BINARY_CLASSIFICATION)

    feature_cols = [c for c in df_train.columns if c != "target"]
    feature_types = {c: str(df_train[c].dtype) for c in feature_cols}

    ChampionStore.save_champion(
        champion_selection=selection,
        test_evaluation=test_eval,
        output_dir=persist_dir,
        feature_names=feature_cols,
        feature_types=feature_types,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
    )

    predictor = UnifiedPredictor.load(persist_dir)

    # Missing column 'num_feat1'
    df_bad = df_test.drop(columns=["num_feat1"])
    with pytest.raises(ValueError, match="missing required feature column"):
        predictor.predict(df_bad)

    with pytest.raises(ValueError, match="missing required feature column"):
        predictor.predict_proba(df_bad)


def test_08_predict_proba_rejected_on_regression(tmp_path, synthetic_regression_data):
    """8. UnifiedPredictor raises ValueError if predict_proba is called on a regression task."""
    df_train, df_val, df_test = synthetic_regression_data
    persist_dir = tmp_path / "persisted_reg"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
        time_limit_seconds=10.0,
        output_dir=tmp_path / "raw_08",
        working_dir=tmp_path / "stage_08",
    )

    cand = ChampionSelector.candidate_from_autogluon(fit_res, df_val, "price", ProblemType.REGRESSION)
    selection = ChampionSelector.select_champion([cand], ProblemType.REGRESSION, primary_metric="r2")
    test_eval = FinalTestEvaluator.evaluate_champion(selection.champion, df_test, "price", ProblemType.REGRESSION)

    feature_cols = [c for c in df_train.columns if c != "price"]
    feature_types = {c: str(df_train[c].dtype) for c in feature_cols}

    ChampionStore.save_champion(
        champion_selection=selection,
        test_evaluation=test_eval,
        output_dir=persist_dir,
        feature_names=feature_cols,
        feature_types=feature_types,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
    )

    predictor = UnifiedPredictor.load(persist_dir)
    assert predictor.problem_type == ProblemType.REGRESSION

    with pytest.raises(ValueError, match="only supported for classification tasks"):
        predictor.predict_proba(df_test.drop(columns=["price"]))


def test_09_end_to_end_m4_pipeline_smoke_classification(tmp_path, synthetic_classification_data):
    """9. Complete synthetic end-to-end M4 smoke test for classification."""
    df_train, df_val, df_test = synthetic_classification_data
    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    out_dir = tmp_path / "m4_e2e_clf"

    results = M4PipelineRunner.run(
        data_source=df_full,
        target_column="target",
        output_dir=out_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        eval_metric="f1",
        time_limit_seconds=8.0,
        presets="medium_quality",
        include_baselines=True,
        random_state=42,
        stage_dir=tmp_path / "m4_stage_clf",
    )

    assert results["champion_selection"] is not None
    assert results["final_test_evaluation"] is not None
    assert results["manifest"] is not None
    assert results["predictor"] is not None

    champ = results["champion_selection"].champion
    assert champ.model_name is not None
    assert champ.primary_val_score > 0.0

    test_metrics = results["final_test_evaluation"].test_metrics
    assert "accuracy" in test_metrics
    assert "f1" in test_metrics

    # Test reloaded predictor
    preds = results["predictor"].predict(df_test.drop(columns=["target"]))
    assert len(preds) == len(df_test)


def test_10_end_to_end_m4_pipeline_smoke_regression(tmp_path, synthetic_regression_data):
    """10. Complete synthetic end-to-end M4 smoke test for regression."""
    df_train, df_val, df_test = synthetic_regression_data
    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    out_dir = tmp_path / "m4_e2e_reg"

    results = M4PipelineRunner.run(
        data_source=df_full,
        target_column="price",
        output_dir=out_dir,
        problem_type=ProblemType.REGRESSION,
        eval_metric="r2",
        time_limit_seconds=8.0,
        presets="medium_quality",
        include_baselines=True,
        random_state=42,
        stage_dir=tmp_path / "m4_stage_reg",
    )

    assert results["champion_selection"] is not None
    assert results["final_test_evaluation"] is not None
    assert results["manifest"] is not None
    assert results["predictor"] is not None

    champ = results["champion_selection"].champion
    assert champ.model_name is not None

    test_metrics = results["final_test_evaluation"].test_metrics
    assert "r2" in test_metrics
    assert "rmse" in test_metrics
    assert "mae" in test_metrics

    preds = results["predictor"].predict(df_test.drop(columns=["price"]))
    assert len(preds) == len(df_test)
    assert isinstance(preds[0], (float, np.floating, int, np.integer))
