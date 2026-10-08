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
    WorkerStatus,
    WorkerJobSpec,
    AutoGluonBackendResult,
)
from backend.engine.backends.subprocess_runner import SubprocessRunner
from backend.engine.backends.autogluon_backend import AutoGluonBackend


@pytest.fixture
def classification_data():
    """Generates synthetic binary classification dataset."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feature_num1": np.random.randn(n),
        "feature_num2": np.random.uniform(10, 50, size=n),
        "feature_cat": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice(["stay", "leave"], size=n, p=[0.6, 0.4]),
    })
    df_train = df.iloc[:40].copy()
    df_val = df.iloc[40:].copy()
    return df_train, df_val


@pytest.fixture
def regression_data():
    """Generates synthetic regression dataset."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "sqft": np.random.uniform(500, 3000, size=n),
        "rooms": np.random.randint(1, 6, size=n),
        "price": np.random.uniform(100000, 500000, size=n),
    })
    df_train = df.iloc[:40].copy()
    df_val = df.iloc[40:].copy()
    return df_train, df_val


def test_01_autogluon_dependency_availability():
    """1. Verify that the dedicated AutoGluon Python environment exists and has autogluon installed."""
    py_exec = AutoGluonBackend.get_python_executable()
    assert Path(py_exec).exists(), f"AutoGluon Python executable not found at: {py_exec}"

    import subprocess
    cmd = [py_exec, "-c", "import autogluon.tabular; print('AG_OK')"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"AutoGluon import failed: {res.stderr}"
    assert "AG_OK" in res.stdout


def test_02_worker_launches_with_dedicated_interpreter():
    """2. Verify that the subprocess worker is launched using the dedicated AutoGluon interpreter."""
    py_exec = AutoGluonBackend.get_python_executable()
    spec = WorkerJobSpec(
        job_id="test_interp_001",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="echo",
        python_executable=py_exec,
        params={"message": "dedicated_env_test"},
    )
    result = SubprocessRunner.run_job(spec, python_executable=py_exec)
    assert result.status == WorkerStatus.SUCCESS
    assert result.payload.get("echo") == "dedicated_env_test"


def test_03_04_05_worker_trains_on_train_only_no_leakage(tmp_path, classification_data):
    """
    3. Worker trains on TRAIN data.
    4. Validation data is NOT included in the fit job.
    5. Test data is NOT included in the fit job.
    """
    df_train, df_val = classification_data
    art_dir = tmp_path / "ag_model_train_only"

    result = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=10.0,
        presets="medium_quality",
        auto_stack=False,
        output_dir=art_dir,
        working_dir=tmp_path / "stage_03",
    )

    assert result.status == WorkerStatus.SUCCESS
    assert result.model_best is not None
    assert Path(result.artifact_path).exists()
    assert result.problem_type == ProblemType.BINARY_CLASSIFICATION

    # Verify that only train rows were written to the IPC staging directory
    staged_train = pd.read_parquet(tmp_path / "stage_03" / "train.parquet")
    assert len(staged_train) == len(df_train)
    assert not (tmp_path / "stage_03" / "val.parquet").exists()
    assert not (tmp_path / "stage_03" / "test.parquet").exists()


def test_06_classification_prediction_output_valid(tmp_path, classification_data):
    """6. Classification prediction output is valid and non-empty."""
    df_train, df_val = classification_data
    art_dir = tmp_path / "ag_model_clf"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=10.0,
        output_dir=art_dir,
        working_dir=tmp_path / "stage_06",
    )
    assert fit_res.status == WorkerStatus.SUCCESS

    preds, probs = AutoGluonBackend.predict(
        df_input=df_val.drop(columns=["target"]),
        artifact_path=art_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=["leave", "stay"],
    )

    assert len(preds) == len(df_val)
    assert set(preds).issubset({"stay", "leave"})
    assert probs is not None
    assert probs.shape == (len(df_val), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)


def test_07_regression_prediction_output_valid(tmp_path, regression_data):
    """7. Regression prediction output is valid and 1D float array."""
    df_train, df_val = regression_data
    art_dir = tmp_path / "ag_model_reg"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
        eval_metric="r2",
        time_limit_seconds=10.0,
        output_dir=art_dir,
        working_dir=tmp_path / "stage_07",
    )
    assert fit_res.status == WorkerStatus.SUCCESS

    preds, probs = AutoGluonBackend.predict(
        df_input=df_val.drop(columns=["price"]),
        artifact_path=art_dir,
        problem_type=ProblemType.REGRESSION,
    )

    assert len(preds) == len(df_val)
    assert probs is None
    assert isinstance(preds[0], (float, np.floating, int, np.integer))


def test_08_probability_class_ordering_normalization(tmp_path, classification_data):
    """8. Probability class ordering is normalized strictly to expected_class_labels order."""
    df_train, df_val = classification_data
    art_dir = tmp_path / "ag_model_order"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=10.0,
        output_dir=art_dir,
        working_dir=tmp_path / "stage_08",
    )
    assert fit_res.status == WorkerStatus.SUCCESS

    # Order 1: ['leave', 'stay']
    _, probs_order1 = AutoGluonBackend.predict(
        df_input=df_val.drop(columns=["target"]),
        artifact_path=art_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=["leave", "stay"],
    )

    # Order 2: ['stay', 'leave']
    _, probs_order2 = AutoGluonBackend.predict(
        df_input=df_val.drop(columns=["target"]),
        artifact_path=art_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=["stay", "leave"],
    )

    assert probs_order1 is not None and probs_order2 is not None
    # Verify column 0 in order1 matches column 1 in order2
    assert np.allclose(probs_order1[:, 0], probs_order2[:, 1], atol=1e-5)
    assert np.allclose(probs_order1[:, 1], probs_order2[:, 0], atol=1e-5)


def test_09_missing_target_is_rejected(tmp_path, classification_data):
    """9. Missing target column in training dataframe raises ValueError."""
    df_train, _ = classification_data
    with pytest.raises(ValueError, match="missing from df_train"):
        AutoGluonBackend.fit(
            df_train=df_train.drop(columns=["target"]),
            target_column="target",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            time_limit_seconds=5.0,
            output_dir=tmp_path / "ag_missing_target",
        )


def test_10_backend_import_failure_reported(tmp_path):
    """10. Backend import failure inside worker returns FAILED result with clear error."""
    # Execute autogluon_worker in the main interpreter (which has no autogluon installed)
    spec = WorkerJobSpec(
        job_id="test_import_err_010",
        worker_module="backend.engine.backends.workers.autogluon_worker",
        job_type="fit",
        time_limit_seconds=10.0,
        python_executable=sys.executable,  # Main python does not have autogluon
        params={
            "train_parquet_path": str(tmp_path / "dummy.parquet"),
            "target_column": "target",
            "artifact_dir": str(tmp_path / "dummy_art"),
        },
    )
    result = SubprocessRunner.run_job(spec, working_dir=tmp_path, python_executable=sys.executable)
    assert result.status == WorkerStatus.FAILED
    assert "not installed" in (result.error_message or "").lower()


def test_11_training_failure_reported(tmp_path):
    """11. Training failure with corrupt data returns FAILED result."""
    df_corrupt = pd.DataFrame({"a": [1, 2], "target": ["single_class", "single_class"]})
    # If binary classification has only 1 class, AutoGluon rejects it
    result = AutoGluonBackend.fit(
        df_train=df_corrupt,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=5.0,
        output_dir=tmp_path / "ag_corrupt",
        working_dir=tmp_path / "stage_11",
    )
    assert result.status == WorkerStatus.FAILED
    assert result.error_message is not None


def test_12_artifact_path_is_unique(tmp_path, classification_data):
    """12. Each fit invocation creates a unique independent artifact directory."""
    df_train, _ = classification_data
    res1 = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=8.0,
        output_dir=tmp_path / "run_ag_001",
        working_dir=tmp_path / "stage_12_1",
    )
    res2 = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=8.0,
        output_dir=tmp_path / "run_ag_002",
        working_dir=tmp_path / "stage_12_2",
    )
    assert res1.artifact_path != res2.artifact_path
    assert Path(res1.artifact_path).exists()
    assert Path(res2.artifact_path).exists()


def test_13_14_saved_model_reload_and_prediction_equality(tmp_path, classification_data):
    """
    13. Saved model reload works from disk.
    14. Reloaded validation predictions match initial recorded predictions.
    """
    df_train, df_val = classification_data
    art_dir = tmp_path / "ag_model_persist"

    fit_res = AutoGluonBackend.fit(
        df_train=df_train,
        target_column="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        time_limit_seconds=10.0,
        output_dir=art_dir,
        working_dir=tmp_path / "stage_13",
    )
    assert fit_res.status == WorkerStatus.SUCCESS

    # First prediction run
    preds_initial, probs_initial = AutoGluonBackend.predict(
        df_input=df_val.drop(columns=["target"]),
        artifact_path=art_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=["leave", "stay"],
    )

    # Second fresh prediction run (simulating independent reload in separate process)
    preds_reloaded, probs_reloaded = AutoGluonBackend.predict(
        df_input=df_val.drop(columns=["target"]),
        artifact_path=art_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        expected_class_labels=["leave", "stay"],
    )

    # Assert exact prediction equality
    np.testing.assert_array_equal(preds_initial, preds_reloaded)
    np.testing.assert_allclose(probs_initial, probs_reloaded, atol=1e-6)


def test_15_timeout_propagation_from_subprocess_runner(tmp_path):
    """15. Subprocess timeout is cleanly propagated and terminates hung worker."""
    py_exec = AutoGluonBackend.get_python_executable()
    spec = WorkerJobSpec(
        job_id="test_timeout_ag_015",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="sleep",
        time_limit_seconds=0.6,
        python_executable=py_exec,
        params={"duration": 10.0},
    )
    result = SubprocessRunner.run_job(spec, working_dir=tmp_path, python_executable=py_exec)
    assert result.status == WorkerStatus.TIMEOUT
    assert "timed out" in (result.error_message or "").lower()
