import os
import sys
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import psutil

from backend.engine.contracts.schemas import (
    ProblemType,
    WorkerStatus,
    WorkerJobSpec,
    WorkerResult,
    AutoGluonBackendResult,
)
from backend.engine.backends.subprocess_runner import SubprocessRunner
from backend.engine.evaluation.evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)


class AutoGluonBackend:
    """
    Parent-Side Adapter for AutoGluon Tabular AutoML.
    Orchestrates execution strictly through the isolated SubprocessRunner infrastructure,
    ensuring zero memory leakage, strict timeout enforcement, and native artifact persistence.
    """

    DEFAULT_VENV_PATH = Path("E:/ML MODEL/.venvs/autogluon/Scripts/python.exe")

    @classmethod
    def get_python_executable(cls) -> str:
        """Resolves the Python interpreter dedicated to the AutoGluon backend."""
        # 1. Project-configured dedicated environment
        if cls.DEFAULT_VENV_PATH.exists():
            return str(cls.DEFAULT_VENV_PATH.resolve())

        # 2. Workspace-relative .venvs/autogluon
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        rel_venv = project_root / ".venvs" / "autogluon" / "Scripts" / "python.exe"
        if rel_venv.exists():
            return str(rel_venv.resolve())

        # 3. Linux/Unix virtualenv structure fallback
        rel_venv_unix = project_root / ".venvs" / "autogluon" / "bin" / "python"
        if rel_venv_unix.exists():
            return str(rel_venv_unix.resolve())

        # 4. Fallback to active runtime interpreter
        logger.warning(
            f"Dedicated AutoGluon venv not found at {cls.DEFAULT_VENV_PATH}. Falling back to sys.executable."
        )
        return str(Path(sys.executable).resolve())

    @classmethod
    def fit(
        cls,
        df_train: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        eval_metric: Optional[str] = None,
        time_limit_seconds: float = 60.0,
        presets: str = "medium_quality",
        auto_stack: bool = False,
        num_cpus: Optional[int] = None,
        num_gpus: int = 0,
        output_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        memory_limit_mb: Optional[float] = None,
        min_available_ram_gb: float = 1.5,
    ) -> AutoGluonBackendResult:
        """
        Fits AutoGluon Tabular on isolated TRAIN data in a dedicated subprocess.

        Parameters:
            df_train: Training DataFrame partition ONLY.
            target_column: Name of target column in df_train.
            problem_type: ProblemType classification or regression.
            eval_metric: Optimization metric (e.g. 'f1', 'r2', 'balanced_accuracy').
            time_limit_seconds: Maximum wall-clock training time limit in seconds.
            presets: AutoGluon quality preset ('medium_quality' recommended for 8GB RAM).
            auto_stack: Whether to enable multi-layer stacking (False recommended for 8GB RAM).
            num_cpus: CPU core allocation limit.
            num_gpus: GPU count (0 for CPU-only).
            output_dir: Persistent storage directory for native AutoGluon artifact.
            working_dir: Temporary directory for IPC parquet files.
            memory_limit_mb: Optional hard RSS memory ceiling for the worker subprocess tree.
            min_available_ram_gb: Pre-flight RAM availability requirement.

        Returns:
            AutoGluonBackendResult containing training summary and artifact path.
        """
        # 1. Validation & Pre-flight Checks
        if target_column not in df_train.columns:
            raise ValueError(
                f"Target column '{target_column}' is missing from df_train. Columns: {list(df_train.columns)}"
            )

        avail_ram_gb = psutil.virtual_memory().available / (1024.0 ** 3)
        if avail_ram_gb < min_available_ram_gb:
            logger.warning(
                f"Low system memory detected ({avail_ram_gb:.2f} GB available < {min_available_ram_gb} GB). "
                f"Proceeding with conservative resource allocation."
            )

        # 2. Setup Staging & Artifact Paths
        run_id = uuid.uuid4().hex[:8]
        stage_dir = Path(working_dir) if working_dir else Path(os.environ.get("TEMP", ".")) / f"ag_run_{run_id}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        if output_dir:
            artifact_path = Path(output_dir).resolve()
        else:
            artifact_path = (stage_dir / "autogluon_model").resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)

        train_parquet = stage_dir / "train.parquet"
        df_train.to_parquet(train_parquet, index=False)

        # 3. Build Worker Job Specification
        python_bin = cls.get_python_executable()
        job_spec = WorkerJobSpec(
            job_id=f"ag_fit_{run_id}",
            worker_module="backend.engine.backends.workers.autogluon_worker",
            job_type="fit",
            time_limit_seconds=float(time_limit_seconds + 45.0),  # Grace period for startup/teardown
            memory_limit_mb=memory_limit_mb,
            python_executable=python_bin,
            params={
                "train_parquet_path": str(train_parquet.resolve()),
                "target_column": target_column,
                "problem_type": problem_type.value if hasattr(problem_type, "value") else str(problem_type),
                "eval_metric": eval_metric,
                "artifact_dir": str(artifact_path),
                "time_limit_seconds": float(time_limit_seconds),
                "presets": presets,
                "auto_stack": auto_stack,
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
                "save_space": True,
            },
        )

        # 4. Dispatch via SubprocessRunner
        logger.info(
            f"[AutoGluonBackend] Launching AutoGluon fit job {job_spec.job_id} on {len(df_train)} rows using {python_bin}..."
        )
        worker_res: WorkerResult = SubprocessRunner.run_job(
            job_spec,
            working_dir=stage_dir,
            python_executable=python_bin,
        )

        # 5. Process & Structure Result
        if worker_res.status != WorkerStatus.SUCCESS:
            err_msg = worker_res.error_message or f"AutoGluon worker terminated with status {worker_res.status}"
            logger.error(f"[AutoGluonBackend] Fit failed: {err_msg}\nSTDERR: {worker_res.stderr}")
            return AutoGluonBackendResult(
                backend_name="autogluon",
                status=worker_res.status,
                model_best=None,
                problem_type=problem_type,
                eval_metric=eval_metric or "default",
                artifact_path=str(artifact_path),
                fit_time_seconds=worker_res.runtime_seconds,
                peak_memory_mb=worker_res.peak_memory_mb,
                leaderboard=[],
                val_metrics={},
                error_message=err_msg,
            )

        payload = worker_res.payload or {}
        return AutoGluonBackendResult(
            backend_name="autogluon",
            status=WorkerStatus.SUCCESS,
            model_best=payload.get("model_best"),
            problem_type=problem_type,
            eval_metric=payload.get("eval_metric", eval_metric or "default"),
            artifact_path=str(artifact_path),
            fit_time_seconds=worker_res.runtime_seconds,
            peak_memory_mb=worker_res.peak_memory_mb,
            class_labels=payload.get("class_labels"),
            leaderboard=payload.get("leaderboard", []),
            val_metrics={},
            autogluon_version=payload.get("autogluon_version"),
            error_message=None,
        )

    @classmethod
    def predict(
        cls,
        df_input: pd.DataFrame,
        artifact_path: Union[str, Path],
        problem_type: ProblemType,
        expected_class_labels: Optional[List[Any]] = None,
        working_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Executes prediction using a saved native AutoGluon artifact via isolated worker.

        Returns:
            Tuple of (predictions: np.ndarray, normalized_probabilities: Optional[np.ndarray]).
        """
        art_path = Path(artifact_path).resolve()
        if not art_path.exists():
            raise FileNotFoundError(f"AutoGluon artifact not found at: {art_path}")

        run_id = uuid.uuid4().hex[:8]
        stage_dir = Path(working_dir) if working_dir else Path(os.environ.get("TEMP", ".")) / f"ag_pred_{run_id}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        input_parquet = stage_dir / "input.parquet"
        df_input.to_parquet(input_parquet, index=False)

        python_bin = cls.get_python_executable()
        job_spec = WorkerJobSpec(
            job_id=f"ag_pred_{run_id}",
            worker_module="backend.engine.backends.workers.autogluon_worker",
            job_type="predict",
            time_limit_seconds=60.0,
            python_executable=python_bin,
            params={
                "artifact_dir": str(art_path),
                "input_parquet_path": str(input_parquet.resolve()),
                "problem_type": problem_type.value if hasattr(problem_type, "value") else str(problem_type),
                "expected_class_labels": expected_class_labels,
            },
        )

        worker_res: WorkerResult = SubprocessRunner.run_job(
            job_spec,
            working_dir=stage_dir,
            python_executable=python_bin,
        )

        if worker_res.status != WorkerStatus.SUCCESS:
            err_msg = worker_res.error_message or f"AutoGluon predict failed with status {worker_res.status}"
            raise RuntimeError(f"AutoGluon prediction failed: {err_msg}\nSTDERR: {worker_res.stderr}")

        payload = worker_res.payload or {}
        preds_list = payload.get("predictions", [])
        probs_list = payload.get("probabilities")

        preds_arr = np.array(preds_list)
        probs_arr = np.array(probs_list) if probs_list is not None else None

        return preds_arr, probs_arr

    @classmethod
    def evaluate(
        cls,
        df_val: pd.DataFrame,
        target_column: str,
        artifact_path: Union[str, Path],
        problem_type: ProblemType,
        expected_class_labels: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluates saved AutoGluon model on independent VALIDATION dataset partition.
        Computes standard metrics using the engine's MetricsEvaluator.
        """
        if target_column not in df_val.columns:
            raise ValueError(f"Target column '{target_column}' missing from validation set.")

        y_true = df_val[target_column].to_numpy()
        df_features = df_val.drop(columns=[target_column])

        preds, probs = cls.predict(
            df_input=df_features,
            artifact_path=artifact_path,
            problem_type=problem_type,
            expected_class_labels=expected_class_labels,
        )

        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            is_binary = problem_type == ProblemType.BINARY_CLASSIFICATION
            return MetricsEvaluator.evaluate_classification(
                y_true=y_true,
                y_pred=preds,
                y_prob=probs,
                is_binary=is_binary,
            )
        elif problem_type == ProblemType.REGRESSION:
            return MetricsEvaluator.evaluate_regression(
                y_true=y_true,
                y_pred=preds,
            )
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
