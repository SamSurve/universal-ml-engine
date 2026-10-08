import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import (
    ProblemType,
    ChampionCandidate,
    FinalTestEvaluationResult,
)
from backend.engine.evaluation.evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)


class FinalTestEvaluator:
    """
    Evaluates the selected champion model on the untouched TEST dataset partition exactly once.
    
    Guarantees:
    - Never modifies or tunes the champion during test evaluation.
    - Test partition is used strictly for final unbiased performance assessment.
    - Preserves data provenance via cryptographic SHA-256 partition hashing.
    """

    @classmethod
    def _compute_dataframe_sha256(cls, df: pd.DataFrame) -> str:
        """Computes a deterministic SHA-256 hash of the DataFrame."""
        hashed_series = pd.util.hash_pandas_object(df, index=False)
        return hashlib.sha256(hashed_series.to_numpy().tobytes()).hexdigest()

    @classmethod
    def evaluate_champion(
        cls,
        champion: ChampionCandidate,
        df_test: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        expected_class_labels: Optional[List[Any]] = None,
        test_sha256: Optional[str] = None,
    ) -> FinalTestEvaluationResult:
        """
        Executes final single-pass evaluation on the untouched TEST partition.

        Parameters:
            champion: The selected winning ChampionCandidate.
            df_test: Untouched test dataset partition.
            target_column: Name of target column.
            problem_type: Task problem type.
            expected_class_labels: Class ordering list for classification.
            test_sha256: Optional precomputed hash of the test partition.

        Returns:
            FinalTestEvaluationResult with full test metrics and provenance data.
        """
        start_time = time.time()

        if df_test.empty:
            raise ValueError("Test DataFrame cannot be empty.")

        if target_column not in df_test.columns:
            raise ValueError(f"Target column '{target_column}' is missing from test DataFrame.")

        actual_sha256 = test_sha256 or cls._compute_dataframe_sha256(df_test)
        y_true = df_test[target_column].to_numpy()
        df_features = df_test.drop(columns=[target_column])

        is_classification = problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]
        is_binary = problem_type == ProblemType.BINARY_CLASSIFICATION

        preds: np.ndarray
        probs: Optional[np.ndarray] = None

        # Predict using champion's native backend artifact or in-memory instance
        if champion.backend_name == "autogluon":
            if not champion.artifact_path:
                raise ValueError("AutoGluon champion candidate is missing artifact_path.")
            from backend.engine.backends.autogluon_backend import AutoGluonBackend

            preds, probs = AutoGluonBackend.predict(
                df_input=df_features,
                artifact_path=champion.artifact_path,
                problem_type=problem_type,
                expected_class_labels=expected_class_labels or champion.class_labels,
            )

        elif champion.model_instance is not None:
            preds = champion.model_instance.predict(df_features)
            if is_classification and hasattr(champion.model_instance, "predict_proba"):
                try:
                    probs = champion.model_instance.predict_proba(df_features)
                except Exception:
                    probs = None
        else:
            raise ValueError(
                f"Cannot evaluate champion '{champion.model_name}' ({champion.backend_name}): "
                f"Neither artifact_path nor model_instance is available."
            )

        # Compute standard project evaluation metrics
        if is_classification:
            test_metrics = MetricsEvaluator.evaluate_classification(
                y_true=y_true,
                y_pred=preds,
                y_prob=probs,
                is_binary=is_binary,
            )
        elif problem_type == ProblemType.REGRESSION:
            test_metrics = MetricsEvaluator.evaluate_regression(
                y_true=y_true,
                y_pred=preds,
            )
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        eval_time = time.time() - start_time
        logger.info(
            f"[FinalTestEvaluator] Evaluated champion '{champion.model_name}' on {len(df_test)} test rows. "
            f"Metrics: {test_metrics}"
        )

        return FinalTestEvaluationResult(
            champion_model_name=champion.model_name,
            champion_backend=champion.backend_name,
            problem_type=problem_type,
            test_metrics=test_metrics,
            test_rows=len(df_test),
            test_sha256=actual_sha256,
            evaluation_time_seconds=eval_time,
        )
