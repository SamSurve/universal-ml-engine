import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from backend.engine.contracts.schemas import ProblemType, LeaderboardEntry, ModelEvaluationResult


class MetricsEvaluator:
    """Computes comprehensive evaluation metrics for classification and regression tasks."""

    @classmethod
    def evaluate_classification(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        is_binary: bool = True,
    ) -> Dict[str, float]:
        from sklearn.preprocessing import LabelEncoder

        # Align y_true and y_pred types to avoid string/number mixing errors
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_true_arr.dtype != y_pred_arr.dtype:
            # If one is string and one is numeric, stringify both
            if np.issubdtype(y_true_arr.dtype, np.number) != np.issubdtype(y_pred_arr.dtype, np.number):
                y_true_arr = y_true_arr.astype(str)
                y_pred_arr = y_pred_arr.astype(str)
            else:
                try:
                    y_pred_arr = y_pred_arr.astype(y_true_arr.dtype)
                except Exception:
                    y_true_arr = y_true_arr.astype(str)
                    y_pred_arr = y_pred_arr.astype(str)

        metrics = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        }

        # Calculate ROC-AUC if probabilities are available
        if y_prob is not None:
            try:
                # Encode y_true numerically for ROC-AUC
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true_arr)

                if is_binary:
                    # In binary classification, y_prob may be 1D or 2D (take column 1)
                    prob_1d = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
                    metrics["roc_auc"] = float(roc_auc_score(y_true_encoded, prob_1d))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true_encoded, y_prob, multi_class="ovr", average="weighted")
                    )
            except Exception:
                metrics["roc_auc"] = 0.0

        return {k: round(v, 4) for k, v in metrics.items()}

    @classmethod
    def evaluate_regression(
        cls, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        }

    @classmethod
    def get_primary_metric_name(
        cls, problem_type: ProblemType, class_imbalance_ratio: float = 1.0
    ) -> str:
        """
        Determines the primary metric for model ranking.
        For imbalanced classification, uses balanced_accuracy or f1, never raw accuracy.
        """
        if problem_type == ProblemType.REGRESSION:
            return "r2"  # Higher is better
        elif problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            # If imbalance ratio > 1.5, prioritize balanced_accuracy/f1
            if class_imbalance_ratio > 1.5:
                return "balanced_accuracy"
            return "f1"
        return "accuracy"

    @classmethod
    def build_leaderboard(
        cls, evaluation_results: List[ModelEvaluationResult], primary_metric: str
    ) -> List[LeaderboardEntry]:
        """
        Sorts models deterministically to build the final competition leaderboard.
        For R2/F1/accuracy/balanced_accuracy: higher is better.
        For MAE/RMSE/MSE: lower is better.
        """
        lower_is_better = primary_metric in ["mae", "mse", "rmse"]

        # Filter out failed models
        successful = [r for r in evaluation_results if r.status == "success"]
        failed = [r for r in evaluation_results if r.status != "success"]

        successful.sort(
            key=lambda x: x.mean_cv_score,
            reverse=not lower_is_better,
        )

        leaderboard: List[LeaderboardEntry] = []
        rank = 1

        for res in successful:
            leaderboard.append(
                LeaderboardEntry(
                    rank=rank,
                    model_name=res.model_name,
                    cv_score_mean=round(res.mean_cv_score, 4),
                    cv_score_std=round(res.std_cv_score, 4),
                    primary_metric=primary_metric,
                    summary_metrics=res.aggregated_metrics,
                    fit_time_seconds=round(res.total_fit_time, 2),
                    status="success",
                )
            )
            rank += 1

        for res in failed:
            leaderboard.append(
                LeaderboardEntry(
                    rank=rank,
                    model_name=res.model_name,
                    cv_score_mean=-999.0 if not lower_is_better else 999999.0,
                    cv_score_std=0.0,
                    primary_metric=primary_metric,
                    summary_metrics={},
                    fit_time_seconds=round(res.total_fit_time, 2),
                    status=f"failed: {res.error_message}",
                )
            )
            rank += 1

        return leaderboard
