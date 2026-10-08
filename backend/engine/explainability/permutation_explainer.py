import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from backend.engine.contracts.schemas import (
    ProblemType,
    FeatureImportanceEntry,
    PermutationImportanceResult,
)
from backend.engine.evaluation.evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)


class PermutationExplainer:
    """
    Lightweight, backend-agnostic Permutation Feature Importance service.

    Guarantees:
    - Evaluates strictly on external VALIDATION partition (never touches TEST).
    - Works with any predictor supporting standard `.predict(DataFrame)`.
    - Supports both Classification (binary & multiclass) and Regression.
    - Accurately tracks primary metric directionality (higher-is-better vs lower-is-better).
    - Avoids heavy SHAP dependencies for ensembles.
    - Gracefully handles edge cases (empty data, constant features, missing columns).
    """

    HIGHER_IS_BETTER_METRICS = {
        "f1",
        "f1_macro",
        "r2",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "precision",
        "recall",
    }

    @classmethod
    def compute_importance(
        cls,
        predictor: Any,
        df_val: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        primary_metric: Optional[str] = None,
        n_repeats: int = 5,
        max_samples: int = 2000,
        random_state: int = 42,
    ) -> PermutationImportanceResult:
        """
        Calculates permutation feature importance on external VALIDATION data.

        Parameters:
            predictor: Any trained model/predictor with `.predict(df_features)` method.
            df_val: External VALIDATION partition containing features and target.
            target_column: Name of the target column.
            problem_type: ProblemType (Binary, Multiclass, or Regression).
            primary_metric: Primary evaluation metric name. If None, derived from problem_type.
            n_repeats: Number of shuffle repetitions per feature.
            max_samples: Maximum validation rows to sample for efficiency.
            random_state: Deterministic random seed.

        Returns:
            PermutationImportanceResult containing ranked FeatureImportanceEntry items.
        """
        start_time = time.time()
        warnings: List[str] = []

        # 1. Edge-Case Validation
        if df_val is None or df_val.empty:
            logger.warning("[PermutationExplainer] Validation DataFrame is empty.")
            return PermutationImportanceResult(
                problem_type=problem_type,
                primary_metric=primary_metric or "unknown",
                baseline_score=0.0,
                importance_entries=[],
                status="empty_dataset",
                warnings=["Validation DataFrame is empty."],
                computation_time_seconds=0.0,
            )

        if target_column not in df_val.columns:
            raise ValueError(f"Target column '{target_column}' is missing from validation DataFrame.")

        feature_cols = [c for c in df_val.columns if c != target_column]
        if not feature_cols:
            logger.warning("[PermutationExplainer] No feature columns found in validation DataFrame.")
            return PermutationImportanceResult(
                problem_type=problem_type,
                primary_metric=primary_metric or "unknown",
                baseline_score=0.0,
                importance_entries=[],
                status="no_features",
                warnings=["Validation DataFrame contains only the target column."],
                computation_time_seconds=0.0,
            )

        # Subsample if validation dataset is exceptionally large
        if len(df_val) > max_samples:
            eval_df = df_val.sample(n=max_samples, random_state=random_state).copy()
            warnings.append(f"Validation data subsampled from {len(df_val)} to {max_samples} rows for permutation efficiency.")
        else:
            eval_df = df_val.copy()

        X_val = eval_df[feature_cols].copy()
        y_val = eval_df[target_column].to_numpy()

        # 2. Determine Primary Metric & Directionality
        is_classification = problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]
        is_binary = problem_type == ProblemType.BINARY_CLASSIFICATION

        if not primary_metric:
            primary_metric = MetricsEvaluator.get_primary_metric_name(problem_type)

        metric_lower = primary_metric.lower()
        higher_is_better = metric_lower in cls.HIGHER_IS_BETTER_METRICS

        # 3. Compute Baseline Performance on Untouched Validation
        try:
            y_pred_base = predictor.predict(X_val)
            y_prob_base = None
            if is_classification and hasattr(predictor, "predict_proba"):
                try:
                    y_prob_base = predictor.predict_proba(X_val)
                except Exception:
                    y_prob_base = None

            if is_classification:
                base_metrics = MetricsEvaluator.evaluate_classification(
                    y_true=y_val,
                    y_pred=y_pred_base,
                    y_prob=y_prob_base,
                    is_binary=is_binary,
                )
            else:
                base_metrics = MetricsEvaluator.evaluate_regression(
                    y_true=y_val,
                    y_pred=y_pred_base,
                )

            baseline_score = float(base_metrics.get(metric_lower, 0.0))
        except Exception as e:
            logger.error(f"[PermutationExplainer] Baseline prediction failed: {e}")
            return PermutationImportanceResult(
                problem_type=problem_type,
                primary_metric=primary_metric,
                baseline_score=0.0,
                importance_entries=[],
                status="prediction_failure",
                warnings=[f"Failed to obtain baseline predictions: {str(e)}"],
                computation_time_seconds=round(time.time() - start_time, 4),
            )

        # 4. Permutation Iteration Across Features
        raw_scores: Dict[str, List[float]] = {}
        feature_mean_scores: Dict[str, float] = {}
        feature_std_scores: Dict[str, float] = {}

        for feat in feature_cols:
            col_scores: List[float] = []
            is_constant_col = (X_val[feat].nunique() <= 1)

            if is_constant_col:
                # Shuffling a constant feature has 0 impact by definition
                raw_scores[feat] = [0.0] * n_repeats
                feature_mean_scores[feat] = 0.0
                feature_std_scores[feat] = 0.0
                continue

            for r in range(n_repeats):
                seed = random_state + r * 1000 + (abs(hash(feat)) % 1000)
                rng = np.random.RandomState(seed)

                # Shuffle feature values
                X_shuffled = X_val.copy()
                X_shuffled[feat] = rng.permutation(X_val[feat].to_numpy())

                try:
                    y_pred_shuf = predictor.predict(X_shuffled)
                    y_prob_shuf = None
                    if is_classification and hasattr(predictor, "predict_proba") and metric_lower == "roc_auc":
                        try:
                            y_prob_shuf = predictor.predict_proba(X_shuffled)
                        except Exception:
                            y_prob_shuf = None

                    if is_classification:
                        shuf_metrics = MetricsEvaluator.evaluate_classification(
                            y_true=y_val,
                            y_pred=y_pred_shuf,
                            y_prob=y_prob_shuf,
                            is_binary=is_binary,
                        )
                    else:
                        shuf_metrics = MetricsEvaluator.evaluate_regression(
                            y_true=y_val,
                            y_pred=y_pred_shuf,
                        )

                    shuf_score = float(shuf_metrics.get(metric_lower, 0.0))

                    if higher_is_better:
                        # Drop in score = baseline - shuffled (higher drop = more important)
                        delta = baseline_score - shuf_score
                    else:
                        # Increase in error = shuffled - baseline (higher error increase = more important)
                        delta = shuf_score - baseline_score

                    col_scores.append(delta)
                except Exception as e:
                    logger.warning(f"[PermutationExplainer] Permutation failed for feature '{feat}' rep {r}: {e}")
                    col_scores.append(0.0)

            raw_scores[feat] = col_scores
            feature_mean_scores[feat] = float(np.mean(col_scores))
            feature_std_scores[feat] = float(np.std(col_scores))

        # 5. Rank Features and Compute Relative Importance
        sorted_features = sorted(
            feature_mean_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        positive_sum = sum(max(0.0, score) for _, score in sorted_features)

        importance_entries: List[FeatureImportanceEntry] = []
        for rank, (feat, mean_score) in enumerate(sorted_features, start=1):
            if positive_sum > 0.0:
                rel_pct = round(100.0 * max(0.0, mean_score) / positive_sum, 2)
            else:
                rel_pct = round(100.0 / len(sorted_features), 2) if sorted_features else 0.0

            std_score = feature_std_scores.get(feat, 0.0)
            importance_entries.append(
                FeatureImportanceEntry(
                    feature_name=feat,
                    raw_feature_name=feat,
                    importance_score=round(mean_score, 4),
                    relative_importance_pct=rel_pct,
                    rank=rank,
                    std_dev=round(std_score, 4),
                )
            )

        comp_time = round(time.time() - start_time, 4)
        logger.info(
            f"[PermutationExplainer] Computed importance for {len(feature_cols)} features "
            f"across {len(eval_df)} validation rows ({n_repeats} repeats) in {comp_time:.2f}s."
        )

        return PermutationImportanceResult(
            problem_type=problem_type,
            primary_metric=primary_metric,
            baseline_score=round(baseline_score, 4),
            importance_entries=importance_entries,
            raw_scores=raw_scores,
            n_repeats=n_repeats,
            sample_size=len(eval_df),
            computation_time_seconds=comp_time,
            status="success",
            warnings=warnings,
        )
