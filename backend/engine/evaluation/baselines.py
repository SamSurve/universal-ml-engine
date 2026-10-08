import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from backend.engine.contracts.schemas import (
    ProblemType,
    ColumnProfile,
    BaselineModelResult,
    BaselineSuiteResult,
)
from backend.engine.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from backend.engine.evaluation.evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """
    Evaluates mandatory standard baseline models (Dummy and Linear) on the Validation set.

    Guarantees:
    - Fit strictly on Train partition (zero data leakage).
    - Evaluated strictly on Validation partition.
    - Test partition is NEVER accessed.
    - Standardized metrics via MetricsEvaluator.
    - Produces structured BaselineSuiteResult for champion selection.
    """

    @classmethod
    def evaluate_baselines(
        cls,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        column_profiles: Optional[Dict[str, ColumnProfile]] = None,
        random_state: int = 42,
    ) -> BaselineSuiteResult:
        """
        Fits and evaluates baseline models on the Train/Validation splits.

        Classification Baselines:
          1. DummyClassifier (most frequent class)
          2. LogisticRegression (with standard scaling and one-hot encoding)

        Regression Baselines:
          1. DummyRegressor (mean target value)
          2. Ridge (with standard scaling and one-hot encoding)
        """
        if df_train.empty or df_val.empty:
            raise ValueError("Train and Validation DataFrames must both be non-empty.")

        if target_column not in df_train.columns or target_column not in df_val.columns:
            raise ValueError(f"Target column '{target_column}' must exist in both Train and Validation sets.")

        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]
        X_val = df_val.drop(columns=[target_column])
        y_val = df_val[target_column]

        is_classification = problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]
        is_binary = problem_type == ProblemType.BINARY_CLASSIFICATION

        # Prepare target encoding for classification
        label_encoder: Optional[LabelEncoder] = None
        if is_classification:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train.astype(str))

            known_classes = set(label_encoder.classes_)
            val_classes = set(y_val.astype(str))
            unseen_classes = val_classes - known_classes
            if unseen_classes:
                raise ValueError(
                    f"Validation set contains unseen target classes {sorted(list(unseen_classes))}. "
                    f"Known classes from training: {sorted(list(known_classes))}."
                )
            y_val_encoded = label_encoder.transform(y_val.astype(str))
        else:
            y_train_encoded = y_train.to_numpy(dtype=float)
            y_val_encoded = y_val.to_numpy(dtype=float)

        # Identify feature types for pipeline builder
        num_cols, cat_cols, date_cols = PreprocessingPipelineBuilder.identify_feature_types(
            X_train, column_profiles or {}
        )

        # Determine primary metric
        imbalance_ratio = 1.0
        if is_classification:
            counts = pd.Series(y_train_encoded).value_counts()
            if len(counts) >= 2:
                imbalance_ratio = float(counts.max() / counts.min())

        primary_metric = MetricsEvaluator.get_primary_metric_name(problem_type, imbalance_ratio)
        higher_is_better = primary_metric in [
            "f1", "f1_macro", "r2", "accuracy", "balanced_accuracy", "roc_auc"
        ]

        # -------------------------------------------------------------
        # 1. Evaluate Dummy Baseline
        # -------------------------------------------------------------
        dummy_start = time.time()
        dummy_name = "Dummy Classifier (Most Frequent)" if is_classification else "Dummy Regressor (Mean)"

        try:
            if is_classification:
                dummy_model = DummyClassifier(strategy="most_frequent")
                dummy_model.fit(X_train, y_train_encoded)
                y_pred_dummy = dummy_model.predict(X_val)

                y_prob_dummy = None
                if hasattr(dummy_model, "predict_proba"):
                    try:
                        y_prob_dummy = dummy_model.predict_proba(X_val)
                    except Exception:
                        y_prob_dummy = None

                dummy_metrics = MetricsEvaluator.evaluate_classification(
                    y_val_encoded, y_pred_dummy, y_prob_dummy, is_binary=is_binary
                )
            else:
                dummy_model = DummyRegressor(strategy="mean")
                dummy_model.fit(X_train, y_train_encoded)
                y_pred_dummy = dummy_model.predict(X_val)
                dummy_metrics = MetricsEvaluator.evaluate_regression(y_val_encoded, y_pred_dummy)

            dummy_fit_time = round(time.time() - dummy_start, 4)
            dummy_score = dummy_metrics[primary_metric]

            dummy_result = BaselineModelResult(
                model_name=dummy_name,
                problem_type=problem_type,
                primary_metric=primary_metric,
                val_score=round(dummy_score, 4),
                val_metrics=dummy_metrics,
                fit_time_seconds=dummy_fit_time,
                status="success",
            )
        except Exception as e:
            logger.error(f"Dummy baseline evaluation failed: {e}")
            dummy_result = BaselineModelResult(
                model_name=dummy_name,
                problem_type=problem_type,
                primary_metric=primary_metric,
                val_score=-999.0 if higher_is_better else 999999.0,
                val_metrics={},
                fit_time_seconds=round(time.time() - dummy_start, 4),
                status="failed",
                error_message=str(e),
            )

        # -------------------------------------------------------------
        # 2. Evaluate Linear / Ridge Baseline (Zero-Leakage Pipeline)
        # -------------------------------------------------------------
        linear_start = time.time()
        linear_name = "Logistic Regression Baseline" if is_classification else "Ridge Regression Baseline"

        try:
            # Build preprocessor fitted strictly on X_train inside the pipeline
            preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                datetime_cols=date_cols,
                scale_numeric=True,  # Linear models require standardized inputs
            )

            if is_classification:
                estimator = LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                )
            else:
                estimator = Ridge(random_state=random_state)

            linear_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            # Fit strictly on train fold
            linear_pipeline.fit(X_train, y_train_encoded)

            # Predict on validation fold
            y_pred_linear = linear_pipeline.predict(X_val)

            if is_classification:
                y_prob_linear = None
                if hasattr(linear_pipeline, "predict_proba"):
                    try:
                        y_prob_linear = linear_pipeline.predict_proba(X_val)
                    except Exception:
                        y_prob_linear = None

                linear_metrics = MetricsEvaluator.evaluate_classification(
                    y_val_encoded, y_pred_linear, y_prob_linear, is_binary=is_binary
                )
            else:
                linear_metrics = MetricsEvaluator.evaluate_regression(y_val_encoded, y_pred_linear)

            linear_fit_time = round(time.time() - linear_start, 4)
            linear_score = linear_metrics[primary_metric]

            linear_result = BaselineModelResult(
                model_name=linear_name,
                problem_type=problem_type,
                primary_metric=primary_metric,
                val_score=round(linear_score, 4),
                val_metrics=linear_metrics,
                fit_time_seconds=linear_fit_time,
                status="success",
            )
        except Exception as e:
            logger.error(f"Linear baseline evaluation failed: {e}")
            linear_result = BaselineModelResult(
                model_name=linear_name,
                problem_type=problem_type,
                primary_metric=primary_metric,
                val_score=-999.0 if higher_is_better else 999999.0,
                val_metrics={},
                fit_time_seconds=round(time.time() - linear_start, 4),
                status="failed",
                error_message=str(e),
            )

        # Rank baselines
        all_baselines = [dummy_result, linear_result]
        valid_baselines = [b for b in all_baselines if b.status == "success"]

        if valid_baselines:
            if higher_is_better:
                best_baseline = max(valid_baselines, key=lambda b: b.val_score)
            else:
                best_baseline = min(valid_baselines, key=lambda b: b.val_score)
            best_baseline_name = best_baseline.model_name
            best_baseline_score = best_baseline.val_score
        else:
            best_baseline_name = linear_name
            best_baseline_score = 0.0

        logger.info(
            f"Baselines Evaluated: {dummy_result.model_name} ({primary_metric}={dummy_result.val_score}) vs "
            f"{linear_result.model_name} ({primary_metric}={linear_result.val_score}) -> Best: {best_baseline_name}"
        )

        return BaselineSuiteResult(
            problem_type=problem_type,
            primary_metric=primary_metric,
            dummy_baseline=dummy_result,
            linear_baseline=linear_result,
            best_baseline_name=best_baseline_name,
            best_baseline_score=best_baseline_score,
            all_baselines=all_baselines,
        )
