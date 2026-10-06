import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from backend.engine.contracts.schemas import (
    ProblemType,
    FoldMetric,
    ModelEvaluationResult,
    ColumnProfile,
)
from backend.engine.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from backend.engine.evaluation.evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)


class CrossValidationRunner:
    """
    Executes leakage-safe cross-validation across all candidate models.

    Key guarantees:
    - 80% dev / 20% holdout split (untouched until final evaluation).
    - StratifiedKFold for classification (preserves class balance across folds).
    - KFold for regression.
    - Preprocessing pipeline is fitted inside each fold (zero data leakage).
    - A failure in an individual model is caught and isolated without aborting the experiment.
    """

    @classmethod
    def split_dev_holdout(
        cls,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        holdout_ratio: float = 0.20,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into 80% development data and 20% final untouched holdout data.
        Uses stratification for classification tasks.
        """
        stratify = None
        if problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]:
            # Stratify only if minimum class count >= 2
            y = df[target_column]
            counts = y.value_counts()
            if counts.min() >= 2:
                stratify = y

        df_dev, df_holdout = train_test_split(
            df,
            test_size=holdout_ratio,
            random_state=random_state,
            stratify=stratify,
        )

        return df_dev.reset_index(drop=True), df_holdout.reset_index(drop=True)

    @classmethod
    def run_cv(
        cls,
        df_dev: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        models: Dict[str, BaseEstimator],
        column_profiles: Dict[str, ColumnProfile],
        n_splits: int = 5,
        random_state: int = 42,
    ) -> Tuple[List[ModelEvaluationResult], Optional[LabelEncoder]]:
        """
        Performs 5-fold cross-validation on the development set for all candidate models.
        """
        X_dev = df_dev.drop(columns=[target_column])
        y_dev = df_dev[target_column]

        # Handle target label encoding for classification if target is non-numeric string/object
        label_encoder = None
        if problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]:
            label_encoder = LabelEncoder()
            y_dev_encoded = label_encoder.fit_transform(y_dev.astype(str))
        else:
            y_dev_encoded = y_dev.to_numpy(dtype=float)

        # Identify feature types
        num_cols, cat_cols, date_cols = PreprocessingPipelineBuilder.identify_feature_types(
            X_dev, column_profiles
        )

        # Calculate class imbalance ratio for classification
        imbalance_ratio = 1.0
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            counts = pd.Series(y_dev_encoded).value_counts()
            if len(counts) >= 2:
                imbalance_ratio = float(counts.max() / counts.min())

        primary_metric = MetricsEvaluator.get_primary_metric_name(problem_type, imbalance_ratio)

        # Configure K-Fold
        # Ensure n_splits does not exceed min class count
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            min_class_count = int(pd.Series(y_dev_encoded).value_counts().min())
            effective_splits = min(n_splits, max(2, min_class_count))
            splitter = StratifiedKFold(
                n_splits=effective_splits, shuffle=True, random_state=random_state
            )
            is_classification = True
            is_binary = problem_type == ProblemType.BINARY_CLASSIFICATION
        else:
            effective_splits = min(n_splits, max(2, len(df_dev) // 2))
            splitter = KFold(
                n_splits=effective_splits, shuffle=True, random_state=random_state
            )
            is_classification = False
            is_binary = False

        results: List[ModelEvaluationResult] = []

        for model_name, model_instance in models.items():
            start_total = time.time()
            fold_metrics: List[FoldMetric] = []
            failed = False
            error_msg = None

            try:
                for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_dev, y_dev_encoded)):
                    X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
                    y_train, y_val = y_dev_encoded[train_idx], y_dev_encoded[val_idx]

                    fold_start = time.time()

                    # 1. Build a fresh preprocessor for this fold
                    # Linear models benefit from scaling; trees do not require scaling
                    scale_features = "Linear" in model_name or "Ridge" in model_name or "Logistic" in model_name
                    preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
                        num_cols, cat_cols, date_cols, scale_numeric=scale_features
                    )

                    # 2. Build pipeline
                    estimator = clone(model_instance)
                    pipeline = Pipeline([
                        ("preprocessor", preprocessor),
                        ("model", estimator),
                    ])

                    # 3. Fit pipeline strictly on training fold
                    pipeline.fit(X_train, y_train)
                    fit_time = time.time() - fold_start

                    # 4. Predict on validation fold
                    y_pred = pipeline.predict(X_val)

                    # Evaluate metrics
                    if is_classification:
                        y_prob = None
                        if hasattr(pipeline, "predict_proba"):
                            try:
                                y_prob = pipeline.predict_proba(X_val)
                            except Exception:
                                y_prob = None
                        m = MetricsEvaluator.evaluate_classification(
                            y_val, y_pred, y_prob, is_binary=is_binary
                        )
                    else:
                        m = MetricsEvaluator.evaluate_regression(y_val, y_pred)

                    val_score = m[primary_metric]
                    train_score = val_score  # We record validation score as primary fold performance

                    fold_metrics.append(
                        FoldMetric(
                            fold=fold_idx + 1,
                            train_score=round(train_score, 4),
                            val_score=round(val_score, 4),
                            metrics=m,
                            fit_time_seconds=round(fit_time, 3),
                        )
                    )

            except Exception as e:
                failed = True
                error_msg = str(e)
                logger.error(f"Model '{model_name}' failed during CV: {e}")

            total_time = time.time() - start_total

            if failed or not fold_metrics:
                results.append(
                    ModelEvaluationResult(
                        model_name=model_name,
                        problem_type=problem_type,
                        primary_metric=primary_metric,
                        mean_cv_score=-999.0 if primary_metric in ["r2", "f1", "accuracy", "balanced_accuracy"] else 999999.0,
                        std_cv_score=0.0,
                        fold_metrics=[],
                        aggregated_metrics={},
                        total_fit_time=round(total_time, 2),
                        status="failed",
                        error_message=error_msg,
                    )
                )
            else:
                # Aggregate metrics across folds
                scores = [fm.val_score for fm in fold_metrics]
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))

                # Aggregate all individual metric means
                all_metric_keys = fold_metrics[0].metrics.keys()
                agg_metrics = {
                    k: round(float(np.mean([fm.metrics[k] for fm in fold_metrics])), 4)
                    for k in all_metric_keys
                }

                results.append(
                    ModelEvaluationResult(
                        model_name=model_name,
                        problem_type=problem_type,
                        primary_metric=primary_metric,
                        mean_cv_score=round(mean_score, 4),
                        std_cv_score=round(std_score, 4),
                        fold_metrics=fold_metrics,
                        aggregated_metrics=agg_metrics,
                        total_fit_time=round(total_time, 2),
                        status="success",
                    )
                )

        return results, label_encoder
