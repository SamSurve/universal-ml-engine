import os
import uuid
import time
import logging
from typing import Union, Optional, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from backend.engine.contracts.schemas import (
    ProblemType,
    ExperimentResult,
    ValidationResult,
    DatasetProfile,
    ProblemDetectionResult,
    LeaderboardEntry,
)
from backend.engine.ingestion.service import IngestionService, DataValidator
from backend.engine.profiler.analyzer import DatasetProfiler
from backend.engine.problem_detection.detector import ProblemDetector
from backend.engine.leakage.leakage_guard import LeakageGuard
from backend.engine.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from backend.engine.models.registry import ModelRegistry
from backend.engine.training.cv_runner import CrossValidationRunner
from backend.engine.evaluation.evaluator import MetricsEvaluator
from backend.engine.artifacts.serializer import ModelArtifact, ArtifactManager

logger = logging.getLogger(__name__)


class AutoMLEngine:
    """
    Unified Orchestrator for the Universal ML Engine.
    Executes the complete end-to-end pipeline:
    Validate -> Clean -> Preprocess -> Detect Problem -> Multi-Model Train -> CV -> Compare -> Select Best -> Holdout Eval -> Save
    """

    @classmethod
    def run(
        cls,
        data_source: Union[str, Path, pd.DataFrame],
        target_column: str,
        output_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        random_state: int = 42,
        n_splits: int = 5,
    ) -> ExperimentResult:
        experiment_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # 1. Ingestion
        if isinstance(data_source, (str, Path)):
            dataset_name = dataset_name or Path(data_source).name
            df_raw = IngestionService.load_dataset(data_source)
        elif isinstance(data_source, pd.DataFrame):
            dataset_name = dataset_name or "in_memory_dataframe"
            df_raw = data_source.copy()
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

        # 2. Validation & Initial Cleaning (record all exclusions)
        df_cleaned, validation_result = DataValidator.validate_and_clean(
            df_raw, target_column=target_column
        )
        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.errors}")

        # 3. Initial Profiling
        initial_profile = DatasetProfiler.profile_dataset(df_cleaned, target_column=target_column)

        # 4. Problem Detection
        problem_detection = ProblemDetector.detect(df_cleaned, target_column=target_column)
        if problem_detection.problem_type == ProblemType.UNKNOWN:
            raise ValueError(
                f"Could not determine problem type for target '{target_column}': {problem_detection.reason}"
            )

        # 5. Leakage Guard & High-Risk Feature Filtering
        df_safe, leakage_exclusions = LeakageGuard.audit_and_filter(
            df_cleaned,
            target_column=target_column,
            problem_type=problem_detection.problem_type,
        )
        validation_result.exclusions.extend(leakage_exclusions)

        # Update profile with safe features
        final_profile = DatasetProfiler.profile_dataset(df_safe, target_column=target_column)

        # 6. Train/Test Split (80% Dev / 20% Untouched Holdout)
        df_dev, df_holdout = CrossValidationRunner.split_dev_holdout(
            df_safe,
            target_column=target_column,
            problem_type=problem_detection.problem_type,
            holdout_ratio=0.20,
            random_state=random_state,
        )

        # 7. Model Candidates Retrieval
        candidate_models = ModelRegistry.get_models(
            problem_type=problem_detection.problem_type,
            random_state=random_state,
        )
        if not candidate_models:
            raise RuntimeError(f"No candidate models available for problem type {problem_detection.problem_type}.")

        # 8. Cross-Validation on Development Data (Strict Zero-Leakage)
        cv_results, label_encoder = CrossValidationRunner.run_cv(
            df_dev=df_dev,
            target_column=target_column,
            problem_type=problem_detection.problem_type,
            models=candidate_models,
            column_profiles=final_profile.column_profiles,
            n_splits=n_splits,
            random_state=random_state,
        )

        # 9. Leaderboard Ranking
        imbalance_ratio = 1.0
        if problem_detection.class_distribution and len(problem_detection.class_distribution) >= 2:
            counts = list(problem_detection.class_distribution.values())
            imbalance_ratio = max(counts) / min(counts)

        primary_metric = MetricsEvaluator.get_primary_metric_name(
            problem_detection.problem_type, class_imbalance_ratio=imbalance_ratio
        )
        leaderboard = MetricsEvaluator.build_leaderboard(cv_results, primary_metric=primary_metric)

        if not leaderboard or leaderboard[0].status != "success":
            raise RuntimeError("All candidate models failed during cross-validation.")

        best_entry = leaderboard[0]
        best_model_name = best_entry.model_name
        best_model_instance = candidate_models[best_model_name]

        # 10. Refit Best Model on 100% of Development Data
        X_dev = df_dev.drop(columns=[target_column])
        y_dev = df_dev[target_column]
        if label_encoder is not None:
            y_dev_encoded = label_encoder.transform(y_dev.astype(str))
        else:
            y_dev_encoded = y_dev.to_numpy(dtype=float)

        num_cols, cat_cols, date_cols = PreprocessingPipelineBuilder.identify_feature_types(
            X_dev, final_profile.column_profiles
        )
        scale_features = "Linear" in best_model_name or "Ridge" in best_model_name or "Logistic" in best_model_name
        refit_preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
            num_cols, cat_cols, date_cols, scale_numeric=scale_features
        )
        final_pipeline = Pipeline([
            ("preprocessor", refit_preprocessor),
            ("model", clone(best_model_instance)),
        ])
        final_pipeline.fit(X_dev, y_dev_encoded)

        # 11. Final Evaluation on Untouched Holdout Data
        X_holdout = df_holdout.drop(columns=[target_column])
        y_holdout = df_holdout[target_column]
        if label_encoder is not None:
            y_holdout_encoded = label_encoder.transform(y_holdout.astype(str))
        else:
            y_holdout_encoded = y_holdout.to_numpy(dtype=float)

        y_holdout_pred = final_pipeline.predict(X_holdout)

        if problem_detection.problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]:
            y_holdout_prob = None
            if hasattr(final_pipeline, "predict_proba"):
                try:
                    y_holdout_prob = final_pipeline.predict_proba(X_holdout)
                except Exception:
                    y_holdout_prob = None
            holdout_metrics = MetricsEvaluator.evaluate_classification(
                y_holdout_encoded,
                y_holdout_pred,
                y_holdout_prob,
                is_binary=(problem_detection.problem_type == ProblemType.BINARY_CLASSIFICATION),
            )
        else:
            holdout_metrics = MetricsEvaluator.evaluate_regression(y_holdout_encoded, y_holdout_pred)

        # 12. Model Artifact Serialization
        artifact = ModelArtifact(
            pipeline=final_pipeline,
            problem_type=problem_detection.problem_type,
            target_column=target_column,
            label_encoder=label_encoder,
            feature_names=list(X_dev.columns),
        )

        run_output_dir = output_dir or f"artifacts/run_{experiment_id}"
        metadata = {
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "problem_type": problem_detection.problem_type.value,
            "confidence": problem_detection.confidence,
            "reason": problem_detection.reason,
            "best_model_name": best_model_name,
            "primary_metric": primary_metric,
            "best_cv_score_mean": best_entry.cv_score_mean,
            "best_cv_score_std": best_entry.cv_score_std,
            "holdout_metrics": holdout_metrics,
            "dev_samples": len(df_dev),
            "holdout_samples": len(df_holdout),
            "features_used": list(X_dev.columns),
            "exclusions_count": len(validation_result.exclusions),
            "exclusions": [
                {"column": ex.column_name, "reason": ex.reason, "action": ex.action}
                for ex in validation_result.exclusions
            ],
            "leaderboard": [
                {
                    "rank": entry.rank,
                    "model": entry.model_name,
                    "cv_score": entry.cv_score_mean,
                    "cv_std": entry.cv_score_std,
                    "metrics": entry.summary_metrics,
                    "fit_time_seconds": entry.fit_time_seconds,
                    "status": entry.status,
                }
                for entry in leaderboard
            ],
            "total_runtime_seconds": round(time.time() - start_time, 2),
            "random_state": random_state,
        }

        saved_model_path, saved_meta_path = ArtifactManager.save_pipeline(
            artifact=artifact,
            metadata=metadata,
            output_dir=run_output_dir,
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            target_column=target_column,
            problem_detection=problem_detection,
            dataset_profile=final_profile,
            validation_result=validation_result,
            dev_shape=df_dev.shape,
            holdout_shape=df_holdout.shape,
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            best_model_cv_score=best_entry.cv_score_mean,
            holdout_metrics=holdout_metrics,
            artifact_path=saved_model_path,
            metadata_path=saved_meta_path,
            random_state=random_state,
        )
